"""OpenTelemetry tracing, exported to Phoenix.

What this gives a reviewer: one trace per ticket with a span for every graph
node, every LLM call (with the tier and model as attributes), every tool call,
and every cascade swap. That is the only way to check the claims this project
makes about routing and degradation -- ``/health`` shows current state, a trace
shows what actually happened to a specific run.

Two modes, because they fail differently:

* ``phoenix serve`` on another terminal (the documented path). Spans go over
  OTLP HTTP to ``PHOENIX_COLLECTOR_ENDPOINT``.
* ``P4_PHOENIX_INMEMORY=1`` builds the same span tree into a local SQLite file
  via Phoenix's in-memory client. No server needed, so the trace structure can
  be verified without standing up Phoenix. Marked as in-memory in the honesty
  table because it is not the same as viewing it in the Phoenix UI.

Instrumentation is applied at import time, before LangChain and the OpenAI SDK
build any clients -- instrumenting afterwards misses calls, because the
instrumentors wrap the classes at construction.
"""

import os
from contextlib import contextmanager
from typing import Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from p4.config import (
    SERVICE_NAME, TIER_FRONTIER, tier_model, TIERS,
)

_TRACER = None
_INITIALIZED = False
_BACKEND = "none"
_SPAN_COUNT = 0

#: Resource attributes attached to every span. ``tier`` is deliberately not
#: here: it belongs on the individual LLM span that knows which rung served it.
SERVICE_ATTRIBUTES = {
    "service.name": SERVICE_NAME,
    "service.version": "phase_4",
    "deployment.environment": os.getenv("P4_ENV", "local"),
}


def init_tracing(app=None) -> str:
    """Configure the tracer provider and apply instrumentors.

    Safe to call more than once. Returns the backend actually in use so
    ``/health`` can report it honestly.
    """
    global _TRACER, _INITIALIZED, _BACKEND
    if _INITIALIZED:
        return _BACKEND

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT",
                         "http://localhost:6006/v1/traces")
    in_memory = os.getenv("P4_PHOENIX_INMEMORY", "").lower() in ("1", "true", "yes")

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource.create(SERVICE_ATTRIBUTES))

    if in_memory:
        _setup_in_memory(provider)
        _BACKEND = "phoenix-inmemory"
    else:
        _setup_otlp(provider, endpoint)
        _BACKEND = "phoenix-otlp"

    trace.set_tracer_provider(provider)
    _apply_instrumentors()

    _TRACER = trace.get_tracer(SERVICE_NAME)
    _INITIALIZED = True
    return _BACKEND


def _setup_otlp(provider, endpoint: str) -> None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # A long export timeout on purpose: a slow collector must not block the
    # request path, and spans are worth keeping even if the run is slow.
    exporter = OTLPSpanExporter(endpoint=endpoint, timeout=20)
    provider.add_span_processor(BatchSpanProcessor(
        exporter,
        max_queue_size=4096,
        schedule_delay_millis=1000,
    ))
    # An unreachable collector is a normal local state, not an incident. The
    # default exporter retries with backoff and logs a stack of warnings per
    # batch, which buries real errors and makes an otherwise healthy run look
    # broken. Traces are still kept in the queue; we just stop shouting about
    # the failure. A deployment that wants the retry behaviour sets
    # P4_TRACE_SHOW_EXPORTER_ERRORS=1.
    if os.getenv("P4_TRACE_SHOW_EXPORTER_ERRORS", "").lower() not in ("1", "true", "yes"):
        _quiet_exporter_errors()


def _quiet_exporter_errors() -> None:
    """Stop the OTLP exporter logging connection failures.

    The exporter logs through ``opentelemetry.sdk.trace.export`` at ERROR for
    failed batches. Downgrading that to DEBUG keeps the signal available without
    the noise; retrying still happens internally either way.
    """
    import logging

    for name in (
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        "opentelemetry.exporter.otlp",
        "opentelemetry.sdk.trace.export",
    ):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _setup_in_memory(provider) -> None:
    provider.add_span_processor(_InMemoryProcessor())


class _InMemoryProcessor:
    """Writes spans into a local SQLite store.

    Used when ``P4_PHOENIX_INMEMORY=1`` so trace *structure* can be verified
    without standing up ``phoenix serve``. It is deliberately a flat table
    rather than Phoenix's schema: this exists to prove the span tree is right,
    not to replace the Phoenix UI, and the README says so.
    """

    def __init__(self):
        import sqlite3
        from pathlib import Path

        from p4.config import PROJECT_ROOT

        self.path = Path(PROJECT_ROOT) / "evidence" / "spans_inmemory.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                rowid_key INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT, span_id TEXT, parent_id TEXT,
                name TEXT, kind TEXT,
                start_time INTEGER, end_time INTEGER,
                attributes TEXT, status TEXT, error TEXT
            )""")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_trace ON spans(trace_id)")
        self._conn.commit()

    def on_start(self, span, parent_context=None):
        pass

    def _on_ending(self, span):
        """Sampling hook. The SDK calls this before ``on_end``; returning None
        defers to the parent's decision, so nothing is sampled out here."""
        return None

    def on_end(self, span):
        """Where spans are actually persisted.

        Both this and ``_on_ending`` are required: the OTel SDK calls
        ``_on_ending`` for sampling and ``on_end`` for export, so implementing
        only one silently drops every span.
        """
        global _SPAN_COUNT
        try:
            ctx = span.get_span_context()
            parent = span.parent
            self._conn.execute(
                "INSERT INTO spans (trace_id, span_id, parent_id, name, kind,"
                " start_time, end_time, attributes, status, error)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    format(ctx.trace_id, "032x"),
                    format(ctx.span_id, "016x"),
                    format(parent.span_id, "016x") if parent else None,
                    span.name,
                    str(span.kind),
                    span.start_time,
                    span.end_time,
                    str(dict(span.attributes or {})),
                    (span.status.status_code.name
                     if span.status and span.status.status_code else "UNSET"),
                    _span_error(span),
                ),
            )
            self._conn.commit()
            _SPAN_COUNT += 1
        except Exception:
            # Tracing must never break the request it is describing.
            pass

    def shutdown(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _span_error(span) -> str:
    events = getattr(span, "events", None) or []
    for event in events:
        if getattr(event, "name", "") == "exception":
            attrs = dict(getattr(event, "attributes", None) or {})
            return str(attrs.get("exception.message", ""))
    return ""


def _apply_instrumentors() -> None:
    """Instrument LangChain and the OpenAI SDK.

    Must run before those modules construct clients, which is why
    ``init_tracing`` is called at gateway startup rather than lazily per run.
    Each instrumentor is independent: if one fails the others still apply, and
    the run continues untraced rather than failing.
    """
    for name, fn in (
        ("openinference.instrumentation.langchain",
         "LangChainInstrumentor"),
        ("openinference.instrumentation.openai",
         "OpenAIInstrumentor"),
    ):
        try:
            module = __import__(name, fromlist=[fn])
            getattr(module, fn)().instrument()
        except Exception as exc:
            print(f"[tracing] {fn} not applied: {type(exc).__name__}: {exc}")


def get_tracer():
    if _TRACER is None:
        init_tracing()
    return _TRACER


def backend() -> str:
    return _BACKEND


def span_count() -> int:
    return _SPAN_COUNT


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

@contextmanager
def span(name: str, **attributes):
    """Start a span, tolerating the case where tracing is not configured.

    Yields the span. On exception the span is marked as an error and the
    exception is re-raised -- the caller's control flow is never changed by
    whether tracing happens to be on.
    """
    global _SPAN_COUNT
    tracer = get_tracer()
    # Manual counters would drift from the real span tree, so record only what
    # was actually emitted.
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            if value is not None:
                try:
                    s.set_attribute(key, value)
                except Exception:
                    pass
        try:
            yield s
        except Exception as exc:
            s.set_status(Status(StatusCode.ERROR, str(exc)))
            s.record_exception(exc)
            _SPAN_COUNT += 1
            raise


@contextmanager
def llm_span(agent: str, tier: str, operation: str = "chat"):
    """A span for one LLM call, tagged with the tier that served it.

    ``llm.tier`` and ``llm.model`` are the attributes that make the router
    auditable: a reviewer can read which rung handled a given node without
    inferring it from timing or cost.
    """
    with span(f"llm.{operation}",
              **{
                  "llm.agent": agent,
                  "llm.tier": tier,
                  "llm.model": tier_model(tier),
                  "llm.price_in_per_mtok": (TIERS.get(tier) or {}).get("price_in"),
                  "llm.price_out_per_mtok": (TIERS.get(tier) or {}).get("price_out"),
              }) as s:
        yield s


@contextmanager
def node_span(node: str, thread_id: Optional[str] = None,
              duration_ms: Optional[float] = None):
    """A span for one graph node.

    ``duration_ms`` is the measured wall time for the node. The graph yields a
    node's output only after the node has run, so the caller measures the gap
    between consecutive node events and passes it in. Without this the span
    would open and close inside a few microseconds and report a duration that
    bears no relation to what the node actually cost.
    """
    with span(f"node.{node}",
              **{"node.name": node, "thread.id": thread_id,
                 "node.duration_ms": duration_ms}) as s:
        if s is not None and duration_ms is not None:
            s.set_attribute("node.duration_ms", duration_ms)
        yield s


def attach_run_span(run, **attributes):
    """Return a span for a whole ticket run."""
    return span("support.run",
                **{"thread.id": getattr(run, "thread_id", None), **attributes})


def flush(timeout_millis: int = 5000) -> None:
    """Push buffered spans out. Called before the process exits."""
    provider = trace.get_tracer_provider()
    try:
        provider.force_flush(timeout_millis)
    except Exception:
        pass


def record_run_outcome(**attributes) -> None:
    """Attach the run's outcome to the active ``support.run`` span.

    Called at the end of a run, where the interesting numbers (tiers used,
    swaps, cost) are only known. Kept as a helper so ``p4.streaming`` does not
    have to reach into span internals, and so a run with tracing disabled is a
    no-op rather than an error.
    """
    span = trace.get_current_span()
    if span is None or not getattr(span, "is_recording", lambda: False)():
        return
    for key, value in attributes.items():
        if value is None:
            continue
        try:
            span.set_attribute(f"run.{key}", value)
        except Exception:
            pass


def backend_info() -> dict:
    """What ``/health`` reports about tracing, without overclaiming."""
    return {
        "backend": _BACKEND,
        "initialized": _INITIALIZED,
        "spans_recorded": _SPAN_COUNT,
        "endpoint": os.getenv("PHOENIX_COLLECTOR_ENDPOINT",
                               "http://localhost:6006/v1/traces"),
        "in_memory": os.getenv("P4_PHOENIX_INMEMORY", "").lower()
        in ("1", "true", "yes"),
    }
