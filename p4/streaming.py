"""Turns the Phase 3 graph into a token-streaming SSE source.

The Phase 3 ``/api/stream`` emitted whole-node events only -- nothing arrived
incrementally, so a client could not tell progress from a stall. This module
produces genuinely incremental output: node boundaries *and* per-token deltas.

The shape of the problem is that the two sources iterate differently.
``graph.stream()`` is a sync iterator over node completions, while token
deltas come from ``llm.stream()`` inside a node. Bridging them means pulling
both onto one thread rather than trying to interleave two async iterators.

Design:

* ``stream_run`` runs the graph on a worker thread and drains its event queue,
  yielding typed events in real time.
* The final answer is generated through :class:`~p4.cascade.CascadeRunner`, so
  the streamed prose carries per-token events *and* is the thing the cascade
  can hot-swap. Node events come from the graph; token events come from the
  cascade.
* The graph runs with ``synthesizer`` skipped -- otherwise its non-streaming
  ``llm.invoke`` would duplicate the answer the cascade is already streaming.
  The streamed synthesis is written back into state so the checkpointer and
  the HITL/forensics paths still see a complete result.
* A cache miss is checked before the graph runs and a completed answer is stored
  afterwards, so a repeat ticket can be served without any model call.
* Each run is graded by :mod:`p4.judge` and measured into :mod:`p4.monitor`, so
  the correctness and latency SLOs are fed by real traffic rather than by
  hand-entered numbers.
"""

import queue
import threading
import time
import uuid
from typing import Iterator, List, Optional

from p4 import events as ev
from p4.cascade import CascadeRunner, make_llm
from p4.config import CASCADE_ORDER, tier_model
from p4.breaker import get_breaker_registry
from p4.router import get_router
from p4 import tracing

_STOP = object()

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}


class StreamRun:
    """Drives one ticket to completion, yielding typed events as they happen."""

    def __init__(
        self,
        thread_id: Optional[str] = None,
        skip_synthesis: bool = True,
        ladder: Optional[List[str]] = None,
        use_cache: bool = True,
        grade: bool = True,
    ):
        self.thread_id = thread_id or str(uuid.uuid4())
        self.skip_synthesis = skip_synthesis
        self.ladder = list(ladder or CASCADE_ORDER)
        # The shared registry, not a per-run one: a tier that just failed must
        # stay marked for the *next* request too.
        self.breakers = get_breaker_registry(self.ladder)
        self.started = time.monotonic()
        self.use_cache = use_cache
        self.grade = grade

        self.tiers_used: List[str] = []
        self.swap_count = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0
        self.node_count = 0
        self.errored = False
        self.cache_hit = False
        #: Findings the streamed answer was synthesized from, captured so the
        #: judge and cache store run against the same evidence the answer used.
        self._findings: list = []
        self.judge_scores: dict = {}
        #: Accumulated deltas, used if the cascade result loses its text.
        self._text: List[str] = []

    # -- timing -------------------------------------------------------------

    def _elapsed_ms(self) -> float:
        return (time.monotonic() - self.started) * 1000.0

    # -- node events from the graph ----------------------------------------

    def _produce(self, ticket: str, out: "queue.Queue") -> None:
        """Run the whole run on a worker thread, pushing every event onto ``out``.

        Both producers are synchronous and blocking -- ``graph.stream`` and
        ``CascadeRunner.stream`` -- so they share one thread and the consumer
        stays responsive. The order matters: the graph's findings are the
        cascade's prompt, so synthesis cannot start until the graph finishes.

        Everything is wrapped in a root ``support.run`` span so one trace per
        ticket contains its nodes, tool calls, and cascade tiers.
        """
        import support_agent as sa

        with tracing.span("support.run",
                          **{"thread.id": self.thread_id,
                             "ticket.length": len(ticket),
                             "cascade.ladder": ",".join(self.ladder)}) as root:
            if root is not None:
                # Do not put the raw ticket in the span: it can contain customer
                # data, and traces are exported off-box.
                root.set_attribute("ticket.preview", ticket[:80])
            self._produce_inner(ticket, out, sa)

    def _produce_inner(self, ticket: str, out: "queue.Queue", sa) -> None:
        """Body of :meth:`_produce`, split out so the root span wraps it."""

        config = {"configurable": {"thread_id": self.thread_id}}
        snapshot = sa.graph.get_state(config)
        is_followup = bool(snapshot and snapshot.values
                           and snapshot.values.get("messages"))
        if is_followup:
            payload = {"messages": [sa.HumanMessage(content=ticket)],
                       "raw_input": ticket}
        else:
            payload = sa.build_initial_state(ticket)
        # PHASE 4: hand synthesis to the gateway so the answer streams token by
        # token instead of arriving as one non-streamed block.
        payload["defer_synthesis"] = bool(self.skip_synthesis)

        # Timestamp of the previous node event. See the timing note below: the
        # gap between consecutive events is a completed node's duration.
        last_event_at = time.monotonic()
        try:
            for event in sa.graph.stream(payload, config=config, subgraphs=True):
                path, inner = (event if isinstance(event, tuple) and len(event) >= 2
                               else (None, event))
                if not isinstance(inner, dict):
                    continue
                for node, node_out in inner.items():
                    if node == "__interrupt__":
                        draft = {}
                        try:
                            payload_v = node_out
                            if isinstance(payload_v, (list, tuple)) and payload_v:
                                v = payload_v[0]
                                draft = getattr(v, "value", v)
                            elif isinstance(payload_v, dict):
                                draft = payload_v
                            if isinstance(draft, dict):
                                draft = draft.get("draft", {})
                        except Exception:
                            draft = {}
                        out.put(ev.Interrupt(
                            thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                            draft=draft if isinstance(draft, dict) else {},
                            instruction="Approve, deny, or edit this GitHub issue creation.",
                        ))
                        continue

                    # Node boundary events.
                    #
                    # `graph.stream()` yields a node's *output* after the node
                    # has already finished, so the wall time between two
                    # consecutive node events is what that node cost. Timing it
                    # the obvious way -- stamping a start when the event
                    # arrives and a stop immediately after -- reports ~0ms for
                    # every node, which is worse than useless: it looks like
                    # measured data and points at the wrong culprit when
                    # someone is trying to find the slow step.
                    label = _node_label(path, node)
                    now = time.monotonic()
                    duration_ms = max(0.0, (now - last_event_at)) * 1000.0
                    last_event_at = now

                    out.put(ev.NodeStart(
                        thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                        node=label,
                    ))

                    tool_calls = _tool_calls(node_out) if isinstance(node_out, dict) else []
                    with tracing.node_span(label, self.thread_id,
                                          duration_ms=duration_ms) as nspan:
                        if nspan is not None and isinstance(node_out, dict):
                            nspan.set_attribute("node.category",
                                                str(node_out.get("category", "")))
                            nspan.set_attribute("node.tool_calls", len(tool_calls))
                        for tc in tool_calls:
                            out.put(ev.ToolCall(
                                thread_id=self.thread_id,
                                elapsed_ms=self._elapsed_ms(),
                                name=tc.get("name", ""),
                                args=tc.get("args", {}) or {},
                            ))
                            with tracing.span(
                                f"tool.{tc.get('name', 'unknown')}",
                                **{"tool.name": tc.get("name", ""),
                                   "tool.args": str(tc.get("args", {}))[:200]}):
                                pass

                    self.node_count += 1
                    out.put(ev.NodeEnd(
                        thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                        node=label,
                        duration_ms=duration_ms,
                        preview=_preview(node_out),
                    ))

            # ---- synthesis, streamed token by token through the cascade ----
            self._synthesize(ticket, out)
        except Exception as exc:
            out.put(exc)
        finally:
            out.put(_STOP)

    def _synthesize(self, ticket: str, out: "queue.Queue") -> None:
        """Stream the final answer through the cascade, one token at a time.

        ``CascadeRunner.stream`` is a blocking sync generator consumer, so it
        runs here on the worker thread; each token it produces is pushed to
        ``out`` immediately, which is what makes the client see incremental
        output instead of one buffered block at the end.
        """
        findings, context = self._collect_findings()
        # Kept on the run so the judge and the cache store see the findings the
        # answer was built from, not just the prose.
        self._findings = findings

        if context.get("suspended"):
            # The graph stopped at the GitHub approval gate. There is nothing to
            # synthesize, and answering here would bypass the human -- the run
            # has not concluded, it is waiting. This takes precedence over any
            # findings already collected: the outage agent may have produced a
            # critical finding, and treating that as a finished analysis would
            # both leak a provisional answer and pre-empt the approval.
            out.put(ev.RunEnd(
                thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                final_response="",
                category=context.get("category", ""),
                tiers_used=[],
                swap_count=0,
                is_followup=bool(context.get("is_followup")),
                awaiting_human=True,
            ))
            return

        # Terminal answers the graph already produced without an LLM call:
        # a blocked request, a filed/denied GitHub issue, or a
        # max-delegations notice. These are the graph's words, not something the
        # cascade should be asked to re-generate -- and re-generating a security
        # refusal would be actively wrong.
        terminal = context.get("terminal_response")
        if terminal:
            for piece in _word_pieces(terminal):
                out.put(ev.Token(
                    thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                    delta=piece, tier="local", model="pre-computed",
                ))
                self.tokens_out += 1
            out.put(ev.RunEnd(
                thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                final_response=terminal,
                category=context.get("category", ""),
                tiers_used=["local"],
                swap_count=0,
                tokens_in=0, tokens_out=self.tokens_out,
            ))
            return

        if not findings:
            # The graph completed but recorded no severity-bearing notes (for
            # example a blocked or unsafe input). Mirror the Phase 3 fallback.
            findings = [{"agent": "system", "finding": "Analysis complete.",
                         "severity": "low", "action": "none"}]

        messages = _synthesis_messages(findings, context.get("raw_input", ticket))

        def on_token(delta, tier, model):
            self._text.append(delta)
            self.tokens_out += 1
            if tier not in self.tiers_used:
                self.tiers_used.append(tier)
            out.put(ev.Token(
                thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                delta=delta, tier=tier, model=model,
            ))

        def on_swap(swap):
            self.swap_count += 1
            swap.thread_id = self.thread_id
            swap.elapsed_ms = self._elapsed_ms()
            out.put(swap)

        # The whole cascade is one span; each tier attempt is a child, so
        # Phoenix shows a frontier span and a standard span under the same
        # trace. That nesting is the evidence for the hot-swap claim.
        with tracing.span("cascade.stream",
                          **{"cascade.ladder": ",".join(self.ladder),
                             "thread.id": self.thread_id}) as parent:
            if parent is not None:
                for key in ("cascade.plan",):
                    parent.set_attribute(key, list(self.ladder))

            def on_tier_start(tier, model):
                return tracing.llm_span("synthesizer", tier)

            # The runner is built inside the cascade span so each tier's span
            # nests as a child of it -- that nesting is the evidence for the
            # hot-swap claim in Phoenix. The callback is defined here for the
            # same reason: it has to be available before the runner is built.
            runner = CascadeRunner(self.ladder, self.breakers,
                                   llm_factory=make_llm,
                                   on_tier_start=on_tier_start)

            result = runner.stream(messages, agent="synthesizer",
                                   emit=on_token, on_swap=on_swap)

        final = result.text or "".join(self._text)
        self.tokens_in = result.prompt_tokens
        self.cost_usd = result.cost_usd
        self._persist_synthesis(final)

        if result.exhausted:
            out.put(ev.RunError(
                thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                message=result.error or "cascade exhausted",
                tiers_tried=list(result.tiers_used),
            ))
            out.put(ev.RunEnd(
                thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                final_response=final,
                category=context.get("category", ""),
                tiers_used=list(result.tiers_used),
                swap_count=self.swap_count,
                total_cost_usd=self.cost_usd,
                tokens_in=self.tokens_in, tokens_out=self.tokens_out,
                degraded=True,
            ))
            return

        out.put(ev.RunEnd(
            thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
            final_response=final,
            category=context.get("category", ""),
            tiers_used=list(result.tiers_used),
            swap_count=self.swap_count,
            total_cost_usd=self.cost_usd,
            tokens_in=self.tokens_in, tokens_out=self.tokens_out,
            degraded=self.swap_count > 0,
        ))

        tracing.record_run_outcome(
            thread_id=self.thread_id,
            category=context.get("category", ""),
            tiers_used=list(result.tiers_used),
            swap_count=self.swap_count,
            cost_usd=self.cost_usd,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            degraded=self.swap_count > 0,
            exhausted=result.exhausted,
        )

    # -- public API ---------------------------------------------------------

    def run(self, ticket: str) -> Iterator[ev.BaseEvent]:
        """Yield typed events for ``ticket``, ending with run.end or run.error."""
        router = get_router()
        primary = router.resolve("synthesizer")

        # --- cache lookup, before any model call -------------------------
        cache = None
        lookup = None
        if self.use_cache:
            from p4.cache import get_cache, record_to_monitor

            cache = get_cache()
            lookup = cache.lookup(ticket, primary["tier"],
                                  prompt_tokens=len(ticket) // 4)
            record_to_monitor(lookup.hit)
            if lookup.hit:
                self.cache_hit = True
                self.cost_usd = 0.0
                yield ev.RunStart(
                    thread_id=self.thread_id,
                    elapsed_ms=self._elapsed_ms(),
                    cascade_plan=[],
                    primary_tier=lookup.tier,
                    primary_model=tier_model(lookup.tier),
                    cache_bypassed=False,
                )
                yield ev.CacheHit(
                    thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                    similarity=lookup.similarity,
                    served_tier=lookup.tier,
                    saved_usd=lookup.estimated_savings_usd,
                )
                for piece in _word_pieces(lookup.response):
                    yield ev.Token(
                        thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                        delta=piece, tier="cache", model="semantic-cache",
                    )
                self._record_metrics(ticket, lookup.response, [],
                                     latency_ms=self._elapsed_ms(),
                                     errored=False, degraded=False)
                yield ev.RunEnd(
                    thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                    final_response=lookup.response,
                    category="cached",
                    tiers_used=["cache"],
                    swap_count=0,
                    total_cost_usd=0.0,
                    saved_usd=lookup.estimated_savings_usd,
                    cache_hit=True,
                )
                return

        yield ev.RunStart(
            thread_id=self.thread_id,
            elapsed_ms=self._elapsed_ms(),
            cascade_plan=list(self.ladder),
            primary_tier=primary["tier"],
            primary_model=primary["model"],
            cache_bypassed=True,
        )
        if lookup is not None:
            yield ev.CacheMiss(
                thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                similarity=lookup.similarity, reason=lookup.reason,
            )

        out: "queue.Queue" = queue.Queue(maxsize=512)
        worker = threading.Thread(
            target=self._produce, args=(ticket, out), daemon=True
        )
        worker.start()

        failure = None
        final_answer = ""
        while True:
            try:
                item = out.get(timeout=120)
            except queue.Empty:
                failure = RuntimeError("run produced no events for 120s")
                break
            if item is _STOP:
                break
            if isinstance(item, Exception):
                failure = item
                continue
            if isinstance(item, ev.RunEnd):
                final_answer = item.final_response
            yield item

        if failure is not None:
            self.errored = True
            self._record_metrics(ticket, final_answer, self._findings,
                                 latency_ms=self._elapsed_ms(), errored=True,
                                 degraded=self.swap_count > 0)
            yield ev.RunError(
                thread_id=self.thread_id, elapsed_ms=self._elapsed_ms(),
                message=f"{type(failure).__name__}: {failure}",
            )
            return

        # Judge, cache, and SLO sampling all happen after the last frame, so
        # none of them can delay or corrupt the answer the client already has.
        self._record_metrics(ticket, final_answer, self._findings,
                             latency_ms=self._elapsed_ms(), errored=False,
                             degraded=self.swap_count > 0)

    # -- helpers ------------------------------------------------------------

    def _record_metrics(self, ticket: str, answer: str, findings: list,
                        *, latency_ms: float, errored: bool,
                        degraded: bool) -> None:
        """Grade the answer, cache it, and feed the SLO tracker.

        Every step is best-effort and separately guarded: a judge outage, an
        embedding failure, or a SQLite error must not change what the customer
        receives, which was already sent by the time this runs.
        """
        from p4.monitor import get_monitor

        scores: dict = {}
        if self.grade and answer and not errored:
            try:
                from p4.judge import get_judge, record_to_monitor

                result = get_judge().grade(ticket, findings, answer)
                scores = {
                    "correctness": result.correctness,
                    "safety": result.safety,
                    "tone": result.tone,
                }
                self.judge_scores = {k: v for k, v in scores.items() if v is not None}
                record_to_monitor(result)
            except Exception as exc:
                print(f"[judge] grading failed: {type(exc).__name__}: {exc}")

        if self.use_cache and answer and not errored and not self.cache_hit:
            try:
                from p4.cache import get_cache

                # A suspended run's empty answer is never stored: a later,
                # different ticket must not be served "still awaiting approval".
                get_cache().store(ticket, answer,
                                  self.tiers_used[-1] if self.tiers_used else "frontier",
                                  self.cost_usd, findings)
            except Exception as exc:
                print(f"[cache] store failed: {type(exc).__name__}: {exc}")

        try:
            get_monitor().tracker.record_run(
                latency_ms=latency_ms, errored=errored, degraded=degraded,
                cost_usd=self.cost_usd, judge_scores=self.judge_scores,
            )
            if self.swap_count:
                get_monitor().tracker.record_breaker_event("cascade_swap")
        except Exception as exc:
            print(f"[monitor] record failed: {type(exc).__name__}: {exc}")

    def _collect_findings(self) -> tuple:
        """Read the findings the graph produced, for the streamed synthesis."""
        import support_agent as sa

        config = {"configurable": {"thread_id": self.thread_id}}
        try:
            snapshot = sa.graph.get_state(config)
        except Exception:
            return [], {}
        if not snapshot or not snapshot.values:
            return [], {}
        values = snapshot.values
        #: The graph's own pending-node tuple is the authoritative HITL signal.
        #: `next_worker` is the supervisor's routing field and is empty on the
        #: hub-and-spoke outage path, so it cannot be used to detect a suspend.
        pending = tuple(snapshot.next or ())

        notes = values.get("internal_notes", []) or []
        # No placeholder here on purpose: the caller must be able to tell
        # "analysis found nothing" from "the graph stopped at an approval gate
        # and never produced findings". A synthetic finding would make a
        # suspended run look like a successful analysis.
        findings = [n for n in notes
                    if isinstance(n, dict) and "severity" in n
                    and n.get("severity") != "none"]
        findings.sort(key=lambda f: _SEVERITY_ORDER.get(f.get("severity", "none"), 5))
        context = {
            "category": values.get("category", ""),
            "raw_input": values.get("raw_input", ""),
            "github_issue_url": values.get("github_issue_url", ""),
            "github_draft": values.get("github_draft", {}),
            "denied": any(isinstance(n, dict) and n.get("type") == "github_denied"
                          for n in notes),
            "system_summary": values.get("system_summary", ""),
            "is_followup": bool(values.get("messages")),
            "suspended": "github_tool_node" in pending,
            "terminal_response": _terminal_response(values, notes),
        }
        return findings, context

    def _persist_synthesis(self, final: str) -> None:
        """Write the streamed answer back so checkpoints and forensics see it."""
        import support_agent as sa

        if not final:
            return
        config = {"configurable": {"thread_id": self.thread_id}}
        try:
            sa.graph.update_state(config, {
                "final_response": final,
                "messages": [sa.AIMessage(content=final)],
                "internal_notes": [{"type": "streamed_synthesis", "source": "p4.sse"}],
            })
        except Exception as exc:
            # Non-fatal: the client already has the answer.
            print(f"[stream] could not persist synthesis: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthesis_messages(findings, raw_input) -> list:
    """Build the cascade's message list from the graph's findings.

    Same content the Phase 3 ``synthesizer_node`` would have sent, so a streamed
    answer is not a different answer than the synchronous path produces.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    import json as _json

    return [
        SystemMessage(content=(
            "Synthesize these findings into one clear, actionable response for "
            "the developer. Be concise and concrete. Do not invent facts that are "
            "not present in the findings."
        )),
        HumanMessage(content=(
            f"Findings:\n{_json.dumps(findings, indent=2, default=str)}\n\n"
            f"Original issue: {raw_input}"
        )),
    ]


def _terminal_response(values, notes):
    """Return a non-empty answer the graph produced without an LLM call.

    Several outcomes bypass synthesis by design -- a blocked request, a filed or
    denied GitHub issue, a supervisor that hit its delegation limit. Handing
    these to the cascade would re-generate (and soften) a security refusal or
    report a GitHub URL the graph has not actually created, so they are passed
    through verbatim.
    """
    note_types = {n.get("type") for n in notes if isinstance(n, dict)}

    if "injection_detected" in note_types or "blocked" in note_types:
        final = values.get("final_response") or ""
        if final:
            return final
    if "github_denied" in note_types:
        return "GitHub issue creation was denied. The issue has not been filed."
    if "max_delegations_reached" in note_types:
        return (values.get("final_response")
                or "Reached the maximum number of delegations for this ticket.")
    if "github_issue_approved" in note_types and values.get("github_issue_url"):
        return f"GitHub issue created: {values['github_issue_url']}"
    if "github_issue" in note_types and values.get("github_issue_url"):
        return f"GitHub issue created: {values['github_issue_url']}"
    return ""


def _word_pieces(text: str, width: int = 6):
    """Split ``text`` into small pieces so a pre-computed answer still streams.

    Emits a handful of real tokens rather than one frame, so a client sees the
    same incremental shape whether the text came from a provider or the graph.
    """
    words = text.split(" ")
    buf: list = []
    for word in words:
        buf.append(word)
        if len(buf) >= width:
            yield " ".join(buf) + " "
            buf = []
    if buf:
        tail = " ".join(buf)
        yield tail


_SUBGRAPH_NODES = {
    "ingress", "classify", "blocked", "summarize", "agent", "tools",
    "respond", "egress",
}


def _node_label(path, node) -> str:
    """Label a node, dropping the subgraph's opaque task id.

    ``graph.stream(subgraphs=True)`` reports subgraph runs as
    ``triage:<task-uuid>`` -- that id is different on every run, so including it
    would make node names unusable as a timeline key. Names are therefore
    normalized to ``triage.ingress`` / ``dev_support.agent``, and the
    subgraph's inner node names are kept apart from master-graph node names
    (a master ``classify`` does not exist today, but the rule should not depend
    on that).
    """
    if not path:
        return node

    parts = []
    for element in (path if isinstance(path, tuple) else (path,)):
        if isinstance(element, tuple):
            parts.extend(str(e) for e in element)
        else:
            parts.append(str(element))

    for part in parts:
        part = part.split(":", 1)[0]
        if part:
            return f"{part}.{node}"
    return node


def _tool_calls(node_out: dict) -> list:
    """Pull tool calls out of a node's state update."""
    calls = []
    for msg in node_out.get("messages", []) or []:
        for tc in getattr(msg, "tool_calls", None) or []:
            calls.append(tc)
    return calls


def _preview(node_out: dict, limit: int = 240) -> str:
    """A short human-readable summary of what a node produced."""
    if not isinstance(node_out, dict):
        return ""
    for key in ("final_response", "category", "next_worker"):
        if key in node_out and node_out[key]:
            return str(node_out[key])[:limit]
    notes = node_out.get("internal_notes")
    if isinstance(notes, list) and notes:
        last = notes[-1]
        if isinstance(last, dict):
            return str(last.get("finding") or last.get("type") or "")[:limit]
    return ""


def sse_from_events(events: Iterator[ev.BaseEvent]) -> Iterator[str]:
    """Render typed events as SSE frames."""
    for event in events:
        yield ev.to_sse(event)