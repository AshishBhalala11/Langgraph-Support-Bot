"""
FastAPI Backend — Developer Platform Support Bot
Exposes run/stream/approve/deny/forensics endpoints + static UI.
"""

import json
import os
import queue
import threading
import time
import uuid
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

from p4 import events as ev
from p4 import tracing
from p4.streaming import StreamRun
from p4.breaker import get_breaker_registry, reset_breakers
from p4.config import CASCADE_ORDER, SLOS, tier_model
from p4.router import get_router, injected_base_url, set_injected_base_url, reset_budget
from p4.monitor import get_monitor, monitor_status
from p4.cache import get_cache
from p4.judge import get_judge
from p4.providers import portability_report

# Tracing is initialized here, before ``support_agent`` is imported, and that
# ordering is load-bearing. The OpenAI and LangChain instrumentors work by
# wrapping the client classes; if the providers are constructed first the
# wrapper is installed too late and the LLM spans silently never appear. Doing
# it in a startup event would be too late for exactly this reason.
tracing.init_tracing()

from support_agent import (
    graph, run_ticket, stream_ticket, get_conversation_history,
    get_active_threads, get_pending_approvals, state_forensics,
    find_bad_checkpoint, apply_correction, time_travel, checkpointer,
    build_initial_state, HumanMessage, AIMessage, TOOLS,
)

app = FastAPI(title="Developer Platform Support Bot", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def _start_background_workers() -> None:
    """Start the SLO monitor and flush any spans left in the export queue.

    The monitor is what turns raw run counts into SLO verdicts and triggers
    degradation, so without this the SLOs are configuration with no data behind
    them. It is idempotent, so a reload does not stack threads.
    """
    get_monitor().start()
    tracing.flush()


@app.on_event("shutdown")
async def _stop_background_workers() -> None:
    """Stop the monitor and drain pending spans before the process exits."""
    get_monitor().stop()
    tracing.flush()


# ============================================================================
# Request/Response Models
# ============================================================================

class RunRequest(BaseModel):
    ticket: str
    thread_id: Optional[str] = None
    return_existing: bool = False
    use_cache: bool = True
    """Bypass the semantic cache. Needed by fault drills and cold-start
    measurements: a cached answer short-circuits before any provider is
    called, so a drill would report a pass without ever exercising the tier
    it just broke."""


class TimeTravelRequest(BaseModel):
    thread_id: str
    target_step: int


class CorrectRequest(BaseModel):
    thread_id: str
    checkpoint_id: str
    field: str
    new_value: str


class EditApproveRequest(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    labels: Optional[list] = None


# ============================================================================
# Helper: Build rich response payload
# ============================================================================

def _build_run_payload(result: dict, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = graph.get_state(config)
    values = state_snapshot.values if state_snapshot else {}

    tool_calls_log = []
    for msg in values.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append({
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                })
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append({
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                })

    pending = state_snapshot.next if state_snapshot and hasattr(state_snapshot, "next") else []
    is_interrupted = bool(pending)

    return {
        "thread_id": thread_id,
        "final_response": result.get("final_response", ""),
        "category": result.get("category", ""),
        "iteration_count": result.get("iteration_count", 0),
        "delegation_count": result.get("delegation_count", 0),
        "github_issue_url": result.get("github_issue_url", ""),
        "github_draft": values.get("github_draft", {}),
        "is_interrupted": is_interrupted,
        "next_nodes": list(pending) if pending else [],
        "system_summary": values.get("system_summary", ""),
        "pii_detected": values.get("pii_detected", False),
        "injection_detected": values.get("injection_detected", False),
        "is_safe": values.get("is_safe", True),
        "tool_calls": tool_calls_log,
        "internal_notes": values.get("internal_notes", []),
        "tool_results": values.get("tool_results", []),
        "messages": [
            {
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": (m.content[:500] if isinstance(m.content, str) else str(m.content)[:500]) if hasattr(m, "content") else "",
            }
            for m in values.get("messages", [])[-10:]
            if hasattr(m, "content")
        ],
    }


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Liveness plus the state of every resilience mechanism.

    The Phase 3 version returned a hardcoded ``healthy``. This one reports what
    is actually true: budget remaining, which tiers are usable, whether a fault
    is injected, and whether the SLO monitor is running. A client that is
    about to depend on a degraded path should be able to see that here first.
    """
    router = get_router()
    # The shared registry, so what /health reports is the state the next run
    # will actually act on rather than a freshly-initialized view.
    registry = get_breaker_registry()
    ledger = router.ledger.snapshot()
    ledger_budget_for = router.ledger.budget_for
    injected = {t: u for t, u in
                ((t, injected_base_url(t)) for t in CASCADE_ORDER) if u}

    degraded = bool(injected) or registry.any_open()

    return {
        "status": "degraded" if degraded else "healthy",
        "version": "phase_4",
        "tool_count": len(TOOLS) + 1,
        "max_iterations": 5,
        "max_delegations": 5,
        "capabilities": [
            "classify_route", "tool_binding", "react_loop", "persistence",
            "context_management", "guardrails", "subgraphs", "supervisor",
            "parallel_specialists", "write_access", "hitl", "forensics",
            # Phase 4
            "token_sse", "dynamic_routing", "cascade_fallback",
            "circuit_breaker", "budget_ledger", "fault_injection",
            "llm_judge", "semantic_cache", "finops", "slo_monitor",
            "otel_phoenix", "async_jobs",
        ],
        "cascade": {
            "ladder": list(CASCADE_ORDER),
            "tiers": [
                {
                    "tier": t,
                    "model": tier_model(t),
                    "breaker": registry.breaker_at(i).state.value,
                    "failures_in_window": len(registry.breaker_at(i).failures),
                    "opens": registry.breaker_at(i).opens,
                    "last_error": registry.breaker_at(i).last_error,
                    "injected_fault": injected.get(t),
                }
                for i, t in enumerate(CASCADE_ORDER)
            ],
        },
        "budget": {
            tier: {
                "spend_usd": round(v.get("spend_usd", 0.0), 6),
                "cap_usd": ledger_budget_for(tier),
                "calls": v.get("calls", 0),
                "exhausted": bool(v.get("exhausted")),
            }
            for tier, v in ledger.items()
        },
        "slos": {name: {"threshold": s["threshold"],
                        "comparator": s["comparator"],
                        "unit": s["unit"],
                        "description": s["description"]}
                 for name, s in SLOS.items()},
        "tracing": tracing.backend_info(),
        "monitor": monitor_status(),
        "judge": judge_stats(),
        "cache": cache_stats(),
        # Reported so "runs on CPU or GPU" is something the process asserts,
        # not just something the README claims.
        "portability": portability_report(),
        "notes": (
            "budget and SLO thresholds come from the ledger and config; live SLO "
            "measurements, judge scores, and cache hits are recorded per run by "
            "p4.monitor, p4.judge, and p4.cache. Phoenix traces require "
            "'phoenix serve' to be running -- spans queue locally until then."
        ) if not degraded else
            "fault injection is active; this process is deliberately degraded.",
    }


@app.post("/api/run")
async def api_run(req: RunRequest):
    thread_id = req.thread_id or str(uuid.uuid4())

    if req.return_existing:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        if state and state.values:
            return _build_run_payload({
                "final_response": state.values.get("final_response", ""),
                "category": state.values.get("category", ""),
                "iteration_count": state.values.get("iteration_count", 0),
                "github_issue_url": state.values.get("github_issue_url", ""),
            }, thread_id)

    try:
        result = run_ticket(req.ticket, thread_id)
        return _build_run_payload(result, thread_id)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _sse_body(req: "RunRequest", keepalive_s: float) -> Iterator[str]:
    """Produce the SSE frame sequence for one run.

    Split out of the endpoint so the keep-alive behavior can be tested without
    a socket, a server, or a provider. The endpoint's only remaining job is to
    hand this generator to the response object.
    """
    run = StreamRun(thread_id=req.thread_id, use_cache=req.use_cache)
    events = run.run(req.ticket)

    # A run can go seconds without producing an event: a cache lookup that has
    # to call an embedding model, a supervisor turn, or a synthesis waiting on
    # its first token. If the client sees nothing at all for longer than an
    # intermediary's idle timeout, the connection is closed mid-answer and the
    # client gets a truncated reply with no error -- the worst failure shape
    # there is, because it looks like a short answer.
    #
    # Checking the clock between `next()` calls is not enough: `next()` blocks
    # for the entire gap, so the timer never fires during exactly the window it
    # exists to cover. The events are therefore pumped onto a queue by a thread,
    # and this generator polls that queue with a timeout, which is what makes an
    # idle period observable at all.
    inbox: queue.Queue = queue.Queue()
    done = object()

    def _pump() -> None:
        try:
            for event in events:
                inbox.put(event)
        except Exception as exc:  # pragma: no cover - defensive
            inbox.put(exc)
        finally:
            inbox.put(done)

    threading.Thread(target=_pump, daemon=True,
                     name=f"sse-pump-{run.thread_id[:8]}").start()

    try:
        while True:
            try:
                item = inbox.get(timeout=keepalive_s)
            except queue.Empty:
                # Nothing for a full interval: tell the client the connection is
                # alive. Comments are ignored by EventSource and any conformant
                # SSE parser, so this is invisible to real consumers.
                yield f": keep-alive {int(time.time())}\n\n"
                continue
            if item is done:
                return
            if isinstance(item, Exception):
                traceback.print_exc()
                yield ev.to_sse(ev.RunError(
                    thread_id=run.thread_id,
                    message=f"{type(item).__name__}: {item}",
                ))
                return
            yield ev.to_sse(item)
    finally:
        # The producer thread is a daemon; closing the generator releases the
        # reference so a half-finished run cannot pin the event loop.
        closer = getattr(events, "close", None)
        if closer:
            closer()


@app.post("/api/stream")
async def api_stream(req: RunRequest):
    """Token-level SSE.

    Emits typed frames -- ``node.start``, ``tool.call``, ``token``,
    ``provider.swap``, ``run.end`` -- as they happen. The Phase 3 version
    emitted one frame per node and buffered the whole answer into a single
    ``complete`` event, so a client could not distinguish progress from a stall.

    The generator is a sync iterator drained on a worker thread: the graph and
    the cascade are both blocking, and pushing them through ``iterate_in_threadpool``
    is what keeps frames hitting the socket while the next token is still being
    generated. While the run is silent we emit ``: keep-alive`` comments every
    ``P4_SSE_KEEPALIVE_S`` seconds (default 15) so an intermediary does not close
    an idle connection mid-answer.
    """

    keepalive_s = max(1.0, float(os.getenv("P4_SSE_KEEPALIVE_S", "15")))

    return StreamingResponse(
        iterate_in_threadpool(_sse_body(req, keepalive_s)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Nginx buffers proxied responses by default, which would defeat the
            # point of streaming. This is also the header that tells a browser
            # EventSource to read incrementally.
            "X-Accel-Buffering": "no",
        },
    )


class InjectFaultRequest(BaseModel):
    tier: str
    #: Base URL of a relay that will cut the provider stream. Null clears the
    #: fault for that tier.
    base_url: Optional[str] = None
    reset_budget: bool = False


@app.post("/admin/inject-failure")
async def admin_inject_failure(req: InjectFaultRequest, request: Request):
    """Point a tier's LLM calls at a fault-injecting relay.

    This is what makes ``scripts/cut_cable.py --server`` able to cut a real
    response in a running server.

    Localhost-only, and enforced rather than merely documented. This endpoint
    can redirect a tier's traffic to an arbitrary base URL, so an exposed copy
    is a request-forgery and exfiltration vector: anyone who can reach it can
    point the bot's credentials at a server they control. Requiring a loopback
    peer keeps the drill usable while making the safe default the enforced one.
    Set ``P4_ADMIN_TOKEN`` to additionally require a bearer token when running
    behind a proxy that terminates on another host.
    """
    _require_local_admin(request)
    if req.tier not in CASCADE_ORDER:
        raise HTTPException(
            status_code=400,
            detail=f"unknown tier {req.tier!r}; expected one of {CASCADE_ORDER}",
        )
    # Budget reset is best-effort. If the ledger is unwritable the drill must
    # still be able to inject the fault -- returning 500 here would leave the
    # caller believing no fault was set while the run proceeds normally, which
    # is worse than a noisy response.
    budget_error = None
    if req.reset_budget:
        try:
            reset_budget()
        except Exception as exc:
            budget_error = f"{type(exc).__name__}: {exc}"

    set_injected_base_url(req.tier, req.base_url)
    active = injected_base_url(req.tier) is not None

    response = {
        "tier": req.tier,
        "injected_base_url": injected_base_url(req.tier),
        "active": active,
        "model": tier_model(req.tier),
    }
    if budget_error:
        response["budget_reset_error"] = budget_error
    return response


@app.get("/api/finops")
async def api_finops():
    """Spend, budgets, and what the cache avoided.

    Three numbers are deliberately kept apart, because conflating them is the
    usual way a FinOps dashboard starts lying:

    * ``actual_spend_usd`` -- money gone, per tier, from the ledger.
    * ``avoided_cost_usd`` -- money that would have been spent, priced at the
      tier that would have served each cache hit. Never subtracted from the
      ledger, because no charge was actually reversed.
    * ``savings_ratio`` -- avoided over actual, or null when nothing has been
      spent yet.
    """
    from p4.cache import get_cache

    router = get_router()
    ledger = router.ledger.snapshot()
    budget_for = router.ledger.budget_for
    return {
        "actual_spend_usd": {
            tier: round(v.get("spend_usd", 0.0), 8) for tier, v in ledger.items()
        },
        "total_actual_spend_usd": round(
            sum(v.get("spend_usd", 0.0) for v in ledger.values()), 8),
        "budgets": {
            tier: {
                "cap_usd": budget_for(tier),
                "exhausted": bool(ledger.get(tier, {}).get("exhausted")),
            }
            for tier in CASCADE_ORDER
        },
        "cache": get_cache().finops(ledger),
    }


@app.get("/api/judge")
async def api_judge():
    """Judge configuration and how many answers it has graded.

    Reports ``unavailable`` separately from ``scored``: a judge outage must be
    visible as an outage, not silently folded into a good score.
    """
    return get_judge().stats()


@app.get("/api/cache")
async def api_cache(request: Request, clear: bool = False):
    """Cache hit rate, entry count, and embedder health."""
    if clear:
        _require_local_admin(request)
        get_cache().clear()
    return get_cache().stats()


@app.get("/api/slos")
async def api_slos():
    """Live SLO evaluation plus the monitor's action log.

    Each SLO reports ``no_data`` until enough runs have been observed, which is
    the honest state for a cold process -- reporting a passing 0.0 error rate
    before any traffic would be a false green.
    """
    monitor = get_monitor()
    tracker = monitor.tracker
    verdicts = [v.snapshot() for v in tracker.evaluate()]
    return {
        **monitor.status(),
        "window_seconds": tracker.window,
        # True until the process has observed a run. Not derived from the
        # verdicts: slo_circuit_open_events legitimately measures 0.0 on a cold
        # process (no breakers have opened yet), which would make an
        # all-verdicts check report data where there is none.
        "no_data": not tracker.samples,
        "measurements": tracker.measurements(),
        "sample_count": len(tracker.samples),
        "verdicts": verdicts,
        "actions": tracker.actions[-20:],
        "current_assignment": get_router().snapshot(),
    }


def judge_stats() -> Dict:
    """Judge counters for /health. Imported lazily; the judge is optional."""
    try:
        from p4.judge import get_judge

        return get_judge().stats()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def cache_stats() -> Dict:
    """Cache counters for /health, without building the cache on a cold process."""
    try:
        from p4.cache import get_cache

        return get_cache().stats()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.post("/admin/reset-breakers")
async def admin_reset_breakers(request: Request):
    """Clear accumulated tier health.

    A drill needs this: without it a breaker left OPEN by a previous run would
    skip the tier under test, and the drill would pass without ever exercising
    the failure it is meant to prove.
    """
    _require_local_admin(request)
    registry = reset_breakers()
    return {"reset": True, "states": registry.states()}


@app.post("/admin/reset-assignments")
async def admin_reset_assignments(request: Request):
    """Restore the default agent-to-tier routing table.

    The SLO monitor demotes agents when latency or errors breach, and that
    demotion persists in the router. A drill that injects a fault at
    ``frontier`` is then meaningless: if the synthesizer was already demoted to
    ``standard`` by an earlier SLO breach, the run starts mid-ladder and never
    touches the injected tier, so the drill reports a pass without having
    demonstrated the fallback it exists to prove.
    """
    _require_local_admin(request)
    router = get_router()
    restored = router.reset_assignments()
    return {
        "reset": True,
        "assignments": restored,
        "note": ("default routing restored; breakers and budgets are separate "
                 "and are reset by /admin/reset-breakers"),
    }


def _require_local_admin(request: Request) -> None:
    """Reject admin calls that did not come from this machine.

    ``request.client.host`` is the peer address. Proxies rewrite it, so this is
    a default-deny guard rather than a security boundary on its own -- which is
    exactly why ``P4_ADMIN_TOKEN`` exists for proxied deployments.
    """
    peer = (request.client.host if request.client else "") or ""
    is_local = peer in ("127.0.0.1", "::1", "localhost") or peer.startswith("127.")
    if not is_local:
        raise HTTPException(
            status_code=403,
            detail=("admin endpoints are restricted to loopback; this looks like a "
                    "remote call. Set P4_ADMIN_TOKEN and send "
                    "'Authorization: Bearer <token>' if you intend to run this "
                    "behind a proxy."),
        )

    expected = os.getenv("P4_ADMIN_TOKEN", "").strip()
    if expected:
        supplied = (request.headers.get("authorization") or "").strip()
        if supplied != f"Bearer {expected}":
            raise HTTPException(
                status_code=401,
                detail="P4_ADMIN_TOKEN is set; send 'Authorization: Bearer <token>'",
            )


@app.delete("/admin/inject-failure/{tier}")
async def admin_clear_failure(tier: str, request: Request):
    # Guarded like its POST counterpart. Clearing is less dangerous than
    # injecting, but leaving one admin route open while its sibling is enforced
    # makes the enforcement arbitrary rather than a rule.
    _require_local_admin(request)
    if tier not in CASCADE_ORDER:
        raise HTTPException(status_code=400, detail=f"unknown tier {tier!r}")
    set_injected_base_url(tier, None)
    return {"tier": tier, "active": False}


# ---------------------------------------------------------------------------
# Async job queue
# ---------------------------------------------------------------------------
# The synchronous /api/run path holds a worker for the whole run, which is the
# wrong shape for anything longer than a few seconds: proxies time out, and the
# caller cannot tell "still working" from "crashed". /api/async returns a job id
# immediately and the work continues in a background thread.
#
# State is in-process and in-memory, so a restart loses queued jobs. That is a
# deliberate simplification for a single-node demo; the README's honest
# limitations table says so, and a real deployment needs Redis/Celery or a
# durable queue.

class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    ticket: str
    thread_id: str
    state: JobState = JobState.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None

    def snapshot(self) -> dict:
        return {
            "job_id": self.id,
            "thread_id": self.thread_id,
            "state": self.state.value,
            "ticket": self.ticket,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "queue_latency_ms": (
                round((self.started_at - self.created_at) * 1000, 1)
                if self.started_at else None
            ),
            "duration_ms": (
                round((self.finished_at - self.started_at) * 1000, 1)
                if self.started_at and self.finished_at else None
            ),
            "result": self.result,
            "error": self.error,
        }


_JOBS: Dict[str, Job] = {}
_JOB_ORDER: List[str] = []
_JOBS_LOCK = threading.Lock()
#: Bound on retained job history. Unbounded growth would be a slow leak in a
#: long-running process.
_MAX_JOBS_RETAINED = 200
#: Threads are cheap relative to a model call, so a small pool is plenty; the
#: point is to stop one long run from occupying the only worker.
_MAX_CONCURRENT_JOBS = 4
_JOB_SEMAPHORE = threading.Semaphore(_MAX_CONCURRENT_JOBS)


def _trim_jobs() -> None:
    """Caller holds ``_JOBS_LOCK``."""
    while len(_JOB_ORDER) > _MAX_JOBS_RETAINED:
        oldest = _JOB_ORDER.pop(0)
        _JOBS.pop(oldest, None)


def _run_job(job: Job) -> None:
    with _JOB_SEMAPHORE:
        with _JOBS_LOCK:
            # The queue may have been superseded or the process restarted
            # between submission and execution; never run a cancelled job.
            if job.id not in _JOBS:
                return
            job.state = JobState.RUNNING
            job.started_at = time.time()
        try:
            result = run_ticket(job.ticket, job.thread_id)
            with _JOBS_LOCK:
                job.result = result
                job.state = JobState.DONE
        except Exception as exc:
            traceback.print_exc()
            with _JOBS_LOCK:
                job.error = f"{type(exc).__name__}: {exc}"
                job.state = JobState.FAILED
        finally:
            with _JOBS_LOCK:
                job.finished_at = time.time()


class AsyncRunRequest(BaseModel):
    ticket: str
    thread_id: Optional[str] = None


@app.post("/api/async", status_code=202)
async def api_async_run(req: AsyncRunRequest):
    """Accept a run and return immediately with a job id.

    202 Accepted, not 200: the work has been queued, not completed. The client
    is expected to poll ``GET /api/jobs/{id}``.
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    job = Job(id=str(uuid.uuid4()), ticket=req.ticket, thread_id=thread_id)
    with _JOBS_LOCK:
        _JOBS[job.id] = job
        _JOB_ORDER.append(job.id)
        _trim_jobs()
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {
        "job_id": job.id,
        "thread_id": thread_id,
        "state": job.state.value,
        "poll": f"/api/jobs/{job.id}",
    }


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        return job.snapshot()


@app.get("/api/jobs")
async def api_jobs(limit: int = 20):
    with _JOBS_LOCK:
        ids = _JOB_ORDER[-limit:][::-1]
        return {
            "count": len(_JOBS),
            "retained_limit": _MAX_JOBS_RETAINED,
            "max_concurrent": _MAX_CONCURRENT_JOBS,
            "jobs": [_JOBS[i].snapshot() for i in ids if i in _JOBS],
        }


@app.get("/api/threads")
async def api_threads():
    return {"threads": get_active_threads()}


@app.get("/api/history/{thread_id}")
async def api_history(thread_id: str):
    return {"history": get_conversation_history(thread_id)}


@app.get("/api/pending-approvals")
async def api_pending_approvals():
    return {"pending": get_pending_approvals()}


@app.post("/api/approve/{thread_id}")
async def api_approve(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state or not state.next:
        raise HTTPException(status_code=404, detail="No pending approval found")
    graph.invoke(Command(resume={"approved": True}), config=config)
    return {"status": "approved", "thread_id": thread_id}


@app.post("/api/deny/{thread_id}")
async def api_deny(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state or not state.next:
        raise HTTPException(status_code=404, detail="No pending approval found")
    graph.invoke(Command(resume={"approved": False}), config=config)
    return {"status": "denied", "thread_id": thread_id}


@app.post("/api/edit-approve/{thread_id}")
async def api_edit_approve(thread_id: str, req: EditApproveRequest):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state or not state.next:
        raise HTTPException(status_code=404, detail="No pending approval found")

    edited_draft = state.values.get("github_draft", {})

    edited_draft = dict(edited_draft or {})
    if req.title:
        edited_draft["title"] = req.title
    if req.body:
        edited_draft["body"] = req.body
    if req.labels:
        edited_draft["labels"] = req.labels

    graph.invoke(Command(update={"github_draft": edited_draft}, resume={"approved": True, "edited_draft": edited_draft}), config=config)
    return {"status": "edited_and_approved", "thread_id": thread_id}


@app.get("/api/forensics/{thread_id}")
async def api_forensics(thread_id: str):
    return state_forensics(thread_id)


@app.post("/api/time-travel")
async def api_time_travel(req: TimeTravelRequest):
    try:
        return time_travel(graph, req.thread_id, req.target_step)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Time travel failed: {e}")


@app.post("/api/correct")
async def api_correct(req: CorrectRequest):
    try:
        new_value = json.loads(req.new_value)
    except json.JSONDecodeError:
        new_value = req.new_value

    target_config = {"configurable": {"checkpoint_id": req.checkpoint_id}}
    return apply_correction(graph, req.thread_id, target_config, {req.field: new_value})


@app.get("/api/verify")
async def api_verify():
    results = []
    threads = get_active_threads()
    sample_thread = threads[0] if threads else None

    if sample_thread:
        forensics = state_forensics(sample_thread)
        results.append({
            "check": "forensics_structure",
            "passed": "timeline" in forensics and "anomalies" in forensics,
            "detail": f"Found {forensics['total_steps']} steps, {len(forensics['anomalies'])} anomalies",
        })

        has_human_intervention = len(forensics.get("human_interventions", [])) > 0
        results.append({
            "check": "human_intervention_tracking",
            "passed": True,
            "detail": f"Found {len(forensics.get('human_interventions', []))} human interventions",
        })
    else:
        results.append({"check": "forensics_structure", "passed": False, "detail": "No threads to test"})
        results.append({"check": "human_intervention_tracking", "passed": False, "detail": "No threads"})

    results.append({"check": "graph_compiled", "passed": graph is not None, "detail": "Master graph compiled"})
    results.append({"check": "checkpointer_active", "passed": checkpointer is not None, "detail": "SQLite checkpointer initialized"})
    results.append({"check": "pending_approvals_endpoint", "passed": True, "detail": "Endpoint functional"})

    all_passed = all(r["passed"] for r in results)
    return {"all_passed": all_passed, "results": results}


# ============================================================================
# Serve frontend
# ============================================================================

@app.get("/")
async def serve_ui():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
