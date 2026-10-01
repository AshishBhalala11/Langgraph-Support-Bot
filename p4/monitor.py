"""SLO monitoring and automated degradation.

The claim this module has to earn is not "it logs metrics" but "it notices a
problem and does something about it". So each SLO has a comparator, a
windowed measurement, and an action:

* ``slo_fallback_rate`` too high          -> demote the preferred tier
* ``slo_error_rate`` too high             -> force the ladder down a rung
* ``slo_stream_latency_p95`` too high     -> force the ladder down a rung
* ``slo_circuit_open_events`` sustained   -> block the tier outright
* ``slo_judge_correctness_floor`` breached -> log loudly, no automatic action

Why some SLOs act and others only alert: a routing change is cheap and
reversible, whereas quietly reducing answer quality is not. Anything that
trades away correctness for latency gets a human in the loop, and that is
recorded in the honesty table rather than presented as autonomous repair.

Windowing is a simple ring buffer of recent samples. A production deployment
would use Prometheus; this keeps the measurement inspectable from
``/monitor`` and unit-testable with an injected clock.
"""

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from p4.breaker import get_breaker_registry
from p4.config import CASCADE_ORDER, SLOS
from p4.router import get_router


def _compare(value: float, threshold: float, comparator: str) -> bool:
    """True when ``value`` violates the SLO (i.e. the SLO is breached)."""
    if comparator == "lt":
        return value >= threshold
    if comparator == "lte":
        return value > threshold
    if comparator == "gte":
        return value < threshold
    if comparator == "gt":
        return value <= threshold
    return False


@dataclass
class Sample:
    at: float
    kind: str
    value: float
    meta: Dict = field(default_factory=dict)


@dataclass
class Verdict:
    slo: str
    #: ``None`` when the window holds no samples for this SLO.
    value: Optional[float]
    threshold: float
    comparator: str
    breached: bool
    action: str = ""
    #: True when there were no samples, so ``breached`` is "unknown", not "pass".
    no_data: bool = False

    def snapshot(self) -> Dict:
        return {
            "slo": self.slo,
            "value": (round(self.value, 4) if self.value is not None else None),
            "threshold": self.threshold,
            "comparator": self.comparator,
            "breached": self.breached,
            "no_data": self.no_data,
            "action_taken": self.action,
        }


class SLOTracker:
    """Ring-buffered samples plus an evaluation pass over the named SLOs."""

    def __init__(self, window_seconds: float = 300.0, max_samples: int = 2000,
                 clock=time.time):
        self.window = window_seconds
        self.clock = clock
        self.samples: Deque[Sample] = deque(maxlen=max_samples)
        self.verdicts: List[Verdict] = []
        self.history: List[Dict] = []
        self.actions: List[Dict] = []
        # Reentrant: snapshot() holds the lock while calling percentile()/rate(),
        # which take it again. A plain Lock deadlocks the monitor thread.
        self._lock = threading.RLock()

    # -- recording ----------------------------------------------------------

    def record(self, kind: str, value: float, **meta) -> None:
        with self._lock:
            self.samples.append(Sample(at=self.clock(), kind=kind, value=value,
                                       meta=meta))

    def record_run(self, *, latency_ms: float, errored: bool, degraded: bool,
                   category: str = "", cost_usd: float = 0.0,
                   judge_scores: Optional[Dict] = None) -> None:
        """Record the handful of numbers one completed run contributes."""
        self.record("latency_ms", latency_ms, category=category)
        self.record("error", 1.0 if errored else 0.0)
        self.record("fallback", 1.0 if degraded else 0.0, category=category)
        self.record("cost_usd", cost_usd)
        for name, score in (judge_scores or {}).items():
            self.record(f"judge_{name}", score)

    def record_breaker_event(self, tier: str) -> None:
        self.record("breaker_open", 1.0, tier=tier)

    # -- measurement --------------------------------------------------------

    def _windowed(self, kind: str) -> List[Sample]:
        with self._lock:
            now = self.clock()
            return [s for s in self.samples
                    if s.kind == kind and now - s.at <= self.window]

    def percentile(self, kind: str, pct: float) -> float:
        values = sorted(s.value for s in self._windowed(kind))
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        # Nearest-rank percentile: for p95 of 20 samples this is the 19th, which
        # is what an SLO on a small window should mean (no interpolation that
        # invents a value nobody observed).
        rank = max(1, int(round(pct / 100.0 * len(values) + 0.5)))
        return values[min(rank, len(values)) - 1]

    def rate(self, kind: str) -> float:
        values = [s.value for s in self._windowed(kind)]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def mean(self, kind: str) -> float:
        return self.rate(kind)

    def _measure(self, name: str) -> Optional[float]:
        """Measure an SLO, or ``None`` when there is no data yet.

        ``None`` matters: before any run has completed, mean correctness is
        0.0, and reporting that as a breach would fire a false alarm on a
        freshly-started process. "No data" is reported as unknown instead.
        """
        spec = SLOS[name]
        if name == "slo_stream_latency_p95":
            return self.percentile("latency_ms", 95) if self._windowed("latency_ms") else None
        if name == "slo_error_rate":
            return self.rate("error") if self._windowed("error") else None
        if name == "slo_fallback_rate":
            return self.rate("fallback") if self._windowed("fallback") else None
        if name == "slo_judge_correctness_floor":
            return self.mean("judge_correctness") if self._windowed("judge_correctness") else None
        if name == "slo_cache_hit_rate_floor":
            return self.mean("cache_hit") if self._windowed("cache_hit") else None
        if name == "slo_circuit_open_events":
            return float(len(self._windowed("breaker_open")))
        return None

    def measure(self, name: str) -> Optional[float]:
        """Public wrapper for :meth:`_measure`.

        ``/api/slos`` needs the raw measured value for every configured SLO, and
        reaching into the private helper from the API would be worse than
        exposing the name it already has.
        """
        return self._measure(name)

    def measurements(self) -> Dict[str, Optional[float]]:
        """Current value of every configured SLO, or ``None`` where unknown."""
        return {name: self._measure(name) for name in SLOS}

    # -- evaluation ---------------------------------------------------------

    def evaluate(self) -> List[Verdict]:
        """Measure every SLO and apply the automatic actions that are due."""
        verdicts: List[Verdict] = []
        for name, spec in SLOS.items():
            value = self._measure(name)
            no_data = value is None
            breached = False if no_data else _compare(
                value, spec["threshold"], spec["comparator"])
            action = ""
            if breached:
                action = self._react(name, value, spec)
            verdicts.append(Verdict(slo=name, value=value,
                                    threshold=spec["threshold"],
                                    comparator=spec["comparator"],
                                    breached=breached, action=action,
                                    no_data=no_data))
        with self._lock:
            self.verdicts = verdicts
            self.history.append({
                "at": self.clock(),
                "verdicts": [v.snapshot() for v in verdicts],
            })
            # Keep the history bounded; a long-running process would otherwise
            # accumulate one entry per tick forever.
            if len(self.history) > 500:
                del self.history[:-500]
        return verdicts

    def _react(self, name: str, value: float, spec: Dict) -> str:
        """Act on a breach. Returns a description of what was done."""
        registry = get_breaker_registry()
        router = get_router()

        try:
            if name == "slo_fallback_rate" and value >= spec["threshold"]:
                # Too many runs are degrading. Stop spending on the tier that
                # keeps failing: demote every agent currently on the top rung.
                moved = []
                for agent in _agents_on(router, CASCADE_ORDER[0]):
                    if CASCADE_ORDER[1] in CASCADE_ORDER:
                        moved.append(router.set_tier(agent, CASCADE_ORDER[1]))
                return (f"demoted {len(moved)} agent(s) to {CASCADE_ORDER[1]}"
                        if moved else "no agent was on the top tier; nothing to demote")

            if name in ("slo_error_rate", "slo_stream_latency_p95"):
                # Move the synthesizer's preferred tier down one rung so the
                # next answer comes from a cheaper, usually faster model.
                current = router.resolve("synthesizer")["tier"]
                if current in CASCADE_ORDER:
                    idx = CASCADE_ORDER.index(current)
                    if idx + 1 < len(CASCADE_ORDER):
                        router.set_tier("synthesizer", CASCADE_ORDER[idx + 1])
                        return (f"demoted synthesizer {current} -> "
                                f"{CASCADE_ORDER[idx + 1]}")
                return "already on the last tier; no cheaper fallback available"

            if name == "slo_circuit_open_events" and value > spec["threshold"]:
                # Sustained breaker opens mean a tier is durably unhealthy.
                # Force it OPEN in the shared registry so the cascade stops
                # paying for it until cooldown and a probe prove otherwise.
                forced = []
                for i, tier in enumerate(CASCADE_ORDER):
                    b = registry.breaker_at(i)
                    if b.opens and b.state.value == "closed":
                        b.record_failure(
                            ConnectionError("SLO monitor: sustained breaker opens"),
                            __import__("p4.breaker", fromlist=["FailureKind"])
                            .FailureKind.RETRYABLE,
                        )
                        forced.append(tier)
                return f"forced {forced} open" if forced else "no tier needed forcing"

            if name in ("slo_judge_correctness_floor", "slo_cache_hit_rate_floor"):
                # Alert-only, deliberately: silently degrading answer quality or
                # cache coverage to hit a number would be the wrong trade.
                return "alert only (no automatic action by design)"

        except Exception as exc:
            return f"action failed: {type(exc).__name__}: {exc}"

        return "no action defined for this SLO"

    # -- inspection ---------------------------------------------------------

    def snapshot(self) -> Dict:
        with self._lock:
            verdicts = [v.snapshot() for v in self.verdicts]
            actions = list(self.actions[-20:])
            return {
                "window_seconds": self.window,
                "samples": len(self.samples),
                "verdicts": verdicts,
                "breached": [v["slo"] for v in verdicts if v["breached"]],
                "unknown": [v["slo"] for v in verdicts if v["no_data"]],
                "recent_actions": actions,
                "raw": {
                    "latency_p50_ms": round(self.percentile("latency_ms", 50), 1),
                    "latency_p95_ms": round(self.percentile("latency_ms", 95), 1),
                    "error_rate": round(self.rate("error"), 4),
                    "fallback_rate": round(self.rate("fallback"), 4),
                    "cache_hit_rate": round(self.rate("cache_hit"), 4),
                    "mean_correctness": round(self.mean("judge_correctness"), 4),
                    "mean_safety": round(self.mean("judge_safety"), 4),
                    "mean_tone": round(self.mean("judge_tone"), 4),
                    "total_cost_usd": round(
                        sum(s.value for s in self._windowed("cost_usd")), 8),
                    "breaker_open_events": len(self._windowed("breaker_open")),
                },
            }

    def note_action(self, action: str, **meta) -> None:
        with self._lock:
            self.actions.append({"at": self.clock(), "action": action, **meta})


def _agents_on(router, tier: str) -> List[str]:
    from p4.config import AGENT_TIER

    return [a for a, t in AGENT_TIER.items() if t == tier]


# ---------------------------------------------------------------------------
# Background monitor
# ---------------------------------------------------------------------------

class SLOMonitor:
    """Evaluates the SLOs on an interval, in a daemon thread."""

    def __init__(self, tracker: Optional[SLOTracker] = None, interval: float = 30.0):
        self.tracker = tracker or SLOTracker()
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.ticks = 0
        self.started_at: Optional[float] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.started_at = time.time()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="p4-slo-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for verdict in self.tracker.evaluate():
                    if verdict.breached and verdict.action:
                        self.tracker.note_action(
                            verdict.action, slo=verdict.slo,
                            value=verdict.value)
                self.ticks += 1
            except Exception as exc:  # never let the monitor kill the app
                self.tracker.note_action(
                    f"monitor error: {type(exc).__name__}: {exc}")
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def status(self) -> Dict:
        running = bool(self._thread and self._thread.is_alive())
        return {
            "running": running,
            "interval_seconds": self.interval,
            "ticks": self.ticks,
            "started_at": self.started_at,
            "uptime_seconds": (time.time() - self.started_at)
            if self.started_at else 0.0,
        }


_MONITOR: Optional[SLOMonitor] = None
_MONITOR_LOCK = threading.Lock()


def get_monitor() -> SLOMonitor:
    global _MONITOR
    with _MONITOR_LOCK:
        if _MONITOR is None:
            interval = float(os.getenv("P4_SLO_INTERVAL", "30"))
            _MONITOR = SLOMonitor(SLOTracker(), interval=interval)
        return _MONITOR


def monitor_status() -> Dict:
    monitor = get_monitor()
    status = monitor.status()
    status["slo_summary"] = monitor.tracker.snapshot()["raw"]
    status["breached"] = monitor.tracker.snapshot()["breached"]
    return status
