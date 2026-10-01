"""Circuit breaker state machine and provider error classifier.

Two responsibilities, deliberately kept together because the second decides
whether the first should even be consulted:

1. ``classify_error`` -- decide whether a failure justifies advancing to the
   next tier. A misconfigured model name raises ``OpenAIInvalidRequestError``
   (HTTP 400). Cascading on that would burn latency and *mask* the misconfig
   behind three tiers of fallbacks, so permanent failures fail fast instead.

2. ``CircuitBreaker`` -- stop hammering a tier that is genuinely down.

The breaker is keyed by TIER INDEX, not model name. Keying by model string is
the reference project's documented bug: two tiers sharing a model collapse
into one breaker's state, and renaming a model silently resets its history.
Index keying means the ladder in ``p4.config.CASCADE_ORDER`` is the single
source of truth about what "tier 1" means.
"""

import threading
import time
from enum import Enum
from typing import Dict, List, Optional

from p4.config import (
    BREAKER_COOLDOWN_SECONDS,
    BREAKER_FAILURE_THRESHOLD,
    BREAKER_WINDOW_SECONDS,
)

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

#: Substrings matched against ``type(exc).__name__`` and the HTTP status.
#: Checked case-insensitively, in order.
_RETRYABLE_MARKERS = (
    "apiconnection",
    "connection",
    "timeout",
    "ratelimit",
    "remoteprotocol",
    "protocol",
    "serviceunavailable",
    "internalserver",
    "overloaded",
)

#: Substrings that mean "stop, this will never succeed".
_PERMANENT_MARKERS = (
    "invalidrequest",
    "authentication",
    "permission",
    "notfound",
    "badrequest",
    "unprocessable",
)


class FailureKind(str, Enum):
    RETRYABLE = "retryable"
    PERMANENT = "permanent"
    #: Not a provider fault at all (bad request from our own code).
    LOCAL = "local"


def classify_error(exc: BaseException) -> FailureKind:
    """Decide whether ``exc`` justifies advancing to a fallback tier.

    HTTP status is consulted first when the exception carries one (the
    ``openai`` SDK attaches ``.status_code``), then the exception class name,
    then ``str(exc)`` as a last resort.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "http_status", None), "__int__", lambda: None)()

    if isinstance(status, int):
        if status in (408, 409, 425, 429):
            return FailureKind.RETRYABLE
        if 500 <= status <= 599:
            return FailureKind.RETRYABLE
        if 400 <= status <= 499:
            return FailureKind.PERMANENT

    name = type(exc).__name__.lower()
    if any(m in name for m in _PERMANENT_MARKERS):
        return FailureKind.PERMANENT
    if any(m in name for m in _RETRYABLE_MARKERS):
        return FailureKind.RETRYABLE

    text = str(exc).lower()
    if any(m in text for m in _PERMANENT_MARKERS):
        return FailureKind.PERMANENT
    if any(m in text for m in _RETRYABLE_MARKERS):
        return FailureKind.RETRYABLE

    # Unknown failure: treat as retryable so a genuinely transient novel error
    # still reaches a healthy tier. Unknown-but-permanent is the less harmful
    # mistake -- it costs a fallback, it does not take down the response.
    return FailureKind.RETRYABLE


# ---------------------------------------------------------------------------
# Breaker state machine
# ---------------------------------------------------------------------------

class BreakerState(str, Enum):
    CLOSED = "closed"        # healthy, traffic flows
    OPEN = "open"            # failing, traffic blocked
    HALF_OPEN = "half_open"  # cooldown elapsed, one probe allowed


class CircuitBreaker:
    """Per-tier circuit breaker.

    CLOSED --(N consecutive retryable failures in window)--> OPEN
    OPEN --(cooldown elapsed)--> HALF_OPEN
    HALF_OPEN --(success)--> CLOSED
    HALF_OPEN --(failure)--> OPEN

    ``allow_request`` is the only gate the cascade consults. A permanent
    failure opens the breaker too, but additionally trips ``self.permanent`` so
    the caller can fail fast instead of advancing tiers.
    """

    def __init__(
        self,
        threshold: int = BREAKER_FAILURE_THRESHOLD,
        window: float = BREAKER_WINDOW_SECONDS,
        cooldown: float = BREAKER_COOLDOWN_SECONDS,
        clock=time.monotonic,
    ):
        self.threshold = threshold
        self.window = window
        self.cooldown = cooldown
        self._clock = clock
        self.state = BreakerState.CLOSED
        self.failures: List[float] = []
        self.successes = 0
        self.opens = 0
        #: Set when the last failure was classified PERMANENT.
        self.permanent = False
        self.last_error: Optional[str] = None
        self._opened_at = 0.0
        #: Set while a HALF_OPEN probe is in flight, so a second concurrent
        #: request is rejected instead of doubling the probe.
        self._probe_in_flight = False
        #: Probes rejected because another request held the slot. Surfaced so a
        #: degraded path is visible rather than silent.
        self.probe_rejections = 0
        #: Guards every state transition. The gateway serves requests on
        #: multiple threads, and check-then-set on ``state`` is not atomic.
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """Whether a call against this tier should be attempted.

        Thread-safe. In HALF_OPEN this grants a slot to exactly one caller; the
        rest are rejected until that caller records success or failure. Without
        the guard a burst of concurrent requests would all probe a tier that is
        known-bad, which is how a recovering provider gets knocked back over.
        """
        with self._lock:
            if self.state == BreakerState.CLOSED:
                return True
            if self.state == BreakerState.OPEN:
                if self._clock() - self._opened_at >= self.cooldown:
                    self.state = BreakerState.HALF_OPEN
                    self._probe_in_flight = True
                    return True
                return False
            # HALF_OPEN: one probe at a time.
            if self._probe_in_flight:
                self.probe_rejections += 1
                return False
            self._probe_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self.successes += 1
            self.failures.clear()
            self.permanent = False
            self.last_error = None
            self._probe_in_flight = False
            if self.state in (BreakerState.OPEN, BreakerState.HALF_OPEN):
                self.state = BreakerState.CLOSED

    def record_failure(self, exc: BaseException, kind: FailureKind) -> BreakerState:
        """Record a failure and transition state."""
        with self._lock:
            self.last_error = f"{type(exc).__name__}: {exc}"
            now = self._clock()
            # Any outcome releases the probe slot: a LOCAL failure is not the
            # probe's fault, so leaving the slot held would wedge the breaker.
            self._probe_in_flight = False

            if kind == FailureKind.PERMANENT:
                # A permanent failure is conclusive: open immediately, do not
                # wait for the threshold, and mark the tier so callers fail fast.
                self.permanent = True
                self._trip(now)
                return self.state

            if kind == FailureKind.LOCAL:
                # Our own bug. Not the provider's fault -- do not trip the breaker.
                return self.state

            if self.state == BreakerState.HALF_OPEN:
                # Probe failed: straight back to OPEN with a fresh cooldown.
                self._trip(now)
                return self.state

            self.failures = [t for t in self.failures if now - t < self.window]
            self.failures.append(now)
            if len(self.failures) >= self.threshold:
                self._trip(now)
            return self.state

    def _trip(self, now: float) -> None:
        # Caller holds ``self._lock``.
        self.state = BreakerState.OPEN
        self._opened_at = now
        self.opens += 1
        self._probe_in_flight = False

    def reset(self) -> None:
        with self._lock:
            self.state = BreakerState.CLOSED
            self.failures.clear()
            self.successes = 0
            self.opens = 0
            self.permanent = False
            self.last_error = None
            self._probe_in_flight = False
            self.probe_rejections = 0

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "state": self.state.value,
                "consecutive_failures": len(self.failures),
                "threshold": self.threshold,
                "successes": self.successes,
                "opens": self.opens,
                "permanent": self.permanent,
                "last_error": self.last_error,
                "probe_in_flight": self._probe_in_flight,
                "probe_rejections": self.probe_rejections,
            }


class BreakerRegistry:
    """One breaker per tier index.

    Keyed by position in the cascade ladder, never by model name.
    """

    def __init__(self, ladder: List[str], **kwargs):
        self.ladder = list(ladder)
        self._by_index: Dict[int, CircuitBreaker] = {
            i: CircuitBreaker(**kwargs) for i in range(len(self.ladder))
        }

    def __len__(self) -> int:
        return len(self.ladder)

    def __iter__(self):
        for i in range(len(self.ladder)):
            yield i, self.ladder[i], self._by_index[i]

    def tier_at(self, index: int) -> str:
        return self.ladder[index]

    def breaker_at(self, index: int) -> CircuitBreaker:
        return self._by_index[index]

    def index_of(self, tier: str) -> int:
        return self.ladder.index(tier)

    def reset(self) -> None:
        for breaker in self._by_index.values():
            breaker.reset()

    def snapshot(self) -> Dict[str, Dict]:
        return {
            self.ladder[i]: breaker.snapshot()
            for i, breaker in self._by_index.items()
        }

    def any_open(self) -> bool:
        return any(
            b.state in (BreakerState.OPEN, BreakerState.HALF_OPEN)
            for b in self._by_index.values()
        )

    def states(self) -> Dict[str, str]:
        return {self.ladder[i]: b.state.value for i, b in self._by_index.items()}


# ---------------------------------------------------------------------------
# Process-global registry
# ---------------------------------------------------------------------------
#
# Breakers only earn their keep if a tier that just failed stays marked as
# failing for *subsequent* requests. A registry created per run would let the
# next request walk straight back into a dead provider, which is precisely the
# failure the breaker is supposed to prevent.
#
# Scope note: this is per-process. Under multiple uvicorn workers each worker
# keeps its own breakers, so an open breaker is not shared across the fleet.
# Sharing them would need a shared store (Redis, or a SQLite table with compare
# and swap) and is listed in the README's honest limitations.

_GLOBAL_REGISTRY: Optional[BreakerRegistry] = None
_GLOBAL_LOCK = threading.Lock()


def get_breaker_registry(ladder: Optional[List[str]] = None) -> BreakerRegistry:
    """The shared breaker registry, created on first use.

    A different ``ladder`` is ignored once the registry exists, because the
    registry's whole purpose is to be a stable mapping from tier position to
    accumulated health; rebuilding it would discard that history. Call
    :func:`reset_breakers` to start over deliberately.
    """
    global _GLOBAL_REGISTRY
    with _GLOBAL_LOCK:
        if _GLOBAL_REGISTRY is None:
            from p4.config import CASCADE_ORDER

            _GLOBAL_REGISTRY = BreakerRegistry(ladder or CASCADE_ORDER)
        return _GLOBAL_REGISTRY


def reset_breakers() -> BreakerRegistry:
    """Clear accumulated health on every tier without replacing the registry.

    Used by the failure drill so each run starts from a known state.
    """
    registry = get_breaker_registry()
    registry.reset()
    return registry


def breaker_snapshot() -> Dict[str, Dict]:
    return get_breaker_registry().snapshot()