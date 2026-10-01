"""Circuit breaker hardening tests.

Two properties that a per-run breaker cannot have:

1. Health is *shared* across requests. A tier that just failed must stay
   marked for the next request, otherwise the breaker is decoration.
2. A HALF_OPEN probe slot is exclusive. Ten concurrent requests must not all
   probe a tier that is known-bad.
"""

import sys
import threading
import time

sys.path.insert(0, ".")

from p4.breaker import (  # noqa: E402
    BreakerRegistry, BreakerState, CircuitBreaker, FailureKind,
    get_breaker_registry, reset_breakers,
)

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def boom():
    """A retryable provider failure, as an exception *object*."""
    return ConnectionError("provider down")


# ---------------------------------------------------------------------------
section("1. HALF_OPEN allows exactly one probe")

clock = [1000.0]
cb = CircuitBreaker(threshold=1, window=60, cooldown=10, clock=lambda: clock[0])
cb.record_failure(boom(), FailureKind.RETRYABLE)
check("breaker is OPEN after threshold", cb.state == BreakerState.OPEN,
      cb.state.value)

check("OPEN rejects before cooldown", cb.allow_request() is False)
clock[0] += 11
check("OPEN allows once cooldown elapsed", cb.allow_request() is True)
check("state is HALF_OPEN", cb.state == BreakerState.HALF_OPEN)
check("second probe is rejected", cb.allow_request() is False)
check("third probe is rejected", cb.allow_request() is False)
check("rejections are counted", cb.snapshot()["probe_rejections"] == 2,
      str(cb.snapshot()["probe_rejections"]))

cb.record_success()
check("probe success closes the breaker", cb.state == BreakerState.CLOSED)
check("requests flow again after recovery", cb.allow_request() is True)
check("a closed breaker admits concurrent traffic",
      cb.allow_request() is True and cb.allow_request() is True,
      "no probe gating while CLOSED")

# Re-open, then confirm the *next* cooldown cycle still grants only one probe.
for _ in range(3):
    cb.record_failure(boom(), FailureKind.RETRYABLE)
check("breaker re-opened", cb.state == BreakerState.OPEN, cb.state.value)
clock[0] += 11
check("re-open allows a fresh probe", cb.allow_request() is True)
check("only one probe after re-open", cb.allow_request() is False)
cb.record_failure(boom(), FailureKind.RETRYABLE)
check("failed probe reopens", cb.state == BreakerState.OPEN, cb.state.value)


# ---------------------------------------------------------------------------
section("2. Concurrency: one probe slot under load")

clock = [1000.0]
cb = CircuitBreaker(threshold=1, window=60, cooldown=10, clock=lambda: clock[0])
cb.record_failure(boom(), FailureKind.RETRYABLE)
clock[0] += 11

granted = []
lock = threading.Lock()
barrier = threading.Barrier(12)


def probe():
    barrier.wait()
    allowed = cb.allow_request()
    with lock:
        granted.append(allowed)


threads = [threading.Thread(target=probe) for _ in range(12)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check("exactly one thread got the probe slot",
      granted.count(True) == 1, f"{granted.count(True)} of 12 granted")
check("everyone else was rejected", granted.count(False) == 11)


# ---------------------------------------------------------------------------
section("3. LOCAL failure releases the probe slot")

clock = [1000.0]
cb = CircuitBreaker(threshold=1, window=60, cooldown=10, clock=lambda: clock[0])
cb.record_failure(boom(), FailureKind.RETRYABLE)
clock[0] += 11
check("probe granted", cb.allow_request() is True)
cb.record_failure(ValueError("bug in our own code"), FailureKind.LOCAL)
check("LOCAL did not reopen the breaker", cb.state == BreakerState.HALF_OPEN,
      cb.state.value)
check("probe slot released after LOCAL",
      cb.snapshot()["probe_in_flight"] is False)
check("another probe can be granted", cb.allow_request() is True)


# ---------------------------------------------------------------------------
section("4. Health is shared across requests")

reset_breakers()
shared = get_breaker_registry()
idx = shared.index_of("frontier")
shared.breaker_at(idx).record_failure(boom(), FailureKind.RETRYABLE)
shared.breaker_at(idx).record_failure(boom(), FailureKind.RETRYABLE)
shared.breaker_at(idx).record_failure(boom(), FailureKind.RETRYABLE)
check("frontier is OPEN after three failures",
      shared.breaker_at(idx).state == BreakerState.OPEN, shared.states())

# A "new request" must see that state. Before the fix, StreamRun built a fresh
# BreakerRegistry and this assertion could not hold.
again = get_breaker_registry()
check("a later request sees the OPEN breaker",
      again.breaker_at(idx).state == BreakerState.OPEN,
      again.states())
check("that request is refused the dead tier",
      again.breaker_at(idx).allow_request() is False)
check("other tiers are unaffected",
      again.breaker_at(again.index_of("standard")).state == BreakerState.CLOSED)

reset_breakers()
check("reset_breakers clears shared state",
      get_breaker_registry().breaker_at(idx).state == BreakerState.CLOSED)
check("the registry object is the same after reset",
      get_breaker_registry() is shared)


# ---------------------------------------------------------------------------
section("5. Registry identity and per-index isolation")

r1 = get_breaker_registry()
r2 = get_breaker_registry()
check("get_breaker_registry is a singleton", r1 is r2)

reg = BreakerRegistry(["frontier", "standard"])
reg.breaker_at(0).record_failure(boom(), FailureKind.RETRYABLE)
reg.breaker_at(0).record_failure(boom(), FailureKind.RETRYABLE)
reg.breaker_at(0).record_failure(boom(), FailureKind.RETRYABLE)
check("tier 0 opened", reg.breaker_at(0).state == BreakerState.OPEN)
check("tier 1 untouched", reg.breaker_at(1).state == BreakerState.CLOSED)
check("any_open() is true", reg.any_open() is True)
check("states() reports both", reg.states() == {"frontier": "open", "standard": "closed"},
      str(reg.states()))
check("tier positions are distinct even for a shared model",
      reg.index_of("frontier") != reg.index_of("standard"))


# ---------------------------------------------------------------------------
section("6. Concurrent recording does not corrupt the count")

reset_breakers()
reg = get_breaker_registry()
i = reg.index_of("utility")
barrier = threading.Barrier(16)


def hammer():
    barrier.wait()
    for _ in range(20):
        reg.breaker_at(i).record_success()


ts = [threading.Thread(target=hammer) for _ in range(16)]
for t in ts:
    t.start()
for t in ts:
    t.join()
check("all successes recorded exactly once",
      reg.breaker_at(i).successes == 320, f"{reg.breaker_at(i).successes}")
reset_breakers()


# ---------------------------------------------------------------------------
passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = len(RESULTS) - passed
print(f"\n{'=' * 70}\npassed {passed}   failed {failed}\n{'=' * 70}")
if failed:
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}   {detail}")
    sys.exit(1)
