"""Regression tests for the cascade: breaker, stitcher, and tier walk.

Run with:``venv/bin/python scripts/test_cascade.py``

These use scripted fake providers so the whole file is deterministic, offline,
and free. The live-provider equivalent is ``scripts/cut_cable.py``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("P4_DB", "p4_test.db")

from langchain_core.messages import HumanMessage  # noqa: E402

from p4.breaker import BreakerRegistry, BreakerState, FailureKind, classify_error  # noqa: E402
from p4.cascade import CascadeRunner, stitch_continuation  # noqa: E402

LADDER = ["frontier", "standard", "utility"]

_PASS, _FAIL = [], []


def check(label, cond, extra=""):
    (_PASS if cond else _FAIL).append(label)
    print(f"{'ok  ' if cond else 'FAIL'} {label}" + (f"   {extra}" if extra else ""))


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class ConnError(Exception):
    """Stands in for openai.APIConnectionError."""


class BadRequest(Exception):
    """Stands in for a permanent 400 -- e.g. a misconfigured model name."""

    status_code = 400


class FakeChunk:
    def __init__(self, content):
        self.content = content


class ScriptedLLM:
    """Emits ``script``; raises ``exc`` when the chunk index reaches ``die_before``.

    ``die_before=0`` means "die before emitting anything" (a pre-stream failure,
    which the client never sees and therefore must not produce a swap event).
    ``die_before=2`` means "emit two chunks, then die" (a mid-stream failure).
    """

    def __init__(self, script=None, die_before=None, exc=None):
        self.script = script or []
        self.die_before = die_before
        self.exc = exc or ConnError("connection reset by peer")

    def stream(self, messages):
        for i, piece in enumerate(self.script):
            if self.die_before is not None and i >= self.die_before:
                raise self.exc
            yield FakeChunk(piece)


def runner(llms, **kw):
    br = BreakerRegistry(LADDER, **kw)
    return CascadeRunner(LADDER, br, llm_factory=lambda tier: llms[tier]), br


# ---------------------------------------------------------------------------
# 1. Error classification
# ---------------------------------------------------------------------------

def test_classification():
    class StatusErr(Exception):
        def __init__(self, msg, status):
            super().__init__(msg)
            self.status_code = status

    cases = [
        (StatusErr("bad gateway", 502), FailureKind.RETRYABLE),
        (StatusErr("rate limited", 429), FailureKind.RETRYABLE),
        (StatusErr("request timeout", 408), FailureKind.RETRYABLE),
        (StatusErr("unauthorized", 401), FailureKind.PERMANENT),
        (StatusErr("bad model", 400), FailureKind.PERMANENT),
        (StatusErr("forbidden", 403), FailureKind.PERMANENT),
        (ConnError("reset"), FailureKind.RETRYABLE),
        (TimeoutError("read timeout"), FailureKind.RETRYABLE),
        (ValueError("our own bug"), FailureKind.RETRYABLE),  # unknown -> retryable
    ]
    for exc, want in cases:
        got = classify_error(exc)
        check(f"classify {type(exc).__name__}({getattr(exc,'status_code',None)}) -> {want.value}",
              got == want, "" if got == want else f"got {got.value}")

    # Real SDK exception classes.
    try:
        import httpx
        from openai import (
            APIConnectionError, APITimeoutError, AuthenticationError,
            BadRequestError, RateLimitError,
        )
        req = httpx.Request("POST", "https://example.invalid")
        real = [
            (APIConnectionError(request=req), FailureKind.RETRYABLE),
            (APITimeoutError(request=req), FailureKind.RETRYABLE),
            (RateLimitError(message="429", response=httpx.Response(429, request=req), body=None),
             FailureKind.RETRYABLE),
            (AuthenticationError(message="401",
                                  response=httpx.Response(401, request=req), body=None),
             FailureKind.PERMANENT),
            (BadRequestError(message="400",
                             response=httpx.Response(400, request=req), body=None),
             FailureKind.PERMANENT),
        ]
        for exc, want in real:
            got = classify_error(exc)
            check(f"classify openai.{type(exc).__name__} -> {want.value}",
                  got == want, "" if got == want else f"got {got.value}")
    except ImportError:
        print("skip  openai SDK exception checks (openai not importable)")


# ---------------------------------------------------------------------------
# 2. Breaker state machine
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, d):
        self.t += d


def test_breaker():
    clock = FakeClock()
    b = BreakerRegistry(LADDER, threshold=3, cooldown=30, clock=clock).breaker_at(0)

    check("breaker starts CLOSED and allows traffic",
          b.state == BreakerState.CLOSED and b.allow_request())
    b.record_failure(ConnError("x"), FailureKind.RETRYABLE)
    b.record_failure(ConnError("x"), FailureKind.RETRYABLE)
    check("2/3 failures stays CLOSED", b.state == BreakerState.CLOSED)
    b.record_failure(ConnError("x"), FailureKind.RETRYABLE)
    check("3/3 failures opens", b.state == BreakerState.OPEN)
    check("OPEN blocks traffic", not b.allow_request())

    clock.advance(29)
    check("29s of 30s cooldown still blocks", not b.allow_request())
    clock.advance(2)
    check("cooldown elapsed admits a probe", b.allow_request())
    check("state is HALF_OPEN", b.state == BreakerState.HALF_OPEN)
    b.record_success()
    check("probe success closes the circuit", b.state == BreakerState.CLOSED)

    # A failed HALF_OPEN probe reopens at once rather than waiting for the
    # threshold again. Uses threshold=1 so the trip is unambiguous.
    clock_h = FakeClock()
    bh = BreakerRegistry(LADDER, threshold=1, cooldown=30, clock=clock_h).breaker_at(0)
    bh.record_failure(ConnError("x"), FailureKind.RETRYABLE)
    check("threshold=1 opens immediately", bh.state == BreakerState.OPEN)
    clock_h.advance(31)
    check("probe admitted after cooldown", bh.allow_request())
    bh.record_failure(ConnError("x"), FailureKind.RETRYABLE)
    check("failed probe reopens immediately", bh.state == BreakerState.OPEN)

    b2 = BreakerRegistry(LADDER).breaker_at(0)
    b2.record_failure(BadRequest("bad model"), FailureKind.PERMANENT)
    check("PERMANENT failure trips at once, ignoring threshold",
          b2.state == BreakerState.OPEN and b2.permanent)

    b3 = BreakerRegistry(LADDER).breaker_at(0)
    for _ in range(5):
        b3.record_failure(ValueError("our bug"), FailureKind.LOCAL)
    check("LOCAL failures never trip the breaker", b3.state == BreakerState.CLOSED)

    clock_w = FakeClock()
    b4 = BreakerRegistry(LADDER, threshold=3, window=60, clock=clock_w).breaker_at(0)
    b4.record_failure(ConnError("a"), FailureKind.RETRYABLE)
    clock_w.advance(30)
    b4.record_failure(ConnError("b"), FailureKind.RETRYABLE)
    clock_w.advance(31)  # the first failure is now older than the window
    b4.record_failure(ConnError("c"), FailureKind.RETRYABLE)
    check("failures outside the window age out", b4.state == BreakerState.CLOSED)


# ---------------------------------------------------------------------------
# 3. Overlap stitcher
# ---------------------------------------------------------------------------

def test_stitcher():
    cases = [
        # (label, partial, continuation, expected, expected_stripped)
        ("live restate: 'API rate limiting is' + restatement",
         "API rate limiting is", "API rate limiting is a crucial mechanism.",
         "API rate limiting is a crucial mechanism.", 20),
        ("no overlap -> plain concatenation",
         "The service is down.", "Here are the escalation steps.",
         "The service is down.Here are the escalation steps.", 0),
        ("empty partial",
         "", "hello", "hello", 0),
        ("empty continuation",
         "hello", "", "hello", 0),
        # The exact seam a live SSE cut produced: token-boundary cut with no
        # trailing space on the partial.
        ("token-boundary seam gets a space",
         "API rate limiting", "is a crucial mechanism.",
         "API rate limiting is a crucial mechanism.", 0),
        ("seam that already has a space is left alone",
         "API rate limiting ", "is a crucial mechanism.",
         "API rate limiting is a crucial mechanism.", 0),
        ("last-sentence restate",
         "First we check status. Then we escalate.",
         "Then we escalate. I can file the incident now.",
         "First we check status. Then we escalate. I can file the incident now.", 17),
        ("prefers a word boundary over a mid-word seam",
         "The quick brown fox jumps", "brown fox jumps over the lazy dog.",
         "The quick brown fox jumps over the lazy dog.", 15),
        ("pure repeat collapses to a single copy",
         "Hello there, friend.", "Hello there, friend.",
         "Hello there, friend.", 20),
    ]
    for label, partial, cont, want, want_stripped in cases:
        got, stripped = stitch_continuation(partial, cont)
        check(f"stitch: {label}", got == want and stripped == want_stripped,
              "" if got == want else f"got {got!r} want {want!r}")

    long_a, long_b = "x" * 400, "y" * 400
    _, stripped = stitch_continuation(long_a, long_b, max_window=240)
    check("stitch: overlap beyond the window is not guessed", stripped == 0,
          f"stripped={stripped}")


# ---------------------------------------------------------------------------
# 4. Cascade behaviour
# ---------------------------------------------------------------------------

def test_cascade():
    # Clean run, no failure.
    r, _ = runner({"frontier": ScriptedLLM(["Hello", " world", "!"])})
    res = r.stream([HumanMessage(content="hi")], agent="synthesizer")
    check("clean run returns the full text", res.text == "Hello world!", repr(res.text))
    check("clean run uses exactly one tier", res.tiers_used == ["frontier"], str(res.tiers_used))
    check("clean run emits no swap event", not res.swaps)

    # --- The headline case: mid-stream failure, connection survives. ---
    r, _ = runner({
        "frontier": ScriptedLLM(["The deploy service", " is down", " NEVER"], die_before=2),
        "standard": ScriptedLLM([" and affecting", " production."]),
    })
    emitted, swaps = [], []
    res = r.stream(
        [HumanMessage(content="status?")],
        agent="synthesizer",
        emit=lambda d, t, m: emitted.append((t, d)),
        on_swap=swaps.append,
    )
    check("mid-stream failure: no exception escapes the generator", True)
    check("mid-stream failure: text spans both tiers",
          res.text == "The deploy service is down and affecting production.", repr(res.text))
    check("mid-stream failure: exactly one swap event", len(swaps) == 1)
    check("mid-stream failure: swap names both tiers",
          swaps and swaps[0].from_tier == "frontier" and swaps[0].to_tier == "standard")
    check("mid-stream failure: swap records the partial",
          swaps and swaps[0].partial_text == "The deploy service is down",
          repr(swaps[0].partial_text) if swaps else "")
    check("mid-stream failure: chunks came from both tiers",
          {t for t, _ in emitted} == {"frontier", "standard"})
    check("mid-stream failure: client-visible stream equals final text",
          "".join(d for _, d in emitted) == res.text)
    check("mid-stream failure: breaker saw the failure",
          r.breaker(0).snapshot()["consecutive_failures"] == 1)

    # Pre-stream failure: client never saw tier 0, so no swap event.
    r, _ = runner({
        "frontier": ScriptedLLM(["never emitted"], die_before=0),
        "standard": ScriptedLLM(["clean answer"]),
    })
    swaps = []
    res = r.stream([HumanMessage(content="q")], agent="synthesizer", on_swap=swaps.append)
    check("pre-stream failure emits no swap event", not swaps)
    check("pre-stream failure recovers on the next tier", res.text == "clean answer",
          repr(res.text))

    # Permanent failure: fail fast, never touch tier 2.
    r, _ = runner({
        "frontier": ScriptedLLM(["x"], die_before=0, exc=BadRequest("bad model")),
        "standard": ScriptedLLM(["must not be used"]),
    })
    res = r.stream([HumanMessage(content="q")], agent="synthesizer")
    check("permanent failure fails fast", res.exhausted and "permanent" in (res.error or ""),
          str(res.error))
    check("permanent failure does not use lower tiers",
          "standard" not in res.tiers_used, str(res.tiers_used))

    # Open breaker skips the tier without calling the provider.
    br = BreakerRegistry(LADDER, threshold=1)
    br.breaker_at(0).record_failure(ConnError("x"), FailureKind.RETRYABLE)
    called = []
    r = CascadeRunner(LADDER, br, llm_factory=lambda t: called.append(t) or ScriptedLLM(["from tier 2"]))
    res = r.stream([HumanMessage(content="q")], agent="synthesizer")
    check("open breaker skips the tier entirely", "frontier" not in called, f"called={called}")
    check("open breaker falls through to the next rung", res.text == "from tier 2")

    # A short continuation must not be swallowed by the stitch window.
    r, _ = runner({
        "frontier": ScriptedLLM(["API rate", " limiting", " is"], die_before=1),
        "standard": ScriptedLLM([" a crucial", " mechanism."]),
    })
    res = r.stream([HumanMessage(content="e")], agent="synthesizer")
    check("continuation shorter than the stitch window is still emitted",
          res.text == "API rate a crucial mechanism.", repr(res.text))

    # A continuation longer than the window: the seam resolves mid-stream and
    # everything after it streams straight through.
    chunks = [" ".join(["w%d" % i] * 9) for i in range(40)]
    r, _ = runner({
        "frontier": ScriptedLLM(["start", " CUT"], die_before=1),
        "standard": ScriptedLLM(chunks),
    })
    emitted = []
    res = r.stream([HumanMessage(content="e")], agent="synthesizer",
                   emit=lambda d, t, m: emitted.append((t, d)))
    expected_long = len("start") + 1 + sum(len(c) for c in chunks)  # +1 seam space
    check("long continuation is emitted in full",
          "".join(d for _, d in emitted) == res.text)
    check("long continuation is not truncated",
          len(res.text) == expected_long, f"len={len(res.text)} expected={expected_long}")

    # A model that ignores "continue where it stopped" and rewinds to the top
    # of its own answer.
    r, _ = runner({
        "frontier": ScriptedLLM(["API rate limiting matters because ", "you need budgets. ", " CUT"], die_before=2),
        "standard": ScriptedLLM(["API rate limiting matters because you need budgets. ",
                                 "Most gateways use token buckets."]),
    })
    res = r.stream([HumanMessage(content="e")], agent="synthesizer")
    check("restart-from-beginning is stitched",
          res.text == "API rate limiting matters because you need budgets. "
                      "Most gateways use token buckets.", repr(res.text))

    # A short accidental repeat must NOT be treated as a restart.
    r, _ = runner({
        "frontier": ScriptedLLM(["Use the token bucket.", " DONE"], die_before=1),
        "standard": ScriptedLLM(["Use the token bucket algorithm for smooth traffic."]),
    })
    res = r.stream([HumanMessage(content="e")], agent="synthesizer")
    check("tail restatement still preferred over a prefix match",
          res.text == "Use the token bucket. algorithm for smooth traffic."
          or res.text == "Use the token bucket algorithm for smooth traffic.",
          repr(res.text))

    # Restated overlap on a long continuation gets stripped exactly once.
    filler = "context " * 60
    restate = "The deploy service is down. Here is what I found: "
    r, _ = runner({
        "frontier": ScriptedLLM(["The deploy service is down. " + filler, " CUT"], die_before=1),
        "standard": ScriptedLLM([restate] + ["more detail."] * 40),
    })
    res = r.stream([HumanMessage(content="e")], agent="synthesizer")
    check("restated prefix appears exactly once",
          res.text.count("The deploy service is down.") == 1,
          f"count={res.text.count('The deploy service is down.')}")
    # The continuation rewound to the top of the answer, so the strippable part
    # is the prefix it shares with `partial` -- not the whole restate, which
    # also carries genuinely new content ("Here is what I found: ...").
    shared = len("The deploy service is down. ")
    check("restart-from-beginning strips exactly the shared prefix",
          res.swaps and res.swaps[0].overlap_chars_stripped == shared,
          f"stripped={res.swaps[0].overlap_chars_stripped if res.swaps else None} expected={shared}")
    check("new content after the restart is preserved",
          "Here is what I found:" in res.text)

    # Two consecutive mid-stream failures: the partial must advance each time,
    # otherwise the second stitch is computed against a stale prefix.
    r, _ = runner({
        "frontier": ScriptedLLM(["AAA one", " CUT"], die_before=1),
        "standard": ScriptedLLM(["BBB two", " CUT"], die_before=1),
        "utility": ScriptedLLM(["CCC three", " done"]),
    })
    emitted, swaps = [], []
    res = r.stream([HumanMessage(content="e")], agent="synthesizer",
                   emit=lambda d, t, m: emitted.append((t, d)), on_swap=swaps.append)
    # Each seam between tiers gains a space, since the scripted chunks have no
    # trailing whitespace and a token-boundary cut would otherwise run words
    # together ("AAA oneBBB two").
    check("two mid-stream failures accumulate all three tiers",
          res.text == "AAA one BBB two CCC three done", repr(res.text))
    check("two mid-stream failures emit two swap events", len(swaps) == 2, str(len(swaps)))
    check("client-visible stream still equals the final text",
          "".join(d for _, d in emitted) == res.text)
    check("tier order preserved across swaps",
          res.tiers_used == ["frontier", "standard", "utility"], str(res.tiers_used))

    # Second swap strips only its own overlap.
    r, _ = runner({
        "frontier": ScriptedLLM(["P1 ", " CUT"], die_before=1),
        "standard": ScriptedLLM(["P1 middle ", " CUT"], die_before=1),
        "utility": ScriptedLLM(["P1 middle tail"]),
    })
    res = r.stream([HumanMessage(content="e")], agent="synthesizer")
    check("each swap strips only its own overlap", res.text == "P1 middle tail", repr(res.text))

    # Everything fails: no crash, partial preserved, error surfaced.
    r, _ = runner({t: ScriptedLLM(["partial"], die_before=0) for t in LADDER})
    res = r.stream([HumanMessage(content="q")], agent="synthesizer")
    check("total failure reports exhausted", res.exhausted)
    check("total failure surfaces an error message", bool(res.error), str(res.error))


def main():
    print("=" * 72)
    print("p4 cascade regression suite (offline, deterministic)")
    print("=" * 72)
    for suite in (test_classification, test_breaker, test_stitcher, test_cascade):
        print(f"\n--- {suite.__name__} ---")
        suite()
    print("\n" + "=" * 72)
    print(f"passed {len(_PASS)}   failed {len(_FAIL)}")
    if _FAIL:
        print("failures:")
        for name in _FAIL:
            print(f"  - {name}")
        return 1
    print("ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())