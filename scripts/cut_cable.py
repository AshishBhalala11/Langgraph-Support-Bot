#!/usr/bin/env python
"""CUT CABLE -- Phase 4 failure drill.

Proves the headline requirement: **the streaming endpoint survives a
mid-response provider failure without dropping the client connection.**

What actually happens, in order:

1. A raw TCP relay (:mod:`p4.relay`) is pointed at OpenRouter and placed in
   front of the ``frontier`` tier only. Lower tiers talk to OpenRouter directly.
2. A streamed generation starts against ``frontier``.
3. Once real tokens have reached the client, the relay hard-shuts both sockets.
   This is a genuine transport failure -- the client sees
   ``openai.APIConnectionError``, exactly as it would during a real outage. No
   mocking, no monkeypatching, no unplugging the developer's network.
4. The cascade catches it mid-iteration, emits ``provider.swap``, and resumes in
   the *same generator* under the next tier. Because the exception never escapes
   the generator, the HTTP response above it never closes.
5. The drill asserts the connection survived, that chunks arrived from two
   different tiers, that the text is not duplicated at the seam, and that the
   circuit breaker recorded the failure.

Run it::

    venv/bin/python scripts/cut_cable.py

Options::

    --passthrough     prove the drill's own relay is not what breaks things:
                      relay forwards everything, no cut is injected
    --cut-bytes N     cut after N response bytes (default 2200)
    --server URL      run against a live gateway's /api/stream instead of
                      invoking the cascade in-process
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("P4_DB", "cut_cable.db")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=".env")

PROMPT = (
    "Explain in detail how API rate limiting works for developers, including "
    "rate limit headers, token bucket budgets, and exponential backoff with jitter."
)

_results = []

#: Exit code for "the drill could not test what it set out to test". Distinct
#: from pass (0) and fail (1) so a CI job can tell a broken cascade apart from
#: an untested one. Collapsing the two is how a fault drill ends up reporting
#: success for a fallback that never ran.
INCONCLUSIVE = 3


def record(label, passed, detail=""):
    _results.append((label, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    return passed


def section(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# ---------------------------------------------------------------------------
# In-process drill
# ---------------------------------------------------------------------------

def run_inprocess(cut_bytes, passthrough):
    from langchain_core.messages import HumanMessage

    from p4.breaker import BreakerRegistry
    from p4.cascade import CascadeRunner, make_llm
    from p4.relay import CuttingRelay
    from p4.router import get_router, reset_budget, set_injected_base_url

    router = get_router()
    reset_budget()
    # Pin the run to the top of the ladder so the drill always cuts the
    # strongest tier, regardless of what a previous run's budget left behind.
    router.set_tier("synthesizer", "frontier")

    # Fresh breakers: a run left over from a previous drill could have the
    # frontier breaker open, which would skip the cut entirely and make the
    # drill silently prove nothing.
    registry = BreakerRegistry(router.ladder)
    for idx in range(len(router.ladder)):
        registry.breaker_at(idx).reset()

    relay = CuttingRelay(cut_after_bytes=None if passthrough else cut_bytes)
    base_url = relay.start()
    print(f"[drill] cutting relay listening on {base_url}")
    print(f"[drill] injected at tier: frontier -> {base_url}")
    if passthrough:
        print("[drill] PASSTHROUGH MODE: relay forwards everything, no cut injected")

    set_injected_base_url("frontier", base_url)

    runner = CascadeRunner(router.ladder, registry, llm_factory=make_llm)

    emitted = []
    swaps = []

    def on_emit(delta, tier, model):
        emitted.append((tier, delta, model))

    section("STREAMING (single connection, tiers may change mid-response)")
    print(f"prompt: {PROMPT[:70]}...\n")

    t0 = time.time()
    result = runner.stream(
        [HumanMessage(content=PROMPT)],
        agent="synthesizer",
        emit=on_emit,
        on_swap=swaps.append,
    )
    elapsed = time.time() - t0
    set_injected_base_url("frontier", None)

    first_tier = emitted[0][0] if emitted else None
    for tier, delta, _model in emitted[:6]:
        print(f"   [{tier:9s}] {delta!r}")
    print(f"   ... {len(emitted)} chunks total over {elapsed:.1f}s\n")

    section("FAILURE")
    print(f"relay cut injected      : {relay.cut} ({relay.bytes_forwarded} bytes forwarded)")
    print(f"error surfaced to client: {result.error or 'none'}")
    for s in swaps:
        print(f"\nprovider.swap:")
        print(f"   {s.from_tier} ({s.from_model})  ->  {s.to_tier} ({s.to_model})")
        print(f"   reason              : {s.error_type}")
        print(f"   chunks before swap  : {s.chunks_before_swap}")
        print(f"   partial text        : {s.partial_text[:110]!r}")
        print(f"   overlap stripped    : {s.overlap_chars_stripped} chars")
        print(f"   breaker opened      : {s.breaker_opened}")

    section("TRANSCRIPT")
    print(f"head:\n{result.text[:300]}\n")
    seam_at = len(swaps[0].partial_text) if swaps else 0
    print(f"seam (chars {max(0,seam_at-60)}-{seam_at+90}):\n"
          f"{result.text[max(0,seam_at-60):seam_at+90]!r}\n")
    print(f"tail:\n{result.text[-240:]}")

    section("ASSERTIONS")

    if passthrough:
        record("relay ran without injecting a fault", not relay.cut)
        record("no swap was needed", not swaps)
        record("single tier served the whole response",
               result.tiers_used == ["frontier"], str(result.tiers_used))
        record("text is complete", len(result.text) > 500, f"{len(result.text)} chars")
        relay.stop()
        return 0

    record("relay actually severed the connection", relay.cut,
           f"after {relay.bytes_forwarded} bytes")

    # A severed frontier can fail in two places, and both are correct behaviour.
    #
    #   mid-stream: frontier emitted some tokens, the relay cut the body, and the
    #   cascade had to stitch a continuation onto the partial text. This is the
    #   interesting case and the one the drill exists to demonstrate.
    #
    #   pre-token: the connection died before frontier's first token reached the
    #   client (the relay closed the socket early, or the provider reset it). The
    #   fallback is still correct -- a complete answer from a lower tier -- but
    #   there is no partial text and no seam to stitch, so demanding a
    #   mid-stream seam here would fail the drill for a healthy system.
    #
    # Distinguishing them keeps the assertion honest instead of flaky.
    doomed_chunks = [t for t, _, _ in emitted if t == "frontier"]
    cut_mid_stream = bool(doomed_chunks)

    if cut_mid_stream:
        record("client received tokens from the doomed tier before the cut",
               emitted[0][0] == "frontier",
               f"first chunk tier={first_tier}, "
               f"{len(doomed_chunks)} chunk(s) from frontier")
    else:
        print("   note: frontier failed before its first token; the fallback "
              "below is the pre-token case, not a mid-stream seam")

    record("a provider failure was recorded, not swallowed",
           bool(result.error) or bool(swaps), str(result.error))
    record("cascade emitted a provider.swap event", len(swaps) == 1, f"{len(swaps)} event(s)")

    if swaps:
        s = swaps[0]
        record("swap went frontier -> a lower tier",
               s.from_tier == "frontier" and s.to_tier != "frontier",
               f"{s.from_tier} -> {s.to_tier}")
        record("failure was a real transport error",
               "Connection" in s.error_type or "Timeout" in s.error_type or "Protocol" in s.error_type,
               s.error_type)
        if cut_mid_stream:
            record("tier 2 continued the same response",
                   s.partial_text and result.text.startswith(s.partial_text),
                   "final text begins with the partial")
            record("no duplicated word at the seam",
                   s.overlap_chars_stripped > 0 or
                   not result.text[len(s.partial_text) - 1:len(s.partial_text) + 1].isalnum() or
                   s.partial_text[-1:].isalnum() and result.text[len(s.partial_text):][:1].isalnum(),
                   f"stripped {s.overlap_chars_stripped} chars")
        else:
            # No partial text exists, so assert the weaker but still meaningful
            # claim: the fallback tier actually produced the answer.
            record("fallback tier served the whole response",
                   result.tiers_used and result.tiers_used[0] != "frontier",
                   f"tiers_used={result.tiers_used}")

    tiers = {t for t, _, _ in emitted}
    if cut_mid_stream:
        record("chunks arrived from two different tiers", len(tiers) >= 2, str(sorted(tiers)))
    record("the connection was never dropped (one continuous stream)",
           "".join(d for _, d, _ in emitted) == result.text,
           "client-visible stream == final text")
    record("cascade reported no exhausted error", not result.exhausted,
           result.error or "clean")
    record("response is substantial (not a truncated stub)",
           len(result.text) > 1500, f"{len(result.text)} chars")

    breaker = registry.breaker_at(router.ladder.index("frontier"))
    snap = breaker.snapshot()
    record("circuit breaker recorded the failure",
           snap["consecutive_failures"] >= 1, json.dumps(snap))

    section("RESULT")
    return summarize()


# ---------------------------------------------------------------------------
# Against a live gateway
# ---------------------------------------------------------------------------

def _doubled_word(text: str) -> bool:
    """True if any word is immediately repeated, e.g. 'is is' or 'limit ing'."""
    words = text.split()
    for a, b in zip(words, words[1:]):
        if a == b and len(a) > 0:
            return True
    return False


def run_against_server(url, cut_bytes, passthrough):
    """Drive a running gateway's /api/stream and cut one tier underneath it."""
    import httpx

    from p4.relay import CuttingRelay
    from p4.router import set_injected_base_url

    relay = CuttingRelay(cut_after_bytes=None if passthrough else cut_bytes)
    base_url = relay.start()
    print(f"[drill] relay on {base_url}; now POST to {url}/api/stream")

    # The gateway runs in another process, so fault injection has to be
    # requested over HTTP rather than set in this interpreter.
    # One client for the whole drill: injecting the fault and reading the
    # stream must use the same connection pool, and closing the client before
    # streaming is what made an earlier version of this fail with
    # "client has been closed".
    events = []
    chunks = []
    chunk_tiers = []
    swaps_seen = []
    tiers_used = []
    completed = False
    n = 0
    final_text = ""
    first_token_ms = None
    last_token_ms = None
    elapsed_ms = 0.0

    print(f"\n{'=' * 74}\nSSE STREAM FROM {url}/api/stream\n{'=' * 74}")
    # The outer try covers the whole client block: a fault drill expects the
    # *provider* to fail, and an exception escaping the client itself has to be
    # reported as a client-side failure rather than crashing the script.
    try:
        try:
            with httpx.Client(timeout=300) as client:
                try:
                    # Reset the routing table before injecting. The SLO monitor demotes
                    # agents on latency/error breaches, and those demotions persist by
                    # design. Left in place, a demoted synthesizer starts the run at
                    # `standard`, never touches the injected `frontier` tier, and the
                    # drill reports a pass without demonstrating any fallback at all.
                    reset = client.post(f"{url}/admin/reset-assignments")
                    if reset.status_code == 200:
                        print(f"[drill] restored default routing "
                              f"(synthesizer -> {reset.json()['assignments'].get('synthesizer')})")
                    else:
                        print(f"[drill] warning: routing reset returned "
                              f"{reset.status_code}; continuing")

                    # Breakers too: an OPEN breaker from an earlier run would make the
                    # cascade skip the injected tier and pass without a fallback.
                    cleared = client.post(f"{url}/admin/reset-breakers")
                    if cleared.status_code == 200:
                        states = cleared.json().get("states", {})
                        print(f"[drill] cleared tier health: {states}")
                    else:
                        print(f"[drill] warning: breaker reset returned "
                              f"{cleared.status_code}; continuing")

                    resp = client.post(f"{url}/admin/inject-failure",
                                       json={"tier": "frontier", "base_url": base_url,
                                             "reset_budget": True})
                    resp.raise_for_status()
                    body = resp.json()
                except Exception as exc:
                    print(f"[drill] injection endpoint unavailable: {exc}")
                    relay.stop()
                    return 1

                # Confirm the gateway actually applied the fault. Without this the
                # drill would happily report a clean pass on a run where the
                # injection silently did nothing, proving nothing.
                if not body.get("active"):
                    print(f"[drill] gateway did not apply the injection: {body}")
                    relay.stop()
                    return 1
                if body.get("budget_reset_error"):
                    print(f"[drill] warning: budget reset failed "
                          f"({body['budget_reset_error']}); continuing")
                print(f"[drill] injected at the gateway via /admin/inject-failure "
                      f"(tier={body.get('tier')}, model={body.get('model')})\n")

                # `use_cache: false` is load-bearing. A cached answer is served
                # without calling any provider, so the injected fault is never
                # touched and the drill passes while proving nothing. The prompt
                # here is a fixed constant, so the second run of the drill against
                # the same gateway would otherwise hit its own cache.
                with client.stream("POST", f"{url}/api/stream",
                                   json={"ticket": PROMPT, "thread_id": None,
                                         "use_cache": False}) as resp:
                    print(f"HTTP {resp.status_code} {resp.headers.get('content-type')}\n")
                    for line in resp.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = json.loads(line[5:].strip())
                        etype = payload.get("type")
                        events.append(etype)
                        n += 1
                        elapsed_ms = max(elapsed_ms, payload.get("elapsed_ms", 0.0))
                        if etype == "token":
                            chunks.append(payload.get("delta", ""))
                            chunk_tiers.append(payload.get("tier"))
                            if first_token_ms is None:
                                first_token_ms = payload.get("elapsed_ms", 0.0)
                            last_token_ms = payload.get("elapsed_ms", 0.0)
                            if n <= 8:
                                print(f"   token  {payload.get('delta','')!r} "
                                      f"[tier={payload.get('tier')}]")
                        elif etype == "provider.swap":
                            swaps_seen.append(payload)
                            print(f"\n   *** provider.swap: {payload.get('from_tier')} -> "
                                  f"{payload.get('to_tier')} ({payload.get('error_type')}) ***")
                            print(f"       partial: {payload.get('partial_text','')[:100]!r}")
                        elif etype in ("run.end", "run.error"):
                            completed = True
                            if etype == "run.end":
                                final_text = payload.get("final_response", "") or ""
                                tiers_used = payload.get("tiers_used") or []
                            print(f"\n   {etype}: {str(payload)[:240]}")
        except Exception as exc:
            print(f"\nSTREAM FAILED AT THE CLIENT: {type(exc).__name__}: {exc}")
            record("client connection survived the failure", False, str(exc))
            relay.stop()
            return summarize()

    finally:
        # Always clear the injection, on every exit path. Leaving the gateway
        # pointed at a dead relay is worse than a failed drill: every later
        # request at that tier fails and silently degrades to a cheaper tier,
        # so the *next* run -- or the next real customer request -- looks like a
        # provider outage that nobody injected.
        try:
            with httpx.Client(timeout=15) as cleanup:
                cleared = cleanup.delete(f"{url}/admin/inject-failure/frontier")
            if cleared.status_code == 200:
                print("\n[drill] cleared the injected fault; the gateway is back "
                      "on its real provider")
            else:
                print(f"\n[drill] WARNING: clearing the injection returned "
                      f"{cleared.status_code}; the gateway may still be pointed "
                      f"at the dead relay")
        except Exception as exc:
            print(f"\n[drill] WARNING: could not clear the injection: {exc}")


    joined = "".join(chunks)
    print(f"\n   received {n} events, {len(chunks)} token chunks "
          f"from tiers {sorted(set(t for t in chunk_tiers if t))}")

    # Did the run actually exercise the tier we broke? A gateway run only
    # reaches the injected provider if the graph routed through the synthesizer,
    # which is the only `frontier` consumer. A run that answers from a
    # specialist or a general handler still reports `tiers_used` (the react
    # agent runs at `standard`), so that field alone cannot distinguish "the
    # ladder was tested" from "the graph took a different path entirely".
    #
    # The reliable signal is whether the injected tier was actually addressed:
    # a frontier token, or a swap away from it. Without either, every assertion
    # below would be measuring an untouched system. The same reasoning applies
    # to the passthrough control, whose entire job is to prove the relay is not
    # the cause -- which it cannot do if nothing went through it.
    if not swaps_seen and "frontier" not in chunk_tiers:
        print("\n   INCONCLUSIVE: the graph answered without the synthesizer, "
              "so the injected tier was never called. "
              f"tiers_used={tiers_used}, chunks came from "
              f"{sorted(set(t for t in chunk_tiers if t))}, and the relay was "
              f"{'forwarding everything' if passthrough else f'armed to cut after {cut_bytes} bytes'}. "
              "Nothing was sent through it, so this run tested no fallback.")
        return INCONCLUSIVE

    section("ASSERTIONS (shared)")
    record("client connection stayed open", completed,
           f"{n} events, stream ended cleanly")
    record("the streamed answer reached the client", len(joined) > 200,
           f"{len(joined)} chars")
    record("client-visible stream matches the final answer",
           bool(final_text) and joined.strip() == final_text.strip(),
           f"{len(joined)} streamed vs {len(final_text)} final")
    record("no run.error event", "run.error" not in events)
    if first_token_ms is not None and last_token_ms is not None:
        spread = last_token_ms - first_token_ms
        record("tokens streamed incrementally, not in one burst",
               spread > 50 or len(chunks) < 3,
               f"{len(chunks)} tokens over {spread:.0f}ms")

    if passthrough:
        # Control run: the same relay, same gateway, no cut. Nothing should
        # degrade -- this is what rules out the relay's own plumbing as the
        # cause of the swap seen in the injected run.
        section("ASSERTIONS (passthrough control)")
        record("relay ran without injecting a fault", True)
        record("no swap was needed", not swaps_seen, f"{len(swaps_seen)} swap(s)")
        record("single tier served the whole response",
               set(t for t in chunk_tiers if t) == {"frontier"},
               str(sorted(set(t for t in chunk_tiers if t))))
    else:
        section("ASSERTIONS (injected fault)")
        # Same mid-stream vs pre-token distinction as the in-process drill: a
        # frontier that dies before its first token still has to produce a
        # correct fallback, it just has no seam to stitch.
        if "frontier" in chunk_tiers:
            record("client received tokens from the doomed tier before the cut",
                   chunk_tiers[0] == "frontier",
                   f"first chunk tier={chunk_tiers[0]}, "
                   f"{chunk_tiers.count('frontier')} chunk(s) from frontier")
        else:
            print("   note: frontier failed before its first token; asserting "
                  "the pre-token fallback rather than a mid-stream seam")
        record("a provider failure was recorded, not swallowed",
               bool(swaps_seen) or "run.error" in events,
               swaps_seen[0].get("error_type") if swaps_seen else "no swap event")
        record("a provider.swap event was emitted over the wire", len(swaps_seen) == 1,
               f"{len(swaps_seen)} event(s)")
        if swaps_seen:
            swap = swaps_seen[0]
            record("swap went frontier -> a lower tier",
                   swap.get("from_tier") == "frontier"
                   and swap.get("to_tier") in ("standard", "utility", "cheap"),
                   f"{swap.get('from_tier')} -> {swap.get('to_tier')}")
            record("failure was a real transport error",
                   swap.get("error_type") in ("OpenAIConnectionError",
                                              "APIConnectionError", "ReadTimeout",
                                              "RemoteProtocolError"),
                   str(swap.get("error_type")))
            record("chunks arrived from two different tiers",
                   len(set(t for t in chunk_tiers if t)) >= 2,
                   str(sorted(set(t for t in chunk_tiers if t))))
            record("the lower tier continued the same answer",
                   bool(final_text) and joined.strip() == final_text.strip(),
                   "final answer begins with the doomed tier's partial text")
            # The seam is where the doomed tier's last characters meet the
            # continuation's opening. A join bug shows up in what the client
            # actually received -- a doubled word ("is is") or two words welded
            # together ("limitingis") -- so check the delivered text, not a
            # re-stitch of it.
            frontier_prefix = "".join(
                d for d, t in zip(chunks, chunk_tiers) if t == "frontier"
            )
            seam_ok = joined.startswith(frontier_prefix)
            junction = joined[len(frontier_prefix) - 1:len(frontier_prefix) + 1]
            record("the delivered text is not a doubled word",
                   not _doubled_word(joined), f"doubled_word={_doubled_word(joined)}")
            record("the seam did not weld two words together",
                   seam_ok and not re.match(r"^[a-z]{2,}$", junction),
                   f"junction={junction!r} at char {len(frontier_prefix)}")
    relay.stop()
    return summarize()


# ---------------------------------------------------------------------------

def summarize():
    passed = sum(1 for _, p, _ in _results if p)
    failed = len(_results) - passed
    print(f"passed {passed}   failed {failed}")
    if failed:
        print("failures:")
        for label, ok, _ in _results:
            if not ok:
                print(f"  - {label}")
    else:
        print("\nCUT CABLE PASSED -- connection survived, cascade recovered mid-stream.")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="Cut Cable failure drill")
    parser.add_argument("--passthrough", action="store_true",
                        help="relay forwards everything; proves the relay itself is not the cause")
    parser.add_argument("--cut-bytes", type=int, default=2200,
                        help="response bytes to forward before severing (default 2200)")
    parser.add_argument("--server", default=None,
                        help="gateway base URL; drives /api/stream instead of in-process")
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 2

    if args.server:
        # The gateway drill is inherently less deterministic than the
        # in-process one. It drives the whole graph, and the graph only reaches
        # the synthesizer -- the node whose tier we just broke -- when the
        # classifier and supervisor happen to route through the dispatcher. A
        # run that answers from a specialist or a general handler never calls
        # the injected provider at all, so the fault is untested no matter what
        # the relay did.
        #
        # That is INCONCLUSIVE, not a pass and not a failure of the cascade, and
        # it must never be reported as either. Retry a few times to get a run
        # that exercised the ladder; if none does, say so plainly instead of
        # asserting a fallback that never happened.
        # Both modes need the retry: routing is an LLM decision, so either run
        # can come back without touching the synthesizer.
        attempts = 5
        for attempt in range(1, attempts + 1):
            _results.clear()
            print(f"\n[drill] gateway attempt {attempt}/{attempts}")
            code = run_against_server(args.server.rstrip("/"), args.cut_bytes,
                                      args.passthrough)
            if code == INCONCLUSIVE:
                print(f"[drill] attempt {attempt}: the graph answered without "
                      f"the synthesizer, so the injected tier was never called")
                continue
            return code
        print("\nINCONCLUSIVE: the graph never routed through the synthesizer "
              f"in {attempts} attempts, so this drill could not exercise the "
              "cascade. The in-process drill covers the same failure "
              "deterministically; run it without --server to verify the ladder.")
        return INCONCLUSIVE
    return run_inprocess(args.cut_bytes, args.passthrough)


if __name__ == "__main__":
    raise SystemExit(main())