"""End-to-end proof of the CPU-only claim, with no hosted API key.

Runs the real app against a local OpenAI-compatible server (see
``stub_openai_server``) with ``OPENROUTER_API_KEY`` explicitly removed from the
environment. If this passes, a reviewer on a laptop with no GPU and no API key
can run the project.

Run:  ./venv/bin/python scripts/test_cpu_only.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_results = []


def check(label, passed, detail=""):
    _results.append((label, passed, detail))
    print(f"{'ok  ' if passed else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    return passed


def main():
    # The whole point: no hosted credential of any kind.
    os.environ.pop("OPENROUTER_API_KEY", None)
    os.environ["P4_PROVIDER"] = "ollama"
    os.environ["P4_OLLAMA_BASE"] = ""  # filled in below
    os.environ["P4_DB"] = "/tmp/p4_cpu_only.db"

    from scripts.stub_openai_server import start_stub

    server, base_url = start_stub()
    os.environ["P4_OLLAMA_BASE"] = base_url
    print(f"[cpu-only] local OpenAI-compatible stub at {base_url}")
    print(f"[cpu-only] OPENROUTER_API_KEY set? "
          f"{bool(os.getenv('OPENROUTER_API_KEY'))}")

    try:
        from p4.config import CASCADE_ORDER
        from p4.providers import (DEFAULTS, available_providers, model_for,
                                  portability_report, provider_config)

        name, base_url, api_key = provider_config()
        check("resolves to the local provider with no API key",
              name == "ollama", f"provider={name}")
        check("the provider declares it needs no credential",
              DEFAULTS[name]["needs_key"] is False, f"needs_key={DEFAULTS[name]['needs_key']}")
        check("the provider declares it needs no GPU",
              DEFAULTS[name]["needs_gpu"] is False)
        check("the client is pointed at the local server",
              base_url == os.environ["P4_OLLAMA_BASE"], base_url)
        check("no api key leaked into the client", not api_key)
        for tier in CASCADE_ORDER:
            check(f"the {tier} tier maps to a local model",
                  bool(model_for(name, tier)), model_for(name, tier))

        available = available_providers()
        check("the local server reports as reachable",
              available["ollama"]["usable"] is True,
              available["ollama"].get("reason") or "reachable")
        check("the hosted provider is correctly reported unavailable",
              available["openrouter"]["usable"] is False,
              available["openrouter"].get("reason", ""))
        check("the GPU provider is present but not required",
              "vllm" in available, "vllm needs a CUDA device")

        report = portability_report()
        check("the report resolves the local provider",
              report["resolved_provider"] == "ollama", str(report["resolved_provider"]))
        check("the report states no GPU is required",
              report["requires_gpu"] is False)

        # Now the part that actually matters: a real streamed request, through
        # the real cascade, with no hosted provider anywhere in the path.
        from p4.streaming import StreamRun

        run = StreamRun(use_cache=False)
        events = list(run.run("Explain how API rate limiting works for developers."))

        kinds = [e.type.value for e in events]
        tokens = [e for e in events if e.type.value == "token"]
        end = next((e for e in events if e.type.value == "run.end"), None)

        check("the graph completed", end is not None,
              f"{len(events)} events, no run.error"
              if "run.error" not in kinds else "run.error present")
        check("tokens actually streamed", len(tokens) > 5, f"{len(tokens)} token frames")
        check("every token frame names a tier",
              all(getattr(t, "tier", None) for t in tokens),
              f"tiers={sorted({getattr(t, 'tier', None) for t in tokens})}")
        if end is not None:
            check("a final response was produced", bool(end.final_response),
                  f"{len(end.final_response or '')} chars")
            check("local inference is priced at zero",
                  (end.total_cost_usd or 0.0) == 0.0, f"cost={end.total_cost_usd}")
            check("tokens were counted from the local response",
                  end.tokens_out > 0, f"in={end.tokens_in} out={end.tokens_out}")
            check("the run did not need a fallback tier",
                  end.tiers_used == ["frontier"], str(end.tiers_used))
        check("no error event was emitted", "run.error" not in kinds)

        from p4.breaker import breaker_snapshot

        snap = breaker_snapshot()
        check("the local tier stayed healthy",
              all(v.get("state") == "closed" for v in snap.values()),
              json.dumps({k: v.get("state") for k, v in snap.items()}))
    finally:
        server.shutdown()

    failed = [label for label, ok, _ in _results if not ok]
    print(f"\npassed {len(_results) - len(failed)}   failed {len(failed)}")
    if failed:
        print("failures:")
        for label in failed:
            print(f"  - {label}")
        return 1
    print("\nCPU-ONLY VERIFIED: the project completed a streamed request with no "
          "GPU and no hosted API key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
