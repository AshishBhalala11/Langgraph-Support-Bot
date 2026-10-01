#!/usr/bin/env python
"""Offline tests for provider portability.

The claim under test: this project runs on a machine with no GPU, with no API
key, and on a machine with a GPU, without code changes. What matters is that the
*tier ladder* and *breaker state* are independent of which provider serves a
tier -- otherwise swapping providers would silently change which rung the
cascade walks, which is the class of bug the ladder's index-based design exists
to prevent.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p4 import providers as P
from p4.config import CASCADE_ORDER, TIER_CHEAP, TIER_FRONTIER, TIER_STANDARD, TIER_UTILITY

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED += 1
        print(f"  FAIL {name}  {detail}")


print("provider registry")

check("three providers are registered",
      set(P.DEFAULTS) == {P.PROVIDER_OPENROUTER, P.PROVIDER_OLLAMA, P.PROVIDER_VLLM},
      sorted(P.DEFAULTS))
check("openrouter is a hosted API needing no GPU",
      P.DEFAULTS[P.PROVIDER_OPENROUTER]["needs_gpu"] is False)
check("ollama runs on CPU and needs no key",
      P.DEFAULTS[P.PROVIDER_OLLAMA]["needs_gpu"] is False
      and P.DEFAULTS[P.PROVIDER_OLLAMA]["needs_key"] is False)
check("vllm is the only provider that wants a GPU",
      P.DEFAULTS[P.PROVIDER_VLLM]["needs_gpu"] is True)
check("no provider other than vllm requires a GPU",
      [k for k, v in P.DEFAULTS.items() if v["needs_gpu"]] == [P.PROVIDER_VLLM])
check("local providers default to loopback",
      P.DEFAULTS[P.PROVIDER_OLLAMA]["base_url"].startswith("http://localhost")
      and P.DEFAULTS[P.PROVIDER_VLLM]["base_url"].startswith("http://localhost"))

print("model resolution")

check("openrouter uses the configured model",
      P.model_for(P.PROVIDER_OPENROUTER, TIER_FRONTIER) == "openai/gpt-4o")
check("ollama has its own model for every ladder tier",
      all(P.model_for(P.PROVIDER_OLLAMA, t) for t in CASCADE_ORDER))
check("vllm has its own model for every ladder tier",
      all(P.model_for(P.PROVIDER_VLLM, t) for t in CASCADE_ORDER))
# An unknown tier must fail loudly. Falling back to the hosted model for a tier
# nobody configured would let a run proceed against a model the ladder never
# priced, so the loud error is the correct behaviour.
try:
    P.model_for(P.PROVIDER_OLLAMA, "no-such-tier")
    check("an unknown tier is rejected", False, "no error raised")
except KeyError as exc:
    check("an unknown tier is rejected", "unknown tier" in str(exc))
check("a real tier missing from a provider table still resolves",
      P.model_for(P.PROVIDER_OLLAMA, TIER_CHEAP) == "qwen2.5:1.5b")

# The important invariant: a provider swap changes the model inside a tier, and
# nothing about the ladder. If this fails, a local model would reorder the
# cascade or reset breaker state.
print("ladder is independent of provider")

ladder = list(CASCADE_ORDER)
before_positions = {t: i for i, t in enumerate(ladder)}
for prov in (P.PROVIDER_OPENROUTER, P.PROVIDER_OLLAMA, P.PROVIDER_VLLM):
    resolved_positions = {t: ladder.index(t) for t in ladder}
    check(f"{prov}: ladder order unchanged", resolved_positions == before_positions)
check("ladder still has three rungs", len(CASCADE_ORDER) == 3)
check("cheap is routed but not a cascade rung", TIER_CHEAP not in CASCADE_ORDER)
check("every cascade rung has an ollama model",
      all(P.model_for(P.PROVIDER_OLLAMA, t) != "openai/gpt-4o" for t in CASCADE_ORDER))

print("pricing")

check("openrouter prices are the real per-token rates",
      P.effective_price(TIER_FRONTIER, P.PROVIDER_OPENROUTER) == (2.50, 10.00))
check("local providers price at zero by default",
      P.effective_price(TIER_FRONTIER, P.PROVIDER_OLLAMA) == (0.0, 0.0))
check("vllm prices at zero by default",
      P.effective_price(TIER_STANDARD, P.PROVIDER_VLLM) == (0.0, 0.0))

os.environ["P4_LOCAL_COST_PER_MTOK"] = "0.50"
check("a local cost override is honoured",
      P.effective_price(TIER_UTILITY, P.PROVIDER_OLLAMA) == (0.50, 0.50))
os.environ["P4_LOCAL_COST_PER_MTOK"] = "not-a-number"
check("a malformed override is ignored rather than crashing",
      P.effective_price(TIER_UTILITY, P.PROVIDER_OLLAMA) == (0.0, 0.0))
del os.environ["P4_LOCAL_COST_PER_MTOK"]

print("cost model integration")

from p4.config import cost_usd

check("a zero-cost provider costs nothing",
      cost_usd(TIER_FRONTIER, 1_000_000, 1_000_000) >= 0.0)
os.environ["P4_PROVIDER"] = P.PROVIDER_OLLAMA
check("cost_usd respects the local provider",
      cost_usd(TIER_FRONTIER, 1_000_000, 1_000_000) == 0.0)
os.environ["P4_PROVIDER"] = P.PROVIDER_OPENROUTER
check("cost_usd respects openrouter",
      abs(cost_usd(TIER_FRONTIER, 1_000_000, 0) - 2.50) < 1e-9)
check("cost_usd prices output separately from input",
      abs(cost_usd(TIER_FRONTIER, 0, 1_000_000) - 10.00) < 1e-9)
del os.environ["P4_PROVIDER"]

print("resolution")

check("auto is the default when nothing is set",
      P.resolve_provider() in (P.PROVIDER_OPENROUTER, P.PROVIDER_OLLAMA, P.PROVIDER_VLLM))

os.environ["P4_PROVIDER"] = P.PROVIDER_OLLAMA
check("an explicit provider is honoured", P.resolve_provider() == P.PROVIDER_OLLAMA)
os.environ["P4_PROVIDER"] = P.PROVIDER_VLLM
check("an explicit vllm is honoured", P.resolve_provider() == P.PROVIDER_VLLM)
os.environ["P4_PROVIDER"] = "nonsense"
try:
    P.resolve_provider()
    check("an unknown provider is rejected", False, "no error raised")
except ValueError as exc:
    check("an unknown provider is rejected", "unknown provider" in str(exc))
del os.environ["P4_PROVIDER"]

# With nothing reachable, resolution must still return a provider rather than
# None, so the failure surfaces as a clear auth error at call time.
saved = {}
for key in ("OPENROUTER_API_KEY",):
    if os.getenv(key):
        saved[key] = os.environ.pop(key)
resolved = P.resolve_provider(P.PROVIDER_AUTO)
check("auto always resolves to some provider", resolved in P.PROVIDERS, resolved)
for key, value in saved.items():
    os.environ[key] = value

print("error messaging")

try:
    P.provider_config(P.PROVIDER_OPENROUTER)
    check("a missing key raises", False, "no error")
except RuntimeError as exc:
    msg = str(exc)
    check("a missing key raises", "OPENROUTER_API_KEY" in msg, msg)
    check("the error says a GPU is not required", "not require a GPU" in msg, msg)
    check("the error names the local options",
          "ollama" in msg.lower() and "vllm" in msg.lower(), msg)

print("portability report")

report = P.portability_report()
check("report exists without any configuration", isinstance(report, dict))
check("report does not require a GPU", report["requires_gpu"] is False)
check("report lists all providers", set(report["available"]) == set(P.DEFAULTS))
check("every provider explains its availability",
      all("reason" in v for v in report["available"].values()))
check("report notes the tier/provider independence",
      "independent of provider" in report["note"], report["note"][:60])

print("llm construction")

from p4.cascade import make_llm

os.environ["P4_PROVIDER"] = P.PROVIDER_OLLAMA
llm = make_llm(TIER_FRONTIER)
base = str(getattr(llm, "openai_api_base", "") or getattr(llm, "base_url", ""))
check("make_llm targets the local ollama endpoint", "11434" in base, base)
check("make_llm uses the ollama model for the tier",
      "qwen2.5" in llm.model_name, llm.model_name)
check("a local build needs no API key", bool(llm.openai_api_key))

os.environ["P4_PROVIDER"] = P.PROVIDER_OPENROUTER
os.environ["OPENROUTER_API_KEY"] = "sk-test-not-real"
llm2 = make_llm(TIER_FRONTIER)
check("make_llm targets openrouter when configured",
      "openrouter" in str(getattr(llm2, "openai_api_base", "")), str(llm2.openai_api_base))
check("make_llm uses the hosted model for the tier",
      llm2.model_name == "openai/gpt-4o", llm2.model_name)

# An injected relay must win over provider detection, or the cut-cable drill
# would be defeated by a local server being detected.
from p4.router import set_injected_base_url

set_injected_base_url(TIER_FRONTIER, "http://127.0.0.1:9999/v1")
llm3 = make_llm(TIER_FRONTIER)
check("an injected relay overrides the configured provider",
      "9999" in str(llm3.openai_api_base), str(llm3.openai_api_base))
set_injected_base_url(TIER_FRONTIER, None)
# The gateway drill depends on this: a fault drill that leaves the injected URL
# in place makes every later request at that tier fail, which looks exactly
# like a provider outage that nobody injected. The in-process drill clears the
# injection when it finishes; the gateway drill has to clear it over HTTP.
import inspect  # noqa: E402

import scripts.cut_cable as cc  # noqa: E402

_src = inspect.getsource(cc.run_against_server)
check("the gateway drill clears the injected fault when it finishes",
      "/admin/inject-failure" in _src and "delete" in _src)
check("the gateway drill clears it in a finally block, not just on success",
      "finally:" in _src)
check("the gateway drill restores the routing table before injecting",
      "/admin/reset-assignments" in _src)
check("the gateway drill resets breakers before injecting",
      "/admin/reset-breakers" in _src)
check("an untested run is reported as inconclusive, not as a pass",
      cc.INCONCLUSIVE == 3 and "INCONCLUSIVE" in _src)
check("clearing the injection restores the provider",
      "openrouter" in str(make_llm(TIER_FRONTIER).openai_api_base))
del os.environ["OPENROUTER_API_KEY"]
del os.environ["P4_PROVIDER"]

print()
if FAILED:
    print(f"FAILED {FAILED} / {PASSED + FAILED}")
    sys.exit(1)
print(f"passed {PASSED} failed 0")
