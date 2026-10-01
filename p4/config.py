"""Tier ladder, live prices, SLO thresholds, and DB paths.

Prices are USD per 1M tokens as served by OpenRouter, captured from
``GET /api/v1/models``. They drive the FinOps cost model (cache savings,
budget enforcement) and the README's dollars-saved figures, so they live in
one place rather than being scattered through the modules that spend money.

Hardware
--------
Nothing in this project requires a GPU. The default provider is the hosted
OpenRouter API, so the system runs on any machine with an API key and no local
model server. A local server is optional and chosen by :mod:`p4.providers`:
Ollama for CPU-only, vLLM if a GPU happens to be present. The tier ladder below
describes capability and cost, not hardware -- ``frontier`` means "the most
capable tier available", whatever is serving it.

Note on vLLM/PagedAttention/quantization: those are documented as standalone
verification only. They are not in the request path on any machine, CPU or GPU,
and the README says so explicitly rather than implying a GPU feature is active.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: Directory holding the p4 package; used for artifacts that must sit next to
#: the code rather than in the process working directory.
PHASE4_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PHASE4_DIR)

PHASE4_DB = os.getenv("P4_DB", "p4.db")
CACHE_DB = os.getenv("P4_CACHE_DB", "cache.db")

#: Tracing service name. Phoenix groups traces by this, and it is what a
#: reviewer selects in the UI.
SERVICE_NAME = os.getenv("P4_SERVICE_NAME", "phase4-support-bot")

# ---------------------------------------------------------------------------
# Model tiers
#
# Ordering is by capability/cost. ``CASCADE_ORDER`` is the fallback ladder and
# is walked BY POSITION INDEX in p4/cascade.py -- never keyed by model string.
# Keying by model name is the documented bug in the reference project: two
# tiers sharing a model would collapse into one, and a renamed model would
# silently lose its breaker state.
# ---------------------------------------------------------------------------

TIER_FRONTIER = "frontier"
TIER_STANDARD = "standard"
TIER_UTILITY = "utility"
TIER_CHEAP = "cheap"

#: Fallback ladder for the cascade, most-preferred first.
CASCADE_ORDER = [TIER_FRONTIER, TIER_STANDARD, TIER_UTILITY]

#: Per-tier model and price table. Prices are USD per 1M tokens.
TIERS = {
    TIER_FRONTIER: {
        "model": "openai/gpt-4o",
        "price_in": 2.50,
        "price_out": 10.00,
        "context_length": 128_000,
    },
    TIER_STANDARD: {
        "model": "google/gemini-2.5-flash",
        "price_in": 0.30,
        "price_out": 2.50,
        "context_length": 1_048_576,
    },
    TIER_UTILITY: {
        "model": "openai/gpt-4o-mini",
        "price_in": 0.15,
        "price_out": 0.60,
        "context_length": 128_000,
    },
    TIER_CHEAP: {
        "model": "google/gemini-2.5-flash-lite",
        "price_in": 0.10,
        "price_out": 0.40,
        "context_length": 1_048_576,
    },
}

#: Which tier handles which agent. Cheap where the task is mechanical
#: (classification, routing decisions, summarization), expensive where the
#: output is user-facing prose.
AGENT_TIER = {
    "classify": TIER_CHEAP,
    "supervisor": TIER_CHEAP,
    "summarization": TIER_CHEAP,
    "general_handler": TIER_UTILITY,
    "api_analysis": TIER_UTILITY,
    "billing_analysis": TIER_UTILITY,
    "outage_analysis": TIER_UTILITY,
    "react_agent": TIER_STANDARD,
    "synthesizer": TIER_FRONTIER,
}

#: Virtual budget per tier for one full demo session, in USD.
DEFAULT_TIER_BUDGET = {
    TIER_FRONTIER: 0.50,
    TIER_STANDARD: 0.20,
    TIER_UTILITY: 0.05,
    TIER_CHEAP: 0.02,
}

#: When a tier's budget is exhausted, drop to this tier instead.
DOWNGRADE_TARGET = {
    TIER_FRONTIER: TIER_STANDARD,
    TIER_STANDARD: TIER_UTILITY,
    TIER_UTILITY: TIER_CHEAP,
    TIER_CHEAP: TIER_CHEAP,
}

# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

CACHE_SIMILARITY_THRESHOLD = 0.92

# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

BREAKER_FAILURE_THRESHOLD = 3       # consecutive failures before opening
BREAKER_WINDOW_SECONDS = 60.0       # window those failures are counted in
BREAKER_COOLDOWN_SECONDS = 30.0     # how long OPEN stays open before HALF_OPEN

# ---------------------------------------------------------------------------
# SLOs  (named, with explicit thresholds -- step 7 of the guide)
# ---------------------------------------------------------------------------

SLOS = {
    "slo_stream_latency_p95": {
        "description": "95th percentile end-to-end streamed response latency",
        "threshold": 8000.0,
        "comparator": "lt",
        "unit": "ms",
    },
    "slo_error_rate": {
        "description": "Fraction of runs ending in error over the window",
        "threshold": 0.02,
        "comparator": "lt",
        "unit": "ratio",
    },
    "slo_judge_correctness_floor": {
        "description": "Rolling mean correctness from the LLM judge",
        "threshold": 0.70,
        "comparator": "gte",
        "unit": "score",
    },
    "slo_cache_hit_rate_floor": {
        "description": "Semantic cache hit rate over the window",
        "threshold": 0.30,
        "comparator": "gte",
        "unit": "ratio",
    },
    "slo_fallback_rate": {
        "description": "Fraction of runs that used a fallback tier",
        "threshold": 0.10,
        "comparator": "lt",
        "unit": "ratio",
    },
    "slo_circuit_open_events": {
        "description": "Sustained circuit-breaker opens over the window",
        "threshold": 0,
        "comparator": "lte",
        "unit": "count",
    },
}

#: Rolling window the daemon evaluates SLOs over.
SLO_WINDOW_SECONDS = 300.0

#: Where fired pages are written, for screenshot evidence.
PAGE_LOG = os.getenv("P4_PAGE_LOG", "evidence/pages.jsonl")

# ---------------------------------------------------------------------------
# LLM-as-judge  (step 6 -- three dimensions, grounded in the mock data)
# ---------------------------------------------------------------------------

JUDGE_DIMENSIONS = ["correctness", "safety", "tone"]

# ---------------------------------------------------------------------------
# vLLM / PagedAttention / quantization
# ---------------------------------------------------------------------------
# NOT in the request path, on any machine. The project runs fully on CPU via
# the hosted API or a local Ollama server; none of this code executes during a
# normal run. The numbers below are the arithmetic a reviewer would ask about,
# recorded so the claim is checkable rather than decorative.

#: Bytes per KV element is 2 (K and V) * bytes_per_element * num_kv_heads
#: * head_dim. The block allocator is not implemented; the figure is what a
#: served fp8-KV cache would hold for a 7B-class model.
QUANTIZATION_NOTES = {
    "fp16_kv_bytes": 2,
    "fp8_kv_bytes": 1,
    "awq_weight_bits": 4,
    "gptq_weight_bits": 4,
    "in_request_path": False,
    "note": (
        "vLLM is an optional local provider (p4/providers.py) and is the only "
        "feature that needs a GPU; the project does not require one. AWQ/GPTQ "
        "4-bit weights halve weight memory versus fp16 and cost a small amount "
        "of accuracy; fp8 KV cache halves KV memory versus fp16. None of this "
        "is applied by this codebase -- no server, no quantization, no "
        "PagedAttention allocator. Recorded as documented arithmetic only."
    ),
}


def tier_model(tier: str) -> str:
    """Resolve a tier name to its model string."""
    spec = TIERS.get(tier)
    if not spec:
        raise KeyError(f"unknown tier: {tier!r} (known: {sorted(TIERS)})")
    return spec["model"]


def tier_price(tier: str) -> tuple:
    """Return (price_in, price_out) USD per 1M tokens for a tier."""
    spec = TIERS.get(tier)
    if not spec:
        raise KeyError(f"unknown tier: {tier!r}")
    return spec["price_in"], spec["price_out"]


def cost_usd(tier: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Cost in USD for one call against a tier.

    Delegates to :func:`p4.providers.effective_price` so that a local provider
    prices at zero while OpenRouter prices at the real rate. This stays the only
    place that converts tokens to money, so the cache's savings figure and the
    router's budget ledger cannot drift apart.
    """
    from p4.providers import effective_price

    price_in, price_out = effective_price(tier)
    return (
        prompt_tokens * price_in / 1_000_000
        + completion_tokens * price_out / 1_000_000
    )