"""Provider selection: hosted API, local CPU, or local GPU.

Why this module exists
----------------------
The tier ladder is about *capability and cost*, not about which vendor serves
it. Hardcoding one vendor would mean the project only runs where that vendor's
API key exists. This module makes the choice explicit so the same code runs:

* **OpenRouter** (default) -- hosted, no local hardware, works everywhere.
  Needs ``OPENROUTER_API_KEY``.
* **Ollama / llama.cpp** -- local, CPU-only, no API key. Slower and smaller
  models, but the project runs with zero credentials and zero GPU.
* **vLLM** -- local, needs a CUDA GPU. The only provider that wants hardware.

The distinction that matters for a reviewer: the *tier* stays meaningful under
every provider. A local 7B model is the ``standard`` tier no matter which server
runs it, because the ladder is walked by tier position and breakers are keyed
by index, not by model string. Swapping providers swaps the model *inside* a
tier; it never reorders the ladder or resets breaker state.

Selection
---------
``P4_PROVIDER`` forces one provider (``openrouter``/``ollama``/``vllm``/
``auto``). Under ``auto``, local providers are used only if their server is
actually reachable, and otherwise it falls back to OpenRouter -- so a missing
Ollama degrades to a working app rather than a crash.

Prices
------
Hosted and local differ in kind, not just amount. OpenRouter bills per token;
a local server bills electricity and GPU-hours. :func:`effective_price` folds
that into a per-token number so the budget ledger stays comparable, and local
providers report ``0.0`` by default so a developer's budget is not silently
consumed by tokens that were free to them. Set ``P4_LOCAL_COST_PER_MTOK`` to
charge a real number when you actually want to model serving cost.
"""

import os
from typing import Dict, Optional, Tuple

from p4.config import (
    TIER_CHEAP, TIER_FRONTIER, TIER_STANDARD, TIER_UTILITY, tier_model,
)

#: Provider identifiers.
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_OLLAMA = "ollama"
PROVIDER_VLLM = "vllm"
PROVIDER_AUTO = "auto"

PROVIDERS = (PROVIDER_OPENROUTER, PROVIDER_OLLAMA, PROVIDER_VLLM)


# ---------------------------------------------------------------------------
# Per-provider defaults
# ---------------------------------------------------------------------------
# ``base_url`` is the OpenAI-compatible endpoint. Ollama, llama.cpp's server,
# and vLLM all expose the OpenAI chat-completions API, which is why the same
# ``ChatOpenAI`` client drives all three -- no separate client code per backend.

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

DEFAULTS: Dict[str, Dict] = {
    PROVIDER_OPENROUTER: {
        "base_url": _OPENROUTER_BASE,
        "env_key": "OPENROUTER_API_KEY",
        "needs_gpu": False,
        "needs_key": True,
        "description": "Hosted API. No local hardware, works on any machine.",
    },
    PROVIDER_OLLAMA: {
        "base_url": os.getenv("P4_OLLAMA_BASE", "http://localhost:11434/v1"),
        "env_key": "P4_OLLAMA_API_KEY",
        "needs_gpu": False,
        "needs_key": False,
        "description": "Local CPU inference. No API key, no GPU required.",
    },
    PROVIDER_VLLM: {
        "base_url": os.getenv("P4_VLLM_BASE", "http://localhost:8000/v1"),
        "env_key": "P4_VLLM_API_KEY",
        "needs_gpu": True,
        "needs_key": False,
        "description": "Local GPU inference. Requires a CUDA device.",
    },
}

#: Per-provider model names, keyed by tier. A provider that cannot serve a
#: given tier's role falls back down the ladder rather than failing the run --
#: the cascade already handles that, so the local table just needs *a* name.
_PROVIDER_MODELS: Dict[str, Dict[str, str]] = {
    PROVIDER_OPENROUTER: {},          # empty: use the configured table as-is
    PROVIDER_OLLAMA: {
        TIER_FRONTIER: "qwen2.5:14b",
        TIER_STANDARD: "qwen2.5:7b",
        TIER_UTILITY: "qwen2.5:3b",
        TIER_CHEAP: "qwen2.5:1.5b",
    },
    PROVIDER_VLLM: {
        TIER_FRONTIER: "Qwen/Qwen2.5-14B-Instruct-AWQ",
        TIER_STANDARD: "Qwen/Qwen2.5-7B-Instruct-AWQ",
        TIER_UTILITY: "Qwen/Qwen2.5-3B-Instruct-AWQ",
        TIER_CHEAP: "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    },
}


def model_for(provider: str, tier: str) -> str:
    """Model string for ``tier`` under ``provider``.

    Falls back to the configured OpenRouter model when the provider has no
    override, so adding a tier does not require touching every provider table.
    """
    table = _PROVIDER_MODELS.get(provider) or {}
    return table.get(tier) or tier_model(tier)


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------

def server_reachable(base_url: str, timeout: float = 1.5) -> bool:
    """Can we actually reach this local server? Cheaper than an LLM call.

    Public because the cache's embedding fallback needs the same check to
    decide whether local semantic matching is possible.

    The status code matters, and not as a detail. A local inference server
    listens on a port that other things also use -- vLLM's default of 8000 is
    the same port this app's own dev server is commonly started on -- and any
    HTTP responder answers a GET. Treating "the socket accepted and something
    replied" as reachable meant a completely unrelated server on port 8000 was
    reported as a working vLLM, so `P4_PROVIDER=auto` selected it, sent
    chat completions to a stranger, and got a 404. The probe asks for the
    provider's own ``/models`` endpoint and requires it to actually succeed.
    """
    try:
        import httpx
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or ""
        # Both an OpenAI-compatible server and Ollama answer /models; a server
        # that is not an inference server will 404 here, which is the signal we
        # want.
        probe = f"{parsed.scheme}://{host}:{port}{path.rstrip('/')}/models"
        resp = httpx.get(probe, timeout=timeout)
        return resp.status_code < 400
    except Exception:
        return False


def available_providers() -> Dict[str, Dict]:
    """Which providers could serve right now, and why not for the others.

    Reported on ``/health`` so a deployment's portability is visible rather
    than assumed.
    """
    out: Dict[str, Dict] = {}
    for name, spec in DEFAULTS.items():
        entry = {
            "description": spec["description"],
            "needs_gpu": spec["needs_gpu"],
            "needs_key": spec["needs_key"],
            "base_url": spec["base_url"],
        }
        if spec["needs_key"]:
            entry["usable"] = bool(os.getenv(spec["env_key"]))
            entry["reason"] = ("" if entry["usable"]
                               else f"{spec['env_key']} is not set")
        else:
            entry["usable"] = server_reachable(spec["base_url"])
            entry["reason"] = ("" if entry["usable"]
                               else f"no server at {spec['base_url']}")
        out[name] = entry
    return out


def resolve_provider(requested: Optional[str] = None) -> str:
    """Pick the provider for this process.

    ``auto`` prefers a reachable local server (cheaper and private) and falls
    back to OpenRouter. It never returns a provider that cannot serve, unless
    nothing at all is available -- in which case OpenRouter is returned so the
    caller gets a clear auth error rather than a connection error.
    """
    requested = (requested or os.getenv("P4_PROVIDER") or PROVIDER_AUTO).lower()
    if requested not in (PROVIDER_AUTO,) + PROVIDERS:
        raise ValueError(
            f"unknown provider {requested!r}; expected one of "
            f"{PROVIDERS} or 'auto'")

    if requested != PROVIDER_AUTO:
        return requested

    # Prefer a local server that is actually up. vLLM is checked before Ollama
    # because a machine with a GPU running vLLM wants vLLM.
    for name in (PROVIDER_VLLM, PROVIDER_OLLAMA):
        spec = DEFAULTS[name]
        if server_reachable(spec["base_url"]):
            return name

    if os.getenv(DEFAULTS[PROVIDER_OPENROUTER]["env_key"]):
        return PROVIDER_OPENROUTER

    # Nothing configured. Return the provider that will produce the most
    # actionable error for the operator.
    for name in (PROVIDER_VLLM, PROVIDER_OLLAMA, PROVIDER_OPENROUTER):
        if server_reachable(DEFAULTS[name]["base_url"]):
            return name
    return PROVIDER_OPENROUTER


def provider_config(provider: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
    """Return ``(provider, base_url, api_key)`` for client construction."""
    provider = resolve_provider(provider)
    spec = DEFAULTS[provider]
    key = os.getenv(spec["env_key"])
    if provider == PROVIDER_OPENROUTER and not key:
        raise RuntimeError(
            "no usable LLM provider. Set OPENROUTER_API_KEY for the hosted API, "
            "or run a local server (ollama on CPU, vLLM on GPU) and set "
            "P4_PROVIDER. This project runs on CPU-only machines; it does not "
            "require a GPU."
        )
    return provider, spec["base_url"], key


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def effective_price(tier: str, provider: Optional[str] = None) -> Tuple[float, float]:
    """USD per 1M tokens for ``tier`` under ``provider``.

    Local providers price at ``0.0`` unless ``P4_LOCAL_COST_PER_MTOK`` is set.
    A developer running Ollama on a laptop is not paying per token, and charging
    them OpenRouter rates would make the budget dashboard meaningless and would
    make a demo look like it is burning money.
    """
    provider = provider or resolve_provider()
    if provider == PROVIDER_OPENROUTER:
        from p4.config import tier_price

        return tier_price(tier)

    override = os.getenv("P4_LOCAL_COST_PER_MTOK")
    if override:
        try:
            price = float(override)
            return price, price
        except ValueError:
            pass
    return 0.0, 0.0


def portability_report() -> Dict:
    """Machine-readable statement of what this deployment can do.

    Included on ``/health`` so "runs on CPU or GPU" is a fact the process
    reports, not a claim in a README that may have drifted from the code.
    """
    try:
        provider, base_url, has_key = provider_config()
        resolved, reason = provider, ""
    except RuntimeError as exc:
        provider, base_url, has_key, resolved, reason = None, None, None, None, str(exc)

    return {
        "resolved_provider": resolved,
        "resolve_error": reason,
        "base_url": base_url,
        "has_api_key": bool(has_key),
        "requires_gpu": False,
        "available": available_providers(),
        "note": (
            "The default provider is the hosted OpenRouter API, so this project "
            "runs on any machine with no GPU. A local server is optional: "
            "Ollama for CPU, vLLM if a GPU is present. The tier ladder and "
            "breaker state are independent of provider."
        ),
    }
