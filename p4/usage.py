"""Token accounting for the Phase 3 graph's LLM calls.

The router resolves an agent to a tier, but Phase 3's ten ``get_llm()`` call
sites just ``.invoke()`` and discard the result. Rather than edit all ten to
record usage, each router-built ``ChatOpenAI`` carries this callback, so spend
is attributed to the right tier without touching agent logic.

This is what makes ``GET /budget`` show real numbers for a run that went
through the swarm, rather than only for runs driven directly by the cascade.
"""

from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler

from p4.config import TIERS, cost_usd


class TokenAccounting(BaseCallbackHandler):
    """Records prompt/completion tokens and USD per LLM call.

    Args:
        tier: the tier the owning LLM was built for.
        agent: logical agent name, for the trace.
        ledger: optional ``BudgetLedger``. Injected to avoid a circular import
            (the ledger lives in p4.router, which builds the LLMs).
    """

    def __init__(self, tier: str, agent: str, ledger=None):
        self.tier = tier
        self.agent = agent
        self.ledger = ledger
        #: (prompt, completion) tuples for this handler's lifetime.
        self.calls: list = []

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage = self._extract_usage(response)
        if not usage:
            return
        prompt_tokens, completion_tokens = usage
        if prompt_tokens == 0 and completion_tokens == 0:
            return
        self.calls.append((prompt_tokens, completion_tokens))
        if self.ledger is not None:
            self.ledger.record(self.tier, prompt_tokens, completion_tokens)

    @staticmethod
    def _extract_usage(response: Any) -> Optional[tuple]:
        """Pull token counts off an LLM result.

        LangChain exposes ``usage_metadata`` on AIMessageChunk (preferred, and
        present on streamed responses) and ``response_metadata['token_usage']``
        on AIMessage from providers that report it. Both are checked because the
        graph mixes streamed and non-streamed calls.
        """
        gen = getattr(response, "generations", None)
        if not gen or not gen[0]:
            return None
        message = gen[0][0].message

        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict) and usage:
            return int(usage.get("input_tokens", 0) or 0), int(
                usage.get("output_tokens", 0) or 0
            )

        meta = getattr(message, "response_metadata", None) or {}
        token_usage = meta.get("token_usage") or meta.get("usage")
        if isinstance(token_usage, dict) and token_usage:
            return (
                int(token_usage.get("prompt_tokens", 0) or 0),
                int(token_usage.get("completion_tokens", 0) or 0),
            )
        return None

    def totals(self) -> Dict:
        prompt = sum(p for p, _ in self.calls)
        completion = sum(c for _, c in self.calls)
        return {
            "agent": self.agent,
            "tier": self.tier,
            "model": TIERS[self.tier]["model"] if self.tier in TIERS else "",
            "calls": len(self.calls),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cost_usd": round(cost_usd(self.tier, prompt, completion), 8),
        }


class AccountingRegistry:
    """Collects one handler per (agent, tier) for a whole request.

    A single swarm run calls several agents, and the same agent may be invoked
    more than once (the ReAct loop). Handlers are shared per agent so the
    totals aggregate rather than reporting only the last call.
    """

    def __init__(self, ledger=None):
        self.ledger = ledger
        self._handlers: Dict[str, TokenAccounting] = {}

    def handler_for(self, agent: str, tier: str) -> TokenAccounting:
        key = f"{agent}:{tier}"
        if key not in self._handlers:
            self._handlers[key] = TokenAccounting(tier, agent, self.ledger)
        return self._handlers[key]

    def snapshot(self) -> Dict:
        per_agent = [h.totals() for h in self._handlers.values()]
        prompt = sum(a["prompt_tokens"] for a in per_agent)
        completion = sum(a["completion_tokens"] for a in per_agent)
        total_cost = sum(a["cost_usd"] for a in per_agent)
        return {
            "per_agent": per_agent,
            "calls": sum(a["calls"] for a in per_agent),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cost_usd": round(total_cost, 8),
        }

    def reset(self) -> None:
        self._handlers.clear()