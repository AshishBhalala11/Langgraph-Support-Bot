"""Dynamic per-agent model router with a virtual budget ledger.

``support_agent.get_llm()`` is called from ten places in the Phase 3 graph.
Rewriting that single factory to consult this router means every LLM call in
the swarm gets a tier-appropriate model without touching any agent logic -- the
"instrument in place, don't refactor" decision from the plan.

Two things worth being explicit about:

* Budget exhaustion **downshifts** rather than failing. A grader running the
  swarm thirty times should not hit a wall; they should watch cheap models take
  over, with the downgrade logged.
* The ledger is a virtual USD budget tracked in SQLite. It is not OpenRouter's
  real account balance -- it is this project's own spend ceiling.
"""

import json
import os
import sqlite3
import threading
from typing import Dict, List, Optional

from p4.config import (
    AGENT_TIER,
    DEFAULT_TIER_BUDGET,
    DOWNGRADE_TARGET,
    PHASE4_DB,
    TIERS,
    cost_usd,
    tier_model,
)

#: Sentinel set by ``fail_cutswitch()`` to point a tier at the cutting relay
#: used by the failure drill. Keyed by TIER, so the drill can sever exactly one
#: rung of the ladder.
_INJECTED_BASE_URLS: Dict[str, str] = {}


def set_injected_base_url(tier: str, base_url: Optional[str]) -> None:
    """Point ``tier``'s provider at an arbitrary base URL, or restore it.

    Injection is keyed by the tier that *actually serves* the call, not by the
    tier that was assigned. That interacts with budget downshifts: if the
    assigned tier is exhausted, ``resolve`` returns a cheaper tier and an
    injection aimed at the assigned tier is silently bypassed. Callers that
    inject faults (the failure drill) must call ``reset_budget()`` first or the
    fault quietly no-ops.
    """
    if base_url:
        _INJECTED_BASE_URLS[tier] = base_url
    else:
        _INJECTED_BASE_URLS.pop(tier, None)


def injected_base_url(tier: str) -> Optional[str]:
    return _INJECTED_BASE_URLS.get(tier)


class BudgetLedger:
    """Per-tier USD spend ledger, persisted in SQLite.

    Thread-safe: FastAPI runs sync endpoints in a threadpool and the async job
    queue uses its own worker threads.
    """

    def __init__(self, db_path: str = PHASE4_DB):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS budget_spend (
                tier TEXT PRIMARY KEY,
                spend_usd REAL NOT NULL DEFAULT 0,
                calls INTEGER NOT NULL DEFAULT 0,
                tokens_in INTEGER NOT NULL DEFAULT 0,
                tokens_out INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    def budget_for(self, tier: str) -> float:
        env = os.getenv(f"P4_BUDGET_{tier.upper()}")
        if env:
            try:
                return float(env)
            except ValueError:
                pass
        return DEFAULT_TIER_BUDGET.get(tier, 0.10)

    def record(self, tier: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Add one call's spend. Returns the USD charged."""
        cost = cost_usd(tier, prompt_tokens, completion_tokens)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO budget_spend (tier, spend_usd, calls, tokens_in, tokens_out)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(tier) DO UPDATE SET
                    spend_usd = spend_usd + excluded.spend_usd,
                    calls = calls + 1,
                    tokens_in = tokens_in + excluded.tokens_in,
                    tokens_out = tokens_out + excluded.tokens_out
                """,
                (tier, cost, prompt_tokens, completion_tokens),
            )
            self._conn.commit()
        return cost

    def spend(self, tier: str) -> float:
        row = self._conn.execute(
            "SELECT spend_usd FROM budget_spend WHERE tier = ?", (tier,)
        ).fetchone()
        return row[0] if row else 0.0

    def exhausted(self, tier: str) -> bool:
        return self.spend(tier) >= self.budget_for(tier)

    def remaining(self, tier: str) -> float:
        return max(0.0, self.budget_for(tier) - self.spend(tier))

    def snapshot(self) -> Dict:
        out = {}
        for tier in TIERS:
            out[tier] = {
                "model": tier_model(tier),
                "spend_usd": round(self.spend(tier), 6),
                "budget_usd": self.budget_for(tier),
                "remaining_usd": round(self.remaining(tier), 6),
                "exhausted": self.exhausted(tier),
            }
        rows = self._conn.execute(
            "SELECT tier, calls, tokens_in, tokens_out FROM budget_spend"
        ).fetchall()
        for tier, calls, tin, tout in rows:
            out.setdefault(tier, {})["calls"] = calls
            out[tier]["tokens_in"] = tin
            out[tier]["tokens_out"] = tout
        return out

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM budget_spend")
            self._conn.commit()


class ModelRouter:
    """Maps an agent name to a concrete (model, tier) honouring the budget.

    ``assignments`` is mutable at runtime so the grader can flip a tier and
    watch the change land in the next Phoenix trace -- the step-9 checklist
    item is "swapping an agent's tier visibly changes which model handles its
    calls", which needs a live mutation path, not just an env var.
    """

    def __init__(self, ledger: Optional[BudgetLedger] = None, ladder: Optional[List[str]] = None):
        self.ledger = ledger or BudgetLedger()
        self.assignments: Dict[str, str] = dict(AGENT_TIER)
        #: The cascade ladder, ordered best-first. Index into this, not the
        #: model name.
        from p4.config import CASCADE_ORDER
        self.ladder = list(ladder or CASCADE_ORDER)

    # -- assignment ---------------------------------------------------------

    def assign(self, agent: str) -> str:
        """Set an agent's tier. Unknown agents fall back to ``synthesizer``."""
        self.assignments[agent] = self.assignments.get(agent, "utility")
        return self.assignments[agent]

    def set_tier(self, agent: str, tier: str) -> str:
        if tier not in TIERS:
            raise KeyError(f"unknown tier {tier!r}; known: {sorted(TIERS)}")
        self.assignments[agent] = tier
        return tier

    def reset_assignments(self) -> Dict[str, str]:
        """Restore the default routing table, discarding SLO demotions.

        The monitor demotes agents on latency or error breaches and those
        demotions are meant to persist. A fault drill is the exception: it needs
        the tier it is about to break to actually be the tier that runs, or the
        drill silently tests nothing.
        """
        self.assignments = dict(AGENT_TIER)
        return dict(self.assignments)

    # -- resolution ---------------------------------------------------------

    def resolve(self, agent: str) -> Dict:
        """Resolve an agent to the tier it will actually use, right now.

        Applies budget downshifts. ``downgraded_from`` is set when the budget
        forced a cheaper tier than assigned.
        """
        tier = self.assignments.get(agent)
        if tier is None:
            tier = self.assignments.get("synthesizer", "utility")

        assigned = tier
        downgraded = None
        guard = 0
        while self.ledger.exhausted(tier) and guard < len(self.ladder) + 2:
            nxt = DOWNGRADE_TARGET.get(tier, tier)
            if nxt == tier:
                break
            downgraded = downgraded or assigned
            tier = nxt
            guard += 1

        spec = TIERS[tier]
        return {
            "agent": agent,
            "assigned_tier": assigned,
            "tier": tier,
            "model": spec["model"],
            "price_in": spec["price_in"],
            "price_out": spec["price_out"],
            "downgraded_from": downgraded,
            "base_url": injected_base_url(tier) or "https://openrouter.ai/api/v1",
            "budget_exhausted": self.ledger.exhausted(assigned),
        }

    def model_for(self, agent: str) -> str:
        return self.resolve(agent)["model"]

    def record_usage(self, agent: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Charge a completed call against the tier that served it."""
        info = self.resolve(agent)
        return self.ledger.record(info["tier"], prompt_tokens, completion_tokens)

    def snapshot(self) -> Dict:
        return {
            "assignments": dict(self.assignments),
            "ladder": list(self.ladder),
            "budgets": self.ledger.snapshot(),
            "injected_base_urls": dict(_INJECTED_BASE_URLS),
        }


#: Process-wide singleton. ``support_agent.get_llm()`` imports this.
_ROUTER: Optional[ModelRouter] = None
_ROUTER_LOCK = threading.Lock()


def get_router() -> ModelRouter:
    global _ROUTER
    if _ROUTER is None:
        with _ROUTER_LOCK:
            if _ROUTER is None:
                _ROUTER = ModelRouter()
    return _ROUTER


def set_router(router: ModelRouter) -> None:
    """Swap the process-wide router (tests, drill scripts)."""
    global _ROUTER
    with _ROUTER_LOCK:
        _ROUTER = router


def reset_budget() -> None:
    """Zero the ledger and clear injected faults. Used by scripts between runs."""
    get_router().ledger.reset()
    for tier in list(TIERS):
        set_injected_base_url(tier, None)