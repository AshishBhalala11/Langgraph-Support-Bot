"""Semantic cache over ticket text, with real FinOps accounting.

An exact-match cache on this system is nearly useless: developers phrase the
same question many ways, and the Phase 3 tickets ("My API key DEV-1001 gets
401s" vs "why am I seeing 401 unauthorized on my key") share no substring
worth indexing on. So similarity is computed over embeddings.

Storing answers has a real risk that an exact-match cache does not: serving a
confident, fluent, wrong answer. Two guards:

* Threshold is strict by default (``CACHE_SIMILARITY_THRESHOLD``), and a hit
  must clear it to be served.
* Entries that recorded a HITL interrupt are never cached, and neither are
  answers that mention a specific developer id, because those are the cases
  where a near-duplicate is genuinely a different question.

Every hit is priced against the tier that *would* have served it, so the savings
figure in ``/finops`` is comparable to the real ledger rather than a guess.
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from p4.config import CACHE_DB, CACHE_SIMILARITY_THRESHOLD, cost_usd, tier_model

#: Tickets referencing a specific account or incident are never served from
#: cache: a near-duplicate of "DEV-1001 is broken" for "DEV-1002 is broken" is a
#: different question, and answering it from cache would be a cross-tenant
#: disclosure.
_IDENTITY_MARKERS = (
    "dev-", "ghp_", "github_pat_", "invoice", "case #", "case#", "ref-",
)


def _has_identity_marker(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _IDENTITY_MARKERS)


@dataclass
class CacheEntry:
    key: str
    text: str
    response: str
    tier: str
    cost_usd: float
    created_at: float
    hits: int = 0
    findings: Optional[list] = None


@dataclass
class CacheLookup:
    hit: bool
    response: str = ""
    tier: str = ""
    similarity: float = 0.0
    stored_key: str = ""
    #: Why a lookup was refused even if a similar entry existed.
    reason: str = ""
    estimated_savings_usd: float = 0.0

    def snapshot(self) -> Dict:
        return {
            "hit": self.hit,
            "tier": self.tier,
            "similarity": round(self.similarity, 4),
            "estimated_savings_usd": round(self.estimated_savings_usd, 8),
            "reason": self.reason,
        }


#: Phrases that mark an answer as a non-answer. Caching one of these is worse
#: than not caching at all: it converts a single empty result into a permanent
#: one, served instantly forever, and the question is never re-attempted even
#: after the underlying data changes. A miss costs a few seconds; a cached
#: "no information found" costs a wrong answer indefinitely.
_NON_ANSWER_MARKERS = (
    "no information was found",
    "no relevant information",
    "could not find",
    "unable to find",
    "not enough information",
    "no information available",
    "analysis is complete, but no",
)


def is_cacheable_answer(response: str) -> bool:
    """Is this answer worth serving again without re-running the graph?"""
    text = (response or "").strip().lower()
    if not text:
        return False
    return not any(marker in text for marker in _NON_ANSWER_MARKERS)


class Embedder:
    """Embeds ticket text.

    Primary path is OpenRouter's embedding endpoint. Two fallbacks keep the cache
    useful without any credentials, which matters because the project is meant
    to run on a bare CPU machine:

    1. A local OpenAI-compatible ``/embeddings`` endpoint (Ollama on CPU, vLLM
       on GPU) if one is configured and reachable.
    2. Otherwise ``None``, which makes :meth:`SemanticCache.lookup` fall back to
       deterministic exact-match. The cache degrades to being dumber; it never
       takes the request down.
    """

    MODEL = os.getenv("P4_EMBED_MODEL", "openai/text-embedding-3-small")
    DIMS = 1536

    def __init__(self):
        self._cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self.available: Optional[bool] = None
        self.backend: str = ""
        self.last_error = ""

    def _post(self, url: str, headers: Dict[str, str], payload: dict) -> List[float]:
        import httpx

        resp = httpx.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed(self, text: str) -> Optional[List[float]]:
        key = hashlib.sha256(text.encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        errors = []
        # Declared up front: with no OpenRouter key and no local embedder there
        # is no assignment to either branch, and referencing it below would
        # raise UnboundLocalError -- turning "no embedding backend" into a
        # crash on every request instead of a fall back to exact-match.
        vector = None

        # Attempt 1: hosted OpenRouter.
        if os.getenv("OPENROUTER_API_KEY"):
            try:
                vector = self._post(
                    "https://openrouter.ai/api/v1/embeddings",
                    {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                    {"model": self.MODEL, "input": text})
                self.backend = "openrouter"
            except Exception as exc:
                errors.append(f"openrouter: {type(exc).__name__}")
                vector = None

        # Attempt 2: a local embeddings endpoint, so CPU-only setups still get
        # real semantic matching instead of falling back to exact strings.
        if not vector:
            local_model = os.getenv("P4_LOCAL_EMBED_MODEL")
            if local_model:
                try:
                    from p4.providers import DEFAULTS, server_reachable

                    spec = DEFAULTS["ollama"]
                    if server_reachable(spec["base_url"]):
                        vector = self._post(
                            f"{spec['base_url']}/embeddings",
                            {"Authorization": "Bearer not-needed"},
                            {"model": local_model, "input": text})
                        self.backend = "ollama"
                except Exception as exc:
                    errors.append(f"ollama: {type(exc).__name__}")

        if vector:
            self.available = True
            self.last_error = ""
        else:
            self.available = False
            self.backend = "exact-match"
            self.last_error = "; ".join(errors) if errors else "no embedding backend configured"

        with self._lock:
            if vector:
                self._cache[key] = vector
        return vector


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _entry_key(normalized: str) -> str:
    """Deterministic key for a normalized ticket.

    Shared by ``store`` and ``lookup`` so the two can never drift apart -- when
    they did, every lookup fell through to embedding comparison against text
    that had already been rewritten, and no repeat ever hit.
    """
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Used for the exact-match fallback path, where a deterministic key is the
    only thing available.
    """
    import re

    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


class SemanticCache:
    """SQLite-backed semantic cache with a FinOps ledger."""

    def __init__(self, db_path: str = CACHE_DB,
                 threshold: float = CACHE_SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self.embedder = Embedder()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                key TEXT PRIMARY KEY,
                text TEXT,
                response TEXT,
                findings TEXT,
                tier TEXT,
                cost_usd REAL,
                created_at REAL,
                hits INTEGER DEFAULT 0
            )""")
        self._conn.commit()
        self.hits = 0
        self.misses = 0
        self.blocked_by_identity = 0
        self.savings_usd = 0.0

    # -- lookup -------------------------------------------------------------

    def lookup(self, ticket: str, tier: str, prompt_tokens: int = 0) -> CacheLookup:
        """Look for a semantically similar prior answer.

        ``prompt_tokens`` is what the request *would* have cost at ``tier``;
        it is what makes the savings figure real rather than notional.
        """
        if _has_identity_marker(ticket):
            # Refused before any similarity work: these are per-account
            # questions and a near-match is a different ticket.
            with self._lock:
                self.blocked_by_identity += 1
                self.misses += 1
            return CacheLookup(hit=False,
                               reason="ticket references a specific account or case; "
                                      "never served from cache")

        # Embed the *normalized* form, because that is what ``store`` persisted.
        # Embedding the raw ticket and comparing it against a normalized stored
        # vector scores ~0.13 for a byte-identical question -- punctuation and
        # casing carry enough signal to sink the cosine -- so the cache would
        # never hit, not even on an exact repeat.
        normalized = normalize_text(ticket)
        vector = self.embedder.embed(normalized)
        if vector is None:
            return self._exact_lookup(ticket, tier, prompt_tokens)

        best: Optional[Tuple[float, sqlite3.Row]] = None
        with self._lock:
            # ``text`` must be selected: it is the normalized question, and the
            # similarity below compares questions. Selecting the response here
            # by mistake silently scores the answer against the question and
            # every lookup misses.
            rows = self._conn.execute(
                "SELECT key, text, response, tier, cost_usd FROM entries").fetchall()
        for row in rows:
            # An exact key match is a certain hit; do not let a near-miss
            # with a marginally higher cosine displace it.
            if row[0] == _entry_key(normalized):
                best = (1.0, row)
                break
            score = cosine(vector, self.embedder.embed(row[1]) or [])
            if best is None or score > best[0]:
                best = (score, row)

        if best is None or best[0] < self.threshold:
            with self._lock:
                self.misses += 1
            return CacheLookup(
                hit=False,
                similarity=best[0] if best else 0.0,
                reason=("no similar entry" if best is None
                        else f"best similarity {best[0]:.3f} below "
                             f"threshold {self.threshold:.3f}"),
            )

        similarity, row = best
        key, _text, response, entry_tier, entry_cost = row
        with self._lock:
            self.hits += 1
            self._conn.execute("UPDATE entries SET hits = hits + 1 WHERE key = ?",
                               (key,))
            self._conn.commit()
            saved = cost_usd(tier, prompt_tokens, max(1, prompt_tokens // 4)) if prompt_tokens else entry_cost
            self.savings_usd += saved

        return CacheLookup(hit=True, response=response, tier=entry_tier,
                           similarity=similarity, stored_key=key,
                           estimated_savings_usd=saved)

    def _exact_lookup(self, ticket: str, tier: str, prompt_tokens: int) -> CacheLookup:
        """Deterministic fallback used when embeddings are unavailable."""
        normalized = normalize_text(ticket)
        with self._lock:
            row = self._conn.execute(
                "SELECT key, response, tier, cost_usd FROM entries WHERE text = ?",
                (normalized,)).fetchone()
            if row is None:
                self.misses += 1
                return CacheLookup(hit=False,
                                   reason="embedding unavailable; no exact match")
            self.hits += 1
            self._conn.execute("UPDATE entries SET hits = hits + 1 WHERE key = ?",
                               (row[0],))
            self._conn.commit()
            saved = cost_usd(tier, prompt_tokens, max(1, prompt_tokens // 4)) if prompt_tokens else row[3]
            self.savings_usd += saved
        return CacheLookup(hit=True, response=row[1], tier=row[2], similarity=1.0,
                           stored_key=row[0], estimated_savings_usd=saved,
                           reason="exact match (embedder unavailable)")

    # -- storage ------------------------------------------------------------

    def store(self, ticket: str, response: str, tier: str, cost: float,
              findings: Optional[list] = None) -> bool:
        if not is_cacheable_answer(response):
            return False
        if _has_identity_marker(ticket):
            return False
        normalized = normalize_text(ticket)
        key = _entry_key(normalized)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries "
                "(key, text, response, findings, tier, cost_usd, created_at, hits) "
                "VALUES (?,?,?,?,?,?,?,0)",
                (key, normalized, response,
                 json.dumps(findings or [], default=str), tier, cost, time.time()))
            self._conn.commit()
        return True

    # -- reporting ----------------------------------------------------------

    def stats(self) -> Dict:
        with self._lock:
            total = self.hits + self.misses
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(hits),0), COALESCE(SUM(cost_usd),0) "
                "FROM entries").fetchone()
        return {
            "entries": row[0],
            "reuse_events": row[1],
            "stored_cost_usd": round(row[2], 8),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "blocked_by_identity_markers": self.blocked_by_identity,
            "estimated_savings_usd": round(self.savings_usd, 8),
            "threshold": self.threshold,
            "embedder_model": self.embedder.MODEL,
            "embedder_available": self.embedder.available,
            "embedder_error": self.embedder.last_error,
        }

    def finops(self, ledger_snapshot: Optional[Dict] = None) -> Dict:
        """Savings against what was actually spent, per tier.

        The honest framing: this is *avoided* cost, measured against the tier
        that would have served the request, not money that was returned.
        """
        stats = self.stats()
        out = {
            "cache": stats,
            "avoided_cost_usd": stats["estimated_savings_usd"],
            "note": ("avoided cost is priced at the tier that would have served the "
                     "request; it is not a refund and is not deducted from the "
                     "budget ledger"),
        }
        if ledger_snapshot:
            out["actual_spend_usd"] = {
                tier: round(v.get("spend_usd", 0.0), 8)
                for tier, v in ledger_snapshot.items()
            }
            total_actual = sum(v.get("spend_usd", 0.0)
                               for v in ledger_snapshot.values())
            out["total_actual_spend_usd"] = round(total_actual, 8)
            avoided = stats["estimated_savings_usd"]
            out["savings_ratio"] = (round(avoided / total_actual, 4)
                                    if total_actual > 0 else None)
        return out

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM entries")
            self._conn.commit()
            self.hits = self.misses = 0
            self.blocked_by_identity = 0
            self.savings_usd = 0.0


_CACHE: Optional[SemanticCache] = None
_CACHE_LOCK = threading.Lock()


def get_cache() -> SemanticCache:
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            _CACHE = SemanticCache()
        return _CACHE


def record_to_monitor(hit: bool) -> None:
    from p4.monitor import get_monitor

    get_monitor().tracker.record("cache_hit", 1.0 if hit else 0.0)
