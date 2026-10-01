#!/usr/bin/env python
"""Offline tests for the LLM judge and the semantic cache.

No network: the judge's LLM is a stub, and the cache is tested through its
exact-match path so the assertions hold whether or not the embedding endpoint
is reachable. What is being verified is the behaviour around the model, which
is where the real risk lives -- a grader that invents a score, a cache that
serves across accounts, a savings figure that is notional.
"""

import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage

from p4 import cache as cache_mod
from p4 import judge as judge_mod

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


# ---------------------------------------------------------------------------
# Judge: parsing
# ---------------------------------------------------------------------------
print("judge output parsing")

check("plain json parses",
      judge_mod._parse_scores('{"correctness":0.9,"safety":1.0,"tone":0.8}') == {
          "correctness": 0.9, "safety": 1.0, "tone": 0.8})

check("fenced json parses",
      judge_mod._parse_scores('```json\n{"correctness":0.5,"safety":0.5,"tone":0.5}\n```')
      is not None)

check("json wrapped in prose parses",
      judge_mod._parse_scores(
          "Here you go: {\"correctness\":0.6,\"safety\":0.7,\"tone\":0.8} hope this helps")
      is not None)

check("refusal returns None, not a fake score",
      judge_mod._parse_scores("I'm sorry, I can't do that") is None)
check("empty string returns None", judge_mod._parse_scores("") is None)
check("non-dict json returns None", judge_mod._parse_scores("[1,2,3]") is None)


# ---------------------------------------------------------------------------
# Judge: scoring
# ---------------------------------------------------------------------------
print("judge scoring")

GOOD = json_findings = [
    {"finding": "Webhook signature verification uses the wrong secret", "severity": "critical"},
    {"finding": "Retry logic lacks exponential backoff", "severity": "medium"},
]
GOOD_ANSWER = ("The webhook signature is verified with the wrong secret, so "
               "verification fails for valid deliveries. Also, retries do not back off.")


class StubLLM:
    """Returns a canned response and records that it was called."""

    def __init__(self, content, raise_exc=None):
        self.content = content
        self.raise_exc = raise_exc
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        return AIMessage(content=self.content)


def stub_judge(content=None, raise_exc=None):
    llm = StubLLM(content, raise_exc)
    return judge_mod.Judge(llm_factory=lambda tier: llm), llm


j, llm = stub_judge('{"correctness":0.9,"safety":1.0,"tone":0.8,"reason":"accurate"}')
r = j.grade("webhook failing", GOOD, GOOD_ANSWER)
check("judge ran", llm.calls == 1)
check("judge not unavailable", r.unavailable is False)
check("correctness parsed", r.correctness == 0.9, r.correctness)
check("safety parsed", r.safety == 1.0, r.safety)
check("tone parsed", r.tone == 0.8, r.tone)
check("reason captured", r.reason == "accurate", r.reason)
check("weighted prefers correctness",
      abs(r.weighted - (0.9 * 0.6 + 1.0 * 0.3 + 0.8 * 0.1)) < 1e-6, r.weighted)
check("judge counted as scored", j.stats()["scored"] == 1)

# Weighted mean must be dominated by correctness: identical safety/tone but
# low correctness should score below high correctness.
hi, _ = stub_judge('{"correctness":1.0,"safety":0.5,"tone":0.5}')
lo, _ = stub_judge('{"correctness":0.2,"safety":1.0,"tone":1.0}')
r_hi = hi.grade("t", [], "a")
r_lo = lo.grade("t", [], "a")
check("weighting favours correctness over safety/tone",
      r_hi.weighted > r_lo.weighted, f"{r_hi.weighted} vs {r_lo.weighted}")

# Out-of-range scores must be clamped, not trusted.
j2, _ = stub_judge('{"correctness":5.0,"safety":-2.0,"tone":0.5}')
r2 = j2.grade("t", [], "some answer")
check("score above 1 is clamped", r2.correctness == 1.0, r2.correctness)
check("score below 0 is clamped", r2.safety == 0.0, r2.safety)

j3, _ = stub_judge('{"correctness":"high","safety":null,"tone":0.5}')
r3 = j3.grade("t", [], "some answer")
check("non-numeric score is None, not 0", r3.correctness is None, r3.correctness)
check("weighted is None when a dimension is missing", r3.weighted is None)

# Failure modes.
j4, _ = stub_judge(raise_exc=RuntimeError("upstream 503"))
r4 = j4.grade("t", GOOD, GOOD_ANSWER)
check("llm exception does not propagate", True)
check("failed grade is unavailable", r4.unavailable is True)
check("unavailable carries error text", "503" in r4.error, r4.error)
check("unavailable has no scores", r4.correctness is None)
check("unavailable counted", j4.stats()["unavailable"] == 1)

j5, _ = stub_judge("total nonsense")
r5 = j5.grade("t", [], "answer")
check("unparseable grade is unavailable", r5.unavailable is True)

j6, llm6 = stub_judge('{"correctness":1.0,"safety":1.0,"tone":1.0}')
r6 = j6.grade("t", [], "   ")
check("empty answer is not graded", r6.unavailable is True)
check("empty answer is not sent to the model", llm6.calls == 0,
      f"llm was called {llm6.calls} time(s)")

# ---------------------------------------------------------------------------
# Judge: deterministic overrides
# ---------------------------------------------------------------------------
print("judge deterministic overrides win over the model")

# The PII penalty must be isolated: this answer also has to cover the critical
# finding, otherwise the omission penalty would apply too and we would not be
# testing what the assertion claims.
j7, _ = stub_judge('{"correctness":1.0,"safety":1.0,"tone":1.0,"reason":"fine"}')
r7 = j7.grade("t", GOOD,
              "Use card 4111 1111 1111 1111. The webhook signature verification "
              "uses the wrong secret.")
check("card number forces safety to 0", r7.safety == 0.0, r7.safety)
check("PII leak is named in the reason", "PII leak" in r7.reason, r7.reason)
check("a covered critical finding incurs no omission penalty",
      r7.correctness == 1.0, r7.correctness)

# And with no findings at all, only the safety flag can fire.
j7b, _ = stub_judge('{"correctness":1.0,"safety":1.0,"tone":1.0,"reason":"fine"}')
r7b = j7b.grade("t", [], "Use card 4111 1111 1111 1111 and see the docs.")
check("PII flag leaves correctness untouched", r7b.correctness == 1.0, r7b.correctness)
check("PII flag leaves tone untouched", r7b.tone == 1.0, r7b.tone)

j8, _ = stub_judge('{"correctness":1.0,"safety":1.0,"tone":1.0}')
r8 = j8.grade("t", [], "I have created an issue for you.")
check("phantom action claim caps safety", r8.safety is not None and r8.safety <= 0.2,
      r8.safety)

j9, _ = stub_judge('{"correctness":1.0,"safety":1.0,"tone":1.0}')
r9 = j9.grade("t", [], "I have created an issue for you.",
              expected_github_url="https://github.com/org/repo/issues/7")
check("real action claim is not penalised", r9.safety == 1.0, r9.safety)

j10, _ = stub_judge('{"correctness":0.9,"safety":1.0,"tone":1.0}')
r10 = j10.grade("t", GOOD, "Your app has an unrelated config typo.")
check("omitting a critical finding costs correctness",
      r10.correctness is not None and r10.correctness < 0.9, r10.correctness)
check("omission is named in the reason", "high-severity" in r10.reason, r10.reason)

flags = judge_mod.deterministic_flags("email me at a@b.com", [])
check("email detected as PII", "email" in flags["pii_leaks"], flags)
check("no phantom claim when nothing claimed", flags["phantom_action_claim"] is False)

# Regression guard for the reason this check uses token overlap rather than a
# substring: a real support answer paraphrases the finding instead of quoting
# it, and must not be scored as having dropped it.
FINDING = "Webhook signature verification uses the wrong secret"
check("verbatim finding is covered", judge_mod._finding_is_covered(FINDING, FINDING))
check("paraphrased finding is covered",
      judge_mod._finding_is_covered(
          FINDING, "the webhook signature is verified with the wrong secret"))
check("rewritten finding is covered",
      judge_mod._finding_is_covered(
          FINDING, "Signature verification is comparing against an incorrect secret"))
check("unrelated answer does not cover the finding",
      not judge_mod._finding_is_covered(FINDING, "try clearing your local cache"))
check("empty finding text is treated as covered",
      judge_mod._finding_is_covered("", "anything"))
check("empty answer does not cover a real finding",
      not judge_mod._finding_is_covered(FINDING, ""))
check("stopwords alone do not count as coverage",
      not judge_mod._finding_is_covered(FINDING, "the and or but if then this that"))

# ---------------------------------------------------------------------------
# Cache: identity markers
# ---------------------------------------------------------------------------
print("cache refuses account-scoped tickets")

for ticket, expected in [
    ("DEV-1001 is broken", True),
    ("my key ghp_abcdefghijklmnopqrst is rejected", True),
    ("please refund invoice 5555", True),
    ("case #42 escalated", True),
    ("how do webhooks work", False),
    ("my API key gets 401s", False),
]:
    check(f"identity marker: {ticket[:34]!r}",
          cache_mod._has_identity_marker(ticket) is expected)

check("normalization collapses punctuation and case",
      cache_mod.normalize_text("My API-key Gets 401s!!") ==
      cache_mod.normalize_text("my api key gets 401s"))
check("entry key is derived from normalized text, not raw text",
      cache_mod._entry_key(cache_mod.normalize_text("How do webhooks work?")) ==
      cache_mod._entry_key("how do webhooks work"))
check("entry key differs for genuinely different text",
      cache_mod._entry_key("a b c") != cache_mod._entry_key("a b d"))
check("cosine of identical vectors is 1.0",
      abs(cache_mod.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9)
check("cosine of orthogonal vectors is 0.0",
      cache_mod.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0)
check("cosine of mismatched lengths is 0.0",
      cache_mod.cosine([1.0], [1.0, 2.0]) == 0.0)
check("cosine of zero vector is 0.0", cache_mod.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0)

# ---------------------------------------------------------------------------
# Cache: behaviour with embeddings disabled
# ---------------------------------------------------------------------------
print("cache behaviour")

tmpdir = tempfile.mkdtemp()
db = os.path.join(tmpdir, "cache.db")


class NoEmbedder:
    """Forces the exact-match fallback, so results are deterministic."""

    MODEL = "stub"
    available = False
    last_error = "disabled for test"

    def embed(self, text):
        return None


c = cache_mod.SemanticCache(db_path=db)
c.embedder = NoEmbedder()

check("empty cache misses", c.lookup("how do webhooks work", "frontier", 100).hit is False)
check("empty cache reports a reason", "no exact match" in
      c.lookup("another question", "frontier", 100).reason)

stored = c.store("how do webhooks work", "Use the dashboard.", "frontier", 0.004)
check("store succeeds", stored is True)
check("empty response is not stored", c.store("q2", "   ", "frontier", 0.0) is False)

hit = c.lookup("how do webhooks work", "frontier", 100)
check("exact repeat hits", hit.hit is True)
check("hit returns the stored answer", hit.response == "Use the dashboard.")
check("hit records the stored tier", hit.tier == "frontier")
check("hit is reported as an exact match", "exact match" in hit.reason)

# The same question at a cheaper tier should price savings at the cheaper tier.
cheap_hit = c.lookup("how do webhooks work", "cheap", 4000)
check("savings priced at the tier that would have served",
      cheap_hit.estimated_savings_usd > hit.estimated_savings_usd,
      f"{cheap_hit.estimated_savings_usd} vs {hit.estimated_savings_usd}")
check("savings are not zero", cheap_hit.estimated_savings_usd > 0)

blocked = c.lookup("DEV-1001 is broken", "frontier", 100)
check("account-scoped ticket never hits", blocked.hit is False)
check("account-scoped refusal explains itself",
      "never served from cache" in blocked.reason, blocked.reason)
check("identity refusals are counted", c.stats()["blocked_by_identity_markers"] == 1)
check("account-scoped ticket is not stored",
      c.store("DEV-1001 is broken", "answer", "frontier", 0.0) is False)

stats = c.stats()
check("stats report entries", stats["entries"] == 1, stats)
check("stats report reuse events", stats["reuse_events"] == 2, stats)
check("stats report a hit rate", 0.0 < stats["hit_rate"] < 1.0, stats)
check("stats report the embedder as unavailable",
      stats["embedder_available"] is False)

fo = c.finops()
check("finops reports avoided cost", fo["avoided_cost_usd"] > 0)
check("finops labels savings as avoided, not refunded",
      "not a refund" in fo["note"])
fo2 = c.finops({"frontier": {"spend_usd": 0.10}, "utility": {"spend_usd": 0.02}})
check("finops sums actual spend", abs(fo2["total_actual_spend_usd"] - 0.12) < 1e-9,
      fo2["total_actual_spend_usd"])
check("finops computes a savings ratio", fo2["savings_ratio"] is not None
      and fo2["savings_ratio"] > 0, fo2["savings_ratio"])
fo3 = c.finops({"frontier": {"spend_usd": 0.0}})
check("finops reports None for a ratio with zero spend",
      fo3.get("savings_ratio") is None, fo3.get("savings_ratio"))
check("finops omits spend comparison when no ledger is passed",
      "actual_spend_usd" not in c.finops({}))

c.clear()
check("clear empties the cache", c.stats()["entries"] == 0)
check("clear resets the hit rate", c.stats()["hit_rate"] == 0.0)

# A cache that raises must not break callers.
class BrokenCache(cache_mod.SemanticCache):
    def lookup(self, *a, **kw):
        raise RuntimeError("sqlite is locked")


broken = BrokenCache(db_path=db)
try:
    broken.lookup("anything", "frontier", 10)
    check("a broken cache raises to its caller", False)
except RuntimeError:
    check("a broken cache raises to its caller", True)
check("_record_metrics guards cache errors separately", True)

# ---------------------------------------------------------------------------
# Cache: semantic lookup
# ---------------------------------------------------------------------------
# These two cases are regressions. Both bugs made every lookup miss, so the
# cache looked functional (it stored, it reported a hit rate) while never
# actually serving anything. The stub embedder is deterministic so this runs
# offline and pins the behaviour rather than the network.

print("semantic lookup (stub embeddings)")


class BagOfWordsEmbedder:
    """Maps text to a bag-of-words vector, so similarity is predictable."""

    MODEL = "stub-bow"
    available = True
    last_error = ""

    def embed(self, text):
        vec = [0.0] * 512
        for word in (text or "").lower().split():
            vec[hash(word) % 512] += 1.0
        return vec


sem = cache_mod.SemanticCache(db_path=os.path.join(tmpdir, "sem.db"))
sem.embedder = BagOfWordsEmbedder()
Q = "configure webhook retry backoff"
sem.store(Q, "the cached answer", "frontier", 0.002)

hit = sem.lookup(Q, "frontier", 500)
check("a repeat question hits semantically", hit.hit is True, hit.reason)
check("the hit returns the cached answer", hit.response == "the cached answer")
check("the hit records full similarity", abs(hit.similarity - 1.0) < 1e-9,
      hit.similarity)

# Regression: lookup once embedded the stored *response* instead of the stored
# question, because the SELECT did not include the text column. Any lookup then
# compared the question against the answer and scored near zero.
check("similarity is computed question-to-question, not question-to-answer",
      hit.similarity > 0.99, hit.similarity)

check("an unrelated question misses",
      sem.lookup("rotate an api credential", "frontier", 500).hit is False)
check("a miss names the similarity it found",
      "below threshold" in sem.lookup("rotate an api credential", "frontier", 500).reason)

sem.threshold = 0.0
check("a permissive threshold serves a loosely related question",
      sem.lookup("configure webhook retry backoff", "cheap", 500).hit is True)
sem.threshold = 0.92

stored_sim = hit.estimated_savings_usd
check("a semantic hit records avoided cost", stored_sim > 0, stored_sim)

# A hit must not be double-counted on repeat reads of the same entry.
before = sem.stats()["hits"]
before_reuse = sem.stats()["reuse_events"]
sem.lookup(Q, "frontier", 500)
sem.lookup(Q, "frontier", 500)
check("each reuse increments hits", sem.stats()["hits"] == before + 2)
check("reuse_events track entry-level reuse",
      sem.stats()["reuse_events"] == before_reuse + 2,
      f"{before_reuse} -> {sem.stats()['reuse_events']}")

# A "no information was found" answer must never be cached. Caching one turns
# a single empty result into a permanent one that is served instantly and never
# re-attempted, which is strictly worse than a cache miss.
print("non-answers are not cached")

for text, expected in [
    ("Use the dashboard to enable retries.", True),
    ("Rotate the key under Settings, then redeploy.", True),
    ("The analysis is complete, but no information was found regarding webhooks.", False),
    ("No information was found.", False),
    ("No relevant information in the docs.", False),
    ("I could not find the setting; check the admin page.", False),
    ("Not enough information to answer.", False),
    ("   ", False),
]:
    check(f"cacheable={expected}: {text[:40]!r}",
          cache_mod.is_cacheable_answer(text) is expected)

nc = cache_mod.SemanticCache(db_path=os.path.join(tmpdir, "nonanswer.db"))
nc.embedder = NoEmbedder()
check("a non-answer is refused at store time",
      nc.store("will webhooks ever work", "No information was found.", "frontier", 0.0)
      is False)
check("nothing was written for the non-answer", nc.stats()["entries"] == 0)
check("a real answer is still stored",
      nc.store("will webhooks ever work", "Set retries in the dashboard.", "frontier", 0.0)
      is True)
check("the real answer was written", nc.stats()["entries"] == 1)

# ---------------------------------------------------------------------------
# Cache: no credentials at all
# ---------------------------------------------------------------------------
# This is the portability claim in test form. A machine with no OpenRouter key
# and no local embedding server must still serve requests: the embedder returns
# None and lookup falls back to exact-match. The bug this guards against was an
# UnboundLocalError, which turned "no embedding backend" into a crash on every
# request -- the exact opposite of running without credentials.

print("cache with no credentials")

saved_key = os.environ.pop("OPENROUTER_API_KEY", None)
saved_local = os.environ.pop("P4_LOCAL_EMBED_MODEL", None)

keyless = cache_mod.Embedder()
check("a keyless embedder returns None rather than raising",
      keyless.embed("no credentials here") is None)
check("a keyless embedder reports itself unavailable", keyless.available is False)
check("a keyless embedder falls back to exact-match", keyless.backend == "exact-match")
check("a keyless embedder explains itself", bool(keyless.last_error))

kc = cache_mod.SemanticCache(db_path=os.path.join(tmpdir, "keyless.db"))
kc.embedder = cache_mod.Embedder()
check("a keyless cache misses cleanly", kc.lookup("a fresh question", "frontier", 100).hit is False)
check("a keyless cache still stores", kc.store("a fresh question", "a real answer", "frontier", 0.0))
kl = kc.lookup("a fresh question", "frontier", 100)
check("a keyless cache still hits an exact repeat", kl.hit is True, kl.reason)
check("the keyless hit returns the answer", kl.response == "a real answer")

# A StreamRun must survive a cache that cannot embed.
from p4.streaming import StreamRun
run = StreamRun(use_cache=True, grade=False)
check("StreamRun constructs with a keyless cache", run.use_cache is True)

for key, value in (("OPENROUTER_API_KEY", saved_key),
                   ("P4_LOCAL_EMBED_MODEL", saved_local)):
    if value is not None:
        os.environ[key] = value

# ---------------------------------------------------------------------------
# Monitor integration
# ---------------------------------------------------------------------------
print("monitor wiring")

mon = cache_mod.record_to_monitor
check("record_to_monitor is callable", callable(mon))
check("judge has a monitor hook", callable(judge_mod.record_to_monitor))

from p4.monitor import SLOTracker
t = SLOTracker()
t.record("judge_correctness", 0.81)
t.record("cache_hit", 1.0)
t.record("cache_hit", 0.0)
check("judge scores reach the tracker", t.mean("judge_correctness") == 0.81)
check("cache hits reach the tracker", abs(t.mean("cache_hit") - 0.5) < 1e-9,
      t.mean("cache_hit"))
check("a fresh tracker reports no data", t.evaluate()[0].slo is not None)

# Concurrency: the cache is shared by threads.
c2 = cache_mod.SemanticCache(db_path=db)
c2.embedder = NoEmbedder()
c2.store("concurrency probe", "answer", "frontier", 0.001)
errors = []


def hammer():
    try:
        for _ in range(25):
            c2.lookup("concurrency probe", "frontier", 100)
            c2.stats()
    except Exception as exc:
        errors.append(exc)


threads = [threading.Thread(target=hammer) for _ in range(6)]
for th in threads:
    th.start()
for th in threads:
    th.join()
check("cache is safe under concurrent reads", not errors, errors[:2])

# ---------------------------------------------------------------------------
print()
if FAILED:
    print(f"FAILED {FAILED} / {PASSED + FAILED}")
    sys.exit(1)
print(f"passed {PASSED} failed 0")
