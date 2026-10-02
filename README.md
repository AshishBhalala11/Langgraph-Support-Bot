# Developer Platform Support Bot

A support-bot service for developer-platform tickets. It triages API, billing,
outage, and bug reports through a LangGraph swarm, streams the answer token by
token, and degrades down a cost-and-capability ladder instead of failing.

**Domain:** Developer Platform Support. Includes a gated GitHub issue write tool
(mock mode without real credentials) and human approval before any write during
a production outage.

---

## Quick start (macOS)

**Requires Python 3.12.** `scipy`, pulled in by `arize-phoenix`, declares
`>=3.12`, so older interpreters fail at install time.

```bash
brew install python@3.12            # skip if `python3.12 --version` works
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env                # then set OPENROUTER_API_KEY in .env

# Optional but recommended: spaCy model for Presidio PII scrubbing (~400 MB).
# On macOS, `python -m spacy download en_core_web_lg` often fails with a
# 0-byte / invalid wheel — install from the release URL instead:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl

python api.py
```

Open [http://localhost:8000](http://localhost:8000).

Optional, for the trace UI: `./venv/bin/phoenix serve`, then
[http://localhost:6006](http://localhost:6006). The app does not depend on it; without a collector
running, traces queue in-process and export on the next start.

Without `en_core_web_lg`, Presidio falls back to regex-based PII scrubbing and
logs that at startup. With the model installed you should see
`[security] Presidio active (spaCy en_core_web_lg)`.

### Configuration


| Variable                 | Default                     | Purpose                                               |
| ------------------------ | --------------------------- | ----------------------------------------------------- |
| `OPENROUTER_API_KEY`     | —                           | Hosted API. Not needed for a local provider.          |
| `P4_PROVIDER`            | `auto`                      | `openrouter` / `ollama` / `vllm` / `auto`             |
| `P4_OLLAMA_BASE`         | `http://localhost:11434/v1` | Local Ollama endpoint                                 |
| `P4_VLLM_BASE`           | `http://localhost:8000/v1`  | Local vLLM endpoint                                   |
| `P4_LOCAL_EMBED_MODEL`   | —                           | e.g. `nomic-embed-text`; enables local semantic cache |
| `P4_ADMIN_TOKEN`         | —                           | Extra bearer token for admin endpoints when proxied   |
| `P4_DB`                  | `p4.db`                     | Ledger / checkpointer database                        |
| `P4_JUDGE_TIER`          | `cheap`                     | Tier the judge grades with                            |
| `P4_LOCAL_COST_PER_MTOK` | `0`                         | Charge local providers a real per-token rate          |
| `P4_PHOENIX_INMEMORY`    | —                           | `1` writes spans to local SQLite instead of OTLP      |
| `P4_SSE_KEEPALIVE_S`     | `15`                        | SSE keep-alive comment interval                       |




### Verify the server came up

```bash
curl -s http://localhost:8000/health | python -m json.tool | grep -A3 portability
```

`portability.resolved_provider` confirms which backend is actually serving,
rather than leaving you to assume.

---



## Testing



### In the browser

[http://localhost:8000](http://localhost:8000) serves a working UI, not a placeholder. It exercises the
same endpoints a client would.


| Do this                                                        | What it proves                                                                            |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Type a ticket, click **Submit Ticket**                         | Full non-streaming request: triage, routing, tool call, answer                            |
| Click **Stream**                                               | Token-level SSE: `node.start` / `node.end` / `tool.call` frames arrive incrementally      |
| **Run Verification Suite**                                     | Calls `/api/verify` and prints the result per check                                       |
| **Load Forensics**                                             | Per-node timings, tokens, cost, swaps for a thread                                        |
| Submit a production-outage ticket, then **Approve** / **Deny** | Human-in-the-loop gate: the run suspends on `awaiting_human` and resumes on your decision |


Approve/Deny is the one worth doing by hand. A ticket that triggers the outage
policy will not answer on its own — it ends with `run.end` carrying
`awaiting_human: true` and an empty `final_response`, and stays suspended until
you decide. If you never click, nothing is written to GitHub.

### From the command line

```bash
./venv/bin/python scripts/test_cascade.py       #  70 assertions
./venv/bin/python scripts/test_breakers.py      #  33 assertions
./venv/bin/python scripts/test_streaming.py     #  45 assertions
./venv/bin/python scripts/test_judge_cache.py   # 123 assertions
./venv/bin/python scripts/test_providers.py     #  51 assertions
./venv/bin/python scripts/test_cpu_only.py      #  22 assertions, no API key
./venv/bin/python scripts/test_streaming.py --live   # +12, needs an API key
```

344 offline assertions in total, all passing.

`test_cpu_only.py` needs no API key: it strips the key from the environment,
starts a local OpenAI-compatible server, and drives a real streamed request
through the real cascade.

The failure drills are separate because they kill network I/O on purpose:

```bash
./venv/bin/python scripts/cut_cable.py                                    # 13/13
./venv/bin/python scripts/cut_cable.py --server http://127.0.0.1:8000     # 14/14
./venv/bin/python scripts/cut_cable.py --server http://127.0.0.1:8000 --passthrough  # 8/8
```

They exit `0` on pass, `1` on failure, and `3` for `INCONCLUSIVE` — a run that
never reached the tier under test, which is a different outcome from passing.

### Endpoint checks with curl

```bash
# is it up, and which provider actually served?
curl -s http://localhost:8000/health | python -m json.tool | grep -A3 portability

# the observability endpoints
for e in api/finops api/judge api/cache api/slos; do
  echo "== $e"; curl -s "http://localhost:8000/$e" | python -m json.tool | head -20
done

# a streamed answer: watch frames arrive one at a time
curl -sN -X POST http://localhost:8000/api/stream \
  -H 'Content-Type: application/json' \
  -d '{"ticket":"Our webhook returned 502 after deploy. What should I check first?","use_cache":false}'

# a non-streamed answer
curl -s -X POST http://localhost:8000/api/run \
  -H 'Content-Type: application/json' \
  -d '{"ticket":"Why is my build timing out?"}' | python -m json.tool
```

What to look for in the stream: many `event: token` frames arriving incrementally
rather than one buffered blob at the end, then exactly one `event: run.end`
carrying a non-empty `final_response`.

Use `"use_cache":false` when you want to watch real provider latency — a cache
hit returns one instant frame and proves nothing about streaming. Asking the same
question twice will hit the cache, which is correct behaviour.

---

## Features

### Core
- Support bot for developer-platform tickets (API, billing, outage, bug, general)
- Browser console with sample tickets and live diagnostics panels
- FastAPI service with sync, streaming, and async job APIs

### Multi-agent system
- Ingress security gate → classifier → category routing
- Supervisor with structured worker selection and delegation limits
- ReAct support agent with tool binding and iteration circuit breaker
- Parallel specialists: API analysis, billing analysis, outage analysis
- Outage dispatcher fan-out (LangGraph `Send`)
- Answer synthesizer (deferred under the cascade so tokens stream from the ladder)
- Context summarization and message deduplication
- Multi-turn threads via SQLite checkpoints (`support.db`)

### Tools & integrations
- Mock CRM: `lookup_developer_account`
- Knowledge base search: `search_knowledge_base`
- Service status board: `check_service_status`
- GitHub issue creation — mock by default; live API when `GITHUB_TOKEN` / `GITHUB_REPO` are set
- Idempotent GitHub write keys; HITL gate on high/critical outage drafts

### Security
- Ingress PII redaction (Presidio + spaCy, or regex fallback)
- Prompt-injection blocking with `BLK-…` reference IDs
- Egress PII and uncertainty flagging
- Cache identity-marker blocking (no cross-tenant / account leakage)

### Human-in-the-loop
- Suspend runs on GitHub write (`awaiting_human`)
- Approve, deny, or edit-and-approve issue drafts
- Pending-approvals listing and UI controls

### Streaming & APIs
- Token-level SSE with typed lifecycle / tool / cache / swap / interrupt events
- Synchronous ticket run and async job submit + poll
- Loopback-guarded admin: fault inject/clear, reset breakers, reset tier assignments

### Reliability & cost
- Three-tier cascade (frontier → standard → utility) with mid-stream continuation stitching
- Per-tier circuit breakers
- Dynamic per-agent model routing + SQLite budget ledger and hard caps
- Semantic answer cache (embeddings or exact match) with savings accounting
- Cut-the-cable TCP relay fault injection and automated failure drills
- Provider portability: OpenRouter, Ollama (CPU), vLLM (GPU), and `auto` fallback

### Quality & ops
- Post-response LLM judge (correctness, safety, tone)
- Rolling SLO evaluation with automatic demotion and alert-only paths
- OpenTelemetry tracing with Phoenix export / in-memory span store
- Token and USD usage accounting
- Checkpoint time-travel branching and state correction APIs

---

## Providers

**No GPU is required.** It runs on a CPU-only machine, a laptop with no local
model server, or a GPU box, with no code changes.


| Provider               | Hardware          | API key | Set                  |
| ---------------------- | ----------------- | ------- | -------------------- |
| `openrouter` (default) | any, incl. no GPU | yes     | —                    |
| `ollama`               | CPU only          | no      | `P4_PROVIDER=ollama` |
| `vllm`                 | needs CUDA GPU    | no      | `P4_PROVIDER=vllm`   |


All three speak the OpenAI chat-completions API, so one client serves every
backend. `P4_PROVIDER=auto` prefers a reachable local server and falls back to
the hosted API, so a missing Ollama degrades to a working app rather than a
crash.

The tier ladder is about **capability and cost, not hardware**. `frontier` means
"the most capable tier available", whatever is serving it. Swapping providers
swaps the model *inside* a tier; it never reorders the ladder or resets breaker
state — `scripts/test_providers.py` asserts exactly that.

Local providers price at `0.0` by default, because a developer running Ollama is
not paying per token and charging OpenRouter rates would make the FinOps
dashboard meaningless. Set `P4_LOCAL_COST_PER_MTOK` to model serving cost if you
want real numbers.

With no key and no local server, startup fails up front rather than at the first
customer request:

```
RuntimeError: no usable LLM provider. Set OPENROUTER_API_KEY for the hosted
API, or run a local server (ollama on CPU, vLLM on GPU) and set P4_PROVIDER.
This project runs on CPU-only machines; it does not require a GPU.
```

---



## Architecture

```
POST /api/stream
  │
  ├─ semantic cache lookup ─── hit? → stream cached answer, $0
  │
  ├─ LangGraph swarm (support_agent.py)
  │    triage → supervisor loop (ReAct agent) | dispatcher fan-out (specialists)
  │    HITL suspends the run before any GitHub write; synthesis is deferred
  │
  ├─ CascadeRunner: frontier → standard → utility ── streams the answer token by token
  │    per-tier breaker, per-tier span, hot-swap on retryable failure
  │
  └─ LLM judge + cache store + SLO sample   (all after the last frame)
```

Judging, caching, and SLO sampling happen **after** the final SSE frame, each
separately guarded. A judge outage, an embedding failure, or a SQLite error
cannot delay or corrupt an answer the customer has already received.

### Project layout

```
api.py              FastAPI app, SSE endpoint, admin routes, static UI
support_agent.py    LangGraph swarm: triage, specialists, supervisor, HITL gate
index.html          Browser UI served at /
p4/
  config.py         Tier ladder, model/price table, thresholds
  providers.py      OpenRouter / Ollama / vLLM selection and reachability
  cascade.py        Tier walking, continuation stitching, swap events
  breaker.py        Per-tier circuit breakers
  router.py         Per-agent tier assignment, SQLite budget ledger
  cache.py          Semantic cache with exact-match fallback
  judge.py          LLM grader
  monitor.py        SLO evaluation and degradation actions
  tracing.py        OpenTelemetry spans, Phoenix export
  streaming.py      Graph pump, typed SSE events, token forwarding
  relay.py          Cutting TCP relay used by the failure drill
  events.py         SSE event schemas
  usage.py          Token and cost accounting
scripts/            Test suites, failure drill, local provider stub
evidence/           Recorded drill output
```



### Endpoints


| Endpoint                                        | Purpose                                           |
| ----------------------------------------------- | ------------------------------------------------- |
| `POST /api/stream`                              | Token SSE, typed events                           |
| `POST /api/run`                                 | Synchronous answer (no cascade)                   |
| `POST /api/async` → `GET /api/jobs/{id}`        | 202 + job polling                                 |
| `GET /health`                                   | Breakers, budgets, SLOs, tracing, portability     |
| `GET /api/finops`                               | Spend, caps, avoided cost                         |
| `GET /api/judge`                                | Judge config and counts                           |
| `GET /api/cache`                                | Hit rate, entries, embedder health                |
| `GET /api/slos`                                 | Live verdicts and degradation actions             |
| `GET /api/threads`, `GET /api/history/{id}`     | Conversation history                              |
| `GET /api/pending-approvals`                    | Runs suspended on the HITL gate                   |
| `POST /api/approve/{id}`, `POST /api/deny/{id}` | Resolve a suspended run                           |
| `GET /api/forensics/{id}`                       | Per-node timings, tokens, cost, anomalies         |
| `POST /api/time-travel`, `POST /api/correct`    | Branch a checkpoint, correct saved state          |
| `GET /api/verify`                               | Runs the verification suite                       |
| `POST /admin/inject-failure`                    | Point a tier at the fault relay (loopback only)   |
| `DELETE /admin/inject-failure/{tier}`           | Restore a tier's real provider (loopback only)    |
| `POST /admin/reset-breakers`                    | Clear tier health (loopback only)                 |
| `POST /admin/reset-assignments`                 | Restore default routing, discarding SLO demotions |


Admin endpoints are **enforced** loopback-only, not merely documented — an
exposed copy could redirect the bot's credentials at an attacker's server. Set
`P4_ADMIN_TOKEN` when behind a proxy.

`reset-assignments` exists because SLO demotions persist by design: with the
synthesizer demoted to `standard`, a drill injecting a fault at `frontier` never
touches the injected tier, and reports a pass having demonstrated nothing.

---



## Failure behaviour

**The cut-cable drill** proves the ladder works by actually breaking it. A raw
TCP relay sits in front of a tier and either fails the connection or cuts the
response mid-body, so the cascade must stitch a continuation from the next tier.

Mid-word cuts are normalized to a word boundary so the continuation does not
produce `limitingis`. The tradeoff is that a genuine mid-word truncation is
reported as a clean split — a real content loss, recorded here rather than
discovered later.

The gateway mode drives the whole graph, and the graph only reaches the
synthesizer — the node whose tier gets broken — when the classifier and
supervisor happen to route through the dispatcher. A run that answers from a
specialist never calls the injected provider at all. That is neither a pass nor
a failure, which is why the drill distinguishes `3` (INCONCLUSIVE) from `1`
(fail) and `0` (pass). It retries up to 5 times first, and clears the injection
in a `finally` block — leaving a tier pointed at a dead relay would make every
subsequent request look like an outage nobody injected. The in-process drill
covers the same failure deterministically, without the routing lottery.

**SLO degradation** is automatic and reversible. A p95 latency or error-rate
breach demotes the affected agent to a cheaper tier and logs the action. Judge
correctness and cache hit rate are deliberately **alert-only**: a bad answer or a
cold cache is a signal to a human, not something to auto-tune away.

On a cold process every SLO reports `no_data` rather than a passing 0.0. A false
green on an unobserved metric is worse than an explicit unknown.

**The judge is not a gate.** `JUDGE_GATE_THRESHOLD` exists but is disabled by
default: a flaky grader must never become a production outage, so the judge is a
regression signal over many runs, not a per-response block.

**A cache hit is not free of risk.** Entries that referenced a specific account
(`DEV-1001`, `ghp_…`, `case #42`) are never stored or served, and "no information
was found" answers are never cached — a cached non-answer would be served forever
and never re-attempted.

---



## What is real, and what is not

Everything in the first table was executed. Everything in the second is
documented arithmetic or design intent, and nothing in the second runs during a
normal request.

### Live-wired and verified


| Capability                | Where                                 | Evidence                                                                                                                                   |
| ------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Token-level SSE           | `p4/streaming.py`, `POST /api/stream` | 227 token frames on one run; incremental across 12 consecutive live runs                                                                   |
| SSE keep-alive            | `api.py`                              | 7 comments during an 11.7s run, interleaved with events; verified a conformant parser ignores them and the tokens still reassemble exactly |
| Provider cascade          | `p4/cascade.py`                       | `test_cascade.py` 70/70; seam stitching, continuation prompts                                                                              |
| Circuit breakers          | `p4/breaker.py`                       | `test_breakers.py` 33/33; process-global, single `HALF_OPEN` probe                                                                         |
| Cut-the-cable drill       | `p4/relay.py`, `scripts/cut_cable.py` | 13/13 in-process, 14/14 gateway, 8/8 passthrough (`evidence/`)                                                                             |
| Dynamic routing + budgets | `p4/router.py`                        | per-agent tier assignment, SQLite ledger, hard caps                                                                                        |
| LLM judge                 | `p4/judge.py`                         | graded a live answer at correctness 0.70; `test_judge_cache.py` 123/123                                                                    |
| Semantic cache            | `p4/cache.py`                         | repeat served in 0.7ms vs 11s, $0 vs $0.00127                                                                                              |
| FinOps                    | `GET /api/finops`                     | actual spend, avoided cost, savings ratio kept distinct                                                                                    |
| SLO monitor               | `p4/monitor.py`                       | demoted the synthesizer on a live p95 latency breach                                                                                       |
| OTel + Phoenix            | `p4/tracing.py`                       | span tree verified; OTLP exporter wired                                                                                                    |
| Provider portability      | `p4/providers.py`                     | `test_providers.py` 51/51; `test_cpu_only.py` 22/22 with no key and no GPU                                                                 |
| Async jobs                | `POST /api/async`                     | 3 concurrent jobs completed; 404 on unknown id                                                                                             |




### Not live — read this before claiming anything


| Item                            | Status                    | Why                                                                                                                                       |
| ------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| vLLM as a provider              | **optional, unexercised** | No GPU was available to test it. The code path exists and is unit-tested with a stub server; it has never talked to a real vLLM.          |
| AWQ / GPTQ quantization         | **documentation only**    | No model was quantized. The bit-width tradeoff is arithmetic, not a measurement. `QUANTIZATION_NOTES` in `p4/config.py`.                  |
| PagedAttention allocator        | **not implemented**       | There is no block allocator in this codebase.                                                                                             |
| Phoenix UI screenshots          | **not captured**          | The collector was not running during the final run. Span *structure* is verified via the local SQLite store.                              |
| Cross-worker breaker state      | **not shared**            | The registry is process-global. Under multiple uvicorn workers each process keeps its own breakers. Use one worker, or an external store. |
| Durable job queue               | **in-memory only**        | Jobs are lost on restart. SQLite-backed jobs would be the next step.                                                                      |
| Breaker/cascade on `/api/run`   | **streamed path only**    | `/api/run` and async jobs call `run_ticket` directly, so they do not cascade, grade, or cache.                                            |
| JudgeScore / SLOTrip SSE events | **schemas only**          | The types exist in `p4/events.py` but are never emitted; judge and SLO results are visible via their own endpoints.                       |


---



## Known limitations

- Single-process breaker state; no shared store across workers.
- In-memory job queue; lost on restart.
- Degradation actions live in router state and do not survive a restart.
- PII scrubbing falls back to regex without `en_core_web_lg` installed (see Quick start for the macOS install command).
- No GPU was available, so vLLM has never been exercised against a real server.
- One GitHub-issue text in the drill evidence is seam-demonstration output, not a
clean answer.

