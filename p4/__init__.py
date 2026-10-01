"""Phase 4 production infrastructure for the Developer Platform Support Bot.

Layer boundaries, in the order the guide requires them be built:

    config      tier ladder, live prices, SLO thresholds
    events      typed SSE event schemas
    telemetry   OpenTelemetry tracer + Phoenix OTLP exporter
    breaker     circuit state machine + error classifier
    router      agent -> tier -> model, budget ledger
    cascade     tier walk, overlap stitcher, mid-stream hot-swap
    cache       semantic cache + FinOps cost model
    judge       LLM-as-judge (correctness / safety / tone)
    monitor     SLO evaluation, tick(), paging
    jobs        durable async job queue

Later modules import earlier ones; no module imports from a later layer, so
each concern stays independently testable.
"""