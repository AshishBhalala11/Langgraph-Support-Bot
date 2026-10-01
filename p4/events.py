"""Typed SSE event schemas for the streaming gateway.

Every event the gateway emits is one of these. Typing them here (rather than
hand-rolling ``json.dumps`` dicts at each yield site) is what lets the
frontend distinguish ``token`` from ``provider.swap`` without string sniffing,
and gives the failure drill something to assert against.

Wire format is ``event: <type>\\ndata: <json>\\n\\n``. The ``type`` field is
repeated inside the JSON body so a client that only parses ``data:`` lines
still gets the discriminant.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Discriminant carried on every SSE frame."""

    RUN_START = "run.start"
    NODE_START = "node.start"
    NODE_END = "node.end"
    TOOL_CALL = "tool.call"
    TOKEN = "token"
    PROVIDER_SWAP = "provider.swap"
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    JUDGE_SCORE = "judge.score"
    SLO_TRIP = "slo.trip"
    INTERRUPT = "interrupt"
    RUN_END = "run.end"
    RUN_ERROR = "run.error"


class BaseEvent(BaseModel):
    """Fields present on every frame regardless of type."""

    type: EventType
    thread_id: Optional[str] = None
    #: Seconds since the request began, so a client can reconstruct timing
    #: without trusting its own clock.
    elapsed_ms: float = 0.0


class RunStart(BaseEvent):
    type: EventType = EventType.RUN_START
    #: Tier ladder the cascade will walk if the run degrades.
    cascade_plan: list[str] = Field(default_factory=list)
    #: Tier actually chosen by the router for the first LLM call.
    primary_tier: Optional[str] = None
    primary_model: Optional[str] = None
    #: False when the request was served from the semantic cache.
    cache_bypassed: bool = True


class NodeStart(BaseEvent):
    type: EventType = EventType.NODE_START
    node: str


class NodeEnd(BaseEvent):
    type: EventType = EventType.NODE_END
    node: str
    duration_ms: float = 0.0
    #: Short preview of what the node produced, for a readable transcript.
    preview: str = ""


class ToolCall(BaseEvent):
    type: EventType = EventType.TOOL_CALL
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Token(BaseEvent):
    type: EventType = EventType.TOKEN
    delta: str
    #: Tier that emitted this chunk. Changes mid-response after a hot-swap,
    #: which is what the failure drill looks for.
    tier: Optional[str] = None
    model: Optional[str] = None


class ProviderSwap(BaseEvent):
    """Emitted when the cascade abandons a tier mid-response.

    ``partial_text`` is what the client already received from the dead tier, so
    a consumer can reconcile its own buffer against ours.
    """

    type: EventType = EventType.PROVIDER_SWAP
    from_tier: str
    to_tier: str
    from_model: str
    to_model: str
    reason: str
    #: Exception class name from the abandoned tier.
    error_type: str = ""
    #: Chunks emitted before the failure.
    chunks_before_swap: int = 0
    #: Characters received from the abandoned tier.
    partial_text: str = ""
    #: Characters of the continuation that duplicated the partial and were
    #: dropped by the overlap stitcher.
    overlap_chars_stripped: int = 0
    #: True when the breaker opened on the abandoned tier.
    breaker_opened: bool = False


class CacheHit(BaseEvent):
    type: EventType = EventType.CACHE_HIT
    similarity: float
    saved_usd: float
    matched_query: str = ""


class CacheMiss(BaseEvent):
    type: EventType = EventType.CACHE_MISS
    similarity: float = 0.0
    stored: bool = False


class JudgeScore(BaseEvent):
    type: EventType = EventType.JUDGE_SCORE
    correctness: float
    safety: float
    tone: float
    overall: float
    reasoning: str = ""


class Interrupt(BaseEvent):
    """HITL gate reached -- the run is suspended awaiting human approval."""

    type: EventType = EventType.INTERRUPT
    draft: dict[str, Any] = Field(default_factory=dict)
    instruction: str = ""


class RunEnd(BaseEvent):
    type: EventType = EventType.RUN_END
    final_response: str = ""
    category: str = ""
    #: Tiers that actually served chunks, in the order they were used.
    tiers_used: list[str] = Field(default_factory=list)
    swap_count: int = 0
    total_cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    #: True when the answer completed on a cheaper tier than it started on.
    degraded: bool = False
    #: True when the run stopped at a human-in-the-loop approval gate, so the
    #: empty final_response is expected rather than a failure.
    awaiting_human: bool = False
    is_followup: bool = False
    #: Cost avoided by serving this answer from the semantic cache.
    saved_usd: float = 0.0
    cache_hit: bool = False


class CacheHit(BaseEvent):
    """A prior answer was similar enough to reuse."""

    type: EventType = EventType.CACHE_HIT
    similarity: float = 0.0
    #: Tier the cached answer was originally produced on.
    served_tier: str = ""
    saved_usd: float = 0.0


class CacheMiss(BaseEvent):
    type: EventType = EventType.CACHE_MISS
    similarity: float = 0.0
    reason: str = ""


class JudgeScore(BaseEvent):
    type: EventType = EventType.JUDGE_SCORE
    correctness: Optional[float] = None
    safety: Optional[float] = None
    tone: Optional[float] = None
    weighted: Optional[float] = None
    reason: str = ""
    tier: str = ""
    #: True when the judge could not run; scores are then absent, not zero.
    unavailable: bool = False


class SLOTrip(BaseEvent):
    """An SLO was breached and the monitor acted."""

    type: EventType = EventType.SLO_TRIP
    slo: str
    value: float = 0.0
    threshold: float = 0.0
    action: str = ""


class RunError(BaseEvent):
    type: EventType = EventType.RUN_ERROR
    message: str
    #: True when every tier in the cascade was exhausted.
    cascade_exhausted: bool = False
    tiers_attempted: list[str] = Field(default_factory=list)
    #: Named ``tiers_tried`` in the emitter; kept as an alias so either
    #: spelling validates.
    tiers_tried: list[str] = Field(default_factory=list)


def to_sse(event: BaseEvent) -> str:
    """Render an event as a complete SSE frame."""
    body = event.model_dump(mode="json")
    return f"event: {event.type.value}\ndata: {event.model_dump_json()}\n\n"