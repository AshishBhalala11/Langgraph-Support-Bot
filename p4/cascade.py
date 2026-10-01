"""Tier fallback cascade with mid-stream hot-swap.

The deliverable this module exists to satisfy: *the streaming endpoint must
survive a mid-response provider failure without dropping the client
connection.* Three pieces make that work.

``stitch_continuation``
    Tier 2 is handed the partial text and re-states its opening. Naive
    concatenation yields ``"API rate limiting isAPI rate limiting is a crucial"``.
    This finds the longest suffix of the partial that matches a prefix of the
    continuation and drops the duplicate.

``stream_with_cascade``
    Wraps ``llm.stream()`` so an exception partway through iteration is caught,
    a ``provider.swap`` event is emitted, and the *same generator* resumes
    under the next tier. Because the exception never escapes the generator, the
    HTTP response above it never closes.

``CascadeRunner``
    Tries tiers in ladder order, consulting the breaker before each attempt.
    Walked by index. Permanent failures trip the breaker and stop the walk
    rather than being papered over by three tiers of fallback.

Verified against a real TCP relay that hard-cuts the socket mid-SSE -- see
``scripts/cut_cable.py``.
"""

import contextlib
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional

from p4.breaker import (
    BreakerRegistry,
    BreakerState,
    CircuitBreaker,
    FailureKind,
    classify_error,
)
from p4.config import CASCADE_ORDER, TIERS
from p4.providers import model_for, resolve_provider
from p4.events import EventType, ProviderSwap
from p4.router import injected_base_url

#: How much of a continuation to buffer before deciding where the seam is.
#: Matches the stitcher's default max_window.
STITCH_WINDOW = 240


# ---------------------------------------------------------------------------
# Overlap-aware stitcher
# ---------------------------------------------------------------------------

#: Minimum characters of shared prefix before we conclude a continuation is
#: restating the answer from the beginning rather than continuing at the seam.
RESTATE_MIN_CHARS = 20


def stitch_continuation(partial: str, continuation: str, max_window: int = 240) -> tuple:
    """Remove the duplication where ``continuation`` repeats text the client already has.

    Returns ``(merged_text, overlap_chars_stripped)``.

    Two distinct restatement shapes are handled, in order:

    1. **Tail restatement** -- the model picks up where it left off but repeats
       the last clause first (``"API rate limiting is"`` + ``"API rate limiting
       is a crucial mechanism"``). The longest suffix of ``partial`` that is also
       a prefix of ``continuation`` is dropped. Word-boundary seams are
       preferred over mid-word ones.

    2. **Restart-from-beginning** -- the model ignores the continuation
       instruction and rewinds to the top of its own answer. Detected by a long
       shared prefix between ``partial`` and ``continuation``; everything up to
       the divergence point is dropped.

    The worst case for either rule is dropping a legitimate verbatim repeat.
    That is benign: the client keeps a single copy of text it already has. The
    opposite mistake -- leaving a doubled clause in the user's face -- is not.
    """
    if not partial:
        return continuation, 0
    if not continuation:
        return partial, 0

    # -- shape 1: overlap between the tail of `partial` and the head of `continuation`
    limit = min(max_window, len(partial), len(continuation))
    for window in range(limit, 0, -1):
        tail = partial[-window:]
        if not continuation.startswith(tail):
            continue
        # Reject mid-word seams when a word-aligned one is available further down
        # the loop; accept it only as a last resort.
        mid_word = (
            window < len(partial)
            and partial[-window - 1].isalnum()
            and tail[0].isalnum()
        )
        if mid_word:
            continue
        return partial + continuation[window:], window
    # Any overlap at all, mid-word seams included.
    for window in range(limit, 0, -1):
        if continuation.startswith(partial[-window:]):
            return partial + continuation[window:], window

    # -- shape 2: continuation rewound to the start of the answer
    shared = 0
    for a, b in zip(partial, continuation):
        if a != b:
            break
        shared += 1
    if shared >= RESTATE_MIN_CHARS:
        return partial + continuation[shared:], shared

    # -- shape 3: genuine continuation with nothing repeated.
    #
    # Normalise the seam whitespace. SSE cuts usually land on a token boundary,
    # so the partial ends mid-sentence with no trailing space ("API rate
    # limiting") and the continuation resumes at the next word ("is a crucial
    # mechanism") -- concatenating those gives "limitingis".
    #
    # Known trade-off: a cut landing *inside* a word ("limit" + "ing") would
    # gain an unwanted space ("limit ing"). Mid-word token splits are far rarer
    # than word-boundary ones, and a missing space is the worse visible defect,
    # so this optimises for the common case. Documented in the README rather
    # than hidden.
    merged = partial + continuation
    if partial and continuation and partial[-1].isalnum() and continuation[0].isalnum():
        merged = partial + " " + continuation
    return merged, 0


# ---------------------------------------------------------------------------
# Cascade result
# ---------------------------------------------------------------------------

@dataclass
class CascadeResult:
    """Outcome of one streamed generation."""

    text: str = ""
    #: Tier per emitted chunk, so a transcript can show which rung served what.
    chunk_tiers: List[str] = field(default_factory=list)
    #: Ordered list of tiers actually used.
    tiers_used: List[str] = field(default_factory=list)
    swaps: List[ProviderSwap] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    exhausted: bool = False
    error: Optional[str] = None

    def snapshot(self) -> Dict:
        return {
            "text": self.text,
            "tiers_used": self.tiers_used,
            "swap_count": len(self.swaps),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "exhausted": self.exhausted,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def make_llm(tier: str, temperature: float = 0, max_retries: int = 0):
    """Build a ``ChatOpenAI`` for ``tier`` against the configured provider.

    ``max_retries=0`` is deliberate: the cascade owns retry policy, and letting
    the SDK retry internally would hide the failure the drill is trying to
    observe.

    One client type serves all three providers (hosted OpenRouter, local
    Ollama, local vLLM) because each exposes the OpenAI chat-completions API.
    That is what makes the project portable: the tier decides *capability*, the
    provider decides *where it runs*, and neither has to know about the other.
    See :mod:`p4.providers`.
    """
    from langchain_openai import ChatOpenAI

    from p4.providers import model_for, provider_config

    provider, base_url, api_key = provider_config()
    # An injected relay wins over the configured provider: that is the whole
    # point of the cut-cable drill, and it must not be defeated by a local
    # server being detected.
    injected = injected_base_url(tier)

    return ChatOpenAI(
        model=model_for(provider, tier),
        base_url=injected or base_url,
        api_key=api_key or "not-needed-for-local",
        temperature=temperature,
        max_retries=max_retries,
        request_timeout=60,
    )


# ---------------------------------------------------------------------------
# Cascade runner
# ---------------------------------------------------------------------------

class CascadeRunner:
    """Walks the tier ladder for a streamed generation.

    ``emit`` is called with each token delta, so the caller can forward it onto
    the SSE stream as it arrives. ``on_swap`` receives a ``ProviderSwap`` event
    when a rung is abandoned mid-response.
    """

    def __init__(
        self,
        ladder: Optional[List[str]] = None,
        breakers: Optional[BreakerRegistry] = None,
        llm_factory: Callable[[str], object] = None,
        on_tier_start: Optional[Callable[[str, str], object]] = None,
    ):
        self.ladder = list(ladder or CASCADE_ORDER)
        self.breakers = breakers or BreakerRegistry(self.ladder)
        self.llm_factory = llm_factory or make_llm
        #: Called with (tier, model) before each rung is attempted; must return a
        #: context manager. Used to open a tracing span per tier so a hot-swap
        #: appears as sibling spans under one cascade span.
        self.on_tier_start = on_tier_start

    def breaker(self, index: int) -> CircuitBreaker:
        return self.breakers.breaker_at(index)

    def _tier_span(self, tier: str, model: str):
        if self.on_tier_start is None:
            return contextlib.nullcontext(None)
        try:
            return self.on_tier_start(tier, model)
        except Exception:
            return contextlib.nullcontext(None)

    def stream(self, messages, agent: str = "synthesizer", emit=None, on_swap=None) -> CascadeResult:
        """Stream a completion, advancing tiers on retryable failure.

        ``emit(delta, tier, model)`` -- called per chunk.
        ``on_swap(ProviderSwap)``    -- called when a rung is abandoned.

        Each tier is asked to generate exactly once. The continuation tier's
        opening is buffered just long enough to find the seam, then the rest
        streams straight through -- so a swap does not cost two generations.
        """
        from p4.config import cost_usd
        from p4.router import get_router

        result = CascadeResult()
        router = get_router()
        # Resolved once per run: every rung in this ladder is served by the same
        # provider, so the model name reported in events and spans must match
        # what make_llm actually built. Reading it from the provider is what
        # keeps a local run from reporting a hosted model it never called.
        provider = resolve_provider()

        assigned = router.resolve(agent)["tier"]
        start_index = self.ladder.index(assigned) if assigned in self.ladder else 0

        prompt_text = _messages_to_text(messages)
        partial = ""
        chunks_so_far = 0
        #: Messages for the current rung. Replaced by a continuation prompt
        #: after a swap.
        current_messages = list(messages)
        is_continuation = False

        for index in range(start_index, len(self.ladder)):
            tier = self.ladder[index]
            breaker = self.breaker(index)
            model = model_for(provider, tier)

            if not breaker.allow_request():
                # Breaker is open: skip this rung without touching the provider.
                continue

            if tier not in result.tiers_used:
                result.tiers_used.append(tier)

            emitted_this_tier = 0
            buffered = ""          # continuation opening, awaiting seam detection
            seam_resolved = not is_continuation   # first tier streams straight out
            received_any = False

            def flush_continuation(piece: str, force: bool = False) -> None:
                """Emit the continuation, dropping any restated overlap.

                Called for the buffered opening and, once the seam is settled,
                for every subsequent chunk. ``force`` resolves the seam
                unconditionally -- used at end-of-stream, where a continuation
                shorter than ``STITCH_WINDOW`` will never otherwise reach the
                client.
                """
                nonlocal buffered, seam_resolved, emitted_this_tier, partial
                if seam_resolved:
                    if piece:
                        partial += piece
                        if emit:
                            emit(piece, tier, model)
                        emitted_this_tier += 1
                    return
                buffered += piece
                if len(buffered) < STITCH_WINDOW and not force:
                    return
                merged, stripped = stitch_continuation(partial, buffered)
                result.swaps[-1].overlap_chars_stripped = stripped
                delta = merged[len(partial):]
                buffered = ""
                seam_resolved = True
                # `partial` must advance, not just the emitted delta -- otherwise
                # a second swap would re-stitch against a stale prefix.
                partial = merged
                result.completion_tokens += 1
                if delta and emit:
                    emit(delta, tier, model)
                    emitted_this_tier += 1

            tier_span = self._tier_span(tier, model)
            try:
                with tier_span as tspan:
                    llm = self.llm_factory(tier)
                    if tspan is not None:
                        tspan.set_attribute("cascade.tier_index", index)
                    for chunk in llm.stream(current_messages):
                        text = getattr(chunk, "content", "") or ""
                        if not text:
                            continue
                        received_any = True

                        if seam_resolved:
                            partial += text
                            result.completion_tokens += 1
                            if emit:
                                emit(text, tier, model)
                            emitted_this_tier += 1
                        else:
                            flush_continuation(text)

                    if tspan is not None:
                        tspan.set_attribute("cascade.chunks", emitted_this_tier)

                if not seam_resolved:
                    # End of stream: resolve the seam even if the whole
                    # continuation was shorter than STITCH_WINDOW, otherwise it
                    # would never be emitted.
                    flush_continuation("", force=True)

                # Tier finished the whole response.
                result.text = partial
                breaker.record_success()
                result.prompt_tokens = _estimate_prompt_tokens(messages)
                result.completion_tokens = max(1, result.completion_tokens)
                result.cost_usd = cost_usd(
                    tier, result.prompt_tokens, result.completion_tokens
                )
                router.record_usage(agent, result.prompt_tokens, result.completion_tokens)
                return result

            except Exception as exc:
                kind = classify_error(exc)
                breaker.record_failure(exc, kind)

                if not seam_resolved:
                    # This rung died before its buffer reached the stitch window.
                    # Force the seam so the text it *did* produce reaches the
                    # client instead of being discarded.
                    flush_continuation("", force=True)

                if kind == FailureKind.PERMANENT:
                    # Misconfiguration, not an outage. Fail fast: falling through
                    # to cheaper tiers would mask a bad model name behind
                    # successful-looking responses.
                    result.text = partial
                    result.error = f"{tier}: permanent failure ({type(exc).__name__}: {exc})"
                    result.exhausted = True
                    return result

                if not received_any and not is_continuation:
                    # Failed before producing anything. The client never saw this
                    # tier, so there is no seam to stitch and no swap to report.
                    result.error = f"{tier}: {type(exc).__name__}"
                    continue

                chunks_so_far += emitted_this_tier
                result.text = partial

                if not self._has_open_tier_after(index):
                    result.error = f"cascade exhausted after {tier}"
                    result.exhausted = True
                    return result

                nxt_tier = self._next_open_tier(index)
                swap = ProviderSwap(
                    from_tier=tier,
                    to_tier=nxt_tier,
                    from_model=model,
                    to_model=model_for(provider, nxt_tier),
                    reason=type(exc).__name__,
                    error_type=type(exc).__name__,
                    chunks_before_swap=chunks_so_far,
                    partial_text=partial,
                    overlap_chars_stripped=0,
                    breaker_opened=breaker.state in (BreakerState.OPEN, BreakerState.HALF_OPEN),
                )
                result.swaps.append(swap)
                if on_swap:
                    on_swap(swap)

                result.tiers_used.append(nxt_tier)
                current_messages = _continuation_messages(prompt_text, partial)
                is_continuation = True

        result.text = partial
        result.error = result.error or "cascade exhausted"
        result.exhausted = True
        return result

    def _next_open_tier(self, after: int) -> str:
        """Name of the next rung whose breaker will admit traffic."""
        for i in range(after + 1, len(self.ladder)):
            if self.breakers.breaker_at(i).allow_request():
                return self.ladder[i]
        return self.ladder[-1]

    def _has_open_tier_after(self, index: int) -> bool:
        return any(
            self.breakers.breaker_at(i).allow_request()
            for i in range(index + 1, len(self.ladder))
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _messages_to_text(messages) -> str:
    parts = []
    for m in messages or []:
        content = getattr(m, "content", m)
        parts.append(content if isinstance(content, str) else str(content))
    return "\n".join(parts)


def _estimate_prompt_tokens(messages) -> int:
    """Rough token estimate (~4 chars/token) for cost accounting.

    Exact counts would need a tokenizer per model family; the FinOps figures
    are estimates and are labelled as such in the README.
    """
    return max(1, len(_messages_to_text(messages)) // 4)


def _continuation_messages(prompt_text: str, partial: str) -> List:
    """Build the follow-up prompt asking tier 2 to continue, not restate.

    The instruction to *continue exactly* is what makes the overlap stitcher a
    small cleanup rather than a load-bearing hack -- the model mostly complies,
    and the stitcher removes the seam when it does not.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(
            content=(
                "You are continuing a response that was cut off mid-sentence "
                "because the upstream connection dropped. Output ONLY the "
                "remaining text. Do NOT repeat, restate, recap, or summarize "
                "anything already written. Do NOT add a preamble. Start exactly "
                "where the delivered text stops, mid-sentence if that is where "
                "it stops."
            )
        ),
        HumanMessage(
            content=(
                f"Original request:\n{prompt_text}\n\n"
                f"Already delivered to the user (verbatim):\n{partial}\n\n"
                "Continue from exactly where that text stops. Output the "
                "continuation only."
            )
        ),
    ]