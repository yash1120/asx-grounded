"""Claude-backed grounded generation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import anthropic
import structlog

from asx_grounded.agent.prompts import SYSTEM_PROMPT, build_user_message
from asx_grounded.config import get_settings
from asx_grounded.models import RetrievedChunk

log = structlog.get_logger()


@dataclass(slots=True)
class GenerationResult:
    text: str
    refused: bool
    refusal_reason: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int


def _parse_refusal(text: str) -> tuple[bool, str]:
    stripped = text.strip()
    if stripped.upper().startswith("REFUSE:"):
        return True, stripped.split(":", 1)[1].strip()
    return False, ""


def generate(query: str, chunks: list[RetrievedChunk], max_tokens: int = 800) -> GenerationResult:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")
    if not chunks:
        return GenerationResult(
            text="REFUSE: No relevant context found in the corpus.",
            refused=True,
            refusal_reason="empty retrieval",
            model=settings.generator_model,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
        )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    user_msg = build_user_message(query, chunks)
    started = time.perf_counter()
    resp = client.messages.create(
        model=settings.generator_model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    text = resp.content[0].text if resp.content else ""
    refused, reason = _parse_refusal(text)
    log.info(
        "generate.done",
        latency_ms=latency_ms,
        refused=refused,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
    return GenerationResult(
        text=text,
        refused=refused,
        refusal_reason=reason,
        model=settings.generator_model,
        latency_ms=latency_ms,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
