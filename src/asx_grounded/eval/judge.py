"""LLM-as-judge using Claude Opus 4.7 with structured reasoning.

The judge sees: the question, the model's full answer, the retrieved chunks
that were available to the model, and the expected behaviour (refusal flag +
free-text notes). It returns a structured verdict the runner aggregates into
the scoreboard.

We use Opus only for evaluation — never for serving — because the eval set is
small (~80 questions) so the cost stays bounded while the grading quality goes
up materially over Sonnet.
"""

from __future__ import annotations

import json
import re

import anthropic
import structlog

from asx_grounded.config import get_settings
from asx_grounded.models import EvalQuestion, JudgeVerdict, QueryResponse, RetrievedChunk

log = structlog.get_logger()


JUDGE_SYSTEM = """You are a strict evaluator of a RAG system answering ASX questions.

You will receive:
- The original QUESTION
- The system's ANSWER (may be a refusal beginning with "REFUSE:")
- The CITATIONS the system emitted
- The retrieved SOURCES the system had access to
- The EXPECTED_REFUSAL flag (true if the question is unanswerable from the corpus)
- The category (answerable, unanswerable, time_bounded, comparative, adversarial)

Return ONLY a JSON object with these fields:

{
  "factually_correct": boolean,    // claims in the answer are accurate given the sources (true if refused appropriately)
  "citation_accuracy": float,      // 0..1 — of cited chunks, fraction that actually support the surrounding claim (1.0 if N/A, e.g. correct refusal)
  "citation_recall": float,        // 0..1 — fraction of clearly-relevant available sources that were cited (1.0 if N/A)
  "refusal_correct": boolean,      // refusal behaviour matches EXPECTED_REFUSAL
  "format_compliant": boolean,     // every factual claim has at least one [chunk_id] citation, or the response is a clean REFUSE:
  "hallucination": boolean,        // ANY claim that is not supported by the cited sources
  "reasoning": string              // 1-3 sentences justifying the above
}

No extra prose, no markdown fences."""


def _sources_block(sources: list[RetrievedChunk]) -> str:
    if not sources:
        return "(none)"
    return "\n\n".join(f"[{s.chunk.chunk_id}] ({s.chunk.asx_code}) {s.chunk.text.strip()}" for s in sources)


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)


def judge(
    question: EvalQuestion,
    response: QueryResponse,
    sources: list[RetrievedChunk],
) -> JudgeVerdict:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    citations_block = "\n".join(f"- {c.chunk_id}" for c in response.citations) if response.citations else "(none)"
    user_msg = (
        f"QUESTION: {question.question}\n"
        f"CATEGORY: {question.category}\n"
        f"EXPECTED_REFUSAL: {question.expected_refusal}\n"
        f"NOTES: {question.notes}\n\n"
        f"ANSWER:\n{response.answer}\n\n"
        f"CITATIONS:\n{citations_block}\n\n"
        f"SOURCES (what the system had access to):\n{_sources_block(sources)}"
    )

    resp = client.messages.create(
        model=settings.judge_model,
        max_tokens=600,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = _strip_fences(resp.content[0].text if resp.content else "{}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("judge.parse_failed", qid=question.qid, error=str(exc), raw=raw[:200])
        data = {}

    return JudgeVerdict(
        qid=question.qid,
        factually_correct=bool(data.get("factually_correct", False)),
        citation_accuracy=float(data.get("citation_accuracy", 0.0)),
        citation_recall=float(data.get("citation_recall", 0.0)),
        refusal_correct=bool(data.get("refusal_correct", False)),
        format_compliant=bool(data.get("format_compliant", False)),
        hallucination=bool(data.get("hallucination", False)),
        reasoning=str(data.get("reasoning", "")),
    )
