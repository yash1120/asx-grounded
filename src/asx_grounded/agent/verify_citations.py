"""Verify that every citation the model emitted (a) exists in the retrieved set
and (b) actually supports the surrounding claim.

Two-stage verification:

  Stage 1 (cheap, always run): regex-extract chunk_ids, confirm each is in the
  retrieved set. Drop or rewrite citations that reference an id we never sent
  to the model — that's a sign of fabrication.

  Stage 2 (LLM, optional): for each (claim, cited_chunks) pair, ask a small
  Claude call whether the cited text supports the claim. This costs money so
  we expose it as ``verify_with_llm=True`` and use it during eval, not on every
  user query in production.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import anthropic
import structlog

from asx_grounded.config import get_settings
from asx_grounded.models import Citation, RetrievedChunk

log = structlog.get_logger()


_CITE_RE = re.compile(r"\[([^\[\]]+?)\]")


@dataclass(slots=True)
class VerifiedAnswer:
    answer_text: str           # rewritten text with invalid citations stripped
    citations: list[Citation]
    fabricated_ids: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


def _extract_chunk_ids(text: str) -> list[tuple[str, str]]:
    """Return list of (full_bracket_content, chunk_id) for each citation."""
    out: list[tuple[str, str]] = []
    for m in _CITE_RE.finditer(text):
        inner = m.group(1).strip()
        for cid in inner.split(","):
            cid = cid.strip()
            if re.match(r"^[A-Za-z0-9_:\-\.]+:\d+$", cid):
                out.append((inner, cid))
    return out


def verify_citations(
    answer_text: str,
    retrieved: list[RetrievedChunk],
    verify_with_llm: bool = False,
) -> VerifiedAnswer:
    retrieved_ids = {r.chunk.chunk_id: r for r in retrieved}
    cited_pairs = _extract_chunk_ids(answer_text)

    fabricated: list[str] = []
    verified: dict[str, Citation] = {}
    for _, cid in cited_pairs:
        if cid not in retrieved_ids:
            fabricated.append(cid)
            continue
        if cid in verified:
            continue
        r = retrieved_ids[cid]
        verified[cid] = Citation(
            chunk_id=cid,
            ann_id=r.chunk.ann_id,
            asx_page_url="",  # filled in by retriever-aware caller (api/main.py)
            verified=True,
            verification_note="present in retrieved set",
        )

    # Strip fabricated ids from the answer text rather than leaving dangling brackets.
    cleaned = answer_text
    for fab in set(fabricated):
        cleaned = re.sub(rf"\[\s*{re.escape(fab)}\s*(?:,[^\]]*)?\]", "", cleaned)
        cleaned = re.sub(rf",\s*{re.escape(fab)}", "", cleaned)
    if fabricated:
        log.warning("verify.fabricated", ids=fabricated)

    unsupported: list[str] = []
    if verify_with_llm and verified:
        unsupported = _llm_check_support(cleaned, retrieved_ids, list(verified.keys()))

    return VerifiedAnswer(
        answer_text=cleaned.strip(),
        citations=list(verified.values()),
        fabricated_ids=fabricated,
        unsupported_claims=unsupported,
    )


_SUPPORT_SYSTEM = """You are an entailment checker. Given a CLAIM and one or more SOURCE excerpts, decide whether the sources support the claim.

Reply with ONLY one of: SUPPORTED, UNSUPPORTED, PARTIAL.
No other text."""


def _llm_check_support(
    answer_text: str,
    retrieved_ids: dict[str, RetrievedChunk],
    cited: list[str],
) -> list[str]:
    settings = get_settings()
    if not settings.anthropic_api_key:
        return []
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Split the answer into sentences for finer-grained checking.
    sentences = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", answer_text) if s.strip()]
    unsupported: list[str] = []
    for sent in sentences:
        ids_in_sent = [cid for _, cid in _extract_chunk_ids(sent)]
        if not ids_in_sent:
            continue
        sources = "\n\n".join(
            f"[{cid}] {retrieved_ids[cid].chunk.text.strip()}"
            for cid in ids_in_sent
            if cid in retrieved_ids
        )
        if not sources:
            continue
        try:
            resp = client.messages.create(
                model=settings.generator_model,
                max_tokens=10,
                system=_SUPPORT_SYSTEM,
                messages=[{"role": "user", "content": f"CLAIM: {sent}\n\nSOURCES:\n{sources}"}],
            )
            verdict = (resp.content[0].text if resp.content else "").strip().upper()
        except anthropic.AnthropicError as exc:
            log.warning("verify.llm_failed", error=str(exc))
            continue
        if verdict.startswith("UNSUPPORTED"):
            unsupported.append(sent)
    return unsupported
