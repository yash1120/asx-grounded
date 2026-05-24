"""Prompts for the grounded agent.

Design choices documented inline so a reviewer (and the blog post) can audit:
  * Citations are REQUIRED in `[chunk_id]` format, one per claim.
  * The model is instructed to refuse rather than guess when context is thin —
    we'd rather over-refuse than hallucinate (we measure both in the eval).
  * Format is strict so the regex-based verifier (verify_citations.py) can parse.
"""

from __future__ import annotations

from asx_grounded.models import RetrievedChunk


SYSTEM_PROMPT = """You answer questions about ASX-listed companies using ONLY the provided context excerpts from ASX continuous-disclosure announcements.

Hard rules:

1. EVERY factual claim you make MUST be followed by a citation in square brackets containing the chunk_id, e.g. [CBA_2025-11-12_001:3]. Multiple citations are separated by commas inside one set of brackets, e.g. [CBA_2025-11-12_001:3, CBA_2025-11-12_001:4].

2. If the provided context does NOT contain enough information to answer confidently, respond with EXACTLY:
   REFUSE: <one short sentence explaining what is missing>
   Do not guess. Do not use general knowledge.

3. Do not invent chunk_ids. Only cite chunk_ids that appear in the <context> block.

4. Keep answers concise: 1-4 short paragraphs. No preamble like "Based on the context...".

5. When the question is unanswerable from the corpus, refuse — even if you could answer from training data.

6. This is not financial advice. Do not editorialise or recommend actions.

Output exactly one of:
- A direct answer with inline [chunk_id] citations after every claim.
- A line starting with `REFUSE: `."""


def render_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a tagged context block the model can cite."""
    lines: list[str] = ["<context>"]
    for c in chunks:
        page = f" page {c.chunk.page_num}" if c.chunk.page_num else ""
        lines.append(
            f'<chunk id="{c.chunk.chunk_id}" company="{c.chunk.asx_code}"{page}>\n'
            f"{c.chunk.text.strip()}\n"
            f"</chunk>"
        )
    lines.append("</context>")
    return "\n".join(lines)


def build_user_message(query: str, chunks: list[RetrievedChunk]) -> str:
    return f"{render_context(chunks)}\n\n<question>{query.strip()}</question>"
