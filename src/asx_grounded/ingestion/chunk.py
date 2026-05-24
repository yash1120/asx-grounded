"""Recursive, sentence-aware chunker for parsed announcement text.

Targets ~600 tokens per chunk with ~100 tokens of overlap. Preserves the
originating page number in chunk metadata so citations can deep-link back
to a specific page in the source PDF later.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import tiktoken

from asx_grounded.ingestion.parse_pdf import ParsedPdf
from asx_grounded.models import Chunk

_TOKENIZER = tiktoken.get_encoding("cl100k_base")
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z\(])")


def _token_count(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _greedy_pack(
    sentences: Iterable[str],
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    sentences = list(sentences)
    i = 0
    while i < len(sentences):
        s = sentences[i]
        s_tokens = _token_count(s)
        if s_tokens > target_tokens:
            # sentence too long — emit alone, no further packing
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_tokens = [], 0
            chunks.append(s)
            i += 1
            continue
        if buf_tokens + s_tokens > target_tokens and buf:
            chunks.append(" ".join(buf))
            # build overlap tail
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(buf):
                t = _token_count(prev)
                if tail_tokens + t > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += t
            buf = tail
            buf_tokens = tail_tokens
        buf.append(s)
        buf_tokens += s_tokens
        i += 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunk_pdf(
    parsed: ParsedPdf,
    asx_code: str,
    target_tokens: int = 600,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Chunk a parsed PDF into retrievable units.

    Chunking is per-page so the page_num metadata is preserved exactly.
    Cross-page context is handled by the overlap from the previous chunk's tail.
    """
    chunks: list[Chunk] = []
    chunk_idx = 0
    for page in parsed.pages:
        if not page.text:
            continue
        sentences = _split_sentences(page.text)
        for chunk_text in _greedy_pack(sentences, target_tokens, overlap_tokens):
            chunks.append(
                Chunk(
                    chunk_id=f"{parsed.ann_id}:{chunk_idx}",
                    ann_id=parsed.ann_id,
                    asx_code=asx_code,
                    chunk_idx=chunk_idx,
                    page_num=page.page_num,
                    text=chunk_text,
                    token_count=_token_count(chunk_text),
                )
            )
            chunk_idx += 1
    return chunks
