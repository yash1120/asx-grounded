"""Cross-encoder reranker on top of hybrid retrieval.

We over-fetch 50 candidates from hybrid, then rerank with bge-reranker-large to
keep only the top ``rerank_top_k`` (default 8). The reranker scores each
(query, chunk) pair jointly, which catches semantic relevance the bi-encoder
embedding step misses.
"""

from __future__ import annotations

from typing import Any

import structlog

from asx_grounded.config import get_settings
from asx_grounded.models import RetrievedChunk

log = structlog.get_logger()


class CrossEncoderReranker:
    """Lazy-loaded BGE reranker."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _ensure(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("rerank.loading_model", model=self._model_name)
            self._model = CrossEncoder(self._model_name, max_length=512)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        settings = get_settings()
        top_k = top_k or settings.rerank_top_k
        model = self._ensure()
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        for c, s in zip(candidates, scores, strict=True):
            c.rerank_score = float(s)
        sorted_ = sorted(candidates, key=lambda x: x.rerank_score or 0.0, reverse=True)
        kept = [c for c in sorted_ if (c.rerank_score or 0.0) >= settings.min_relevance_score]
        return (kept or sorted_)[:top_k]
