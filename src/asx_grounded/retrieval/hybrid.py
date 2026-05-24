"""Hybrid retrieval: BM25 (lexical) + Qdrant (vector), fused with Reciprocal Rank Fusion.

BM25 catches exact-match terms (ASX codes, regulatory phrases) the embedding model
under-weights; vector catches paraphrases. RRF (k=60) is a parameter-free fuser that
beats weighted-sum in practice and is the default starting point in the literature.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from rank_bm25 import BM25Okapi

from asx_grounded.config import get_settings
from asx_grounded.ingestion.embed import Embedder
from asx_grounded.models import Chunk, RetrievedChunk
from asx_grounded.retrieval.filters import MetadataFilter

log = structlog.get_logger()


RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


@dataclass(slots=True)
class _Doc:
    chunk_id: str
    ann_id: str
    asx_code: str
    chunk_idx: int
    page_num: int | None
    text: str
    headline: str
    released_at: str | None
    asx_page_url: str


class HybridRetriever:
    def __init__(self, bm25_corpus_path: Path) -> None:
        settings = get_settings()
        self._settings = settings
        self._qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self._embedder = Embedder(settings.embedding_model)
        self._docs: list[_Doc] = []
        self._by_chunk: dict[str, _Doc] = {}
        self._bm25: BM25Okapi | None = None
        self._load_bm25(bm25_corpus_path)

    def _load_bm25(self, corpus_path: Path) -> None:
        if not corpus_path.exists():
            log.warning("bm25.corpus_missing", path=str(corpus_path))
            self._bm25 = BM25Okapi([["__empty__"]])
            return
        tokens: list[list[str]] = []
        with corpus_path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                d = _Doc(
                    chunk_id=rec["chunk_id"],
                    ann_id=rec["ann_id"],
                    asx_code=rec["asx_code"],
                    chunk_idx=rec["chunk_idx"],
                    page_num=rec.get("page_num"),
                    text=rec["text"],
                    headline=rec.get("headline", ""),
                    released_at=rec.get("released_at"),
                    asx_page_url=rec.get("asx_page_url", ""),
                )
                self._docs.append(d)
                self._by_chunk[d.chunk_id] = d
                tokens.append(_tokenize(d.text))
        if not tokens:
            tokens = [["__empty__"]]
        self._bm25 = BM25Okapi(tokens)
        log.info("bm25.loaded", docs=len(self._docs))

    def _qdrant_filter(self, mfilter: MetadataFilter | None) -> qdrant_models.Filter | None:
        if mfilter is None:
            return None
        must: list[qdrant_models.FieldCondition] = []
        if mfilter.asx_codes:
            must.append(
                qdrant_models.FieldCondition(
                    key="asx_code",
                    match=qdrant_models.MatchAny(any=list(mfilter.asx_codes)),
                )
            )
        if mfilter.released_after:
            must.append(
                qdrant_models.FieldCondition(
                    key="released_at",
                    range=qdrant_models.Range(gte=mfilter.released_after.isoformat()),
                )
            )
        if mfilter.released_before:
            must.append(
                qdrant_models.FieldCondition(
                    key="released_at",
                    range=qdrant_models.Range(lte=mfilter.released_before.isoformat()),
                )
            )
        return qdrant_models.Filter(must=must) if must else None

    def _vector_search(self, query: str, k: int, mfilter: MetadataFilter | None) -> list[tuple[str, float]]:
        vec = self._embedder.encode([query])[0]
        hits = self._qdrant.search(
            collection_name=self._settings.qdrant_collection,
            query_vector=vec,
            query_filter=self._qdrant_filter(mfilter),
            limit=k,
            with_payload=True,
        )
        return [(h.payload["chunk_id"], float(h.score)) for h in hits if h.payload]

    def _bm25_search(self, query: str, k: int, mfilter: MetadataFilter | None) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._docs:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            ((self._docs[i].chunk_id, float(scores[i]), self._docs[i]) for i in range(len(self._docs))),
            key=lambda t: t[1],
            reverse=True,
        )
        out: list[tuple[str, float]] = []
        for chunk_id, score, doc in ranked:
            if mfilter is not None and not mfilter.allows(doc.asx_code, doc.released_at):
                continue
            out.append((chunk_id, score))
            if len(out) >= k:
                break
        return out

    @staticmethod
    def _rrf(rankings: Iterable[list[tuple[str, float]]]) -> list[tuple[str, float]]:
        totals: dict[str, float] = defaultdict(float)
        for ranking in rankings:
            for rank, (chunk_id, _) in enumerate(ranking, start=1):
                totals[chunk_id] += 1.0 / (RRF_K + rank)
        return sorted(totals.items(), key=lambda t: t[1], reverse=True)

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        mfilter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        k = k or self._settings.retrieval_top_k
        vec_hits = self._vector_search(query, k, mfilter)
        bm25_hits = self._bm25_search(query, k, mfilter)
        fused = self._rrf([vec_hits, bm25_hits])[:k]

        out: list[RetrievedChunk] = []
        for chunk_id, score in fused:
            doc = self._by_chunk.get(chunk_id)
            if doc is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id=doc.chunk_id,
                        ann_id=doc.ann_id,
                        asx_code=doc.asx_code,
                        chunk_idx=doc.chunk_idx,
                        page_num=doc.page_num,
                        text=doc.text,
                    ),
                    score=score,
                )
            )
        return out

    def page_url(self, chunk_id: str) -> str:
        doc = self._by_chunk.get(chunk_id)
        return doc.asx_page_url if doc else ""
