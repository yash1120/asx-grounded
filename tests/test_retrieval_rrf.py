from __future__ import annotations

from asx_grounded.retrieval.hybrid import HybridRetriever


def test_rrf_fuses_two_rankings() -> None:
    # Pure-function test against the static method; no Qdrant required.
    vec_ranking = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25_ranking = [("c", 12.0), ("a", 10.0), ("d", 5.0)]
    fused = HybridRetriever._rrf([vec_ranking, bm25_ranking])

    ids = [cid for cid, _ in fused]
    # 'a' and 'c' both appear high in both rankings → should top the fused list.
    assert ids[:2] == ["a", "c"] or ids[:2] == ["c", "a"]
    assert "d" in ids
    assert "b" in ids


def test_rrf_handles_singleton_ranking() -> None:
    fused = HybridRetriever._rrf([[("a", 1.0)]])
    assert fused[0][0] == "a"
    assert fused[0][1] > 0
