from __future__ import annotations

from asx_grounded.retrieval.hybrid import RRF_K, HybridRetriever, _tokenize


def _rrf_score(rank: int) -> float:
    """Expected RRF contribution for a single appearance at 1-based ``rank``."""
    return 1.0 / (RRF_K + rank)


# --------------------------------------------------------------------------- #
# _rrf edge cases
# --------------------------------------------------------------------------- #


def test_rrf_empty_input_returns_empty() -> None:
    assert HybridRetriever._rrf([]) == []


def test_rrf_all_rankings_empty_returns_empty() -> None:
    assert HybridRetriever._rrf([[], []]) == []


def test_rrf_single_empty_ranking_returns_empty() -> None:
    assert HybridRetriever._rrf([[]]) == []


def test_rrf_full_overlap_same_order_doubles_scores() -> None:
    # Identical rankings: every id keeps its rank in both lists, so each score
    # is exactly twice the single-appearance contribution, and order is preserved.
    ranking = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    fused = HybridRetriever._rrf([ranking, ranking])

    ids = [cid for cid, _ in fused]
    assert ids == ["a", "b", "c"]
    scores = {cid: score for cid, score in fused}
    assert scores["a"] == 2 * _rrf_score(1)
    assert scores["b"] == 2 * _rrf_score(2)
    assert scores["c"] == 2 * _rrf_score(3)
    # Monotonically decreasing.
    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_full_overlap_reversed_order_ties_all_ids() -> None:
    # Two members in mirror-image order: each id sits at rank 1 in one list and
    # rank 2 in the other, so both totals are identical -> a tie.
    forward = [("a", 0.9), ("b", 0.8)]
    reverse = [("b", 0.2), ("a", 0.3)]
    fused = HybridRetriever._rrf([forward, reverse])

    scores = [score for _, score in fused]
    expected = _rrf_score(1) + _rrf_score(2)
    assert all(abs(score - expected) < 1e-12 for score in scores)
    assert {cid for cid, _ in fused} == {"a", "b"}


def test_rrf_disjoint_rankings_keeps_all_and_ranks_by_position() -> None:
    # No shared ids: each id scores its single contribution; top of each list
    # outranks lower entries, top-1 entries tie with each other.
    vec = [("a", 0.9), ("b", 0.5)]
    bm25 = [("x", 9.0), ("y", 1.0)]
    fused = HybridRetriever._rrf([vec, bm25])

    assert {cid for cid, _ in fused} == {"a", "b", "x", "y"}
    scores = {cid: score for cid, score in fused}
    # Rank-1 entries score equally; rank-2 entries score equally and lower.
    assert scores["a"] == scores["x"] == _rrf_score(1)
    assert scores["b"] == scores["y"] == _rrf_score(2)
    assert scores["a"] > scores["b"]
    # The two rank-1 ids sort ahead of the two rank-2 ids.
    assert set([cid for cid, _ in fused][:2]) == {"a", "x"}


def test_rrf_tie_breaks_on_insertion_order() -> None:
    # Two ids that never co-occur but land at the same rank tie on score; the
    # sort is stable so the first one inserted (seen) wins the ordering.
    fused = HybridRetriever._rrf([[("first", 1.0)], [("second", 1.0)]])
    assert [cid for cid, _ in fused] == ["first", "second"]
    assert fused[0][1] == fused[1][1] == _rrf_score(1)


def test_rrf_overlapping_id_beats_singletons() -> None:
    # An id present in both rankings accumulates two contributions and should
    # outrank ids that appear only once, even at rank 1.
    vec = [("shared", 0.9), ("only_vec", 0.8)]
    bm25 = [("only_bm25", 5.0), ("shared", 4.0)]
    fused = HybridRetriever._rrf([vec, bm25])

    assert fused[0][0] == "shared"
    scores = {cid: score for cid, score in fused}
    assert scores["shared"] == _rrf_score(1) + _rrf_score(2)
    assert scores["shared"] > scores["only_vec"]
    assert scores["shared"] > scores["only_bm25"]


def test_rrf_score_uses_k_constant() -> None:
    # Pin the exact arithmetic so a change to RRF_K is caught by a test.
    fused = HybridRetriever._rrf([[("a", 0.0)]])
    assert fused[0][1] == 1.0 / (RRF_K + 1)


# --------------------------------------------------------------------------- #
# _tokenize
# --------------------------------------------------------------------------- #


def test_tokenize_lowercases_and_splits_on_whitespace() -> None:
    assert _tokenize("CBA Dividend Announcement") == ["cba", "dividend", "announcement"]


def test_tokenize_collapses_runs_of_whitespace() -> None:
    assert _tokenize("  CBA   \t  dividend\n payment  ") == ["cba", "dividend", "payment"]


def test_tokenize_empty_string_yields_empty_list() -> None:
    assert _tokenize("") == []


def test_tokenize_whitespace_only_yields_empty_list() -> None:
    assert _tokenize("   \t\n  ") == []


def test_tokenize_does_not_strip_punctuation() -> None:
    # The tokeniser only splits/lowercases; punctuation stays attached.
    assert _tokenize("BHP's $2.40 dividend.") == ["bhp's", "$2.40", "dividend."]
