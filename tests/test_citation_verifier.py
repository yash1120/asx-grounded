from __future__ import annotations

from asx_grounded.agent.verify_citations import verify_citations

from tests.conftest import make_retrieved


def test_extracts_valid_citations() -> None:
    retrieved = [
        make_retrieved("CBA:0", "CBA_2026-02-01", "CBA announced a $2.40 dividend."),
        make_retrieved("CBA:1", "CBA_2026-02-01", "Payment date is 28 March 2026."),
    ]
    answer = "CBA announced a $2.40 dividend [CBA:0]. The payment date is 28 March 2026 [CBA:1]."
    verified = verify_citations(answer, retrieved, verify_with_llm=False)

    assert {c.chunk_id for c in verified.citations} == {"CBA:0", "CBA:1"}
    assert verified.fabricated_ids == []
    assert "CBA:0" in verified.answer_text
    assert "CBA:1" in verified.answer_text


def test_strips_fabricated_citations() -> None:
    retrieved = [make_retrieved("CBA:0", "CBA_2026-02-01", "Real chunk text.")]
    answer = "Claim one [CBA:0]. Fake claim [BHP:99]."
    verified = verify_citations(answer, retrieved, verify_with_llm=False)

    assert "BHP:99" in verified.fabricated_ids
    assert "BHP:99" not in verified.answer_text
    assert "[CBA:0]" in verified.answer_text
    assert {c.chunk_id for c in verified.citations} == {"CBA:0"}


def test_handles_multi_citation_brackets() -> None:
    retrieved = [
        make_retrieved("CBA:0", "CBA_2026-02-01", "Text A."),
        make_retrieved("CBA:1", "CBA_2026-02-01", "Text B."),
    ]
    answer = "Combined claim [CBA:0, CBA:1]."
    verified = verify_citations(answer, retrieved, verify_with_llm=False)
    assert {c.chunk_id for c in verified.citations} == {"CBA:0", "CBA:1"}


def test_no_citations_means_no_verified_citations() -> None:
    retrieved = [make_retrieved("CBA:0", "CBA_2026-02-01", "Text.")]
    verified = verify_citations("Bare claim without any citation.", retrieved, verify_with_llm=False)
    assert verified.citations == []
    assert verified.fabricated_ids == []
