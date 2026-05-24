from __future__ import annotations

from asx_grounded.retrieval.filters import _regex_filter


def test_extracts_explicit_asx_code() -> None:
    f = _regex_filter("What did CBA announce yesterday?")
    assert "CBA" in f.asx_codes


def test_ignores_common_stopwords() -> None:
    f = _regex_filter("Show ALL announcements for THE company")
    assert "ALL" not in f.asx_codes
    assert "THE" not in f.asx_codes


def test_extracts_iso_date_range() -> None:
    f = _regex_filter("Between 2026-03-01 and 2026-03-31 what did BHP announce?")
    assert f.released_after is not None
    assert f.released_before is not None
    assert f.released_after.year == 2026 and f.released_after.month == 3 and f.released_after.day == 1
    assert "BHP" in f.asx_codes


def test_empty_query_yields_empty_filter() -> None:
    f = _regex_filter("Any announcements about climate?")
    assert f.empty
