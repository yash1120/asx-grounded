"""Parser regression tests against a real live ASX listing response.

The fixture ``tests/fixtures/asx_cba_listing.json`` was captured from the live
Markit Digital endpoint on 2026-05-26 (sprint S1, story S1-02). Any future
ASX-side schema drift will be caught here before it breaks ingestion.
"""

from __future__ import annotations

import json
from pathlib import Path

from asx_grounded.ingestion.fetch_asx import _parse_item

_FIXTURE = Path(__file__).parent / "fixtures" / "asx_cba_listing.json"


def _load_envelope() -> dict:
    # utf-8-sig transparently strips a BOM if the fixture was saved by a
    # BOM-injecting editor (e.g. PowerShell's default Out-File).
    body = json.loads(_FIXTURE.read_text(encoding="utf-8-sig"))
    return body["data"]


def test_fixture_has_expected_top_level_keys() -> None:
    env = _load_envelope()
    assert env["symbol"] == "CBA"
    assert "items" in env and isinstance(env["items"], list)
    assert len(env["items"]) >= 1
    assert env["displayName"]


def test_parse_item_maps_markit_shape_to_announcement() -> None:
    env = _load_envelope()
    item = env["items"][0]
    company = env["displayName"]
    ann = _parse_item(item, "CBA", company)

    assert ann is not None
    assert ann.asx_code == "CBA"
    assert ann.company_name == company
    assert ann.ann_id == item["documentKey"]
    assert ann.headline == item["headline"]
    assert ann.is_price_sensitive == bool(item["isPriceSensitive"])
    assert ann.pdf_url.endswith(item["documentKey"])
    assert "displayAnnouncement.do" in ann.asx_page_url
    assert ann.released_at.year >= 2025  # fixture is from 2026


def test_parse_item_returns_none_on_missing_document_key() -> None:
    bad = {"date": "2026-01-01T00:00:00.000Z", "headline": "x", "isPriceSensitive": False}
    assert _parse_item(bad, "CBA", "Test Co") is None


def test_parse_item_returns_none_on_missing_date() -> None:
    bad = {"documentKey": "abc-123", "headline": "x", "isPriceSensitive": False}
    assert _parse_item(bad, "CBA", "Test Co") is None


def test_all_fixture_items_parse_or_fail_cleanly() -> None:
    env = _load_envelope()
    company = env["displayName"]
    parsed = [_parse_item(item, "CBA", company) for item in env["items"]]
    # Every item in the live fixture should parse successfully.
    assert all(a is not None for a in parsed)
    # ann_ids should be unique.
    ann_ids = [a.ann_id for a in parsed if a is not None]
    assert len(ann_ids) == len(set(ann_ids))
