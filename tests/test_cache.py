from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from asx_grounded.cache import QueryCache, normalise_question
from asx_grounded.models import Citation, QueryResponse


def _make_response(query: str = "What is CBA's dividend?") -> QueryResponse:
    return QueryResponse(
        query=query,
        answer="CBA declared a fully-franked dividend of $2.40 per share.",
        citations=[
            Citation(
                chunk_id="CBA_001:3",
                ann_id="CBA_001",
                asx_page_url="https://www.asx.com.au/markets/company/CBA",
                verified=True,
                verification_note="supported by chunk text",
            )
        ],
        refused=False,
        retrieval_debug={"candidates": 5},
        latency_ms=1234,
        model="claude-sonnet-4-6",
    )


def _cache(tmp_path: Path, ttl_seconds: int = 7 * 24 * 60 * 60) -> QueryCache:
    return QueryCache(db_path=tmp_path / "query_cache.sqlite", ttl_seconds=ttl_seconds)


def test_cold_get_returns_none(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    assert cache.get("What is CBA's dividend?", "v1") is None


def test_set_get_round_trips_response(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    resp = _make_response()
    cache.set("What is CBA's dividend?", "v1", resp)

    got = cache.get("What is CBA's dividend?", "v1")
    assert got is not None
    # Full structural round-trip via the pydantic models.
    assert got == resp
    assert got.model_dump() == resp.model_dump()
    assert got.citations[0].chunk_id == "CBA_001:3"
    assert got.citations[0].verified is True
    assert got.latency_ms == 1234


def test_normalisation_collapses_trivial_variants(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    resp = _make_response()
    cache.set("What is CBA's dividend?", "v1", resp)

    # Different case + missing punctuation must hit the same key.
    got = cache.get("what is cba's dividend", "v1")
    assert got is not None
    assert got == resp


def test_normalisation_handles_whitespace_and_punctuation(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    resp = _make_response()
    cache.set("What is CBA's dividend?", "v1", resp)

    # Extra/irregular whitespace and trailing punctuation still collapse.
    got = cache.get("  What   is  CBA's   dividend??!  ", "v1")
    assert got is not None
    assert got == resp


def test_normalise_question_helper() -> None:
    assert normalise_question("What is CBA's dividend?") == "what is cbas dividend"
    assert normalise_question("  Hello,   WORLD!  ") == "hello world"
    assert normalise_question("What is CBA's dividend?") == normalise_question("what is cba's dividend")


def test_different_corpus_version_is_a_miss(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    resp = _make_response()
    cache.set("What is CBA's dividend?", "v1", resp)

    # Same question, different corpus version -> not the same entry.
    assert cache.get("What is CBA's dividend?", "v2") is None
    # The original entry is still present.
    assert cache.get("What is CBA's dividend?", "v1") is not None


def test_ttl_expiry_returns_none(tmp_path: Path) -> None:
    cache = _cache(tmp_path, ttl_seconds=60)
    resp = _make_response()
    cache.set("What is CBA's dividend?", "v1", resp)

    # Fresh entry is a hit.
    assert cache.get("What is CBA's dividend?", "v1") is not None

    # Backdate the stored row past the TTL instead of sleeping.
    with sqlite3.connect(cache.db_path) as conn:
        conn.execute("UPDATE query_cache SET created_at = ?", (time.time() - 3600,))

    assert cache.get("What is CBA's dividend?", "v1") is None


def test_expired_entry_is_purged_on_read(tmp_path: Path) -> None:
    cache = _cache(tmp_path, ttl_seconds=60)
    cache.set("What is CBA's dividend?", "v1", _make_response())

    with sqlite3.connect(cache.db_path) as conn:
        conn.execute("UPDATE query_cache SET created_at = ?", (time.time() - 3600,))

    # A miss on expiry should also delete the dead row.
    assert cache.get("What is CBA's dividend?", "v1") is None
    with sqlite3.connect(cache.db_path) as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
    assert remaining == 0


def test_set_overwrites_existing_entry(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.set("What is CBA's dividend?", "v1", _make_response())
    updated = _make_response()
    updated.answer = "Updated answer."
    cache.set("What is CBA's dividend?", "v1", updated)

    got = cache.get("What is CBA's dividend?", "v1")
    assert got is not None
    assert got.answer == "Updated answer."
    # Still a single row for that key.
    with sqlite3.connect(cache.db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
    assert count == 1


def test_default_db_path_parent_is_created(tmp_path: Path) -> None:
    nested = tmp_path / "data" / "processed" / "query_cache.sqlite"
    QueryCache(db_path=nested)
    assert nested.parent.is_dir()
