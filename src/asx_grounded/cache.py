"""Query cache for grounded /query responses.

Every /query call hits the Claude API, costing money and latency. This module
caches QueryResponse objects keyed by a hash of (normalised question +
corpus_version) so repeat and trivially-varied queries are served from a local
SQLite file instead of re-running generation.

Backend is stdlib sqlite3 (no new dependency). A fresh connection is opened per
call so the cache is safe to share across FastAPI's threadpool workers.

# WIRE-IN: api/main.py — add the cache to the /query handler.
#
# 1. At module level (near the other Path constants, ~line 33), import and
#    construct the cache once. Use a corpus version derived from the BM25 corpus
#    file's mtime so the cache self-invalidates whenever you re-embed:
#
#        from asx_grounded.cache import QueryCache, corpus_version_from_path
#        _QUERY_CACHE = QueryCache()
#
# 2. Inside `def query(req, request)`, right after the `_State.retriever is None`
#    guard (~line 76, before `started = time.perf_counter()`), check the cache:
#
#        _corpus_version = corpus_version_from_path(BM25_CORPUS)
#        _cached = _QUERY_CACHE.get(req.question, _corpus_version)
#        if _cached is not None:
#            return _cached
#
# 3. Replace each `return QueryResponse(...)` / `return ...` at the end of the
#    handler so the response is cached before it is returned, e.g.:
#
#        resp = QueryResponse(...)
#        _QUERY_CACHE.set(req.question, _corpus_version, resp)
#        return resp
#
#    (Cache both the empty-retrieval refusal and the generated answer. Optionally
#    skip caching when req.verify_with_llm differs from the default by folding
#    that flag into the corpus_version string you pass in.)
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from pathlib import Path

import structlog

from asx_grounded.models import QueryResponse

log = structlog.get_logger()

DEFAULT_CACHE_PATH = Path("data/processed/query_cache.sqlite")
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days

# Strip everything that is not a word character or whitespace. Combined with
# casefold + whitespace collapse this maps trivial variants to one key, e.g.
# "What is CBA's dividend?" and "what is cba's dividend" -> "what is cbas dividend".
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalise_question(question: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace.

    Used so trivial variants of the same question share a cache key.
    """
    lowered = question.casefold()
    no_punct = _PUNCT_RE.sub("", lowered)
    collapsed = _WS_RE.sub(" ", no_punct)
    return collapsed.strip()


def _cache_key(question: str, corpus_version: str) -> str:
    """SHA-256 over the normalised question and the caller's corpus version."""
    normalised = normalise_question(question)
    payload = f"{corpus_version}\x00{normalised}".encode()
    return hashlib.sha256(payload).hexdigest()


def corpus_version_from_path(path: Path) -> str:
    """Derive a corpus-version string from a corpus file's size + mtime.

    Convenience helper for callers: the cache self-invalidates whenever the
    underlying corpus file is rebuilt. Missing file -> a stable "absent" token.
    """
    try:
        st = path.stat()
    except OSError:
        return "absent"
    return f"{int(st.st_size)}-{int(st.st_mtime)}"


class QueryCache:
    """SQLite-backed cache mapping (question, corpus_version) -> QueryResponse.

    Thread-safe for FastAPI's threadpool: a new sqlite3 connection is opened for
    every get/set call rather than sharing one across threads.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_CACHE_PATH,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        # A short busy timeout lets concurrent threadpool writers serialise
        # instead of raising "database is locked".
        return sqlite3.connect(self.db_path, timeout=30.0)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    key           TEXT PRIMARY KEY,
                    question      TEXT NOT NULL,
                    corpus_version TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at    REAL NOT NULL
                )
                """
            )

    def get(self, question: str, corpus_version: str) -> QueryResponse | None:
        """Return the cached QueryResponse or None on a miss / expiry.

        Expired rows are deleted lazily on read.
        """
        key = _cache_key(question, corpus_version)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json, created_at FROM query_cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            response_json, created_at = row
            if time.time() - created_at > self.ttl_seconds:
                conn.execute("DELETE FROM query_cache WHERE key = ?", (key,))
                return None

        try:
            return QueryResponse.model_validate_json(response_json)
        except ValueError:
            # Corrupt/stale-schema payload — treat as a miss and drop it.
            log.warning("query_cache.decode_failed", key=key)
            with self._connect() as conn:
                conn.execute("DELETE FROM query_cache WHERE key = ?", (key,))
            return None

    def set(self, question: str, corpus_version: str, resp: QueryResponse) -> None:
        """Insert or replace the cached response for this question + corpus version."""
        key = _cache_key(question, corpus_version)
        response_json = resp.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_cache (key, question, corpus_version, response_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    question = excluded.question,
                    corpus_version = excluded.corpus_version,
                    response_json = excluded.response_json,
                    created_at = excluded.created_at
                """,
                (key, question, corpus_version, response_json, time.time()),
            )
