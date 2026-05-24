"""Query → metadata filter extraction.

Parses entities (ASX codes, date ranges) out of a natural-language query so
that retrieval can restrict to the relevant slice of the corpus before fusing
with semantic search.

Uses two layers:
  1. Cheap regex for explicit ASX codes and ISO dates (works most of the time).
  2. Claude Sonnet 4.6 fallback when the query contains company *names* or
     vague date references ("last quarter", "since March").

The LLM call is bypassed entirely when the regex layer already returns enough.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

import anthropic
import structlog

from asx_grounded.config import get_settings

log = structlog.get_logger()


_ASX_CODE_RE = re.compile(r"\b([A-Z]{3,4})\b")
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


# Known false positives — uppercase 3-letter words that are not ASX codes.
_CODE_STOPWORDS = {
    "ASX",
    "AUD",
    "USD",
    "CEO",
    "CFO",
    "COO",
    "AGM",
    "EGM",
    "FY",
    "EPS",
    "PDF",
    "USA",
    "ANZ",
    "ABN",
    "ACN",
    "GST",
    "RBA",
    "API",
    "QLD",
    "NSW",
    "VIC",
    "WA",
    "SA",
    "TAS",
    "NT",
    "ACT",
    "AND",
    "FOR",
    "THE",
    "ALL",
    "NEW",
}


@dataclass(slots=True)
class MetadataFilter:
    asx_codes: set[str] = field(default_factory=set)
    released_after: datetime | None = None
    released_before: datetime | None = None

    @property
    def empty(self) -> bool:
        return not self.asx_codes and not self.released_after and not self.released_before

    def allows(self, code: str, released_iso: str | None) -> bool:
        if self.asx_codes and code not in self.asx_codes:
            return False
        if released_iso and (self.released_after or self.released_before):
            try:
                d = datetime.fromisoformat(released_iso.replace("Z", "+00:00"))
            except ValueError:
                return True
            if self.released_after and d < self.released_after:
                return False
            if self.released_before and d > self.released_before:
                return False
        return True


def _regex_filter(query: str) -> MetadataFilter:
    f = MetadataFilter()
    for m in _ASX_CODE_RE.findall(query):
        if m not in _CODE_STOPWORDS:
            f.asx_codes.add(m)
    iso_dates = _ISO_DATE_RE.findall(query)
    if len(iso_dates) == 1:
        f.released_after = datetime.fromisoformat(iso_dates[0]).replace(tzinfo=UTC)
    elif len(iso_dates) >= 2:
        f.released_after = datetime.fromisoformat(iso_dates[0]).replace(tzinfo=UTC)
        f.released_before = datetime.fromisoformat(iso_dates[1]).replace(tzinfo=UTC)
    return f


_NAME_HINT_RE = re.compile(
    r"\b(commonwealth bank|cba|westpac|wbc|nab|anz|bhp|rio tinto|csl|woolworths|wesfarmers|"
    r"telstra|macquarie|fortescue|origin|santos|qantas|seek|rea)\b",
    re.IGNORECASE,
)
_DATE_HINT_RE = re.compile(
    r"\b(last|past|since|between|before|after|q[1-4]|fy\d{2,4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)


def _needs_llm(query: str, regex_filter: MetadataFilter) -> bool:
    if not regex_filter.empty:
        return False
    return bool(_NAME_HINT_RE.search(query) or _DATE_HINT_RE.search(query))


_LLM_SYSTEM = """You extract metadata filters from a user's question about ASX announcements.

Return ONLY a JSON object with these optional keys:
- "asx_codes": array of 3-4 letter uppercase ASX tickers (e.g. ["CBA","BHP"])
- "released_after": ISO date "YYYY-MM-DD"
- "released_before": ISO date "YYYY-MM-DD"

If a key is not clearly implied by the question, OMIT it. Do not invent codes."""


def _llm_filter(query: str) -> MetadataFilter:
    settings = get_settings()
    if not settings.anthropic_api_key:
        return MetadataFilter()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        resp = client.messages.create(
            model=settings.generator_model,
            max_tokens=200,
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        raw = resp.content[0].text if resp.content else "{}"
        # Strip code fences if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(raw)
    except (anthropic.AnthropicError, json.JSONDecodeError, IndexError) as exc:
        log.warning("filter.llm_failed", error=str(exc))
        return MetadataFilter()
    f = MetadataFilter()
    for c in data.get("asx_codes", []) or []:
        if isinstance(c, str) and c.isalpha() and 3 <= len(c) <= 4:
            f.asx_codes.add(c.upper())
    for key in ("released_after", "released_before"):
        val = data.get(key)
        if isinstance(val, str):
            with contextlib.suppress(ValueError):
                setattr(f, key, datetime.fromisoformat(val).replace(tzinfo=UTC))
    return f


def extract_filter(query: str) -> MetadataFilter:
    """Best-effort metadata filter for a natural-language query."""
    f = _regex_filter(query)
    if _needs_llm(query, f):
        llm_f = _llm_filter(query)
        f.asx_codes |= llm_f.asx_codes
        f.released_after = f.released_after or llm_f.released_after
        f.released_before = f.released_before or llm_f.released_before
    return f
