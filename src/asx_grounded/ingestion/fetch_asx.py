"""Fetch announcement metadata + PDFs for a set of ASX-listed companies.

ASX's public-facing endpoint shape has changed historically; this module isolates
that integration behind a single :class:`AsxClient`. If ASX changes the API,
only this file needs updating.

CLI usage::

    python -m asx_grounded.ingestion.fetch_asx --codes CBA,BHP,WBC --days 30
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import structlog
import typer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from asx_grounded.config import get_settings
from asx_grounded.models import Announcement, AnnouncementType

log = structlog.get_logger()
app = typer.Typer(help="Fetch ASX announcements + PDFs.")


ASX_LISTING_URL = "https://www.asx.com.au/asx/1/company/{code}/announcements"


class RateLimiter:
    """Simple async token-bucket-ish limiter — one request per ``min_interval`` seconds."""

    def __init__(self, per_sec: float) -> None:
        self._min_interval = 1.0 / max(per_sec, 0.01)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class AsxClient:
    def __init__(self, user_agent: str, rate_limit_per_sec: float) -> None:
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
        self._limiter = RateLimiter(rate_limit_per_sec)

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError,)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    async def _get(self, url: str, **params: Any) -> httpx.Response:
        await self._limiter.acquire()
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        return resp

    async def list_announcements(self, code: str, count: int = 50) -> list[dict[str, Any]]:
        url = ASX_LISTING_URL.format(code=code.upper())
        try:
            resp = await self._get(url, count=count, pageNum=0)
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            log.warning("asx.listing.failed", code=code, error=str(exc))
            return []
        return data.get("data", []) if isinstance(data, dict) else []

    async def download_pdf(self, url: str, dest: Path) -> bool:
        try:
            resp = await self._get(url)
        except httpx.HTTPError as exc:
            log.warning("asx.pdf.failed", url=url, error=str(exc))
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return True


def _classify(headline: str, is_price_sensitive: bool) -> AnnouncementType:
    """Heuristic classifier — good enough for filtering; refine later if needed."""
    h = headline.lower()
    if any(k in h for k in ["dividend", "distribution"]):
        return AnnouncementType.DIVIDEND
    if any(k in h for k in ["annual report", "half year", "quarterly", "appendix 4"]):
        return AnnouncementType.PERIODIC_REPORT
    if "substantial" in h and "holder" in h:
        return AnnouncementType.SUBSTANTIAL_HOLDER
    if is_price_sensitive:
        return AnnouncementType.PRICE_SENSITIVE
    return AnnouncementType.GENERAL


def _parse_item(item: dict[str, Any], code: str) -> Announcement | None:
    try:
        ann_id = str(item.get("id") or item["document_release_date"] + "_" + code)
        headline = item.get("header", "").strip() or "Untitled"
        released = item.get("document_release_date") or item.get("release_date")
        if not released:
            return None
        released_at = datetime.fromisoformat(released.replace("Z", "+00:00"))
        pdf_url = item.get("url") or item.get("pdf_url") or ""
        if not pdf_url:
            return None
        is_ps = bool(item.get("price_sensitive", False))
        return Announcement(
            ann_id=ann_id,
            asx_code=code.upper(),
            company_name=item.get("issuer_full_name") or item.get("company") or code,
            headline=headline,
            released_at=released_at,
            announcement_type=_classify(headline, is_ps),
            is_price_sensitive=is_ps,
            pdf_url=pdf_url,
            asx_page_url=item.get("page_url") or pdf_url,
            pages=int(item.get("number_of_pages", 0) or 0),
        )
    except (KeyError, ValueError, TypeError) as exc:
        log.warning("asx.parse.failed", code=code, error=str(exc))
        return None


async def fetch_for_code(
    client: AsxClient,
    code: str,
    out_dir: Path,
    since: datetime,
) -> list[Announcement]:
    raw = await client.list_announcements(code)
    out: list[Announcement] = []
    for item in raw:
        ann = _parse_item(item, code)
        if ann is None or ann.released_at < since:
            continue
        pdf_dest = out_dir / "pdfs" / code.upper() / f"{ann.ann_id}.pdf"
        if not pdf_dest.exists():
            ok = await client.download_pdf(ann.pdf_url, pdf_dest)
            if not ok:
                continue
        out.append(ann)
        log.info("ingest.fetched", code=code, ann_id=ann.ann_id, headline=ann.headline[:60])
    return out


async def fetch_all(codes: list[str], days: int, out_dir: Path) -> list[Announcement]:
    settings = get_settings()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    client = AsxClient(settings.asx_user_agent, settings.asx_rate_limit_per_sec)
    try:
        results: list[Announcement] = []
        for code in codes:
            results.extend(await fetch_for_code(client, code, out_dir, since))
        return results
    finally:
        await client.aclose()


def _write_manifest(announcements: list[Announcement], out_dir: Path) -> Path:
    manifest = out_dir / "announcements.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as fh:
        for ann in announcements:
            fh.write(ann.model_dump_json() + "\n")
    return manifest


@app.command()
def cli(
    codes: str = typer.Option(..., "--codes", help="Comma-separated ASX codes, e.g. CBA,BHP,WBC"),
    days: int = typer.Option(30, "--days", help="Lookback window in days"),
    out: Path = typer.Option(Path("data/raw"), "--out", help="Output directory"),
) -> None:
    """Fetch announcements + PDFs for the given codes into ``--out``."""
    code_list = [c.strip().upper() for c in codes.split(",") if c.strip()]
    announcements = asyncio.run(fetch_all(code_list, days, out))
    manifest = _write_manifest(announcements, out)
    typer.echo(f"Fetched {len(announcements)} announcements → {manifest}")


if __name__ == "__main__":
    app()
