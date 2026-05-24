"""Embed chunks with bge-large-en-v1.5 and upsert to Qdrant.

The embedder is loaded lazily so importing this module doesn't pull torch into
processes that don't need it (e.g. the API container in production can use a
pre-built collection without ever loading the model).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import structlog
import typer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from asx_grounded.config import get_settings
from asx_grounded.ingestion.chunk import chunk_pdf
from asx_grounded.ingestion.parse_pdf import parse_pdf
from asx_grounded.models import Announcement, Chunk

log = structlog.get_logger()
app = typer.Typer(help="Embed parsed chunks and push to Qdrant.")


class Embedder:
    """Lazy wrapper around sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _ensure(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("embed.loading_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure()
        embs = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, e)) for e in embs]


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qdrant_models.VectorParams(
            size=dim,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    log.info("qdrant.collection_created", name=name, dim=dim)


def _to_point(chunk: Chunk, vec: list[float], ann: Announcement | None) -> qdrant_models.PointStruct:
    payload: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "ann_id": chunk.ann_id,
        "asx_code": chunk.asx_code,
        "chunk_idx": chunk.chunk_idx,
        "page_num": chunk.page_num,
        "text": chunk.text,
        "token_count": chunk.token_count,
    }
    if ann is not None:
        payload.update(
            {
                "headline": ann.headline,
                "released_at": ann.released_at.isoformat(),
                "announcement_type": ann.announcement_type.value,
                "is_price_sensitive": ann.is_price_sensitive,
                "asx_page_url": ann.asx_page_url,
            }
        )
    return qdrant_models.PointStruct(
        id=abs(hash(chunk.chunk_id)) % (10**18),
        vector=vec,
        payload=payload,
    )


def _load_announcements(manifest_path: Path) -> dict[str, Announcement]:
    out: dict[str, Announcement] = {}
    if not manifest_path.exists():
        return out
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ann = Announcement.model_validate_json(line)
            out[ann.ann_id] = ann
    return out


def _iter_pdfs(data_dir: Path) -> Iterable[tuple[str, str, Path]]:
    pdf_root = data_dir / "pdfs"
    if not pdf_root.exists():
        return
    for code_dir in pdf_root.iterdir():
        if not code_dir.is_dir():
            continue
        for pdf in code_dir.glob("*.pdf"):
            yield code_dir.name, pdf.stem, pdf


def embed_all(data_dir: Path, batch_size: int = 64) -> int:
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    ensure_collection(client, settings.qdrant_collection, settings.embedding_dim)

    embedder = Embedder(settings.embedding_model)
    manifest = _load_announcements(data_dir / "announcements.jsonl")

    pending_chunks: list[Chunk] = []
    pending_ann: list[Announcement | None] = []
    embedded = 0

    def flush() -> None:
        nonlocal embedded
        if not pending_chunks:
            return
        vecs = embedder.encode([c.text for c in pending_chunks])
        points = [_to_point(c, v, a) for c, v, a in zip(pending_chunks, vecs, pending_ann, strict=True)]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        embedded += len(points)
        pending_chunks.clear()
        pending_ann.clear()

    for code, ann_id, pdf_path in _iter_pdfs(data_dir):
        try:
            parsed = parse_pdf(pdf_path, ann_id=ann_id)
        except Exception as exc:  # pragma: no cover — pdfplumber raises a wide variety
            log.warning("parse.failed", path=str(pdf_path), error=str(exc))
            continue
        if parsed.image_only:
            continue
        chunks = chunk_pdf(parsed, asx_code=code)
        ann = manifest.get(ann_id)
        for c in chunks:
            pending_chunks.append(c)
            pending_ann.append(ann)
            if len(pending_chunks) >= batch_size:
                flush()
        log.info("embed.processed", ann_id=ann_id, chunks=len(chunks))
    flush()

    log.info("embed.done", total=embedded)
    return embedded


def write_bm25_corpus(data_dir: Path, out_path: Path) -> int:
    """Snapshot the chunk corpus to disk for the BM25 in-process index.

    Qdrant holds vectors; BM25 needs the raw tokens locally. We write a
    JSONL with chunk_id, ann_id, asx_code, text — sufficient for the
    retrieval layer to build BM25 at startup.
    """
    manifest = _load_announcements(data_dir / "announcements.jsonl")
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for code, ann_id, pdf_path in _iter_pdfs(data_dir):
            try:
                parsed = parse_pdf(pdf_path, ann_id=ann_id)
            except Exception:
                continue
            if parsed.image_only:
                continue
            chunks = chunk_pdf(parsed, asx_code=code)
            ann = manifest.get(ann_id)
            for c in chunks:
                rec = {
                    "chunk_id": c.chunk_id,
                    "ann_id": c.ann_id,
                    "asx_code": c.asx_code,
                    "chunk_idx": c.chunk_idx,
                    "page_num": c.page_num,
                    "text": c.text,
                    "headline": ann.headline if ann else "",
                    "released_at": ann.released_at.isoformat() if ann else None,
                    "asx_page_url": ann.asx_page_url if ann else "",
                }
                fh.write(json.dumps(rec) + "\n")
                count += 1
    return count


@app.command()
def cli(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data/raw"),
    bm25_out: Annotated[Path, typer.Option("--bm25-out")] = Path("data/processed/bm25_corpus.jsonl"),
) -> None:
    """Parse all PDFs in ``--data-dir``, embed, upsert to Qdrant, and snapshot BM25 corpus."""
    embedded = embed_all(data_dir)
    bm25 = write_bm25_corpus(data_dir, bm25_out)
    typer.echo(f"Embedded {embedded} chunks; wrote {bm25} rows to {bm25_out}")


if __name__ == "__main__":
    app()
