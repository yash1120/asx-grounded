"""OCR fallback for image-only ASX announcement PDFs.

Some ASX announcements are scanned images with no embedded text layer, so
``parse_pdf`` flags them ``image_only=True`` and ``embed.py`` skips them,
silently dropping that content from the corpus. This module renders each page
to a bitmap and runs OCR over it, producing the *same* ``ParsedPage`` /
``ParsedPdf`` shape so the output drops straight into the existing chunker.

Design notes:
  * Rendering uses **pypdfium2** (PDFium bindings) rather than ``pdf2image`` so
    we don't need the separate Poppler binary on the host — only the Tesseract
    binary is an external dependency, and its absence degrades gracefully.
  * OCR text reuses ``parse_pdf._scrub`` so boilerplate (cover-page line,
    "page X of Y", bare page numbers, copyright) is trimmed identically to the
    text-extraction path.
  * If *any* OCR/render dependency (the ``pytesseract``/``pypdfium2`` imports or
    the Tesseract binary itself) is missing or fails, we log a warning and
    return an empty ``image_only=True`` ``ParsedPdf`` — never raising, so a bad
    scan can never crash the ingestion loop.

# DEPENDENCY: add to pyproject.toml [project].dependencies:
#   "pypdfium2>=4.30"   (pure-wheel PDF page rendering, no Poppler needed)
#   "pytesseract>=0.3.13"   (thin wrapper around the Tesseract OCR engine)
# Plus the system Tesseract binary must be on PATH (e.g. `choco install
# tesseract` on Windows / `apt-get install tesseract-ocr` on Linux). If it is
# absent, ocr_pdf degrades to a no-op rather than failing.

# WIRE-IN: in embed.py, replace the two `if parsed.image_only: continue` guards
# (in embed_all and write_bm25_corpus) with an OCR retry, e.g.:
#   from asx_grounded.ingestion.ocr import ocr_pdf
#   if parsed.image_only:
#       parsed = ocr_pdf(pdf_path, ann_id=ann_id)
#       if parsed.image_only:
#           continue
"""

from __future__ import annotations

from pathlib import Path

import structlog

from asx_grounded.ingestion.parse_pdf import ParsedPage, ParsedPdf, _scrub

log = structlog.get_logger()

# Render scale passed to PDFium: 300 DPI / 72 PDF-pt-per-inch ≈ 4.17. Higher
# DPI gives Tesseract cleaner glyphs at the cost of memory/time.
_RENDER_SCALE = 300 / 72


def ocr_pdf(pdf_path: Path, ann_id: str) -> ParsedPdf:
    """Render each page of ``pdf_path`` to an image and OCR it.

    Returns a ``ParsedPdf`` with ``image_only=False`` when OCR yields any text.
    On a missing/broken OCR engine (or any other failure) logs a warning and
    returns an empty ``image_only=True`` ``ParsedPdf`` so the pipeline never
    crashes on an unreadable scan.
    """
    try:
        import pypdfium2 as pdfium
        import pytesseract
    except Exception as exc:  # ImportError, or a broken native install
        log.warning("ocr.unavailable", ann_id=ann_id, path=str(pdf_path), error=str(exc))
        return _empty(ann_id)

    pages: list[ParsedPage] = []
    total_chars = 0
    image_only = True

    try:
        document = pdfium.PdfDocument(str(pdf_path))
        try:
            for i, page in enumerate(document, start=1):
                bitmap = page.render(scale=_RENDER_SCALE)
                image = bitmap.to_pil()
                raw = pytesseract.image_to_string(image) or ""
                cleaned = _scrub(raw)
                if cleaned:
                    image_only = False
                pages.append(ParsedPage(page_num=i, text=cleaned))
                total_chars += len(cleaned)
        finally:
            document.close()
    except Exception as exc:  # render failure, missing Tesseract binary, etc.
        log.warning("ocr.failed", ann_id=ann_id, path=str(pdf_path), error=str(exc))
        return _empty(ann_id)

    if image_only:
        log.warning("ocr.no_text", ann_id=ann_id, path=str(pdf_path))
    else:
        log.info("ocr.done", ann_id=ann_id, pages=len(pages), chars=total_chars)

    return ParsedPdf(
        ann_id=ann_id,
        pages=pages,
        total_chars=total_chars,
        image_only=image_only,
    )


def _empty(ann_id: str) -> ParsedPdf:
    """A no-text result that keeps the caller's `image_only` skip path intact."""
    return ParsedPdf(ann_id=ann_id, pages=[], total_chars=0, image_only=True)
