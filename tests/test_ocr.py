"""Tests for the image-only OCR fallback.

Real OCR needs the Tesseract binary plus the ``pypdfium2``/``pytesseract``
wheels, none of which we want to require in CI. So every test injects *fake*
``pypdfium2`` and ``pytesseract`` modules into ``sys.modules`` (or makes the
import raise) and asserts ``ocr_pdf`` behaves: it never crashes, reuses
``ParsedPage`` with correct page ordering, and sets ``image_only`` based purely
on whether any text came back.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from asx_grounded.ingestion.ocr import ocr_pdf
from asx_grounded.ingestion.parse_pdf import ParsedPage, ParsedPdf


# --------------------------------------------------------------------------- #
# Fakes that imitate the pypdfium2 + pytesseract surface ocr_pdf actually uses.
# --------------------------------------------------------------------------- #
class _FakeBitmap:
    def __init__(self, marker: str) -> None:
        self._marker = marker

    def to_pil(self) -> str:
        # ocr_pdf only forwards this object to image_to_string; a str stand-in
        # is enough for the fake OCR engine below.
        return self._marker


class _FakePage:
    def __init__(self, marker: str) -> None:
        self._marker = marker

    def render(self, scale: float) -> _FakeBitmap:
        # ``scale`` is accepted only for signature parity with pypdfium2.
        del scale
        return _FakeBitmap(self._marker)


class _FakeDocument:
    """Iterable PdfDocument stand-in. ``page_texts`` is the OCR output per page."""

    closed = False

    def __init__(self, page_texts: list[str]) -> None:
        self._pages = [_FakePage(t) for t in page_texts]

    def __iter__(self):
        return iter(self._pages)

    def close(self) -> None:
        type(self).closed = True


def _install_fake_engine(monkeypatch: pytest.MonkeyPatch, page_texts: list[str]) -> None:
    """Register fake ``pypdfium2`` + ``pytesseract`` modules in sys.modules.

    ``_FakePage.render().to_pil()`` returns the page marker string, and the fake
    ``image_to_string`` simply echoes that marker back as the OCR result — so
    ``page_texts`` is exactly what ocr_pdf will scrub and store per page.
    """
    fake_pdfium = SimpleNamespace(PdfDocument=lambda _path: _FakeDocument(page_texts))
    fake_tesseract = SimpleNamespace(image_to_string=lambda image: image)
    monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract)


# --------------------------------------------------------------------------- #
# Graceful degradation when the OCR engine is unavailable.
# --------------------------------------------------------------------------- #
def test_graceful_noop_when_engine_import_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If pypdfium2/pytesseract can't be imported, return empty + image_only."""
    # Ensure any previously-imported real/fake modules can't satisfy the import.
    monkeypatch.delitem(sys.modules, "pypdfium2", raising=False)
    monkeypatch.delitem(sys.modules, "pytesseract", raising=False)

    real_import = builtins.__import__

    def _fail_import(name: str, *args, **kwargs):
        if name in {"pypdfium2", "pytesseract"}:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_import)

    result = ocr_pdf(Path("nonexistent.pdf"), ann_id="NO_ENGINE")

    assert isinstance(result, ParsedPdf)
    assert result.ann_id == "NO_ENGINE"
    assert result.pages == []
    assert result.total_chars == 0
    assert result.image_only is True


def test_graceful_noop_when_render_or_ocr_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mid-run failure (e.g. missing Tesseract binary) must not propagate."""

    def _boom(image):  # the Tesseract wrapper blows up on first call
        raise RuntimeError("tesseract is not installed or it's not in your PATH")

    fake_pdfium = SimpleNamespace(PdfDocument=lambda _path: _FakeDocument(["page one"]))
    fake_tesseract = SimpleNamespace(image_to_string=_boom)
    monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract)

    result = ocr_pdf(Path("scan.pdf"), ann_id="BROKEN_BINARY")

    assert isinstance(result, ParsedPdf)
    assert result.pages == []
    assert result.image_only is True
    assert result.total_chars == 0


# --------------------------------------------------------------------------- #
# Happy path: OCR yields text.
# --------------------------------------------------------------------------- #
def test_pages_are_parsed_pages_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    page_texts = [
        "Commonwealth Bank announces a fully-franked dividend of $2.40 per share.",
        "Second scanned page with the record and payment dates.",
        "Third scanned page of supplementary material.",
    ]
    _install_fake_engine(monkeypatch, page_texts)

    result = ocr_pdf(Path("scan.pdf"), ann_id="OCR_001")

    assert isinstance(result, ParsedPdf)
    assert result.ann_id == "OCR_001"
    assert len(result.pages) == 3
    assert all(isinstance(p, ParsedPage) for p in result.pages)
    # page_num must be 1-based and strictly ordered.
    assert [p.page_num for p in result.pages] == [1, 2, 3]
    # text survives the scrub for these non-boilerplate lines.
    assert "fully-franked dividend" in result.pages[0].text
    assert result.total_chars == sum(len(p.text) for p in result.pages)


def test_image_only_false_when_ocr_yields_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_engine(monkeypatch, ["Some real recovered text from the scan."])

    result = ocr_pdf(Path("scan.pdf"), ann_id="OCR_TEXT")

    assert result.image_only is False
    assert result.total_chars > 0


# --------------------------------------------------------------------------- #
# image_only stays True when OCR recovers nothing usable.
# --------------------------------------------------------------------------- #
def test_image_only_true_when_ocr_returns_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only OCR output is scrubbed to empty -> still image_only."""
    _install_fake_engine(monkeypatch, ["", "   \n  \t ", "\n\n"])

    result = ocr_pdf(Path("blank_scan.pdf"), ann_id="OCR_BLANK")

    assert len(result.pages) == 3  # pages are still emitted, just empty
    assert [p.page_num for p in result.pages] == [1, 2, 3]
    assert all(p.text == "" for p in result.pages)
    assert result.image_only is True
    assert result.total_chars == 0


def test_boilerplate_only_page_scrubbed_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page whose only OCR text is ASX boilerplate yields no usable text."""
    # These lines all match parse_pdf._BOILERPLATE_PATTERNS.
    _install_fake_engine(monkeypatch, ["ASX Market Announcement\npage 1 of 1\n42"])

    result = ocr_pdf(Path("boilerplate.pdf"), ann_id="OCR_BOILER")

    assert len(result.pages) == 1
    assert result.pages[0].text == ""
    assert result.image_only is True
