from __future__ import annotations

from asx_grounded.ingestion.parse_pdf import _scrub

# _scrub is module-private boilerplate-stripping; import it directly and feed it
# raw strings (no PDF / pdfplumber needed).


def test_scrub_removes_asx_announcement_cover_line() -> None:
    raw = "ASX Announcement\nCommonwealth Bank declares a dividend of $2.40 per share."
    out = _scrub(raw)
    assert "ASX Announcement" not in out
    assert "Commonwealth Bank declares a dividend of $2.40 per share." in out


def test_scrub_removes_asx_market_announcement_variant() -> None:
    raw = "ASX Market Announcement 1 January 2026\nSubstantive disclosure content here."
    out = _scrub(raw)
    assert "ASX Market Announcement" not in out
    assert "Substantive disclosure content here." in out


def test_scrub_is_case_insensitive_for_cover_line() -> None:
    raw = "asx announcement\nReal body text."
    out = _scrub(raw)
    assert "asx announcement" not in out
    assert "Real body text." in out


def test_scrub_removes_page_x_of_y_footer() -> None:
    raw = "Body line one.\nPage 2 of 7\nBody line two."
    out = _scrub(raw)
    assert "Page 2 of 7" not in out
    assert "Body line one." in out
    assert "Body line two." in out


def test_scrub_removes_bare_page_numbers() -> None:
    raw = "First paragraph of disclosure.\n3\nSecond paragraph of disclosure."
    out = _scrub(raw)
    lines = out.splitlines()
    assert "3" not in lines  # the standalone page-number line is gone
    assert "First paragraph of disclosure." in out
    assert "Second paragraph of disclosure." in out


def test_scrub_keeps_numbers_embedded_in_text() -> None:
    # A bare-number line is boilerplate, but a number inside a sentence is content.
    raw = "The dividend is $2.40 per share.\n42\nRecord date is 14 February 2026."
    out = _scrub(raw)
    assert "$2.40" in out
    assert "14 February 2026" in out
    assert "42" not in out.splitlines()


def test_scrub_removes_copyright_line() -> None:
    raw = "Material disclosure text.\n© 2026 Commonwealth Bank. All rights reserved.\nMore disclosure."
    out = _scrub(raw)
    assert "All rights reserved" not in out
    assert "©" not in out
    assert "Material disclosure text." in out
    assert "More disclosure." in out


def test_scrub_copyright_is_case_insensitive() -> None:
    raw = "Body.\n© 2026 BHP Group. ALL RIGHTS RESERVED.\nBody two."
    out = _scrub(raw)
    assert "RIGHTS RESERVED" not in out
    assert "Body." in out
    assert "Body two." in out


def test_scrub_collapses_three_or_more_blank_lines() -> None:
    raw = "Paragraph one.\n\n\n\n\nParagraph two."
    out = _scrub(raw)
    assert "\n\n\n" not in out
    assert out == "Paragraph one.\n\nParagraph two."


def test_scrub_strips_leading_and_trailing_whitespace() -> None:
    raw = "\n\n   Content in the middle.   \n\n"
    out = _scrub(raw)
    assert out == "Content in the middle."


def test_scrub_combines_all_boilerplate_rules() -> None:
    raw = (
        "ASX Announcement\n"
        "Page 1 of 3\n"
        "Commonwealth Bank of Australia declares a fully-franked dividend of $2.40.\n"
        "1\n"
        "The record date is 14 February 2026.\n"
        "© 2026 Commonwealth Bank of Australia. All rights reserved.\n"
    )
    out = _scrub(raw)

    # Boilerplate gone.
    assert "ASX Announcement" not in out
    assert "Page 1 of 3" not in out
    assert "All rights reserved" not in out
    assert "1" not in out.splitlines()
    # Substance retained.
    assert "Commonwealth Bank of Australia declares a fully-franked dividend of $2.40." in out
    assert "The record date is 14 February 2026." in out
    # No oversized blank runs left behind by the substitutions.
    assert "\n\n\n" not in out
    assert out == out.strip()


def test_scrub_clean_text_is_unchanged() -> None:
    raw = "A perfectly clean paragraph with no boilerplate."
    assert _scrub(raw) == raw


def test_scrub_empty_string() -> None:
    assert _scrub("") == ""


def test_scrub_only_boilerplate_yields_empty() -> None:
    raw = "ASX Announcement\nPage 1 of 2\n5\n"
    assert _scrub(raw) == ""
