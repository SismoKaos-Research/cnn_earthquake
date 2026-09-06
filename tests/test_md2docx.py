"""`sk docx` must finish, and must be callable the way `sk` calls it.

Two defects shipped here together, and neither was visible: `main()` took its
paths positionally while `sk` calls `main()` with none, so `sk docx` had never
once run; and the converter's last branch could fail to advance, so it spun at
99% CPU on `docs/report.md` for as long as it was given.

Both survived because `python-docx` was not installed anywhere the suite ran,
so every test that touched this module skipped. It is a dev dependency now.
"""
import signal

import pytest

pytest.importorskip("docx")

from sismokaos.reporting.md2docx import convert


def _convert_under_a_deadline(src, dst, seconds=20):
    """Runs the converter, failing loudly rather than hanging the suite."""
    def bang(signum, frame):
        raise TimeoutError("md2docx did not terminate -- a branch is not advancing")

    old = signal.signal(signal.SIGALRM, bang)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        convert(str(src), str(dst))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def test_a_table_row_orphaned_from_its_header_does_not_hang(tmp_path):
    """The exact shape at `docs/report.md:1889`.

    A blank line inside a table leaves the rows below it with no header and no
    separator above them. Every branch declines the line, including the
    paragraph accumulator, which breaks on `|` before collecting anything.
    """
    src = tmp_path / "in.md"
    src.write_text(
        "# Heading\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n"
        "\n"
        "| 3 | 4 |\n"          # orphaned: no header, no separator above
        "| 5 | 6 |\n"
        "\nafter\n")
    _convert_under_a_deadline(src, tmp_path / "out.docx")
    assert (tmp_path / "out.docx").exists()


@pytest.mark.parametrize("body", [
    "|\n", "| \n", "|only|\n", ">\n", "```\nunclosed\n", "1.\n", "-\n",
    "![](missing.png)\n", "#\n", "| a |\n| b |\n",
])
def test_no_input_shape_stops_the_loop(tmp_path, body):
    """Every branch must advance `i`, whatever the line looks like."""
    src = tmp_path / "in.md"
    src.write_text(body)
    _convert_under_a_deadline(src, tmp_path / "out.docx", seconds=10)


def test_the_repos_own_report_converts(tmp_path):
    """The document that actually hung, end to end."""
    from pathlib import Path

    report = Path(__file__).resolve().parents[1] / "docs" / "report.md"
    if not report.exists():
        pytest.skip("docs/report.md is not in this checkout")
    _convert_under_a_deadline(report, tmp_path / "report.docx", seconds=120)
    assert (tmp_path / "report.docx").stat().st_size > 20_000
