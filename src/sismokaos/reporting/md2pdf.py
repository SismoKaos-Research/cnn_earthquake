"""Markdown -> PDF for the TÜBİTAK reports (no pandoc on this box).

Lives in the package for the same reason `md2docx.py` does: the reports are
regenerated often, the scratchpad does not survive a session, and a renderer
that has to be rewritten each time will not produce the same document twice.

Run through uv, since neither dependency belongs in this package:

    uv run --with markdown --with weasyprint sk pdf in.md out.pdf
    sk pdf in.md out.pdf                        # same thing, if both are installed

The stylesheet is deliberately plain and is the point of the file: A4 with page
numbers, tables that do not split across pages, and headings that do not strand
themselves at the bottom of one. `DejaVu Serif` is named first because the
reports are in Turkish and the fallback stack must cover ı, İ, ş, ğ and ç.
"""
import sys

CSS_TEXT = """
@page { size: A4; margin: 2.2cm 2cm; @bottom-center { content: counter(page);
        font-size: 9pt; color: #666; } }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 10.5pt;
       line-height: 1.45; color: #111; }
h1 { font-size: 18pt; margin: 0 0 .3em; line-height: 1.2; }
h2 { font-size: 13pt; margin: 1.4em 0 .4em; border-bottom: 1px solid #ccc;
     padding-bottom: .15em; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 1.1em 0 .3em; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: .7em 0; font-size: 9.5pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 4px 7px; text-align: left; }
th { background: #eee; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt;
       background: #f4f4f4; padding: 1px 3px; }
pre { background: #f6f6f6; padding: .6em .8em; font-size: 8.5pt;
      page-break-inside: avoid; overflow-wrap: break-word; }
blockquote { margin: .7em 0 .7em 1em; padding-left: .8em;
             border-left: 3px solid #ccc; color: #444; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.2em 0; }
"""


def render(src, dst):
    """Converts one Markdown file to PDF. Returns the output path."""
    import markdown
    from weasyprint import CSS, HTML

    body = markdown.markdown(open(src, encoding="utf-8").read(),
                             extensions=["tables", "fenced_code", "sane_lists"])
    HTML(string=body).write_pdf(dst, stylesheets=[CSS(string=CSS_TEXT)])
    return dst


def main():
    """Renders `argv[1]` to `argv[2]`."""
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    print("wrote", render(sys.argv[1], sys.argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
