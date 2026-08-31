# TÜBİTAK report sources

Moved here from `~/Desktop` on 2026-08-31 so they are version-controlled and
diffable. **The Desktop paths still work** — they are symlinks into this
directory, so the existing editing workflow is unchanged.

| file | what it is |
|---|---|
| `tubitak_rapor_v2.md` | **current** — the review/reference copy, freshened 2026-08-31 |
| `tubitak_rapor_v2_taslak.md` | earlier draft |
| `tubitak_rapor_bolum_2_5.md` | section 2.5 working document |
| `catalog_mlp_architecture_report.md` | the catalog-MLP architecture write-up (source of the 0.5886 figure) |
| `detector_onepager.md` | one-page detector summary |
| `short_window_cnn_detector_report.md` | short-window detector write-up |
| `rapor_sekiller/` | figures (3 PNGs) |

`.docx` renders are **not** tracked — they are generated from the Markdown and
would only churn the history. Regenerate with `scripts/md2docx.py`. They remain
on the Desktop where they were.

## Why the sources moved

`~/Desktop` is not a git repository. On 2026-08-31 a backup of
`tubitak_rapor_v2.md` was deleted on the incorrect assumption that git held the
original; nothing was lost, but there was no version history to fall back on.
There is now.
