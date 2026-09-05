# Experiments

Everything here was written to answer one question once. It is kept because the
answers are cited in `docs/`, not because it is expected to run again unchanged.

**`reproduce/`** — the exact command sequences behind published results. Several
call the sibling project's `seismic-cli` rather than anything in this repo, and
`docs/REPRODUCE_ponly.md` and the experiment write-ups reference them by path.
They are deliberately NOT folded into a CLI: their value is being a verbatim
record of what was run, and a subcommand that reformats the arguments loses that.

**`analyses/`** — one-off analysis scripts. Each answered a specific question and
its result is written up in `docs/`; the code is here so the number can be
re-derived, not because it is a tool.

Reusable tooling lives in `scripts/`. If something here gets run a third time,
that is the signal it belongs there instead.
