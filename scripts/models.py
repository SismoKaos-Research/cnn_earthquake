"""What networks exist, what they consume, and what their flags are.

Twelve model classes are spread across `seismolib/model/` and five trainers,
and the only way to find out what any of them takes was to read its
constructor. `seismolib.model.registry` is the table; this prints it.

    sk models                       # every architecture, grouped by input kind
    sk models --family dual         # just the ones a dual-tensor task can train
    sk models dual-channel          # one model's branches, flags and defaults
    sk models --spec trained_model_ponly_natural
                                    # the spec saved beside those checkpoints

The last form is the one worth remembering. Seven evaluation scripts ask you to
re-enter a trained model's geometry by hand ("Must match the checkpoints'
training run"), and entering it wrong builds a different network. Runs that
write a `model.json` can be read back instead of remembered.
"""
import argparse
import sys

from seismolib.model.registry import (ARCHITECTURES, FAMILIES, REGISTRY,
                                      ModelSpec, by_family)

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", default=None,
                   help="show one architecture in full")
    p.add_argument("--family", default=None, choices=sorted(FAMILIES),
                   help="restrict the listing to one input kind")
    p.add_argument("--spec", default=None, metavar="DIR",
                   help="print the model.json saved in a checkpoint directory")
    return p.parse_args()


def show_one(a):
    """Prints one architecture: what it takes, its branches, and every flag."""
    print(f"{BOLD}{a.key}{OFF}  {DIM}({a.family}){OFF}")
    print(f"  {a.summary}")
    print(f"  {DIM}inputs {OFF} {a.inputs}")
    print(f"  {DIM}source {OFF} {a.source}")
    if a.notes:
        print(f"  {DIM}note   {OFF} {a.notes}")
    if a.branches:
        marked = " ".join(f"{b}*" if b == a.default_branch else b for b in a.branches)
        print(f"\n  {BOLD}--model-branch{OFF} {marked}"
              + (f"   {DIM}(also {', '.join(a.branch_aliases)}){OFF}"
                 if a.branch_aliases else ""))
        print(f"    {wrap(a.branch_help, 4)}")
    else:
        print(f"\n  {DIM}no branches -- --model-branch does not apply{OFF}")
    print(f"\n  {BOLD}flags{OFF}")
    for p in a.params:
        choices = f"  {{{','.join(p.choices)}}}" if p.choices else ""
        print(f"    {p.flag:<16}{p.type.__name__:<7}default {p.default!r}{choices}")
        print(f"      {DIM}{wrap(p.help, 6)}{OFF}")


def wrap(text, indent, width=78):
    """Wraps help text to the terminal, continuing at `indent` columns."""
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width - indent:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    out.append(line)
    return ("\n" + " " * indent).join(out)


def main():
    """Lists the registry, one architecture, or a saved spec."""
    args = parse_args()

    if args.spec:
        spec = ModelSpec.load(args.spec)
        if spec is None:
            print(f"{args.spec} has no model.json -- it predates the registry, or "
                  f"the run that wrote it did not call ModelSpec.save().",
                  file=sys.stderr)
            return 1
        print(f"{BOLD}{spec.describe()}{OFF}\n")
        a = spec.arch
        for k, v in sorted(spec.params.items()):
            d = a.param(k)
            mark = "" if d is None or v == d.default else "   <-- not the default"
            print(f"  {'--' + k.replace('_', '-'):<16}{v!r}{mark}")
        return 0

    if args.model:
        if args.model not in REGISTRY:
            print(f"unknown model {args.model!r}; run `sk models` for the list",
                  file=sys.stderr)
            return 2
        show_one(REGISTRY[args.model])
        return 0

    shown = by_family(args.family) if args.family else list(ARCHITECTURES)
    for fam in FAMILIES:
        group = [a for a in shown if a.family == fam]
        if not group:
            continue
        print(f"{BOLD}{fam}{OFF}  {DIM}{FAMILIES[fam]}{OFF}")
        for a in group:
            branches = "|".join(a.branches) if a.branches else "-"
            print(f"  {a.key:<26}{a.summary}")
            print(f"  {'':<26}{DIM}--model-branch {branches}{OFF}")
        print()
    print(f"{DIM}`sk models <name>` for one model's flags; "
          f"`sk models --spec DIR` for a trained one's.{OFF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
