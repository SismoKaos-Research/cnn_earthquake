"""The subcommand parser `scripts/continuous_false_alarms.py` presents.

Each command module declares its own `NAME`, `HELP` and `add_args(q)`, so this
only has to put them in order. The order is the one the tool has always printed
and is the order the phases run in, not alphabetical: you build a baseline,
scan with it, then report, and the two-station and timing readings come off the
same scores afterwards.

The parser description is passed in rather than taken from this module, because
`--help` must print the same thing whether the tool was reached as
`python3 scripts/continuous_false_alarms.py` or as `sk falsealarm`.
"""
import argparse

from seismolib.continuous import (baseline, coincidence, report, scan, timing,
                                  verify)

COMMANDS = (baseline, scan, report, coincidence, timing, verify)


def build_parser(description=None):
    p = argparse.ArgumentParser(description=description or __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for mod in COMMANDS:
        add_args = mod.add_args
        add_args(sub.add_parser(mod.NAME, help=mod.HELP))
    return p


def main(argv=None, description=None):
    args = build_parser(description).parse_args(argv)
    return {m.NAME: m.run for m in COMMANDS}[args.cmd](args)
