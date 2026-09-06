"""What does the 6 s detector cost in false alarms on continuous data?

Every detection number in this repo comes from a *curated* benchmark: balanced
classes, arrival-anchored positives, and negatives mined from a noise pool.
`docs/TODO.md` 2.3 asks the question that benchmark cannot answer -- put the
detector on an uninterrupted station record and count the alarms it raises when
nothing is happening.

The benchmark's own answer is a straight extrapolation and it is alarming: at
threshold 0.5 the 3-seed `1d/cnn-lstm` ensemble puts 141 of 7,906 test noise
windows on the wrong side, an FPR of 1.78%. A day holds 14,400 non-overlapping
6 s windows, so that rate is **257 false alarms per day** -- roughly one every
five minutes. Whether continuous noise actually behaves like mined noise is the
measurement.

**The catalogue bounds this from one side only.** An alarm that matches no
catalogued event is either a false positive or a real earthquake AFAD never
listed, and this script cannot tell them apart. Every "false alarm" figure it
prints is therefore an UPPER bound. That cuts the way you would hope -- a
detector worth deploying is one whose upper bound is already small.

Two detectors are scored, because they answer different questions of the same
record and the expensive part -- reading and filtering -- is shared:

    6s      the headline detector, window [P-2.0, P+4.0], 0.9896 AUC against a
            0.9049 amplitude floor. The 1.78% FPR above is its own, so this is
            the arm that actually tests the 257-per-day extrapolation.
    ponly   window [P-2.0, P+1.4], 0.8712 against a 0.6679 floor. Read this as
            **P-phase detection, not early warning**: the 1.4 s tail excludes S
            only where S-P exceeds 1.4 s, so calling it early warning would
            smuggle in a distance condition the window itself imposes. Its
            recall is reported against distance for that reason.

Three phases, because the second is the expensive one and should not be redone
to change a guard window:

    baseline  sample the archive, build the per-component (mu, sigma) the
              training windows were standardized against
    scan      window the whole record, score every window, write (t, p) per
              chunk and arm
    report    associate one arm's scores with the catalogue and tabulate

    sk falsealarm baseline \\
        --zips 'afad_raw/MANT/*.zip' --out mant_baseline.json
    sk falsealarm scan \\
        --zips 'afad_raw/MANT/*.zip' --baseline-json mant_baseline.json \\
        --arm 6s:6.0:trained_model_branch1d_asinh:cnn-lstm \\
        --arm ponly:3.4:trained_model_ponly_matched:cnn-lstm \\
        --out-dir scores_mant
    sk falsealarm report \\
        --scores 'scores_mant/6s/*.npz' --station MANT \\
        --stations-csv catalogs/istasyon_katalog.csv \\
        --catalog catalogs/catalog_current.csv --out-prefix mant_fa_6s

The preprocessing here is not a reimplementation-by-eye: it is the same order of
operations `seismic_cli.core` applies when it builds a training window (detrend
twice, 5% Hann taper, 4th-order 1-45 Hz bandpass, resample to 100 Hz, then
standardize against the station baseline), applied to whole blocks of windows at
a time instead of one at a time. `verify` checks that claim against real dataset
tensors rather than asserting it.

--- how this file is put together ---

Each command module declares its own `NAME`, `HELP` and `add_args(q)`, so this
only has to put them in order. The order is the one the tool has always printed
and is the order the phases run in, not alphabetical: you build a baseline,
scan with it, then report, and the two-station and timing readings come off the
same scores afterwards.
"""
import argparse

from sismokaos.continuous import (baseline, coincidence, report, scan, timing,
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


if __name__ == "__main__":
    raise SystemExit(main() or 0)
