"""Consolidates per-hour .npy struct-array files (E/N/Z fields, produced by
either the real 5Hz pipeline or a gap-only preprocessing step whose script
is not in this repo) into one
pre-allocated, memory-mappable array plus a companion hours.npy timestamp
file. Interpolates gaps ONCE here (same logic as raw_cnn_lstm_forecast.py's
load_hourly_raw) instead of every training run redoing it from scratch.

The point: load_hourly_raw currently requires the WHOLE archive to fit in
RAM at once -- fine at 5Hz (a few GB total) but not at native rate (100Hz,
~20x the samples/hour; BODT's full gap-only archive is ~74GB, more than
this machine's 14GB RAM). A consolidated .npy file can be opened with
np.load(path, mmap_mode='r') instead, so only the hours a given batch
actually touches get pulled into RAM -- see load_hourly_raw_consolidated in
raw_cnn_lstm_forecast.py.

Disk-safe by construction: this machine doesn't have room to hold both the
full source directory and the full consolidated output at once, so each
source .npy is deleted immediately after its hour is written and flushed
into the (sparse -- doesn't consume real disk until written) consolidated
array, rather than deleting everything at the end. Safe to interrupt and
re-run: the full hour list is written to hours.npy up front, before any
deletion happens, and a missing source file is treated as "already
consolidated" and skipped -- so a partial run just resumes where it left off.

Usage:
    python consolidate_hourly_raw.py \\
        --data-root .../data/aegean_bodt_2024_2026_gaponly \\
        --output-dir .../data/aegean_bodt_2024_2026_gaponly_consolidated \\
        --hour-samples 360000 --delete-source

Not imported by anything else -- standalone preprocessing step. Its output
(raw.npy + hours.npy) is read back by
raw_cnn_lstm_forecast.py:load_hourly_raw_consolidated, used from both
raw_cnn_lstm_forecast.py and raw100hz_cnn_lstm_forecast.py via `--consolidated`.
"""

import argparse
import calendar
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_DATE_DIR_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")


def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="Consolidate per-hour .npy files into one mmap-able array.")
    p.add_argument("--data-root", required=True, help="Directory of YYYY_MM_DD/*.npy hourly files.")
    p.add_argument("--output-dir", required=True,
                  help="Destination for raw.npy (the big array) and hours.npy (timestamps).")
    p.add_argument("--hour-samples", type=int, required=True,
                  help="Samples per hour per channel (18000 for 5Hz, 360000 for native 100Hz).")
    p.add_argument("--delete-source", action="store_true",
                  help="Delete each source .npy immediately after it's safely written and flushed "
                       "into the consolidated array. Off by default (dry run otherwise) since it's "
                       "destructive -- only pass it once you trust the output.")
    p.add_argument("--flush-every", type=int, default=1,
                  help="Flush (fsync-equivalent) the memmap every N hours. 1 is safest (every hour) "
                       "-- only raise it if you've verified the source archive is fully regenerable "
                       "and want fewer flush stalls.")
    return p.parse_args()


def scan_entries(data_root: Path):
    """Finds every hourly source .npy file under `data_root`.

    Cheap -- no array data touched, just filename parsing.

    Args:
        data_root: Directory of YYYY_MM_DD/*.npy hourly files.

    Returns:
        List of (hour_dt, npy_path) tuples, sorted chronologically.
    """
    date_dirs = sorted(d for d in data_root.iterdir() if d.is_dir() and _DATE_DIR_RE.match(d.name))
    entries = []
    for date_dir in date_dirs:
        for npy_path in sorted(date_dir.glob("*.npy")):
            parts = npy_path.stem.split("_")
            if len(parts) < 2:
                continue
            try:
                hour_dt = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S")
            except ValueError:
                continue
            entries.append((hour_dt, npy_path))
    entries.sort(key=lambda e: e[0])
    return entries


def load_hour(npy_path: Path, hour_samples: int) -> np.ndarray:
    """Loads and gap-interpolates one hour's raw waveform file.

    Same logic as raw_cnn_lstm_forecast.py's load_hourly_raw, just per-file
    instead of over a pre-loaded array.

    Args:
        npy_path: Path to one hour's struct-array .npy file (E/N/Z fields).
        hour_samples: Expected samples per hour per channel; the output is
            padded/truncated to this length if the source disagrees.

    Returns:
        float32 array, shape (3, hour_samples), gaps linearly interpolated
        then anything left over zeroed.
    """
    struct = np.load(npy_path)
    out = np.empty((3, hour_samples), dtype=np.float32)
    for c, comp in enumerate(("E", "N", "Z")):
        x = (struct[comp].astype(np.float32) if comp in struct.dtype.names
            else np.full(len(struct), np.nan, dtype=np.float32))
        if len(x) != hour_samples:
            fixed = np.full(hour_samples, np.nan, dtype=np.float32)
            n = min(hour_samples, len(x))
            fixed[:n] = x[:n]
            x = fixed
        nan = np.isnan(x)
        if nan.any() and (~nan).sum() > 3:
            x[nan] = np.interp(np.flatnonzero(nan), np.flatnonzero(~nan), x[~nan])
        out[c] = np.nan_to_num(x, nan=0.0)
    return out


def main():
    """Builds (or resumes) the consolidated raw.npy + hours.npy pair from
    the source hourly .npy tree, deleting each source file after it's
    written when `--delete-source` is passed."""
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.npy"
    hours_path = out_dir / "hours.npy"

    # hours.npy, once written, is the permanent hour-to-array-index mapping
    # and the sole source of truth for n_hours -- NOT a re-scan of data_root,
    # which shrinks after every deletion on a resumed run. Deriving n_hours
    # from a re-scan was an earlier bug here: it made a resumed run compute a
    # smaller n_hours than the already-allocated raw.npy and reject it as a
    # false "shape mismatch". Caught by simulating a mid-run interruption on
    # a small copy before trusting this on the real archive.
    if hours_path.exists():
        hours_arr = np.load(hours_path)
        n_hours = len(hours_arr)
        print(f"[INFO] resuming from existing manifest {hours_path}: {n_hours} hours")
    else:
        print(f"[INFO] scanning {data_root} ...")
        entries = scan_entries(data_root)
        n_hours = len(entries)
        if n_hours == 0:
            print("[ERROR] no hourly .npy files found under data-root -- nothing to do.")
            return
        print(f"[INFO] {n_hours} hours, {entries[0][0]} .. {entries[-1][0]}")
        # calendar.timegm (not datetime.timestamp(), which interprets a naive
        # datetime as LOCAL time) treats the naive filename-derived timestamp
        # as a literal UTC-numbered value, matching load_hourly_raw's own
        # hour_index -- which never does any timezone conversion either, it
        # just uses strptime's naive datetimes as-is. On this machine (UTC+3)
        # timestamp() silently shifted every hour by -3 on the round trip
        # through hours.npy; caught by comparing against load_hourly_raw on a
        # small sample before trusting this on the real archive.
        hours_arr = np.array([calendar.timegm(h.timetuple()) for h, _ in entries], dtype=np.int64)
        np.save(hours_path, hours_arr)
        print(f"[INFO] wrote {hours_path}")

    if raw_path.exists():
        print(f"[INFO] {raw_path} already exists, opening for resume (mode='r+')")
        raw_mm = np.lib.format.open_memmap(raw_path, mode="r+")
        if raw_mm.shape != (n_hours, 3, args.hour_samples):
            print(f"[ERROR] existing {raw_path} has shape {raw_mm.shape}, expected "
                 f"{(n_hours, 3, args.hour_samples)} -- refusing to resume into a mismatched "
                 "array. Delete it and start over if the source set genuinely changed.")
            return
    else:
        print(f"[INFO] pre-allocating {raw_path} ({n_hours} x 3 x {args.hour_samples} float32, "
             f"~{n_hours * 3 * args.hour_samples * 4 / 1e9:.1f}GB, sparse until written)")
        raw_mm = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32,
                                           shape=(n_hours, 3, args.hour_samples))

    # Whatever's still actually on disk right now (a resumed run has fewer
    # of these than n_hours -- the rest were already processed and deleted).
    path_by_hour = {h: p for h, p in scan_entries(data_root)}

    done = skipped = 0
    for h in range(n_hours):
        hour_dt = datetime.fromtimestamp(int(hours_arr[h]), tz=timezone.utc).replace(tzinfo=None)
        npy_path = path_by_hour.get(hour_dt)
        if npy_path is None:
            # Either deleted by an earlier (interrupted) run of this same
            # script -- its data is already in raw_mm, nothing to redo --
            # or never existed for this hour in the first place, in which
            # case raw_mm[h] is whatever open_memmap zero-initialized (only
            # possible on a first run, since hours_arr only ever contains
            # hours scan_entries actually found).
            skipped += 1
            continue
        raw_mm[h] = load_hour(npy_path, args.hour_samples)
        done += 1
        if done % args.flush_every == 0:
            raw_mm.flush()
        if args.delete_source:
            npy_path.unlink()
        if done % 100 == 0:
            print(f"[{h + 1}/{n_hours}] {hour_dt}  (done={done} skipped={skipped})")

    raw_mm.flush()
    print(f"[DONE] {n_hours} hours total ({done} newly written, {skipped} already done from a "
         f"prior run). Consolidated array: {raw_path}")


if __name__ == "__main__":
    main()
