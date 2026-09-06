"""
Peak ground motion from a 3-second window: the network, against the floor.

Replicates the model side of Nurtas et al. (ACDSA 2025) -- Conv1D -> BiLSTM ->
attention over a (300, 3) window, predicting peak ground motion -- on our own
response-corrected corpus, with the non-neural baseline the paper never ran
already measured and committed (`groundmotion_baselines.py`, commit 4987317).

**The question this script exists to answer**, stated before it was run:

    Does the WAVEFORM SHAPE carry information about future peak ground motion
    beyond the peak amplitude of the window itself?

Everything below is arranged so that question gets a clean answer.

--------------------------------------------------------------------------
Why the input is peak-normalised by default
--------------------------------------------------------------------------

The stored tensors are response-corrected physical velocity in cm/s, spanning
four orders of magnitude. Two ways to feed that to a network:

  * `--input-norm none` -- raw physical amplitudes. The network must handle the
    dynamic range itself, and amplitude reaches it only through the waveform.
  * `--input-norm peak` (DEFAULT) -- divide each window by its own peak vector
    magnitude, and pass `log10(that peak)` in as an auxiliary scalar.

The default is deliberate and it is what makes the comparison sharp. Under it
the network receives exactly the baseline's feature (the log peak) as a scalar,
PLUS the normalised shape. The linear baseline is then a strict special case of
this model: ignore the convolutional features, use the aux path. So

    CNN <= baseline  =>  the waveform shape adds nothing, and the shortfall is
                         optimisation or overfitting, not information
    CNN >  baseline  =>  the gain is attributable to shape, because amplitude
                         was already handed over

This follows report.md 6.3, where supplying amplitude as an auxiliary scalar
was worth +0.0874 AUC on a representation that had destroyed it. Here amplitude
is not destroyed, but separating scale from shape still makes the attribution
unambiguous rather than leaving the two entangled in one tensor.

--------------------------------------------------------------------------
Protocol
--------------------------------------------------------------------------

* **Rows are the baselines' rows.** `load()` is imported from
  `groundmotion_baselines` rather than reimplemented, so the network and the
  floor are scored on an identical set after identical, label-independent
  quality rules. A comparison across different row sets is not a comparison.
* **Test is touched once.** Early stopping and checkpoint selection both use
  validation only. The paper reports its headline on the split it early-stopped
  on, which is the specific thing not to copy.
* **Multiple seeds by default.** report.md 6.6 showed concretely that a
  single-seed margin under ~0.01-0.02 on this project can overstate an effect,
  understate it, or report the wrong sign. The spread across seeds is printed
  next to the mean, and the floor is printed on the same line.
* **Both metric spaces.** MAE and R2 in log space and, back-transformed, in
  linear space -- because the paper's unexplained ANN R2 of -10.08 reproduces
  here exactly when a log-space model is scored in linear space.

Usage:
    python cnn_groundmotion.py --dataset-dir ../seismic_cli/data/dataset_groundmotion_3s
    python cnn_groundmotion.py --target pga_fwd --arch cnn --no-aux

Not imported by anything else -- standalone script.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from sismokaos.forecasting.cnn_lstm import LSTMAttentionBranch
from sismokaos.groundmotion.groundmotion_baselines import TARGETS, load
from sismokaos.metrics import print_report, regression_report
from sismokaos.model.registry import add_model_args, spec_from_args
from sismokaos.training import seed_everything

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def respilt(d, how, seed=42, ratios=(0.70, 0.15, 0.15)):
    """Re-partitions rows without moving any tensor on disk.

    The manifest's original `split` names the DIRECTORY a tensor lives in, so it
    is preserved as `file_split` and only the LOGICAL split changes.

    **Neither grouping is clean, and that is the point of offering both.**
    The label is a property of the (event, station) pair, so:

      * `event`   -- the generator's default. Events are disjoint, so the source
                     term cannot leak, but 149 of 154 stations appear in more
                     than one split, and site response is a per-station term a
                     network can learn and reuse.
      * `station` -- stations are disjoint, so site response cannot leak, but now
                     ONE EARTHQUAKE recorded at a train station and a test
                     station shares its source term across the split. That is
                     the leak `regression.py` warns is usually the worse one for
                     a regression target.
      * `both`    -- station-disjoint, then every val/test row whose event also
                     appears in train is DROPPED. Neither term can leak. Costs
                     rows, and the count dropped is reported rather than hidden.

    Args:
        d: Manifest DataFrame with 'split', 'station_key', and 'event_id'
            columns.
        how: Grouping to use -- "event" (unchanged), "station", or "both".
        seed: Seed for the station shuffle used by "station"/"both".
        ratios: Target (train, val, test) row-count fractions for the
            station partition.

    Returns:
        A copy of `d` with 'file_split' added (the original directory-based
        split) and, for "station"/"both", 'split' reassigned to the new
        station-disjoint grouping (rows dropped for "both").
    """
    d = d.copy()
    if "file_split" not in d:
        d["file_split"] = d["split"]
    if how == "event":
        return d

    rng = random.Random(seed)
    stations = sorted(set(d.station_key))
    rng.shuffle(stations)
    size = d.station_key.value_counts().to_dict()
    total = len(d)
    targets = {s: r * total for s, r in zip(("train", "val", "test"), ratios)}
    running = {s: 0 for s in targets}
    assign = {}
    for st in stations:
        best = max(targets, key=lambda s: (targets[s] - running[s]) / max(targets[s], 1.0))
        assign[st] = best
        running[best] += size[st]
    d["split"] = d.station_key.map(assign)

    if how == "both":
        train_events = set(d.loc[d.split == "train", "event_id"])
        clash = (d.split != "train") & d.event_id.isin(train_events)
        print(f"[split] doubly-disjoint: dropping {int(clash.sum())} val/test rows whose "
              f"event also appears in train")
        d = d[~clash].copy()
    return d


def report_split(d, how):
    """Prints the row counts and any train/test event or station overlap.

    Args:
        d: Manifest DataFrame after `respilt`, with 'split', 'event_id',
            and 'station_key' columns.
        how: Grouping name to print (as returned by `respilt`'s `how` arg).
    """
    tr, te = d[d.split == "train"], d[d.split == "test"]
    shared_ev = len(set(tr.event_id) & set(te.event_id))
    shared_st = len(set(tr.station_key) & set(te.station_key))
    print(f"[split] grouping='{how}'  train {len(tr)}  val {int((d.split=='val').sum())}  "
          f"test {len(te)}")
    print(f"[split]   events shared train/test : {shared_ev}"
          f"   ({'LEAK: source term' if shared_ev else 'clean'})")
    print(f"[split]   stations shared          : {shared_st}"
          f"   ({'LEAK: site response' if shared_st else 'clean'})")
    print(f"[split]   test stations unseen in train: "
          f"{len(set(te.station_key) - set(tr.station_key))}/{te.station_key.nunique()}")


def preload(df, dataset_dir, input_norm):
    """Loads every window into one array up front.

    43k windows of (3, 300) float32 is ~156 MB, so the whole corpus fits in
    memory and per-epoch disk reads would dominate runtime for no reason.

    Args:
        df: Manifest rows to load, with 'file_split' and 'filename' columns.
        dataset_dir: Dataset root directory (contains a subdirectory per
            file_split).
        input_norm: "peak" divides each window by its own peak vector
            magnitude before returning it; "none" leaves it as raw physical
            amplitude.

    Returns:
        Tuple of (X, log_peak): `X` is a float32 array shape (n, 3, 300)
        (normalized if `input_norm == "peak"`); `log_peak` is a float32
        array shape (n,), the per-window peak of the vector magnitude
        BEFORE normalisation -- recomputed from the tensor rather than read
        from the manifest, so the scalar the model sees provably describes
        the tensor the model sees.
    """
    n = len(df)
    X = np.empty((n, 3, 300), dtype=np.float32)
    log_peak = np.empty(n, dtype=np.float32)
    for i, (split, fname) in enumerate(zip(df.file_split, df.filename)):
        t = torch.load(Path(dataset_dir) / split / fname, weights_only=True).numpy()
        mag = np.sqrt((t.astype(np.float64) ** 2).sum(axis=0))
        pk = float(mag.max())
        log_peak[i] = np.log10(pk) if pk > 0 else -12.0
        if input_norm == "peak" and pk > 0:
            t = t / pk
        X[i] = t
    return X, log_peak


def build_tensors(df, X, log_peak, target_col, aux_cols, use_aux, aux_stats=None):
    """Assembles (x, aux, y) tensors; aux is standardised with TRAIN statistics only.

    Args:
        df: Manifest rows matching `X`/`log_peak` row-for-row.
        X: Preloaded window array, shape (n, 3, 300) (see `preload`).
        log_peak: Per-window log10 peak amplitude, shape (n,) (see `preload`).
        target_col: Column in `df` holding the regression target.
        aux_cols: Auxiliary scalar column names to assemble into `aux`;
            `"__log_peak__"` is replaced by `log_peak` rather than read from
            `df`.
        use_aux: If False, returns an empty (n, 0) aux tensor regardless of
            `aux_cols`.
        aux_stats: Optional (mean, std) tuple to standardize `aux` with; if
            None, computed from this call's own `aux` (the train split
            should pass None; val/test must reuse the train split's stats).

    Returns:
        Tuple of (x, aux, y, aux_stats): `x` is `X` as a tensor, `aux` is
        the (possibly empty) standardized auxiliary tensor, `y` is the
        target tensor, and `aux_stats` is the (mean, std) tuple used (either
        the one passed in or the one just computed).
    """
    y = df[target_col].to_numpy(np.float32)
    if not use_aux:
        aux = np.zeros((len(df), 0), np.float32)
        return torch.from_numpy(X), torch.from_numpy(aux), torch.from_numpy(y), aux_stats

    cols = []
    for c in aux_cols:
        cols.append(log_peak if c == "__log_peak__" else df[c].to_numpy(np.float32))
    aux = np.column_stack(cols).astype(np.float32)
    if aux_stats is None:
        aux_stats = (aux.mean(0), aux.std(0) + 1e-8)
    aux = (aux - aux_stats[0]) / aux_stats[1]
    return torch.from_numpy(X), torch.from_numpy(aux), torch.from_numpy(y), aux_stats


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Conv1dTrunk(nn.Module):
    """
    Conv1D feature extractor over the (3, 300) sequence.

    A 1D trunk rather than reusing `RegressionSeismicCNN`'s Conv2d stack: the
    input here is a genuine time series, not an image, and a 3x3 kernel over a
    height-1 tensor would be mostly padding. This also matches the paper's
    Conv1D front end.
    """

    def __init__(self, in_ch=3, width=32, dropout=0.2):
        """Initializes the 3-stage strided Conv1D trunk.

        Args:
            in_ch: Number of input channels.
            width: Base channel width; the three conv stages use `width`,
                `width*2`, `width*4` channels.
            dropout: Dropout used after the second stage.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, padding=3, bias=False),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.MaxPool1d(2),                                   # 300 -> 150
            nn.Conv1d(width, width * 2, 5, padding=2, bias=False),
            nn.BatchNorm1d(width * 2), nn.GELU(),
            nn.MaxPool1d(2),                                   # 150 -> 75
            nn.Dropout(dropout),
            nn.Conv1d(width * 2, width * 4, 3, padding=1, bias=False),
            nn.BatchNorm1d(width * 4), nn.GELU(),
            nn.MaxPool1d(2),                                   # 75 -> 37
        )
        self.out_ch = width * 4

    def forward(self, x):
        """Runs the trunk over one batch.

        Args:
            x: Input sequence batch, shape (batch, in_ch, 300).

        Returns:
            Tensor of shape (batch, out_ch, 37).
        """
        return self.net(x)


class GroundMotionNet(nn.Module):
    """
    Conv1D trunk, optional BiLSTM+attention, then a head over [features, aux].

    `arch="cnn_lstm"` is the paper's stack; `arch="cnn"` pools the trunk
    directly and exists as the ablation that says whether the recurrent part
    earns its parameters.
    """

    def __init__(self, arch="cnn_lstm", n_aux=0, width=32, hidden=64,
                 dropout=0.2, heads=4):
        """Initializes the trunk, optional LSTM+attention branch, and head.

        Args:
            arch: "cnn_lstm" adds an `LSTMAttentionBranch` over the trunk's
                output sequence (the paper's stack); "cnn" pools the trunk
                directly instead (the recurrent-part ablation).
            n_aux: Width of an auxiliary scalar vector concatenated onto the
                pooled features before the head. 0 disables the aux path.
            width: Base channel width forwarded to `Conv1dTrunk`.
            hidden: LSTM hidden size (per direction, when `arch="cnn_lstm"`)
                and head hidden width.
            dropout: Dropout used throughout the trunk, branch, and head.
            heads: Number of attention heads (when `arch="cnn_lstm"`).
        """
        super().__init__()
        self.trunk = Conv1dTrunk(width=width, dropout=dropout)
        self.arch = arch
        if arch == "cnn_lstm":
            self.seq = LSTMAttentionBranch(self.trunk.out_ch, hidden=hidden,
                                           heads=heads, dropout=dropout)
            feat = self.seq.out_dim
        else:
            self.seq = None
            feat = self.trunk.out_ch
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat + n_aux, hidden), nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, aux):
        """Runs the trunk, optional sequence branch, and head.

        Args:
            x: Input window batch, shape (batch, 3, 300).
            aux: Auxiliary scalar batch, shape (batch, n_aux). Ignored (via
                its own zero width) when `n_aux` is 0.

        Returns:
            Tensor of shape (batch,) -- a single raw regression output per
            window.
        """
        h = self.trunk(x)
        h = self.seq(h.transpose(1, 2)) if self.seq is not None else h.mean(dim=2)
        if aux.shape[1]:
            h = torch.cat([h, aux], dim=1)
        return self.head(h).squeeze(1)


# ---------------------------------------------------------------------------
# Metrics and training
# ---------------------------------------------------------------------------

def metrics(y_log, p_log):
    """Computes MAE/R2 in both log space and back-transformed linear space.

    Args:
        y_log: True target values, in log10 space.
        p_log: Predicted target values, in log10 space.

    Returns:
        Dict with keys "MAE_log", "R2_log", "MAE_lin", "R2_lin" (floats).
    """
    lin_t, lin_p = 10.0 ** y_log, 10.0 ** p_log
    return {"MAE_log": mean_absolute_error(y_log, p_log),
            "R2_log": r2_score(y_log, p_log),
            "MAE_lin": mean_absolute_error(lin_t, lin_p),
            "R2_lin": r2_score(lin_t, lin_p)}


@torch.no_grad()
def predict(model, loader, device):
    """Runs a trained model over a loader and collects its predictions.

    Args:
        model: Trained `GroundMotionNet`.
        loader: DataLoader yielding (x, aux, y) batches.
        device: torch device to run inference on.

    Returns:
        float32 numpy array of predictions, one per sample in `loader`.
    """
    model.eval()
    out = []
    for x, aux, _ in loader:
        out.append(model(x.to(device), aux.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def train_one(args, data, n_aux, device, seed):
    """Trains a single seed. Early stopping and selection use VALIDATION only.

    Args:
        args: Parsed CLI args (uses batch_size, arch, width, hidden,
            dropout, lr, weight_decay, epochs, log_every, patience).
        data: Tuple of (train, val, test) tensor tuples, each
            (x, aux, y), as produced by `build_tensors`.
        n_aux: Width of the auxiliary scalar vector.
        device: torch device to train on.
        seed: Random seed for init/shuffling.

    Returns:
        Tuple of (metrics_dict, best_val_MAE_log, test_predictions), where
        `metrics_dict` is `metrics`'s output on the test split using the
        best (by validation MAE) epoch's weights.
    """
    seed_everything(seed)
    torch.cuda.manual_seed_all(seed)

    tr, va, te = data
    train_loader = DataLoader(TensorDataset(*tr), batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(*va), batch_size=512)
    test_loader = DataLoader(TensorDataset(*te), batch_size=512)

    model = args.model_spec.build(aux_dim=n_aux).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.L1Loss()                      # MAE in log space, the headline metric

    best, best_state, bad = np.inf, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, aux, y in train_loader:
            x, aux, y = x.to(device), aux.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            lossf(model(x, aux), y).backward()
            opt.step()
        sched.step()

        v = mean_absolute_error(va[2].numpy(), predict(model, val_loader, device))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if ep % args.log_every == 0 or ep == 1:
            print(f"      epoch {ep:3d}  val MAE_log {v:.4f}  (best {best:.4f})")
        if bad >= args.patience:
            print(f"      early stop at epoch {ep} (no val gain for {args.patience})")
            break

    model.load_state_dict(best_state)
    p = predict(model, test_loader, device)
    return metrics(te[2].numpy(), p), best, p


def floor_on_same_rows(tr_df, te_df, target_col, amp_col):
    """The baselines, refit on exactly the rows the network was given.

    The third is the one that matters for attribution. Site response is a
    per-station additive term in log space, and a network can identify a station
    from its noise floor and spectral character even inside a peak-normalised
    window -- 149 of 154 stations appear in more than one split, with ~173 train
    windows each. A plain linear model structurally CANNOT express that term, so
    part of any CNN margin over it is per-station calibration rather than
    waveform shape. Adding station as a categorical gives the floor the same
    ability and makes the remaining margin attributable.

    Args:
        tr_df: Training-split manifest rows, with `target_col`, `amp_col`,
            'log_dist', and 'station_key' columns.
        te_df: Test-split manifest rows, same columns.
        target_col: Column holding the regression target (log space).
        amp_col: Column holding the log peak amplitude scalar.

    Returns:
        Tuple of (metrics_by_name, predictions_by_name): dicts keyed by
        floor name ("log peak amplitude", "amplitude + log distance",
        "amplitude + distance + station"), so the delta can be stratified
        per row afterwards.
    """
    out, preds = {}, {}
    y_te = te_df[target_col].to_numpy()

    for name, feats in (("log peak amplitude", [amp_col]),
                        ("amplitude + log distance", [amp_col, "log_dist"])):
        m = LinearRegression().fit(tr_df[feats], tr_df[target_col])
        p = m.predict(te_df[feats])
        out[name], preds[name] = metrics(y_te, p), p

    cats = sorted(set(tr_df.station_key))
    idx = {s: i for i, s in enumerate(cats)}

    def design(df):
        """Builds the [amplitude, log_dist, one-hot station] design matrix.

        Args:
            df: Manifest rows with `amp_col`, 'log_dist', and 'station_key'
                columns.

        Returns:
            float64 array, shape (len(df), 2 + len(cats)). A station not in
            the training set's `cats` gets an all-zero one-hot (zero effect).
        """
        X = np.zeros((len(df), 2 + len(cats)))
        X[:, 0] = df[amp_col].to_numpy()
        X[:, 1] = df["log_dist"].to_numpy()
        for r, s in enumerate(df.station_key):
            if s in idx:                    # unseen station -> zero effect
                X[r, 2 + idx[s]] = 1.0
        return X

    m = LinearRegression().fit(design(tr_df), tr_df[target_col])
    p = m.predict(design(te_df))
    name = "amplitude + distance + station"
    out[name], preds[name] = metrics(y_te, p), p
    return out, preds


# ---------------------------------------------------------------------------

def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="Peak ground motion CNN vs the scalar floor.")
    p.add_argument("--dataset-dir", default="../seismic_cli/data/dataset_groundmotion_3s")
    p.add_argument("--target", default="pgv_fwd", choices=list(TARGETS))
    # --arch is kept as an alias of --model-branch: `run_groundmotion_experiments.sh`
    # passes `--arch cnn_lstm`, and groundmotion_summary.py reads an `arch`
    # column out of the result CSVs, so both the flag and the underscored value
    # have to survive. main() writes the underscored spelling back onto
    # args.arch for exactly that reason.
    add_model_args(p, models=["groundmotion"])
    p.add_argument("--input-norm", default="peak", choices=["peak", "none"],
                   help="peak: divide by the window's own peak and pass log(peak) as aux.")
    p.add_argument("--no-aux", action="store_true",
                   help="Ablation: waveform only, no scalars at all.")
    p.add_argument("--no-distance", action="store_true",
                   help="Drop log_dist from aux, keeping only the amplitude scalar.")
    p.add_argument("--split-by", default="event", choices=["event", "station", "both"],
                   help="event: generator default. station: site response cannot leak, but "
                        "events become shared. both: neither leaks, at the cost of rows.")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed-split", type=int, default=42,
                   help="Seed for the station partition (--split-by station/both).")
    p.add_argument("--out-csv", default="groundmotion_cnn_results.csv")
    return p.parse_args()


def main():
    """Loads the dataset/baselines, trains the seed ensemble, and reports
    both metric spaces against the strongest floor."""
    args = parse_args()
    # Resolved once: several report lines and the result CSV's `arch` column
    # read args.arch, and the underscored spelling is what existing summaries
    # already contain.
    args.model_spec = spec_from_args(args)
    args.arch = args.model_spec.branch.replace("-", "_")
    lin_col, log_col, unit, amp_col, degenerate = TARGETS[args.target]

    manifest = Path(args.dataset_dir) / "manifest.csv"
    d = load(manifest)                       # identical rows to the baselines
    d = d.dropna(subset=[log_col, amp_col, "log_dist"]).reset_index(drop=True)
    d = respilt(d, args.split_by, seed=args.seed_split).reset_index(drop=True)
    report_split(d, args.split_by)

    print(f"\n=== target {args.target} ({unit})  arch {args.arch}  "
          f"input-norm {args.input_norm} ===")
    if degenerate:
        print("  !! DEGENERATE TARGET: its window contains the input window.")
        print("     Amplitude bounds it below; do not quote against the _fwd numbers.")

    aux_cols = [] if args.no_aux else (
        ["__log_peak__"] if args.no_distance else ["__log_peak__", "log_dist"])
    print(f"  aux: {aux_cols or 'NONE (waveform only)'}")

    print("  loading tensors...")
    X, log_peak = preload(d, args.dataset_dir, args.input_norm)

    parts, stats = {}, None
    for s in ("train", "val", "test"):
        m = (d.split == s).to_numpy()
        t = build_tensors(d[m].reset_index(drop=True), X[m], log_peak[m], log_col,
                          aux_cols, not args.no_aux, aux_stats=stats)
        parts[s], stats = t[:3], t[3]
    n_aux = parts["train"][1].shape[1]
    print(f"  train {len(parts['train'][2])}  val {len(parts['val'][2])}  "
          f"test {len(parts['test'][2])}  n_aux {n_aux}")

    tr_df, te_df = d[d.split == "train"], d[d.split == "test"]
    floor, floor_preds = floor_on_same_rows(tr_df, te_df, log_col, amp_col)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nparam = sum(p.numel() for p in args.model_spec.build(aux_dim=n_aux).parameters())
    print(f"  device {device} | parameters {nparam:,}")

    rows, runs, preds = [], [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"\n  --- seed {seed} ---")
        m, vbest, p = train_one(args, (parts["train"], parts["val"], parts["test"]),
                                n_aux, device, seed)
        print(f"      test MAE_log {m['MAE_log']:.4f}  R2_log {m['R2_log']:.4f}")
        runs.append(m)
        preds.append(p)
        rows.append({"target": args.target, "arch": args.arch, "seed": seed,
                     "split_by": args.split_by,
                     "input_norm": args.input_norm, "n_aux": n_aux,
                     "val_MAE_log": vbest, **m})

    # Seed-averaged predictions, so the stratification describes the reported
    # mean model rather than one arbitrary seed.
    cnn_pred = np.mean(preds, axis=0) if preds else None

    print(f"\n{'='*88}")
    print(f"RESULT  target {args.target}   arch {args.arch}   "
          f"{len(runs)} seeds   test n={len(parts['test'][2])}")
    print(f"{'='*88}")
    print(f"{'model':32s} {'MAE_log':>16s} {'R2_log':>16s} {'R2_lin':>12s}")
    print("-" * 88)
    for name, f in floor.items():
        print(f"{name:32s} {f['MAE_log']:16.4f} {f['R2_log']:16.4f} {f['R2_lin']:12.4f}")
    mae = np.array([r["MAE_log"] for r in runs])
    r2 = np.array([r["R2_log"] for r in runs])
    r2l = np.array([r["R2_lin"] for r in runs])
    label = f"CNN ({args.arch})"
    print(f"{label:32s} {mae.mean():8.4f} +-{mae.std():5.4f} "
          f"{r2.mean():8.4f} +-{r2.std():5.4f} {r2l.mean():12.4f}")
    print("-" * 88)

    # The floor to beat is the STRONGEST one, which is the station-augmented
    # model -- comparing against the weakest available reference is the habit
    # this project exists to avoid.
    best_name = min(floor, key=lambda k: floor[k]["MAE_log"])
    best_floor = floor[best_name]
    if cnn_pred is not None:
        print_report(f"CNN ({args.arch}) -- full metric set, seed-averaged (test set, log target)",
                    regression_report(parts["test"][2].numpy(), cnn_pred))

    delta = best_floor["MAE_log"] - mae.mean()
    print(f"  vs strongest floor ({best_name}):")
    print(f"    MAE_log  CNN {mae.mean():.4f} vs {best_floor['MAE_log']:.4f}"
          f"   delta {delta:+.4f}  {'CNN better' if delta > 0 else 'CNN worse'}")
    print(f"    R2_lin   CNN {r2l.mean():.4f} vs {best_floor['R2_lin']:.4f}"
          f"   delta {r2l.mean() - best_floor['R2_lin']:+.4f}"
          f"  {'CNN better' if r2l.mean() > best_floor['R2_lin'] else 'CNN worse'}")

    # Both spaces are reported because they can disagree, and when they do the
    # disagreement IS the finding -- this is the same log-vs-linear inversion
    # that produced the paper's unexplained ANN R2 of -10.08.
    if (delta > 0) != (r2l.mean() > best_floor["R2_lin"]):
        print("\n  !! THE TWO METRIC SPACES DISAGREE. The winner in log space is the")
        print("     loser in linear space, on identical rows and predictions. Do NOT")
        print("     report 'the CNN beats the floor' unqualified -- state both. This is")
        print("     the same inversion diagnosed in the paper's -10.08 ANN result, and")
        print("     linear R2 is the space the paper reports its headline in.")

    if abs(delta) < 2 * mae.std() and len(runs) > 1:
        print(f"\n  NOTE: |delta| is within 2 seed-sigma ({2 * mae.std():.4f}); "
              f"report.md 6.6 says a margin this size is not established.")
    if args.input_norm == "peak" and not args.no_aux:
        print("  The linear floor is a strict special case of this model (aux path"
              "\n  alone), so a shortfall is optimisation, not missing information.")

    _stratify_delta(te_df, te_df[log_col].to_numpy(), cnn_pred, floor_preds[best_name])

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\n[write] {args.out_csv}")


def _stratify_delta(te_df, y, cnn_pred, floor_pred):
    """Prints where the CNN's advantage actually comes from.

    `peak_in_input` matters most. The window spans [arr-0.6, arr+2.4], so at
    short distance it CONTAINS the S arrival -- the network can read directly
    whether the forward window will catch S or only coda. That is real
    information, but it is information about the S-P moveout confound rather
    than about ground motion, so a gain concentrated in one stratum means
    something different from a gain spread evenly.

    Args:
        te_df: Test-split manifest rows, with 'peak_in_input' and (if
            present) 'magnitude' columns.
        y: True target values (log space), aligned with `te_df`.
        cnn_pred: Seed-averaged CNN predictions (log space), or None to
            skip (nothing was trained).
        floor_pred: Strongest floor's predictions (log space), aligned with
            `te_df`.
    """
    if cnn_pred is None:
        return
    e_cnn = np.abs(cnn_pred - y)
    e_flr = np.abs(floor_pred - y)
    print("\n  Where the CNN's advantage sits (MAE_log, CNN vs strongest floor):")
    print(f"     {'stratum':28s} {'n':>6s} {'CNN':>8s} {'floor':>8s} {'delta':>8s}")
    strata = [("peak inside input window", te_df.peak_in_input.astype(bool).to_numpy()),
              ("peak after input window", ~te_df.peak_in_input.astype(bool).to_numpy())]
    if "magnitude" in te_df:
        strata += [("M < 3.0", (te_df.magnitude < 3.0).to_numpy()),
                   ("M >= 3.0", (te_df.magnitude >= 3.0).to_numpy())]
    for name, m in strata:
        if m.sum() > 30:
            print(f"     {name:28s} {int(m.sum()):6d} {e_cnn[m].mean():8.4f} "
                  f"{e_flr[m].mean():8.4f} {e_flr[m].mean() - e_cnn[m].mean():+8.4f}")


if __name__ == "__main__":
    main()
