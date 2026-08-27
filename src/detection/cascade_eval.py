"""End-to-end cascade: detector decides, magnitude regressor sizes what it kept.

Stage 1 is `cnn_lstm_classify.py`'s spectrogram CNN (event vs noise); stage 2 is
`cnn_lstm_regression.py`'s `2d+aux` regressor (magnitude). Each stage keeps its
own tensors: the detector's `img` is a 33x38 STFT (n_fft 64) and the regressor's
is 129x10 (n_fft 256), so a window is looked up in both datasets **by filename**
rather than re-encoded here.

**Why this script exists rather than a joint metric bolted onto either trainer.**
Chaining the two creates three quantities neither stage measures on its own:

  1. What reaches stage 2 at all -- the detector's recall *is* the cascade's
     ceiling, since a missed event never gets a magnitude.
  2. The selection effect on stage 2 -- reported both ways, on accepted events
     and on every event in the evaluation set. The expectation is that gating
     flatters the regressor, since the detector keeps the loud windows; on this
     benchmark it measured at -0.0039 MAE, i.e. negligible and slightly the
     other way. Quote the all-events figure regardless: the misses are a real
     cost of the cascade even when they do not bias the error.
  3. What stage 2 would emit on a false positive.

**On (3), the honest answer is that it cannot be computed for this regressor,
and that is a finding rather than a gap in this script.** The `2d+aux` model
consumes `aux = (log_snr, log_distance)`, and `log_distance` is the epicentral
distance to a catalogued hypocentre. A false positive has no catalogue entry,
so one of the model's two auxiliary inputs is undefined for exactly the case a
deployed cascade must handle. A deployable stage 2 needs either `--channels 2d`
(no aux) or a distance estimated from the waveform.

**Leakage.** The evaluation set is not simply the detector's test split. The
regressor was trained with `--split-by detector`, which drops val/test rows
whose event also appears in its training set; those dropped events are in the
regressor's train split and must not be scored here. Only the regressor's own
test rows are used for the event class. Noise is unrestricted -- the regressor
never saw any noise window under any split.

Usage:
    python3 src/detection/cascade_eval.py \\
        --detector-dir  .../dataset_specdual_catalog_6s_matched_hard \\
        --magnitude-dir .../dataset_magreg_catalog_6s \\
        --detector-ckpt-dir  trained_model_detect_6s_matched \\
        --magnitude-ckpt     trained_model_magreg_grid/best_...detector_seed43....pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.cnn_lstm_classify import DualChannelBinaryNet, RamDualTensorDataset
from magnitude import cnn_lstm_regression as reg
from seismolib.checkpoints import find_checkpoints, run_identity

# Stage 1 is the spectrogram branch by design (see the module docstring), so the
# checkpoint filter and the model constructor must agree on it. Naming it once
# keeps them from drifting apart.
STAGE1_CHANNELS = "2d"


def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="End-to-end detector -> magnitude cascade.")
    p.add_argument("--detector-dir", required=True,
                   help="Detection dataset root (train/val/test + manifest.csv).")
    p.add_argument("--magnitude-dir", required=True,
                   help="Magnitude dataset root, built on the same source windows.")
    p.add_argument("--detector-ckpt-dir", required=True,
                   help="Directory of stage-1 checkpoints. One arm's worth is "
                        "ensembled; a directory holding more than one is an error, "
                        "not an average.")
    p.add_argument("--detector-fusion", default="linear",
                   help="Stage-1 fusion arm to select from --detector-ckpt-dir.")
    p.add_argument("--detector-branch-1d", default=None,
                   help="Stage-1 1D arm, to disambiguate a directory holding "
                        "several. Unset selects on channels and fusion alone, "
                        "which is what pre-`--branch-1d` checkpoints need.")
    p.add_argument("--magnitude-ckpt", required=True,
                   help="Stage-2 checkpoint, trained with --split-by detector.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Stage-1 probability above which a window is passed to stage 2.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--fusion-dim", type=int, default=96)
    p.add_argument("--reg-hidden", type=int, default=64)
    p.add_argument("--reg-fusion-dim", type=int, default=128)
    return p.parse_args()


@torch.no_grad()
def stage1_scores(ckpt_dir, ds, args, device):
    """Probability-averaged detector ensemble over one arm's checkpoints.

    Returns:
        Tuple of (probs, labels, filenames), each aligned to `ds.samples`.
    """
    seq_shape, img_shape = ds.sample_shapes()
    ckpts = find_checkpoints(ckpt_dir, STAGE1_CHANNELS, args.detector_fusion,
                             args.detector_branch_1d)
    print(f"[stage 1] ensembling {len(ckpts)} checkpoint(s) from "
          f"{run_identity(ckpts[0].name)}")

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    per_ckpt = []
    for c in ckpts:
        model = DualChannelBinaryNet(seq_shape[-1], img_shape[0], hidden=args.hidden,
                                     fusion_dim=args.fusion_dim,
                                     channels=STAGE1_CHANNELS).to(device)
        model.load_state_dict(torch.load(c, weights_only=True))
        model.eval()
        probs = []
        for seq, img, _ in loader:
            out = model(seq.to(device), img.to(device))
            probs.extend(torch.sigmoid(out).float().cpu().squeeze(1).tolist())
        per_ckpt.append(np.asarray(probs))
        print(f"           {Path(c).name.split('_seed')[-1]:>12}  done")

    labels = np.asarray([lbl for _, lbl in ds.samples])
    names = [Path(f).name for f, _ in ds.samples]
    return np.mean(per_ckpt, axis=0), labels, names


@torch.no_grad()
def stage2_predict(rows, root, aux_stats, ckpt, args, device):
    """Runs the magnitude regressor over `rows` of the magnitude manifest.

    Returns:
        Tuple of (y_true, y_pred) magnitude arrays.
    """
    ds = reg.DualMagnitudeDataset(rows, Path(root), aux_stats=aux_stats)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    seq0, img0, aux0, _ = ds[0]
    model = reg.DualChannelRegressionNet(seq0.shape[-1], img0.shape[0], aux0.numel(),
                                         hidden=args.reg_hidden,
                                         fusion_dim=args.reg_fusion_dim,
                                         channels="2d+aux").to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()

    true, pred = [], []
    for seq, img, aux, y in loader:
        out = model(seq.to(device), img.to(device), aux.to(device))
        pred.extend(out.float().cpu().tolist())
        true.extend(y.tolist())
    return np.asarray(true), np.asarray(pred)


def main():
    """Runs both stages over the shared evaluation set and reports the join."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- the evaluation set -------------------------------------------------
    mag = pd.read_csv(Path(args.magnitude_dir) / "manifest.csv")
    mag["log_distance"] = np.log(mag["distance_km"].clip(lower=1.0))
    reg.AUX_COLUMNS = reg.detect_aux_columns(mag)
    mag = reg.resplit(mag, "detector",
                      detector_manifest=str(Path(args.detector_dir) / "manifest.csv"))

    # Stage 2's aux standardization must reuse ITS OWN train split's statistics,
    # exactly as training did -- refitting on the evaluation rows would leak.
    train_stats = reg.DualMagnitudeDataset(
        mag[mag.split == "train"], Path(args.magnitude_dir)).aux_stats()
    mag_test = mag[mag.split == "test"].copy()
    scorable = set(mag_test.filename)
    print(f"[eval set] {len(mag_test)} event windows are scorable by stage 2 "
          f"(the rest of the detector's test events are in stage 2's training split)")

    det_ds = RamDualTensorDataset(f"{args.detector_dir}/test")
    probs, labels, names = stage1_scores(args.detector_ckpt_dir, det_ds, args, device)

    # Events restricted to stage 2's test rows; noise unrestricted.
    is_event = labels == 1
    keep = np.array([(not e) or (n in scorable) for e, n in zip(is_event, names)])
    probs, labels, names = probs[keep], labels[keep], list(np.asarray(names)[keep])
    n_ev, n_no = int((labels == 1).sum()), int((labels == 0).sum())
    print(f"[eval set] {n_ev} event + {n_no} noise windows\n")

    # -- stage 1 ------------------------------------------------------------
    accept = probs > args.threshold
    tp = int((accept & (labels == 1)).sum())
    fp = int((accept & (labels == 0)).sum())
    fn = int((~accept & (labels == 1)).sum())
    tn = int((~accept & (labels == 0)).sum())
    recall = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    print("=" * 66)
    print(f"STAGE 1  detector @ threshold {args.threshold}")
    print("=" * 66)
    print(f"  ROC-AUC {roc_auc_score(labels, probs):.4f}")
    print(f"  TP {tp}   FN {fn}   FP {fp}   TN {tn}")
    print(f"  recall {recall:.4f}   precision {prec:.4f}   "
          f"false-positive rate {fp / max(1, fp + tn):.4f}")
    print(f"  -> {fn} event(s) never reach stage 2 and can never receive a magnitude")

    # -- stage 2 ------------------------------------------------------------
    acc_names = {n for n, a, l in zip(names, accept, labels) if a and l == 1}
    rows_all = mag_test[mag_test.filename.isin(
        {n for n, l in zip(names, labels) if l == 1})]
    rows_acc = mag_test[mag_test.filename.isin(acc_names)]

    y_all, p_all = stage2_predict(rows_all, args.magnitude_dir, train_stats,
                                  args.magnitude_ckpt, args, device)
    y_acc, p_acc = stage2_predict(rows_acc, args.magnitude_dir, train_stats,
                                  args.magnitude_ckpt, args, device)
    mae_all = float(np.abs(y_all - p_all).mean())
    mae_acc = float(np.abs(y_acc - p_acc).mean())

    print("\n" + "=" * 66)
    print("STAGE 2  magnitude regressor")
    print("=" * 66)
    print(f"  on every event in the eval set   n={len(y_all):5d}   MAE {mae_all:.4f}")
    print(f"  on events the detector accepted  n={len(y_acc):5d}   MAE {mae_acc:.4f}")
    delta = mae_all - mae_acc
    print(f"  selection effect: {delta:+.4f} MAE")
    # The expectation going in was that accepted-only would flatter the
    # regressor, since the detector keeps the loud windows. Measured on this
    # benchmark it does not: the effect is near zero and has run slightly the
    # other way. Report whichever sign the data shows, and quote the
    # all-events figure regardless -- the misses are a real cost of the
    # cascade even when they do not bias the MAE.
    if abs(delta) < 0.01:
        print("  -> negligible: the detector's misses are not the events stage 2")
        print("     finds easy, so gating barely shifts the magnitude error here.")
    elif delta > 0:
        print("  -> accepted-only flatters stage 2; quote the all-events figure.")
    else:
        print("  -> accepted-only is the HARDER subset; gating did not cherry-pick")
        print("     the easy events on this benchmark.")

    # -- end to end ---------------------------------------------------------
    print("\n" + "=" * 66)
    print("END TO END")
    print("=" * 66)
    within = {t: float((np.abs(y_acc - p_acc) <= t).sum()) for t in (0.2, 0.3, 0.5)}
    for t, c in within.items():
        print(f"  events detected AND sized within {t} magnitude units: "
              f"{c / max(1, len(y_all)) * 100:5.1f}%  ({int(c)}/{len(y_all)})")
    print(f"  false positives admitted: {fp}")
    print("\n  Stage 2 is NOT applied to those false positives: it consumes")
    print("  aux=(log_snr, log_distance), and log_distance is the distance to a")
    print("  catalogued hypocentre, which a false positive does not have. A")
    print("  deployable stage 2 needs --channels 2d, or a waveform-derived distance.")


if __name__ == "__main__":
    main()
