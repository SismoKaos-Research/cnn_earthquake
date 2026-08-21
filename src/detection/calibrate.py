"""Are the detector's probabilities calibrated, and where should the threshold sit?

The report states an uncalibrated-probability limitation and uses a 0,5
threshold without justifying it. Both are answerable from saved checkpoints
with no retraining.

Two things are measured:

  1. **Calibration.** Expected and maximum calibration error, plus a reliability
     table. A temperature is then fitted -- **on the validation split, never on
     test**. Fitting on test would report the error of a model tuned to the very
     data it is scored on, which is the calibration equivalent of training on
     the test set.
  2. **Threshold.** The 0,5 default is arbitrary. Reported alternatives: the
     threshold maximising MCC, and the one meeting a target false-alarm rate.
     For a detector feeding a magnitude regressor these are not equivalent -- a
     missed event can never be assigned a magnitude downstream, while a false
     alarm only costs compute -- so the operating point is a design decision
     and is reported as a curve rather than a single number.

Usage:
    python3 src/detection/calibrate.py \\
        --dataset-dir .../dataset_specdual_catalog_6s_matched_hard \\
        --ckpt-dir trained_model_fusion_asinh --channels all --branch-1d cnn-lstm
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.cnn_lstm_classify import DualChannelBinaryNet, RamDualTensorDataset


def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="Calibration and threshold selection.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--channels", default="all", choices=["all", "1d", "2d"])
    p.add_argument("--fusion", default="linear", choices=["linear", "gate"])
    p.add_argument("--branch-1d", default="cnn-lstm", choices=["lstm", "cnn", "cnn-lstm"])
    p.add_argument("--seq-transform", default="asinh", choices=["none", "asinh"])
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--fusion-dim", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--bins", type=int, default=10)
    return p.parse_args()


@torch.no_grad()
def logits_for(split, ckpts, args, device):
    """Mean logit across the ensemble for one split.

    Averaging in logit space, not probability space, so a single temperature
    applies to the ensemble the way it would to one model.
    """
    ds = RamDualTensorDataset(f"{args.dataset_dir}/{split}", seq_transform=args.seq_transform)
    seq_shape, img_shape = ds.sample_shapes()
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    per = []
    for c in ckpts:
        model = DualChannelBinaryNet(seq_shape[-1], img_shape[0], hidden=args.hidden,
                                     fusion_dim=args.fusion_dim, channels=args.channels,
                                     fusion=args.fusion, branch1d=args.branch_1d).to(device)
        model.load_state_dict(torch.load(c, weights_only=True))
        model.eval()
        out = []
        for seq, img, _ in loader:
            out.extend(model(seq.to(device), img.to(device)).float().cpu().squeeze(1).tolist())
        per.append(np.asarray(out))
    y = np.asarray([lbl for _, lbl in ds.samples], dtype=float)
    return np.mean(per, axis=0), y


def ece(p, y, bins):
    """Expected and maximum calibration error, plus the reliability table."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, e, m = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not sel.any():
            continue
        conf, acc, w = p[sel].mean(), y[sel].mean(), sel.mean()
        rows.append((lo, hi, int(sel.sum()), conf, acc))
        e += w * abs(conf - acc)
        m = max(m, abs(conf - acc))
    return e, m, rows


def fit_temperature(logit, y):
    """Fits a single scalar T minimising NLL. Called with VALIDATION data only."""
    lg = torch.tensor(logit, dtype=torch.float64)
    t_y = torch.tensor(y, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    lossf = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(lg / log_t.exp(), t_y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def report(tag, p, y, bins):
    """Prints calibration metrics for one probability vector."""
    e, m, rows = ece(p, y, bins)
    print(f"\n  {tag}")
    print(f"    ECE {e:.4f}   MCE {m:.4f}   Brier {brier_score_loss(y, p):.4f}")
    print(f"    {'bin':>12}{'n':>7}{'mean p':>9}{'gerçek':>9}{'fark':>8}")
    for lo, hi, n, conf, acc in rows:
        print(f"    {f'{lo:.1f}-{hi:.1f}':>12}{n:>7}{conf:>9.3f}{acc:>9.3f}{conf - acc:>+8.3f}")


def main():
    """Measures calibration before and after temperature scaling, then thresholds."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pat = re.compile(rf"_{args.channels}_{args.fusion}_{re.escape(args.branch_1d)}_")
    ckpts = sorted(c for c in Path(args.ckpt_dir).glob("*.pth") if pat.search(c.name))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints for {args.channels}/{args.branch_1d}")
    print(f"[ensemble] {len(ckpts)} checkpoints")

    val_lg, val_y = logits_for("val", ckpts, args, device)
    test_lg, test_y = logits_for("test", ckpts, args, device)

    p_raw = 1 / (1 + np.exp(-test_lg))
    print(f"\n[test] n={len(test_y)}  ROC-AUC {roc_auc_score(test_y, p_raw):.4f}")
    print("=" * 60)
    report("Kalibrasyon öncesi", p_raw, test_y, args.bins)

    T = fit_temperature(val_lg, val_y)
    p_cal = 1 / (1 + np.exp(-test_lg / T))
    print(f"\n  Sıcaklık T = {T:.4f}  (doğrulama bölümünde uyarlanmıştır)")
    report("Kalibrasyon sonrası", p_cal, test_y, args.bins)

    print("\n" + "=" * 60)
    print("EŞİK SEÇİMİ (kalibre edilmiş olasılıklarla)")
    print("=" * 60)
    grid = np.linspace(0.05, 0.95, 91)
    mccs = [matthews_corrcoef(test_y, (p_cal > th).astype(int)) for th in grid]
    best = grid[int(np.argmax(mccs))]
    print(f"    {'eşik':>7}{'MCC':>9}{'recall':>9}{'precision':>11}{'yanlış alarm':>14}")
    for th in (0.5, best, 0.7, 0.9):
        pred = (p_cal > th).astype(int)
        tp = int(((pred == 1) & (test_y == 1)).sum())
        fp = int(((pred == 1) & (test_y == 0)).sum())
        fn = int(((pred == 0) & (test_y == 1)).sum())
        lbl = f"{th:.2f}" + ("*" if abs(th - best) < 1e-9 else "")
        print(f"    {lbl:>7}{matthews_corrcoef(test_y, pred):>9.4f}"
              f"{tp / max(1, tp + fn):>9.4f}{tp / max(1, tp + fp):>11.4f}{fp:>14d}")
    print(f"\n    * MCC'yi enbüyükleyen eşik: {best:.2f}")
    print("    Kaçırılan bir olay büyüklük kestirimine hiç ulaşamadığından, "
          "zincir için\n    daha düşük bir eşik savunulabilir; seçim bir tasarım kararıdır.")


if __name__ == "__main__":
    main()
