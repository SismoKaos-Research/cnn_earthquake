"""
Combined "when and how big" prediction: the validated dual-channel network
for P(M >= threshold within horizon_days), paired with a simple ridge
regression for the magnitude of that event if one occurs -- catalog_forecast_
report.md section 5's recommendation, not a compromise. Three attempts at a
neural magnitude head (higher loss weight, more training patience, 2x the
training data) all lost to this same ridge floor; see that report for the
full comparison. This script is the two-part system actually recommended,
not a fourth attempt at the thing that didn't work.

Usage:
    python catalog_forecast_predict.py \\
        --dataset-dir ../seismic_cli/data/dataset_catalog_forecast \\
        --model-path trained_model_cnnlstm_forecast_maghead/best_cnnlstm_forecast.pth

Not imported by anything else -- standalone script.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from sismokaos.forecasting.cnn_lstm_forecast import (AUX_FEATURES, DenseWindowDataset,
                                           DualChannelForecastNet, safe_auc)


def parse_args():
    """Parses command-line arguments.

    Returns:
        argparse.Namespace with the script's CLI options.
    """
    p = argparse.ArgumentParser(description="Combined binary + magnitude forecast.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--model-path", default="trained_model_cnnlstm_forecast_maghead/best_cnnlstm_forecast.pth",
                  help="Checkpoint from cnn_lstm_forecast.py -- only its binary head is used; "
                       "the checkpoint's own (unused) magnitude head is ignored on purpose.")
    p.add_argument("--channels", default="all", choices=["all", "1d", "2d", "aux", "1d+aux", "2d+aux"],
                  help="Must match how --model-path was trained.")
    p.add_argument("--out-csv", default="catalog_forecast_predictions.csv")
    return p.parse_args()


def main():
    """Loads the checkpoint's binary head plus a fitted ridge magnitude
    floor, predicts both on the test split, writes a per-window CSV, and
    prints overall and per-zone AUC/MAE.

    Returns:
        None. Writes `args.out_csv` and prints a sample of predictions plus
        overall and per-zone metrics as a side effect.
    """
    args = parse_args()
    root = Path(args.dataset_dir)
    manifest = pd.read_csv(root / "manifest.csv")

    train_ds = DenseWindowDataset(manifest, root, "train")
    test_ds = DenseWindowDataset(manifest, root, "test", stats=train_ds.stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualChannelForecastNet(train_ds.seq_dim, train_ds.img_shape[0], train_ds.aux_dim,
                                   channels=args.channels).to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
    model.eval()

    loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    scores = []
    with torch.no_grad():
        for seq, img, aux, y, mag in loader:
            logit, _ = model(seq.to(device), img.to(device), aux.to(device))  # magnitude head unused
            scores.extend(torch.sigmoid(logit).cpu().tolist())
    scores = np.array(scores)

    floor_cols = [AUX_FEATURES.index(c) for c in ("max_mag", "mean_mag", "b_value", "log_rate")]
    train_pos = train_ds.labels.astype(bool)
    train_aux = train_ds.standardized_aux_matrix()[train_pos][:, floor_cols]
    train_mag = train_ds.next_magnitude[train_pos]
    ridge = Ridge(alpha=1.0).fit(train_aux, train_mag)

    test_aux_all = test_ds.standardized_aux_matrix()[:, floor_cols]
    mag_pred_all = ridge.predict(test_aux_all)

    out = test_ds.rows[["region", "end_time", "label", "next_magnitude"]].copy()
    out["p_event"] = scores
    out["predicted_magnitude_if_event"] = mag_pred_all
    out.to_csv(args.out_csv, index=False)

    print(f"[write] {args.out_csv} ({len(out)} rows)")
    print("\nSample (10 rows):")
    print(out.head(10).to_string(index=False))

    pos = out.label.astype(bool)
    mae = float(np.mean(np.abs(out.loc[pos, "next_magnitude"] - out.loc[pos, "predicted_magnitude_if_event"])))
    print(f"\nOn actual positive test windows (n={int(pos.sum())}): "
          f"magnitude MAE {mae:.3f} (ridge floor; see catalog_forecast_report.md section 5 "
          f"for why this, not a neural head, is the recommended magnitude predictor).")

    print("\n--- Per zone (test set) ---")
    print(f"{'zone':9s} {'n':>5s} {'pos rate':>9s} {'AUC (when)':>11s} {'n_pos':>6s} {'MAE (how big)':>14s}")
    for zone in sorted(out.region.unique()):
        zdf = out[out.region == zone]
        zpos = zdf.label.astype(bool)
        auc = safe_auc(zdf.label.to_numpy(), zdf.p_event.to_numpy())
        n_pos_zone = int(zpos.sum())
        if n_pos_zone >= 5:
            zmae = float(np.mean(np.abs(zdf.loc[zpos, "next_magnitude"] - zdf.loc[zpos, "predicted_magnitude_if_event"])))
            mae_str = f"{zmae:.3f}"
        else:
            mae_str = "n/a"
        print(f"{zone:9s} {len(zdf):5d} {zdf.label.mean():9.3f} {auc:11.4f} {n_pos_zone:6d} {mae_str:>14s}")


if __name__ == "__main__":
    main()
