"""Does adding catalogue-derived columns to chaos features create a leak?

Two ELZG reports (LSTM and GRU, 2026-07-23) report AUC ~0.90 and precision
~99% forecasting seismic risk from chaos and entropy features. Our own work
found the same feature family BELOW a persistence floor -- LSTM 0.5244, GRU
0.5709, TCN 0.5204 against 0.5823. Both cannot be right about the same claim.

The reports' feature list carries four columns the chaos features do not:
Latitude, Longitude, Mesafe_Derecesi and Deprem_Sayisi. Those exist only on
rows where an event occurred (7,731 of 53,565), and the gaps were filled
"ileriye ve geriye dönük" -- forward AND backward. Backward fill copies an
event's own coordinates into the rows preceding it, which is the label.

This runs their method on our data, changing one thing at a time:

    A  chaos features only                        (our published setup)
    B  + event columns, ffill then bfill          (their setup)
    C  + event columns, ffill only                (causal; isolates the bfill)

If B is far above A and C, the performance comes from the fill, not the
features -- and both teams' results are reconciled rather than contradictory.

    python3 scripts/replicate_event_feature_leak.py --epochs 40
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_score, recall_score

from forecasting import chaos_dataset as cd
from seismolib.catalog import STATION_COORDS, haversine_km
from seismolib.metrics import safe_auc

EVENT_COLS = ["Latitude", "Longitude", "Mesafe_Derecesi", "Deprem_Sayisi"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", default="/home/hogib/Projects/Sismokaos/sismokaos-cli/"
                   "dataset_features_chaos_q1_5hz/bodt_q1_chaos_5hz_features.parquet")
    p.add_argument("--catalog", default="/home/hogib/Projects/Sismokaos/"
                   "data_downloader/catalogs/catalog_current.csv")
    p.add_argument("--station", default="BODT")
    p.add_argument("--seq-len", type=int, default=24,
                   help="hours of context; 24 = the 1-day window they call optimal")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--band-spec", default=None,
                   help="custom bands as 'radius:mag,radius:mag,...' e.g. "
                        "'100:2.0,300:3.0,500:5.0,1000:6.0'. Implies --bands.")
    p.add_argument("--bands", action="store_true",
                   help="use the distance-graded magnitude label instead of the "
                        "flat M>=2.5 / 400 km one. Moves the positive class AND "
                        "the persistence floor, so compare both numbers.")
    p.add_argument("--label", choices=["forecast", "concurrent"], default="forecast",
                   help="forecast: an event in the next 6 h, our published task. "
                        "concurrent: an event in THIS hour -- 'risk durumu', which "
                        "is what the reports' wording describes and is the "
                        "condition under which the fill can leak")
    return p.parse_args()


def event_columns(idx, catalog_path, station, radius_km=cd.RADIUS_KM,
                  min_magnitude=cd.MIN_MAGNITUDE):
    """The four catalogue columns, populated ONLY on hours containing an event.

    This is the shape the reports describe: location and count are recorded
    "yalnızca sismik hareket anlarında" and absent everywhere else.
    """
    lat0, lon0 = STATION_COORDS[station]
    cat = pd.read_csv(catalog_path, encoding="utf-8-sig")
    cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    cat = cat.dropna(subset=["t", "Magnitude", "Latitude", "Longitude"])
    d = haversine_km(lat0, lon0, cat.Latitude.values, cat.Longitude.values)
    cat = cat[(cat.Magnitude >= min_magnitude) & (d <= radius_km)].copy()
    cat["dist_km"] = haversine_km(lat0, lon0, cat.Latitude.values, cat.Longitude.values)
    cat["hour"] = cat.t.dt.floor("h")

    g = cat.groupby("hour")
    out = pd.DataFrame(index=idx, columns=EVENT_COLS, dtype=float)
    out.loc[out.index.intersection(g.size().index), "Deprem_Sayisi"] = \
        g.size().reindex(out.index).values[np.isin(out.index, g.size().index)]
    first = g.first()
    for src, dst in (("Latitude", "Latitude"), ("Longitude", "Longitude")):
        s = first[src].reindex(out.index)
        out[dst] = s.values
    # "Mesafe derecesi": epicentral distance in degrees, 1 deg ~ 111.19 km.
    out["Mesafe_Derecesi"] = (g["dist_km"].min().reindex(out.index) / 111.19).values
    out["Deprem_Sayisi"] = g.size().reindex(out.index).values
    return out


class LSTMNet(nn.Module):
    """Their architecture: two LSTM layers of 64, dropout 0.4, one linear head."""

    def __init__(self, n_feat):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, 64, num_layers=2, batch_first=True, dropout=0.4)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def sequences(X, y, L):
    """Windows of L hours ending at each labelled hour."""
    n = len(X) - L + 1
    idx = np.arange(L)[None, :] + np.arange(n)[:, None]
    return X[idx], y[L - 1:]


def run_arm(name, feats, labels, dsp, args, device):
    X = feats.to_numpy(dtype=float)
    y = labels.astype(np.float32)
    Xs, ys = sequences(X, y, args.seq_len)
    dsp_s = dsp[args.seq_len - 1:]

    # Chronological split -- the charitable reading of their 80/20. A random
    # split over sliding windows would put near-duplicate rows on both sides,
    # which would flatter every arm equally and prove nothing about the fill.
    cut = int(len(Xs) * 0.8)
    tr, te = slice(0, cut), slice(cut, len(Xs))

    flat = Xs[tr].reshape(-1, Xs.shape[-1])
    med = np.nanmedian(flat, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    mu = np.nanmean(np.where(np.isfinite(flat), flat, med), axis=0)
    sd = np.nanstd(np.where(np.isfinite(flat), flat, med), axis=0)
    sd = np.where(sd > 0, sd, 1.0)

    def prep(a):
        a = np.where(np.isfinite(a), a, med)
        return ((a - mu) / sd).astype(np.float32)

    Xtr, Xte = torch.tensor(prep(Xs[tr])), torch.tensor(prep(Xs[te]))
    ytr, yte = torch.tensor(ys[tr]), ys[te]

    aucs, aps, precs, recs = [], [], [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        net = LSTMNet(Xs.shape[-1]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=device))
        ntr = len(Xtr)
        vcut = int(ntr * 0.9)
        best, bad, best_state = np.inf, 0, None
        for ep in range(args.epochs):
            net.train()
            perm = torch.randperm(vcut)
            for i in range(0, vcut, 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                loss = lossf(net(Xtr[b].to(device)), ytr[b].to(device))
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = lossf(net(Xtr[vcut:].to(device)), ytr[vcut:].to(device)).item()
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xte.to(device))).cpu().numpy()
        aucs.append(safe_auc(yte, p))
        aps.append(average_precision_score(yte, p))
        precs.append(precision_score(yte, p > 0.5, zero_division=0))
        recs.append(recall_score(yte, p > 0.5, zero_division=0))

    floor = safe_auc(yte, -dsp_s[te], oriented=True)
    print(f"{name:<44} AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f}   "
          f"AP {np.mean(aps):.4f}   P {np.mean(precs):.3f}   R {np.mean(recs):.3f}"
          f"   floor {floor:.4f}")
    return np.mean(aucs), floor


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.band_spec:
        bands = tuple(tuple(float(x) for x in part.split(":"))
                      for part in args.band_spec.split(","))
    else:
        bands = cd.MAGNITUDE_BANDS if args.bands else None
    feats, labels, dsp, idx = cd.build(args.parquet, args.catalog,
                                       station=args.station, bands=bands)
    ev = event_columns(idx, args.catalog, args.station)

    if args.label == "concurrent":
        # "Is there seismic activity in this hour" -- the state the reports
        # appear to be classifying. Note this is exactly the set of hours the
        # event columns are populated on, before any filling, which is what
        # makes the direction of the fill decisive rather than cosmetic.
        labels = ev["Deprem_Sayisi"].notna().to_numpy().astype(int)

    print(f"{len(feats)} hourly rows, {labels.mean():.1%} positive, "
          f"{feats.shape[1]} chaos features, label={args.label}, "
          f"scheme={bands if bands else 'flat M>=2.5/400km'}, device={device}\n")
    filled = ev.notna().any(axis=1).sum()
    print(f"event columns populated on {filled} of {len(ev)} hours "
          f"({filled/len(ev):.1%}) -- theirs: 7,731 of 53,565 (14.4%)\n")

    arms = [
        ("A  chaos only", feats),
        ("B  + event cols, ffill+bfill (their method)",
         feats.join(ev.ffill().bfill())),
        ("C  + event cols, ffill only (causal)", feats.join(ev.ffill())),
    ]
    results = {}
    for name, f in arms:
        results[name] = run_arm(name, f, labels, dsp, args, device)
    print()
    a = results["A  chaos only"][0]
    b = results["B  + event cols, ffill+bfill (their method)"][0]
    c = results["C  + event cols, ffill only (causal)"][0]
    print(f"B - A = {b - a:+.4f}   (what the four columns add, their way)")
    print(f"C - A = {c - a:+.4f}   (what they add causally)")
    print(f"B - C = {b - c:+.4f}   (what the BACKWARD fill alone adds)")


if __name__ == "__main__":
    main()
