"""Do chaotic features forecast anything the previous event's timing does not?

That is the question, and it is narrower than "can this predict earthquakes".
The persistence rule -- an event happened recently, so another is more likely --
already scores ~0.54 on the chosen cell. A model is only interesting if it beats
that, and only *useful* if it beats it using something other than the same
recency information.

**Model capacity is chosen from the effective sample size, not the row count.**
There are ~4,300 hourly rows, but the label is "any M>=2.5 within 400 km in the
next 6 h" and the cell contains **232 qualifying events**. Neighbouring hours
share the same event, so the independent episodes number in the low hundreds.
Gradient-boosted trees at that scale are a defensible choice; a deep sequence
model is not, and the ruled-out GRU experiment is what that looks like when it
is tried anyway. A linear model is fitted alongside as the honest low-capacity
control -- if it matches the trees, the trees are not finding structure.

**The ablation is the actual experiment.** Every model is fitted twice: with
`log1p_dsp` and without. Chaos features that only help when recency is hidden
have not added information, they have recovered a fraction of what was withheld.

**Splits are walk-forward with an embargo.** Hourly rows are autocorrelated and
one event marks six consecutive positive hours, so a random split trains on the
answer. The embargo drops the rows adjacent to each boundary.

Usage:
    python3 src/forecasting/chaos_forecast.py \\
        --parquet ~/Projects/sismokaos-cli/.../bodt_q1_chaos_5hz_features.parquet \\
        --catalog ~/Projects/Sismokaos/data_downloader/catalogs/catalog_current.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

from forecasting.chaos_dataset import (HORIZON_HOURS, MIN_MAGNITUDE, RADIUS_KM,
                                       build, persistence_scores)
from seismolib.metrics import safe_auc
from seismolib.splits import walk_forward_splits


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--parquet", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--horizon-hours", type=float, default=HORIZON_HOURS)
    p.add_argument("--shape", action="store_true",
                   help="Add within-hour slope/half-difference/argmax/lag-1 autocorrelation. Tests whether the trajectory inside an hour carries anything the four summary statistics discard.")
    p.add_argument("--lags", action="store_true",
                   help="Add lagged levels and deltas (1/3/6/12/24 h) for the top-scoring columns. Stands in for the recurrent half of a CNN+LSTM: whether the trajectory ACROSS hours carries anything the current hour does not.")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--embargo-hours", type=int, default=24,
                   help="Rows dropped after each split boundary. Must exceed the "
                        "horizon, or the last training hours share an event with "
                        "the first test hours.")
    p.add_argument("--num-boost-round", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-features", type=int, default=15)
    return p.parse_args()


def fit_lgbm(xtr, ytr, xva, yva, xte, args):
    """Small, heavily regularised GBM -- capacity set by ~232 effective events."""
    import lightgbm as lgb
    params = dict(objective="binary", metric="auc", learning_rate=0.03,
                  num_leaves=7, min_data_in_leaf=80, feature_fraction=0.5,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                  verbosity=-1, seed=args.seed)
    ds_tr = lgb.Dataset(xtr, label=ytr)
    ds_va = lgb.Dataset(xva, label=yva, reference=ds_tr)
    booster = lgb.train(params, ds_tr, num_boost_round=args.num_boost_round,
                        valid_sets=[ds_va],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
    return booster.predict(xte, num_iteration=booster.best_iteration), booster


def fit_logreg(xtr, ytr, xva, yva, xte, args):
    """Low-capacity control. Medians impute, then standardise -- both fitted on
    train only, because fitting either on the full column leaks test rows."""
    from sklearn.linear_model import LogisticRegression
    med = np.nanmedian(xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    def prep(x):
        x = np.where(np.isfinite(x), x, med)
        return (x - mu) / sd

    mu = np.where(np.isfinite(xtr), xtr, med).mean(axis=0)
    sd = np.where(np.isfinite(xtr), xtr, med).std(axis=0)
    sd[sd == 0] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(penalty="l2", C=0.05, max_iter=2000,
                               random_state=args.seed)
        m.fit(prep(xtr), ytr)
    return m.predict_proba(prep(xte))[:, 1], m


def main():
    args = parse_args()
    feats, y, dsp, idx = build(os.path.expanduser(args.parquet),
                               os.path.expanduser(args.catalog),
                               horizon_hours=args.horizon_hours, shape=args.shape, lags=args.lags)
    pers = persistence_scores(dsp)
    log1p_dsp = np.log1p(np.nan_to_num(dsp, nan=np.nanmax(dsp)))

    print(f"cell        M>={MIN_MAGNITUDE}, {RADIUS_KM:g} km, {args.horizon_hours:g} h")
    print(f"hours       {len(y):,}  positives {int(y.sum()):,} ({y.mean():.1%})")
    print(f"features    {feats.shape[1]:,} chaos columns (+1 when dsp is included)")
    print(f"folds       {args.folds} walk-forward, {args.embargo_hours} h embargo\n")

    base = feats.to_numpy(dtype=float)
    variants = {"chaos only": base,
                "chaos + dsp": np.column_stack([base, log1p_dsp])}
    names = list(feats.columns) + ["log1p_dsp"]

    folds = walk_forward_splits(np.arange(len(y)), args.folds, labels=y,
                                embargo=args.embargo_hours)
    rows, importances = [], []
    for k, (tr, va, te) in enumerate(folds, 1):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[va])) < 2:
            print(f"[fold {k}] skipped -- a split is single-class")
            continue
        floor = safe_auc(y[te], pers[te], oriented=True)
        for vname, x in variants.items():
            for mname, fit in (("logreg", fit_logreg), ("lgbm", fit_lgbm)):
                p, model = fit(x[tr], y[tr], x[va], y[va], x[te], args)
                rows.append(dict(fold=k, features=vname, model=mname,
                                 auc=safe_auc(y[te], p, oriented=False), floor=floor,
                                 n_test=len(te), pos=float(y[te].mean())))
                if mname == "lgbm" and vname == "chaos + dsp":
                    importances.append(pd.Series(model.feature_importance("gain"),
                                                 index=names))
        rows.append(dict(fold=k, features="-", model="persistence floor",
                         auc=floor, floor=floor, n_test=len(te), pos=float(y[te].mean())))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no scoreable folds -- widen the cell or add days")

    print(f"{'fold':>5}{'n':>7}{'pos':>7}{'floor':>9}  {'features':14}{'model':18}"
          f"{'AUC':>9}{'captured':>10}")
    for _, r in df.sort_values(["fold", "features", "model"]).iterrows():
        cap = (r.auc - r.floor) / (1 - r.floor) if r.floor < 1 else np.nan
        print(f"{r.fold:>5}{r.n_test:>7}{r.pos:>7.2f}{r.floor:>9.4f}  "
              f"{r.features:14}{r.model:18}{r.auc:>9.4f}{cap:>9.1%}")

    print("\n=== mean across folds ===")
    agg = df[df.model != "persistence floor"].groupby(["features", "model"]).agg(
        auc=("auc", "mean"), floor=("floor", "mean")).reset_index()
    agg["captured"] = (agg.auc - agg.floor) / (1 - agg.floor)
    fl = df[df.model == "persistence floor"].auc.mean()
    print(f"{'persistence floor':34}{fl:>9.4f}")
    for _, r in agg.sort_values("auc", ascending=False).iterrows():
        print(f"{r.features + ' / ' + r.model:34}{r.auc:>9.4f}{r.captured:>10.1%}")

    if importances:
        imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"\n=== mean LightGBM gain, chaos + dsp (top {args.top_features}) ===")
        total = imp.sum()
        for name, v in imp.head(args.top_features).items():
            print(f"  {name:38}{v:>12.1f}{v / total:>8.1%}")
        if "log1p_dsp" in imp.index:
            print(f"\n  log1p_dsp alone accounts for {imp['log1p_dsp'] / total:.1%} "
                  f"of total gain, rank {list(imp.index).index('log1p_dsp') + 1} "
                  f"of {len(imp)}")


if __name__ == "__main__":
    main()
