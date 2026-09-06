"""Does ANY (context, horizon) configuration of these features forecast anything?

The question this answers is not "is my model good" but "is there a
configuration in which these features carry a forecast at all". That is the
answerable version, and a clean negative over a grid is a result -- it closes
the question instead of leaving it open for the next person to re-litigate with
one more architecture.

**Context length is the parameter, because it is what every architecture
proposal has actually been about.** A CNN over the 50 s stream is a claim that
sub-hour shape matters. A 7-day hierarchy of 24 h embeddings is a claim that a
week of context matters. Both are claims about how much history the model should
see, and both can be tested without building either network: aggregate the
trailing window, hand it to a model whose capacity is not the constraint, and
see whether anything clears the floor. `experiment_chaos_forecast_2026-08-27.md`
established that capacity is not the constraint -- an L2 logistic regression
matched gradient-boosted trees to within 0.007.

**Each horizon has its own floor and its own positive rate**, so raw AUC is not
comparable across columns of the output. Headroom captured is.

**A cell that fails the viability filter is reported, not hidden.** A horizon
whose positive rate is 95% is not evidence about the features; it is an
unusable label, and saying so is part of the answer.

Usage:
    python3 src/forecasting/chaos_config_sweep.py \\
        --parquet ~/Projects/sismokaos-cli/.../bodt_q1_chaos_5hz_features.parquet \\
        --catalog ~/Projects/Sismokaos/data_downloader/catalogs/catalog_current.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from forecasting.chaos_dataset import (ID_COLUMNS, RADIUS_KM, MIN_MAGNITUDE,
                                       load_events, persistence_scores)
from seismolib.catalog import (count_events_in_window, days_since_prev_major,
                               label_hours)
from seismolib.metrics import safe_auc
from seismolib.splits import walk_forward_splits

# 1 h is "the current hour only", the configuration already tested. 168 h is the
# 7-day hierarchy proposal. The rest fill in between.
CONTEXT_HOURS = (1, 6, 24, 72, 168)
HORIZON_HOURS = (6, 24, 72)

# A cell outside this positive-rate band cannot be scored meaningfully whatever
# the features do -- the same filter label_sweep.py applies.
POS_RATE_RANGE = (0.05, 0.85)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--parquet", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--contexts", type=int, nargs="+", default=list(CONTEXT_HOURS))
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZON_HOURS))
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--embargo-hours", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def hourly_means(parquet_path):
    """One row per hour, one column per feature. The base for every context."""
    df = pd.read_parquet(parquet_path)
    df = df.assign(hour_start=pd.to_datetime(df["Zaman_Dk"], unit="m").dt.floor("h"))
    cols = [c for c in df.columns if c not in ID_COLUMNS]
    out = df.groupby("hour_start")[cols].mean().sort_index()
    # Reindex to a continuous hourly range so a rolling window cannot reach
    # across the archive's outage as though no time had passed.
    return out.reindex(pd.date_range(out.index.min(), out.index.max(), freq="h"))


def context_features(hourly, hours):
    """Trailing-window summary of the last `hours` hours, ending at each row.

    Returns:
        DataFrame of `{col}_ctxmean` / `_ctxstd` / `_ctxslope`, aligned to
        `hourly.index`.

    At `hours == 1` this is the current hour and the spread terms are dropped,
    which is the already-tested configuration. Windows are closed on the right,
    so a row summarises history up to and including itself and never the future.
    """
    if hours <= 1:
        return hourly.add_suffix("_ctxmean")
    r = hourly.rolling(f"{hours}h", min_periods=max(2, hours // 4))
    parts = {"_ctxmean": r.mean(), "_ctxstd": r.std()}
    # Slope over the window, as the covariance of value with time divided by the
    # variance of time -- a trailing trend, which is the thing a recurrent layer
    # is usually credited with noticing.
    t = pd.Series(np.arange(len(hourly), dtype=float), index=hourly.index)
    tr = t.rolling(f"{hours}h", min_periods=max(2, hours // 4))
    # ddof=0 to match the covariance below, which is E[xy] - E[x]E[y] and so is
    # a POPULATION quantity. Pandas defaults var() to ddof=1, and mixing the two
    # scales every slope by (n-1)/n -- a bias that depends on window length,
    # which would make slopes incomparable across exactly the context lengths
    # this sweep exists to compare.
    tm, tv = tr.mean(), tr.var(ddof=0)
    cov = (hourly.mul(t, axis=0).rolling(f"{hours}h",
           min_periods=max(2, hours // 4)).mean()).sub(parts["_ctxmean"].mul(tm, axis=0))
    parts["_ctxslope"] = cov.div(tv.replace(0, np.nan), axis=0)
    return pd.concat([p.add_suffix(s) for s, p in parts.items()], axis=1)


def score_cell(x, y, dsp, args):
    """Walk-forward AUC and floor for one (context, horizon) cell.

    Returns:
        Tuple of (mean model AUC, mean floor, n scoreable folds).
    """
    import lightgbm as lgb
    pers = persistence_scores(dsp)
    folds = walk_forward_splits(np.arange(len(y)), args.folds, labels=y,
                                embargo=args.embargo_hours)
    aucs, floors = [], []
    for tr, va, te in folds:
        if len(np.unique(y[te])) < 2 or len(np.unique(y[va])) < 2:
            continue
        params = dict(objective="binary", metric="auc", learning_rate=0.03,
                      num_leaves=7, min_data_in_leaf=80, feature_fraction=0.5,
                      bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                      verbosity=-1, seed=args.seed)
        ds_tr = lgb.Dataset(x[tr], label=y[tr])
        b = lgb.train(params, ds_tr, num_boost_round=300,
                      valid_sets=[lgb.Dataset(x[va], label=y[va], reference=ds_tr)],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
        aucs.append(safe_auc(y[te], b.predict(x[te], num_iteration=b.best_iteration)))
        floors.append(safe_auc(y[te], pers[te], oriented=True))
    if not aucs:
        return np.nan, np.nan, 0
    return float(np.mean(aucs)), float(np.mean(floors)), len(aucs)


def main():
    args = parse_args()
    hourly = hourly_means(os.path.expanduser(args.parquet))
    events = load_events(os.path.expanduser(args.catalog))
    idx = pd.DatetimeIndex(hourly.index)
    dsp = days_since_prev_major(idx, events)
    log1p_dsp = np.log1p(np.nan_to_num(dsp, nan=np.nanmax(dsp)))

    print(f"cell        M>={MIN_MAGNITUDE}, {RADIUS_KM:g} km")
    print(f"hours       {len(idx):,}  ({idx[0]} .. {idx[-1]})")
    print(f"grid        {len(args.contexts)} contexts x {len(args.horizons)} horizons, "
          f"{args.folds} walk-forward folds each\n")

    rows = []
    for ctx in args.contexts:
        feats = context_features(hourly, ctx)
        x = np.column_stack([feats.to_numpy(dtype=float), log1p_dsp])
        for hz in args.horizons:
            # label_hours, not a raw forward count: the features at each hour
            # cover [t, t+1h], so a forward count from t counts an event the
            # model can already see. Same fix as chaos_dataset.build.
            y = label_hours(idx, events, hz / 24.0)
            pos = float(y.mean())
            usable = POS_RATE_RANGE[0] <= pos <= POS_RATE_RANGE[1]
            if not usable:
                rows.append(dict(context_h=ctx, horizon_h=hz, n_features=x.shape[1],
                                 pos_rate=pos, auc=np.nan, floor=np.nan,
                                 captured=np.nan, folds=0, note="unusable positive rate"))
                continue
            auc, floor, n = score_cell(x, y, dsp, args)
            rows.append(dict(context_h=ctx, horizon_h=hz, n_features=x.shape[1],
                             pos_rate=pos, auc=auc, floor=floor,
                             captured=(auc - floor) / (1 - floor) if floor < 1 else np.nan,
                             folds=n, note=""))
            print(f"  ctx {ctx:>3} h / hz {hz:>3} h  ->  AUC {auc:.4f}  "
                  f"floor {floor:.4f}  captured {(auc - floor) / (1 - floor):+.1%}")

    df = pd.DataFrame(rows)
    print("\n=== headroom captured, by context and horizon ===")
    piv = df.pivot(index="context_h", columns="horizon_h", values="captured")
    print(piv.map(lambda v: "  n/a  " if pd.isna(v) else f"{v:+7.1%}").to_string())
    print("\n=== floor per horizon (identical down each column by construction) ===")
    print(df.pivot(index="context_h", columns="horizon_h", values="floor")
            .map(lambda v: "  n/a " if pd.isna(v) else f"{v:.4f}").to_string())

    ok = df.dropna(subset=["captured"])
    if len(ok):
        best = ok.loc[ok.captured.idxmax()]
        print(f"\nbest cell: context {int(best.context_h)} h, horizon "
              f"{int(best.horizon_h)} h -- captured {best.captured:+.1%} "
              f"(AUC {best.auc:.4f} vs floor {best.floor:.4f})")
        print(f"cells above +5% captured: {int((ok.captured > 0.05).sum())} of {len(ok)}")
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
