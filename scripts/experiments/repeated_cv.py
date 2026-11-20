"""Repeated temporal CV, confidence intervals, paired tests vs k-sigma (task 3).

Setting: 5 contiguous temporal folds (12-step gap), data before 20 March,
10 events + 10 failures per test fold at magnitude 4, repeated over injection
seeds 0..S-1. For each detector and metric we report

  * mean +/- std over all (fold, seed) runs,
  * a 95% t-interval over the 5 fold means (runs inside a fold share data, so
    the fold is the unit of replication),
  * the paired difference to k-sigma: mean, 95% CI, paired t-test p-value
    and exact sign-flip p-value over fold means, and wins out of all runs.

    python scripts/experiments/repeated_cv.py --seeds 10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common as C
from stats import mean_ci, paired_comparison

METRICS = [("auc", "AUC"), ("auc_event", "AUC events"), ("auc_failure", "AUC failures"),
           ("precision_at_k", "precision@k")]
BASELINE = "ksigma"


def run(fb: C.FoldBench, seeds=range(10), reuse: pd.DataFrame | None = None) -> pd.DataFrame:
    """Evaluate every seed at the default setting; reuse rows already computed."""
    have = set() if reuse is None else set(reuse.seed.unique())
    parts = [] if reuse is None else [reuse]
    for s in seeds:
        if s in have:
            continue
        print(f"  repeated CV seed {s}", flush=True)
        parts.append(fb.evaluate(seed=s))
    df = pd.concat(parts, ignore_index=True)
    return df[df.seed.isin(list(seeds))]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for det, g in df.groupby("detector"):
        fm = g.groupby("fold")[[m for m, _ in METRICS]].mean()
        row = {"detector": det, "n_runs": len(g), "n_folds": len(fm)}
        for m, _ in METRICS:
            mu, lo, hi = mean_ci(fm[m])
            row.update({f"{m}_mean": g[m].mean(), f"{m}_std": g[m].std(ddof=1),
                        f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {d: i for i, d in enumerate(C.DETECTOR_ORDER)}
    return out.sort_values("detector", key=lambda s: s.map(order)).reset_index(drop=True)


def paired_vs_baseline(df: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    rows = []
    wide = df.pivot_table(index=["fold", "seed"], columns="detector",
                          values=[m for m, _ in METRICS])
    for det in sorted(set(df.detector) - {baseline},
                      key=lambda d: C.DETECTOR_ORDER.index(d) if d in C.DETECTOR_ORDER else 99):
        for m, _ in METRICS:
            a, b = wide[(m, det)], wide[(m, baseline)]
            fold_a = a.groupby(level="fold").mean()
            fold_b = b.groupby(level="fold").mean()
            res = paired_comparison(fold_a.values, fold_b.values)
            rows.append({"detector": det, "metric": m, **res,
                         "run_wins": int((a.values > b.values).sum()), "n_runs": len(a)})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, comp: pd.DataFrame):
    """Left: AUC per detector with 95% CI. Right: difference to k-sigma."""
    C.style()
    fig, axes = C.plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
    dets = [d for d in C.DETECTOR_ORDER if d in set(df.detector)]
    mets = METRICS[:3]
    offs = np.linspace(-0.22, 0.22, len(mets))
    shade = {"auc": 1.0, "auc_event": 0.65, "auc_failure": 0.35}
    ax = axes[0]
    for j, (m, lab) in enumerate(mets):
        for i, d in enumerate(dets):
            fm = df[df.detector == d].groupby("fold")[m].mean()
            mu, lo, hi = mean_ci(fm)
            ax.errorbar(i + offs[j], mu, yerr=[[mu - lo], [hi - mu]], fmt=C.MARKERS[d],
                        color=C.COLORS[d], alpha=shade[m], capsize=3, ms=7,
                        label=lab if i == 0 else None, mec="white", mew=0.8)
    ax.set_xticks(range(len(dets)), [C.LABELS[d].replace(" (baseline)", "\n(baseline)")
                                     for d in dets])
    ax.set_ylabel("ROC AUC")
    ax.set_title("Detection AUC, 95% CI over folds")
    ax.grid(axis="x", visible=False)
    leg = ax.legend(loc="lower right", fontsize=8, title="dot shade", title_fontsize=8)
    for h in leg.legend_handles:
        h.set_color(C.INK2)

    ax = axes[1]
    sub = comp[comp.metric.isin([m for m, _ in mets])]
    others = [d for d in dets if d != BASELINE]
    for j, (m, lab) in enumerate(mets):
        for i, d in enumerate(others):
            r = sub[(sub.detector == d) & (sub.metric == m)].iloc[0]
            ax.errorbar(r.mean_diff, i + offs[j], xerr=[[r.mean_diff - r.ci_lo], [r.ci_hi - r.mean_diff]],
                        fmt=C.MARKERS[d], color=C.COLORS[d], alpha=shade[m], capsize=3, ms=7,
                        mec="white", mew=0.8)
    ax.axvline(0, color=C.INK2, lw=1)
    ax.set_yticks(range(len(others)), [C.LABELS[d] for d in others])
    ax.invert_yaxis()
    ax.set_xlabel("AUC minus k-sigma AUC (paired by fold)")
    ax.set_title("Difference to the k-sigma baseline")
    ax.grid(axis="y", visible=False)
    fig.text(0.99, -0.06, "Per detector, left to right / top to bottom: all anomalies, "
             "events, failures", ha="right", fontsize=8, color=C.INK2)
    axes[0].set_ylim(top=1.0)
    fig.subplots_adjust(wspace=0.38)
    return C.savefig(fig, "repeated_cv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--detectors", nargs="+", default=None)
    args = ap.parse_args()
    fb = C.FoldBench(C.load(), C.implemented_detectors(args.detectors))
    df = run(fb, range(args.seeds))
    C.write_csv(df, "repeated_cv_runs")
    summ, comp = summarise(df), paired_vs_baseline(df)
    C.write_csv(summ, "repeated_cv_summary")
    C.write_csv(comp, "repeated_cv_vs_ksigma")
    print(summ.round(3).to_string(index=False))
    print(comp.round(4).to_string(index=False))
    print("figure:", plot(df, comp))


if __name__ == "__main__":
    main()
