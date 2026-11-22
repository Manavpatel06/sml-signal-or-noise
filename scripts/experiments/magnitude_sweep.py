"""Graded difficulty: detection AUC vs injected magnitude (Role 6, task 2).

For every implemented detector and magnitude in {1, 2, 3, 4, 6} (units of the
node's training standard deviation), inject 10 events + 10 failures into each
test fold, repeat over several injection seeds, and plot AUC overall, on
events only and on failures only against magnitude. Points are the mean over
folds and seeds; the band is the 95% CI over fold means (see stats.py).

    python scripts/experiments/magnitude_sweep.py --seeds 5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common as C
from stats import mean_ci

MAGNITUDES = (1.0, 2.0, 3.0, 4.0, 6.0)
PANELS = [("auc", "All anomalies"), ("auc_event", "Events"), ("auc_failure", "Failures")]


def run(fb: C.FoldBench, seeds=range(5), magnitudes=MAGNITUDES) -> pd.DataFrame:
    parts = []
    for m in magnitudes:
        for s in seeds:
            print(f"  magnitude {m:g}  seed {s}", flush=True)
            parts.append(fb.evaluate(seed=s, magnitude=m, tag={"magnitude": m}))
    return pd.concat(parts, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI (over fold means) per detector x magnitude x metric."""
    rows = []
    for (det, m), g in df.groupby(["detector", "magnitude"]):
        fm = g.groupby("fold")[[p for p, _ in PANELS]].mean()
        row = {"detector": det, "magnitude": m}
        for col, _ in PANELS:
            mu, lo, hi = mean_ci(fm[col])
            row.update({col: g[col].mean(), f"{col}_lo": lo, f"{col}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summ: pd.DataFrame):
    C.style()
    fig, axes = C.plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    dets = [d for d in C.DETECTOR_ORDER if d in set(summ.detector)]
    for ax, (col, title) in zip(axes, PANELS):
        ax.axhline(0.5, color=C.INK2, lw=0.8, ls=":")
        for d in dets:
            g = summ[summ.detector == d].sort_values("magnitude")
            ax.fill_between(g.magnitude, g[f"{col}_lo"], g[f"{col}_hi"],
                            color=C.COLORS[d], alpha=0.12, lw=0)
            ax.plot(g.magnitude, g[col], **C.line_kw(d))
        ax.set_title(title)
        ax.set_xlabel("Injected magnitude (training SDs)")
        ax.set_xticks(sorted(summ.magnitude.unique()))
    axes[0].set_ylabel("Detection ROC AUC")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].annotate("chance", (summ.magnitude.min(), 0.5), xytext=(2, 3),
                     textcoords="offset points", fontsize=8, color=C.INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(dets), bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Detection AUC vs anomaly magnitude (mean over folds and seeds, 95% CI band)",
                 y=1.02)
    return C.savefig(fig, "magnitude_sweep")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--detectors", nargs="+", default=None)
    args = ap.parse_args()
    data = C.load()
    fb = C.FoldBench(data, C.implemented_detectors(args.detectors))
    df = run(fb, range(args.seeds))
    C.write_csv(df, "magnitude_sweep_runs")
    summ = summarise(df)
    C.write_csv(summ, "magnitude_sweep")
    print(summ.round(3).to_string(index=False))
    print("figure:", plot(summ))


if __name__ == "__main__":
    main()
