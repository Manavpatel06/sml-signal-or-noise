"""Detection AUC per anomaly kind, with and without quiet-period placement.

Each kind is injected on its own (20 episodes per test fold, magnitude 4), so
the AUC is that kind against normal cells:
    event, lagged_event (1-3 step lag between neighbours),
    failure (noise), stuck (frozen reading), drift (slow linear drift).
Placement: anywhere (v0) or quiet_only=True (calm periods, observed centre).

    python scripts/experiments/hard_variants.py --seeds 3
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common as C
from stats import mean_ci

KIND_ARGS = {
    "event": dict(n_events=20, n_failures=0),
    "lagged_event": dict(n_events=0, n_failures=0, n_lagged_events=20),
    "failure": dict(n_events=0, n_failures=20),
    "stuck": dict(n_events=0, n_failures=0, n_stuck=20),
    "drift": dict(n_events=0, n_failures=0, n_drift=20),
}
KIND_LABELS = {"event": "event", "lagged_event": "lagged event", "failure": "noise failure",
               "stuck": "stuck-at", "drift": "slow drift"}
PLACEMENTS = {"anywhere": {}, "quiet": {"quiet_only": True}}


def run(fb: C.FoldBench, seeds=range(3)) -> pd.DataFrame:
    parts = []
    for place, pkw in PLACEMENTS.items():
        for kind, kkw in KIND_ARGS.items():
            for s in seeds:
                print(f"  {place:>8} {kind:<12} seed {s}", flush=True)
                parts.append(fb.evaluate(seed=s, tag={"kind": kind, "placement": place},
                                         **kkw, **pkw))
    return pd.concat(parts, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (place, kind, det), g in df.groupby(["placement", "kind", "detector"]):
        mu, lo, hi = mean_ci(g.groupby("fold").auc.mean())
        rows.append({"placement": place, "kind": kind, "detector": det, "auc": mu,
                     "auc_ci_lo": lo, "auc_ci_hi": hi,
                     "precision_at_k": g.precision_at_k.mean()})
    return pd.DataFrame(rows)


def plot(summ: pd.DataFrame):
    C.style()
    dets = [d for d in C.DETECTOR_ORDER if d in set(summ.detector)]
    kinds = list(KIND_ARGS)
    fig, axes = C.plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, sharey=True)
    w = 0.8 / len(dets)
    for ax, place in zip(axes, PLACEMENTS):
        s = summ[summ.placement == place]
        for i, d in enumerate(dets):
            g = s[s.detector == d].set_index("kind").reindex(kinds)
            x = np.arange(len(kinds)) - 0.4 + w * (i + 0.5)
            ax.bar(x, g.auc - 0.2, bottom=0.2, width=w * 0.88, color=C.COLORS[d],
                   label=C.LABELS[d], hatch="//" if d == "ksigma" else None,
                   edgecolor="white", linewidth=0.6)
            ax.errorbar(x, g.auc, yerr=[g.auc - g.auc_ci_lo, g.auc_ci_hi - g.auc],
                        fmt="none", ecolor=C.INK, elinewidth=0.8, capsize=1.5)
        ax.axhline(0.5, color=C.INK2, lw=0.8, ls=":")
        ax.set_ylim(0.2, 1.0)
        ax.set_ylabel("ROC AUC")
        ax.set_title("Placement: " + ("anywhere in the test fold" if place == "anywhere"
                                      else "quiet periods only (quiet_only=True)"),
                     loc="left")
        ax.grid(axis="x", visible=False)
    axes[1].set_xticks(range(len(kinds)), [KIND_LABELS[k] for k in kinds])
    axes[1].set_xlabel("Injected anomaly kind (20 episodes per fold, magnitude 4)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(dets), bbox_to_anchor=(0.5, -0.05))
    return C.savefig(fig, "hard_variants")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--detectors", nargs="+", default=None)
    args = ap.parse_args()
    fb = C.FoldBench(C.load(), C.implemented_detectors(args.detectors))
    df = run(fb, range(args.seeds))
    C.write_csv(df, "hard_variants_runs")
    summ = summarise(df)
    C.write_csv(summ, "hard_variants")
    print(summ.pivot_table(index=["placement", "kind"], columns="detector", values="auc")
          .round(3).to_string())
    print("figure:", plot(summ))


if __name__ == "__main__":
    main()
