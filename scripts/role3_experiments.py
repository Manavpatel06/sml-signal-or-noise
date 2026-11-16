"""Role 3 experiments: PCA explained variance and score vs k; HMM vs number of states.

Writes CSV tables and PNG figures to results/role3/:
  pca_explained_variance.csv / .png   variance explained vs k (per fold)
  pca_k_sweep.csv / pca_auc_vs_k.png  bench AUC vs number of components
  hmm_states.csv / hmm_auc_vs_states.png   bench AUC vs S, per-node and shared

Uses the same setup as scripts/run_bench.py (data before 2004-03-20, 5 folds,
magnitude 4, seed 0), so numbers are comparable with the bench table.

Example:
    python scripts/role3_experiments.py              # everything (~10 min)
    python scripts/role3_experiments.py --skip-hmm   # PCA only (~1 min)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from signoise.config import HUMID, LIGHT, RESULTS_DIR, TEMP  # noqa: E402
from signoise.data import load_intel  # noqa: E402
from signoise.data.splits import temporal_folds  # noqa: E402
from signoise.detectors import HMMDetector, PCADetector  # noqa: E402
from signoise.evaluation import run_detection_bench  # noqa: E402
from signoise.imputation.simple import ffill_impute  # noqa: E402
from signoise.preprocess import Standardizer  # noqa: E402

METRICS = {"auc": "All anomalies", "auc_event": "Events", "auc_failure": "Failures"}
COLORS = {"auc": "#2a78d6", "auc_event": "#eb6834", "auc_failure": "#1baf7a"}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED)


def bench(data, detector, args):
    """Run the shared bench for one detector instance; returns per-fold rows."""
    return run_detection_bench(
        data, detectors=[detector], n_folds=args.folds, seed=args.seed, verbose=False,
        inject_kwargs={"magnitude": args.magnitude, "n_events": 10, "n_failures": 10})


def metric_lines(ax, df, x):
    """One line per metric (mean over folds) with a direct label at the right end."""
    g = df.groupby(x)[list(METRICS)].mean()
    for m, label in METRICS.items():
        ax.plot(g.index, g[m], color=COLORS[m], lw=2, marker="o", ms=5, label=label)
        ax.annotate(label, (g.index[-1], g[m].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)


def explained_variance(data, args, out):
    """Cumulative explained variance of the training window of every fold."""
    ch = [TEMP, HUMID, LIGHT]
    rows = []
    for f, (tr, _) in enumerate(temporal_folds(data.shape[0], args.folds)):
        Xtr = ffill_impute(data.values[tr])
        Z = Standardizer(per_node=True).fit_transform(Xtr)[:, :, ch]
        det = PCADetector(n_components=None).fit(Z)
        cum = np.cumsum(det.explained_variance_ratio_)
        rows += [{"fold": f, "k": k + 1, "cumulative": c} for k, c in enumerate(cum)]
        print(f"fold {f}: 95% variance needs k={det.n_components_}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "pca_explained_variance.csv", index=False)

    # Plot the variance LEFT UNEXPLAINED on a log scale: the cumulative curve
    # saturates near 1 after two or three components and hides the structure.
    df["unexplained"] = np.clip(1.0 - df["cumulative"], 1e-12, None)
    g = df[df.k <= args.k_max_plot].groupby("k")["unexplained"]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.fill_between(g.min().index, g.min(), g.max(), color=COLORS["auc"], alpha=0.15,
                    lw=0, label="range over folds")
    ax.plot(g.mean().index, g.mean(), color=COLORS["auc"], lw=2, marker="o", ms=3,
            label="mean over folds")
    ax.axhline(0.05, color=MUTED, lw=1, ls="--")
    ax.annotate("95% explained", (args.k_max_plot, 0.05), xytext=(0, 4),
                textcoords="offset points", ha="right", color=MUTED, fontsize=9)
    ax.axvline(5, color=MUTED, lw=1, ls=":")
    ax.annotate("default k = 5", (5, g.max().max()), xytext=(4, -10),
                textcoords="offset points", color=MUTED, fontsize=9)
    style(ax, "PCA: variance left unexplained by the top k components",
          "number of components k", "fraction unexplained (log)")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pca_explained_variance.png", dpi=150)
    plt.close(fig)


def pca_k_sweep(data, args, out, ks_ref):
    rows = []
    for k in args.ks:
        r = bench(data, PCADetector(n_components=k), args)
        r["k"] = k
        rows.append(r)
        print(f"pca k={k:>3}: auc={r['auc'].mean():.3f}")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "pca_k_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    metric_lines(ax, df, "k")
    ax.axhline(ks_ref, color=MUTED, lw=1, ls="--")
    ax.annotate(f"k-sigma baseline (all) {ks_ref:.2f}", (8, ks_ref), xytext=(0, -12),
                textcoords="offset points", color=MUTED, fontsize=9)
    style(ax, "PCA detector: ROC AUC vs number of components (mean of folds)",
          "number of components k", "ROC AUC")
    ax.set_xscale("log")
    ax.set_xticks(args.ks)
    ax.set_xticklabels([str(k) for k in args.ks])
    ax.minorticks_off()
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_xlim(right=max(args.ks) * 1.8)
    fig.tight_layout()
    fig.savefig(out / "pca_auc_vs_k.png", dpi=150)
    plt.close(fig)
    return df


def hmm_states(data, args, out, ks_ref):
    rows = []
    for shared in (False, True):
        for S in args.states:
            r = bench(data, HMMDetector(n_states=S, shared=shared, seed=args.seed), args)
            r["n_states"], r["model"] = S, "shared" if shared else "per-node"
            rows.append(r)
            print(f"hmm S={S} {r['model'].iloc[0]:>8}: auc={r['auc'].mean():.3f} "
                  f"({r['seconds'].mean():.0f}s per fold)")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "hmm_states.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=SURFACE)
    for ax, model in zip(axes, ("per-node", "shared")):
        metric_lines(ax, df[df.model == model], "n_states")
        ax.axhline(ks_ref, color=MUTED, lw=1, ls="--")
        style(ax, f"HMM, {model} model", "number of hidden states S",
              "ROC AUC" if model == "per-node" else "")
        ax.set_xticks(args.states)
        ax.set_xlim(min(args.states) - 0.3, max(args.states) + 1.6)
    axes[0].annotate(f"k-sigma (all) {ks_ref:.2f}", (min(args.states), ks_ref), xytext=(0, -12),
                     textcoords="offset points", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, loc="lower left", ncols=3, fontsize=8)
    fig.suptitle("HMM detector: ROC AUC vs number of states (mean of folds)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "hmm_auc_vs_states.png", dpi=150)
    plt.close(fig)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--magnitude", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--end", default="2004-03-20")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 50])
    ap.add_argument("--k-max-plot", type=int, default=40)
    ap.add_argument("--states", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--skip-hmm", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_DIR / "role3"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_intel().between(None, args.end)
    print(f"data: {data.shape}")

    ks_ref = bench(data, "ksigma", args)["auc"].mean()
    explained_variance(data, args, out)
    pca = pca_k_sweep(data, args, out, ks_ref)
    summary = [pca.groupby("k")[list(METRICS) + ["precision_at_k"]].mean().round(3)]
    if not args.skip_hmm:
        hmm = hmm_states(data, args, out, ks_ref)
        summary.append(hmm.groupby(["model", "n_states"])[list(METRICS) + ["precision_at_k"]]
                       .mean().round(3))
    print(f"\nk-sigma baseline auc: {ks_ref:.3f}")
    for s in summary:
        print("\n" + s.to_string())
    print(f"\nwrote tables and figures to {out}")


if __name__ == "__main__":
    main()
