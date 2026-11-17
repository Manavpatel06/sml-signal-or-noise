"""Role 2 experiments: KNN and K-means sweeps, elbow plot, feature and clip study.

Writes CSV tables and PNG figures to results/role2/:
  clip_sweep.csv / clip_sweep.png            AUC vs feature clip (KNN k=5, K-means K=8)
  knn_k_sweep.csv / knn_auc_vs_k.png         AUC vs k for KNN
  kmeans_K_sweep.csv / kmeans_auc_vs_K.png   AUC vs K for K-means
  kmeans_elbow.csv / kmeans_elbow.png        training inertia vs K
  features.csv / features.png                "channels" vs "channels+diff" (AUC, precision@k)
  reference_bench.csv                        k-sigma (and optionally PCA, HMM) per fold
  summary.csv                                one row per configuration

Uses the same setup as scripts/run_bench.py (data before 2004-03-20, 5 folds,
magnitude 4, seed 0), so numbers are comparable with the bench table. Every
configuration sees the same injected anomalies (same seeds), so differences
between rows are paired. All figures show the mean over folds with +-1 standard
deviation across folds as error bars.

The clip value used for the "clipped" series is chosen by the clip sweep on the
same folds that are reported, so report it as tuned.

Examples:
    python scripts/role2_experiments.py                   # everything (~25 min)
    python scripts/role2_experiments.py --with-reference  # also bench PCA and HMM (+~8 min)
    python scripts/role2_experiments.py --plots-only      # redraw figures from the CSVs
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
from matplotlib.lines import Line2D  # noqa: E402

from signoise.config import HUMID, LIGHT, RESULTS_DIR, TEMP  # noqa: E402
from signoise.data import load_intel  # noqa: E402
from signoise.data.splits import temporal_folds  # noqa: E402
from signoise.detectors import KMeansDetector, KNNDetector  # noqa: E402
from signoise.detectors.base import point_features  # noqa: E402
from signoise.evaluation import run_detection_bench  # noqa: E402
from signoise.imputation.simple import ffill_impute  # noqa: E402
from signoise.preprocess import Standardizer  # noqa: E402

METRICS = {"auc": "All anomalies", "auc_event": "Events", "auc_failure": "Failures"}
COLORS = {"auc": "#2a78d6", "auc_event": "#eb6834", "auc_failure": "#1baf7a"}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
REPORT_COLS = list(METRICS) + ["precision_at_k"]
DODGE = {"auc": -0.07, "auc_event": 0.0, "auc_failure": 0.07}  # keeps error bars apart


# ------------------------------------------------------------------ plotting helpers

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


def end_labels(ax, items, x):
    """Write series names right of the last point, pushed apart so none overlap."""
    lo, hi = ax.get_ylim()
    gap = 0.055 * (hi - lo)
    placed = []
    for y, text, color in sorted(items):
        if placed and y < placed[-1][0] + gap:
            y = placed[-1][0] + gap
        placed.append((y, text, color))
    for y, text, color in placed:
        # opaque background so the dashed baseline never runs through the text
        ax.text(x, y, text, va="center", ha="left", color=INK, fontsize=9, zorder=5,
                bbox={"fc": SURFACE, "ec": "none", "pad": 1.5})


def baseline_legend(fig, ks_ref):
    """One shared legend entry for the k-sigma line, below the panels."""
    handle = Line2D([0], [0], color=MUTED, lw=1, ls="--")
    fig.legend([handle], [f"k-sigma baseline, all anomalies (AUC {ks_ref:.2f})"],
               loc="lower center", frameon=False, fontsize=9, labelcolor=MUTED)
    fig.text(0.995, 0.01, "points: mean over folds; bars: +-1 std across folds",
             ha="right", va="bottom", fontsize=8, color=MUTED)


def metric_lines(ax, df, x, labels, ks_ref):
    """Three AUC series vs a categorical x column, mean +- std over folds.

    ``labels`` fixes the x order. Positions are 0..n-1 so unevenly spaced
    settings (1, 5, 10, 20) are drawn evenly.
    """
    pos = {str(l): i for i, l in enumerate(labels)}
    g = df.assign(_x=df[x].astype(str).map(pos)).groupby("_x")[list(METRICS)]
    mean, std = g.mean(), g.std(ddof=1)
    items = []
    for m, name in METRICS.items():
        ax.errorbar(mean.index + DODGE[m], mean[m], yerr=std[m], color=COLORS[m], lw=2,
                    marker="o", ms=5, capsize=3, elinewidth=1, capthick=1)
        items.append((mean[m].iloc[-1], name, COLORS[m]))
    ax.axhline(ks_ref, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_xlim(-0.4, len(labels) - 1 + 1.7)
    ax.margins(y=0.05)
    end_labels(ax, items, len(labels) - 1 + 0.2)


def ordered(series):
    """Unique values in order of first appearance."""
    return list(dict.fromkeys(series.astype(str)))


# ------------------------------------------------------------------ figures

def plot_clip(df, ks_ref, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, facecolor=SURFACE)
    for ax, name in zip(axes, ("knn", "kmeans")):
        d = df[df.detector == name]
        metric_lines(ax, d, "clip", ordered(d["clip"]), ks_ref)
        style(ax, f"{name}: AUC vs feature clip", "clip |z| at (none = no clipping)",
              "ROC AUC" if name == "knn" else "")
    baseline_legend(fig, ks_ref)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "clip_sweep.png", dpi=150)
    plt.close(fig)


def plot_param(df, col, name, title, xlabel, ks_ref, out):
    values = sorted(df[col].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, facecolor=SURFACE)
    for ax, clip in zip(axes, ordered(df["clip"])):
        d = df[df["clip"].astype(str) == clip]
        metric_lines(ax, d, col, values, ks_ref)
        style(ax, f"{title}, clip = {clip}", xlabel, "ROC AUC" if ax is axes[0] else "")
    baseline_legend(fig, ks_ref)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / f"{name}_auc_vs_{col}.png", dpi=150)
    plt.close(fig)


def plot_elbow(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=SURFACE)
    for ax, clip in zip(axes, ordered(df["clip"])):
        g = df[df["clip"].astype(str) == clip].groupby("K")["inertia_per_point"]
        ax.fill_between(g.min().index, g.min(), g.max(), color=COLORS["auc"], alpha=0.15, lw=0)
        ax.plot(g.mean().index, g.mean(), color=COLORS["auc"], lw=2, marker="o", ms=4)
        style(ax, f"K-means elbow, clip = {clip}", "number of clusters K",
              "mean squared distance to centroid (train)")
        ax.set_yscale("log")
        ax.minorticks_off()
    fig.text(0.995, 0.01, "line: mean over folds; band: min to max across folds",
             ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "kmeans_elbow.png", dpi=150)
    plt.close(fig)


def plot_features(feats, ref, out):
    """Grouped bars: detector x (features, clip), one panel per metric, with the
    k-sigma / PCA / HMM means drawn as dashed reference lines."""
    panels = [("auc", "All anomalies (AUC)"), ("auc_event", "Events (AUC)"),
              ("auc_failure", "Failures (AUC)"), ("precision_at_k", "precision@k")]
    clips = ordered(feats["clip"])
    combos = [(f, c) for f in ("channels", "channels+diff") for c in clips]
    shade = {("channels", clips[0]): "#9db7da", ("channels", clips[1]): "#2a78d6",
             ("channels+diff", clips[0]): "#f0b69b", ("channels+diff", clips[1]): "#eb6834"}
    ref_style = {"ksigma": ("k-sigma", MUTED, "--"), "pca": ("PCA (k=5)", "#6b4ca3", "-."),
                 "hmm": ("HMM (per-node)", "#1baf7a", ":")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    width = 0.2
    for ax, (m, title) in zip(axes.ravel(), panels):
        for j, (f, c) in enumerate(combos):
            for i, det in enumerate(("knn", "kmeans")):
                d = feats[(feats.detector == det) & (feats.features == f) &
                          (feats["clip"].astype(str) == c)][m]
                sd = d.std(ddof=1)
                # lower whisker stops at 0 so precision@k never shows a negative value
                ax.bar(i + (j - 1.5) * width, d.mean(), width * 0.92,
                       yerr=[[min(sd, d.mean())], [sd]], color=shade[(f, c)], capsize=3,
                       error_kw={"lw": 1, "ecolor": INK})
        for det, (label, color, ls) in ref_style.items():
            r = ref[ref.detector == det][m]
            if len(r):
                ax.axhline(r.mean(), color=color, lw=1.2, ls=ls)
        style(ax, title, "", "")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["KNN (k=5)", "K-means (K=8)"], color=INK)
        ax.margins(y=0.08)
    handles = [plt.Rectangle((0, 0), 1, 1, color=shade[k]) for k in combos]
    labels = [f"{f}, clip {c}" for f, c in combos]
    for det, (label, color, ls) in ref_style.items():
        if (ref.detector == det).any():
            handles.append(Line2D([0], [0], color=color, lw=1.2, ls=ls))
            labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncols=4, frameon=False, fontsize=9)
    fig.text(0.995, 0.005, "bars: mean over folds; error bars: +-1 std across folds",
             ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "features.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ experiments

def bench(data, detector, args):
    """Run the shared bench for one detector instance; returns per-fold rows."""
    return run_detection_bench(
        data, detectors=[detector], n_folds=args.folds, seed=args.seed, verbose=False,
        inject_kwargs={"magnitude": args.magnitude, "n_events": 10, "n_failures": 10})


def sweep(data, args, make, values, label_col):
    """Bench ``make(v)`` for each v; returns the stacked per-fold table."""
    rows = []
    for v in values:
        r = bench(data, make(v), args)
        r[label_col] = "none" if v is None else v
        rows.append(r)
        print(f"  {label_col}={v}: auc={r['auc'].mean():.3f} event={r['auc_event'].mean():.3f} "
              f"failure={r['auc_failure'].mean():.3f} p@k={r['precision_at_k'].mean():.3f}")
    return pd.concat(rows, ignore_index=True)


def clip_sweep(data, args, out):
    print("clip sweep (KNN k=5, K-means K=8)")
    frames = []
    for name, make in (("knn", lambda c: KNNDetector(k=5, clip=c)),
                       ("kmeans", lambda c: KMeansDetector(n_clusters=8, clip=c))):
        print(f" {name}")
        df = sweep(data, args, make, args.clips, "clip")
        df["detector"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "clip_sweep.csv", index=False)
    return df


def best_clip(df, name):
    g = df[df.detector == name].groupby("clip")["auc"].mean()
    return float(g.drop(index="none", errors="ignore").idxmax())


def param_sweep(data, args, out, name, make, values, col, stem):
    """Sweep one hyper-parameter with and without clipping; save the csv."""
    print(f"{name} {col} sweep")
    frames = []
    for clip in (None, args.clip_used[name]):
        print(f" clip={clip}")
        df = sweep(data, args, lambda v, c=clip: make(v, c), values, col)
        df["clip"] = "none" if clip is None else clip
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / f"{stem}.csv", index=False)
    return df


def elbow(data, args, out):
    """Mean squared distance to the nearest centroid on each fold's training window."""
    print("k-means elbow (training inertia vs K)")
    ch = [TEMP, HUMID, LIGHT]
    rng = np.random.default_rng(args.seed)
    rows = []
    for f, (tr, _) in enumerate(temporal_folds(data.shape[0], args.folds)):
        Z = Standardizer(per_node=True).fit_transform(ffill_impute(data.values[tr]))[:, :, ch]
        P = point_features(Z)
        P = P[rng.choice(len(P), min(len(P), args.elbow_points), replace=False)]
        for clip in (None, args.clip_used["kmeans"]):
            Pc = P if clip is None else np.clip(P, -clip, clip)
            for K in args.elbow_K:
                det = KMeansDetector(n_clusters=K, seed=args.seed)
                det._fit_points(Pc)
                rows.append({"fold": f, "clip": "none" if clip is None else clip, "K": K,
                             "inertia_per_point": det.inertia_ / len(Pc)})
        print(f"  fold {f} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "kmeans_elbow.csv", index=False)
    return df


def feature_study(data, args, out):
    print("features: channels vs channels+diff")
    rows = []
    for feat in ("channels", "channels+diff"):
        for clip in (None, args.clip_used["knn"]):
            for name, det in (("knn", KNNDetector(k=5, features=feat, clip=clip)),
                              ("kmeans", KMeansDetector(n_clusters=8, features=feat,
                                                        clip=clip))):
                r = bench(data, det, args)
                r["features"], r["detector"] = feat, name
                r["clip"] = "none" if clip is None else clip
                rows.append(r)
                print(f"  {name:>6} {feat:>13} clip={clip}: auc={r['auc'].mean():.3f}")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "features.csv", index=False)
    return df


def reference_bench(data, args, out):
    """Per-fold results of the baseline (and, with --with-reference, PCA and HMM)."""
    specs = ["ksigma"] + (["pca", "hmm"] if args.with_reference else [])
    print(f"reference bench: {specs}")
    r = run_detection_bench(
        data, detectors=specs, n_folds=args.folds, seed=args.seed, verbose=False,
        inject_kwargs={"magnitude": args.magnitude, "n_events": 10, "n_failures": 10})
    r.to_csv(out / "reference_bench.csv", index=False)
    return r


def write_summary(clips, knn, km, feats, out):
    parts = {
        "clip": clips.groupby(["detector", "clip"])[REPORT_COLS].mean(),
        "knn_k": knn.groupby(["clip", "k"])[REPORT_COLS].mean(),
        "kmeans_K": km.groupby(["clip", "K"])[REPORT_COLS].mean(),
        "features": feats.groupby(["detector", "features", "clip"])[REPORT_COLS].mean(),
    }
    # the studies have different index levels, so stack them with a "study" column
    pd.concat([v.round(3).reset_index().assign(study=k) for k, v in parts.items()],
              ignore_index=True).to_csv(out / "summary.csv", index=False)
    for k, v in parts.items():
        print(f"\n{k}\n{v.round(3).to_string()}")


def draw_all(out):
    """Redraw every figure from the saved CSVs."""
    read = lambda n: pd.read_csv(out / n, dtype={"clip": str})  # noqa: E731
    ref = read("reference_bench.csv")
    ks_ref = ref[ref.detector == "ksigma"]["auc"].mean()
    plot_clip(read("clip_sweep.csv"), ks_ref, out)
    plot_param(read("knn_k_sweep.csv"), "k", "knn", "KNN: AUC vs k", "neighbours k", ks_ref, out)
    plot_param(read("kmeans_K_sweep.csv"), "K", "kmeans", "K-means: AUC vs K", "clusters K",
               ks_ref, out)
    plot_elbow(read("kmeans_elbow.csv"), out)
    plot_features(read("features.csv"), ref, out)
    print(f"wrote figures to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--magnitude", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--end", default="2004-03-20")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--Ks", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--clips", type=lambda s: None if s == "none" else float(s), nargs="+",
                    default=[None, 20.0, 10.0, 6.0, 4.0])
    ap.add_argument("--clip", type=float, default=None,
                    help="clip for the clipped series; default = best of the clip sweep")
    ap.add_argument("--elbow-K", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24])
    ap.add_argument("--elbow-points", type=int, default=60000)
    ap.add_argument("--with-reference", action="store_true",
                    help="also bench PCA and the per-node HMM for the comparison figure (slow)")
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw the figures from the CSVs already in --out")
    ap.add_argument("--out", default=str(RESULTS_DIR / "role2"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        draw_all(out)
        return

    data = load_intel().between(None, args.end)
    print(f"data: {data.shape}")

    ref = reference_bench(data, args, out)
    print(f"k-sigma baseline auc: {ref[ref.detector == 'ksigma']['auc'].mean():.3f}")

    clips = clip_sweep(data, args, out)
    args.clip_used = {n: args.clip if args.clip is not None else best_clip(clips, n)
                      for n in ("knn", "kmeans")}
    print(f"clip used for the clipped series: {args.clip_used}")

    knn = param_sweep(data, args, out, "knn", lambda k, c: KNNDetector(k=k, clip=c),
                      args.ks, "k", "knn_k_sweep")
    km = param_sweep(data, args, out, "kmeans",
                     lambda K, c: KMeansDetector(n_clusters=K, clip=c),
                     args.Ks, "K", "kmeans_K_sweep")
    elbow(data, args, out)
    feats = feature_study(data, args, out)
    write_summary(clips, knn, km, feats, out)
    draw_all(out)


if __name__ == "__main__":
    main()
