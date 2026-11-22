"""Role 4 experiments: EM imputation masking study, GMM BIC curve, GMM K sweep.

Writes CSV tables and PNG figures to results/role4/:
  masking_rmse.csv / masking_rmse.png   RMSE per channel, EM vs ffill, block 1/6/24
  gmm_bic.csv / gmm_bic.png             BIC vs K on each fold's training window
  gmm_k_sweep.csv / gmm_auc_vs_k.png    bench AUC vs K, with and without diff features
  gmm_variants.csv                      GMM variants vs k-sigma, paired over folds

Same setup as scripts/run_bench.py (data before 2004-03-20, 5 folds, magnitude 4,
seed 0), so numbers are comparable with the bench table.

Masking study. For every fold the imputers are fitted on the training window and
evaluated on the test window: 10% of the OBSERVED test entries are hidden (in
runs of `block` steps) and recovered. Two settings are reported:
  raw    the test window as it is. Failing sensors report absurd values (120 C,
         -3000 % humidity) that persist in time, so a forward fill copies them
         and no model of neighbouring nodes can predict them.
  clean  readings outside config.VALID_RANGE or with a training-window robust
         |z| > 4 are removed first, so the numbers measure recovery of normal
         behaviour.

Example:
    python scripts/role4_experiments.py                 # everything
    python scripts/role4_experiments.py --skip-gmm      # masking study only
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
from scipy import stats  # noqa: E402

from signoise.config import CHANNELS, HUMID, LIGHT, RESULTS_DIR, TEMP  # noqa: E402
from signoise.data import load_intel  # noqa: E402
from signoise.data.splits import temporal_folds  # noqa: E402
from signoise.detectors.gmm import GMMDetector, bic_curve, elbow_k  # noqa: E402
from signoise.evaluation import run_detection_bench  # noqa: E402
from signoise.imputation import EMGaussianImputer, ffill_impute, masking_benchmark  # noqa: E402
from signoise.preprocess import Standardizer  # noqa: E402

COLORS = {"em": "#2a78d6", "ffill": "#eb6834", "auc": "#2a78d6",
          "auc_event": "#eb6834", "auc_failure": "#1baf7a"}
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


# --------------------------------------------------------------- masking study
def masking_study(data, args, out):
    rows = []
    for f, (tr, te) in enumerate(temporal_folds(data.shape[0], args.folds)):
        Xtr, Xte = data.values[tr], data.values[te]
        em = EMGaussianImputer().fit(Xtr)
        print(f"  imputer fold {f}: iterations {[len(t) for t in em.loglik_]}, "
              f"converged {em.converged_}")
        z = em.scaler_.transform(Xte)  # training-window robust z-scores
        keep = em._plausible(Xte) & (np.abs(z) <= em.clip_z)
        Xte_clean = np.where(keep, Xte, np.nan)
        for setting, truth in (("raw", Xte), ("clean", Xte_clean)):
            for block in args.blocks:
                res = {"em": masking_benchmark(truth, em.transform, frac=0.1,
                                               block=block, seed=args.seed + f),
                       "ffill": masking_benchmark(truth, ffill_impute, frac=0.1,
                                                  block=block, seed=args.seed + f)}
                for method, per_channel in res.items():
                    for c, r in per_channel.items():
                        rows.append({"fold": f, "setting": setting, "block": block,
                                     "channel": CHANNELS[c], "method": method,
                                     "rmse": r["rmse"], "mae": r["mae"], "n": r["n"]})
        print(f"masking fold {f} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "masking_rmse_per_fold.csv", index=False)
    table = (df.groupby(["setting", "channel", "block", "method"])[["rmse", "mae"]]
             .mean().unstack("method").round(3))
    table.columns = [f"{m}_{k}" for k, m in table.columns]
    table = table.reset_index()
    table["em_wins_rmse"] = table["em_rmse"] < table["ffill_rmse"]
    table.to_csv(out / "masking_rmse.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), facecolor=SURFACE)
    for i, setting in enumerate(("clean", "raw")):
        for j, ch in enumerate(CHANNELS[:3]):
            ax = axes[i, j]
            sub = table[(table.setting == setting) & (table.channel == ch)]
            x = np.arange(len(sub))
            ax.bar(x - 0.2, sub.em_rmse, 0.4, color=COLORS["em"], label="EM")
            ax.bar(x + 0.2, sub.ffill_rmse, 0.4, color=COLORS["ffill"], label="ffill")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{b}" for b in sub.block])
            style(ax, f"{ch} ({setting})", "gap length (steps of ~5 min)",
                  "RMSE" if j == 0 else "")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Masking study: RMSE of hidden entries, EM vs forward fill",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "masking_rmse.png", dpi=150)
    plt.close(fig)
    return table


# --------------------------------------------------------------------- GMM
def training_points(data, tr):
    """Standardised detector input (T, N, 3) for one fold's training window."""
    Xtr = ffill_impute(data.values[tr])
    return Standardizer(per_node=True).fit_transform(Xtr)[:, :, [TEMP, HUMID, LIGHT]]


def gmm_bic(data, args, out):
    rng = np.random.default_rng(args.seed)
    rows = []
    for f, (tr, _) in enumerate(temporal_folds(data.shape[0], args.folds)):
        Z = training_points(data, tr)
        P = Z.reshape(-1, Z.shape[2])
        P = np.clip(P, -args.clip, args.clip)
        P = P[rng.choice(len(P), size=min(len(P), 20000), replace=False)]
        curve = bic_curve(P, ks=args.bic_ks, seed=args.seed, max_points=None)
        rows += [{"fold": f, "k": k, "bic": b} for k, b in curve.items()]
        print(f"bic fold {f}: lowest at K = {min(curve, key=curve.get)}, "
              f"elbow (90% of drop) at K = {elbow_k(curve)}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "gmm_bic.csv", index=False)
    g = df.groupby("k")["bic"]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.fill_between(g.min().index, g.min(), g.max(), color=COLORS["auc"], alpha=0.15, lw=0)
    ax.plot(g.mean().index, g.mean(), color=COLORS["auc"], lw=2, marker="o", ms=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.bic_ks)
    ax.set_xticklabels([str(k) for k in args.bic_ks])
    ax.minorticks_off()
    style(ax, f"GMM: BIC vs number of components (clip +-{args.clip:g}, lower is better)",
          "number of components K", "BIC (mean, band = range over folds)")
    fig.tight_layout()
    fig.savefig(out / "gmm_bic.png", dpi=150)
    plt.close(fig)
    return df


def gmm_k_sweep(data, args, out, ks_ref):
    rows = []
    for features in ("channels", "channels+diff"):
        for k in args.ks:
            det = GMMDetector(n_components=k, clip=args.clip, features=features,
                              seed=args.seed)
            r = run_detection_bench(
                data, detectors=[det], n_folds=args.folds, seed=args.seed,
                verbose=False,
                inject_kwargs={"magnitude": args.magnitude, "n_events": 10,
                               "n_failures": 10})
            r["k"], r["features"] = k, features
            rows.append(r)
            print(f"gmm {features:>13} K={k:>2}: auc={r['auc'].mean():.3f}")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out / "gmm_k_sweep.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=SURFACE)
    for ax, features in zip(axes, ("channels", "channels+diff")):
        g = df[df.features == features].groupby("k")[["auc", "auc_event", "auc_failure"]].mean()
        for m, label in (("auc", "All"), ("auc_event", "Events"), ("auc_failure", "Failures")):
            ax.plot(g.index, g[m], color=COLORS[m], lw=2, marker="o", ms=4, label=label)
        ax.axhline(ks_ref, color=MUTED, lw=1, ls="--")
        style(ax, f"features = {features}", "number of components K",
              "ROC AUC" if features == "channels" else "")
    axes[0].annotate(f"k-sigma (all) {ks_ref:.2f}", (min(args.ks), ks_ref),
                     xytext=(0, -12), textcoords="offset points", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, loc="center right")
    fig.suptitle("GMM detector: ROC AUC vs K (mean of folds)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "gmm_auc_vs_k.png", dpi=150)
    plt.close(fig)
    return df


def gmm_variants(data, args, out):
    """Compare GMM settings with k-sigma on the same folds (paired)."""
    variants = {"ksigma": "ksigma"}
    for feat in ("channels", "channels+diff"):
        for clip in (None, 6.0):
            variants[f"gmm {feat} clip={clip}"] = (
                "gmm", {"features": feat, "clip": clip, "n_components": 5,
                        "seed": args.seed})
    rows = []
    for name, spec in variants.items():
        r = run_detection_bench(
            data, detectors=[spec], n_folds=args.folds, seed=args.seed, verbose=False,
            inject_kwargs={"magnitude": args.magnitude, "n_events": 10, "n_failures": 10})
        r["variant"] = name
        rows.append(r)
    df = pd.concat(rows, ignore_index=True)
    ks = df[df.variant == "ksigma"].sort_values("fold")
    out_rows = []
    for name in variants:
        x = df[df.variant == name].sort_values("fold")
        d_auc = x["auc"].to_numpy() - ks["auc"].to_numpy()
        p = (stats.ttest_rel(x["auc"], ks["auc"]).pvalue if name != "ksigma" else np.nan)
        out_rows.append({
            "variant": name, "auc": x["auc"].mean(), "auc_event": x["auc_event"].mean(),
            "auc_failure": x["auc_failure"].mean(),
            "precision_at_k": x["precision_at_k"].mean(),
            "d_auc_vs_ksigma": d_auc.mean(), "paired_t_p": p,
            "folds_won": int((d_auc > 0).sum())})
    table = pd.DataFrame(out_rows).round(3)
    table.to_csv(out / "gmm_variants.csv", index=False)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--magnitude", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--end", default="2004-03-20")
    ap.add_argument("--blocks", type=int, nargs="+", default=[1, 6, 24])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8],
                    help="K grid for the AUC sweep")
    ap.add_argument("--bic-ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                    help="K grid for the BIC curve")
    ap.add_argument("--clip", type=float, default=10.0, help="GMM feature clip")
    ap.add_argument("--skip-gmm", action="store_true")
    ap.add_argument("--skip-masking", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_DIR / "role4"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_intel().between(None, args.end)
    print(f"data: {data.shape}")
    if not args.skip_masking:
        table = masking_study(data, args, out)
        print("\nmasking study (mean over folds):")
        print(table.to_string(index=False))
    if not args.skip_gmm:
        bic = gmm_bic(data, args, out)
        print("\nBIC (mean over folds):")
        print(bic.groupby("k")["bic"].mean().round(0).to_string())
        print("\nGMM variants vs k-sigma (paired over folds):")
        print(gmm_variants(data, args, out).to_string(index=False))
        ks_ref = run_detection_bench(
            data, detectors=["ksigma"], n_folds=args.folds, seed=args.seed,
            verbose=False,
            inject_kwargs={"magnitude": args.magnitude, "n_events": 10,
                           "n_failures": 10})["auc"].mean()
        sweep = gmm_k_sweep(data, args, out, ks_ref)
        print(f"\nk-sigma baseline auc: {ks_ref:.3f}")
        print(sweep.groupby(["features", "k"])[
            ["auc", "auc_event", "auc_failure", "precision_at_k"]].mean().round(3)
            .to_string())
    print(f"\nwrote tables and figures to {out}")


if __name__ == "__main__":
    main()
