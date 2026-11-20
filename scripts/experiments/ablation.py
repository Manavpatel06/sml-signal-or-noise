"""Stage C ablation and random-neighbour control (Role 6 task 4, with Role 5).

Per temporal fold and injection seed:
  1. inject labelled events and failures into BOTH the train window
     (seed + 100 + fold) and the test window (seed + fold), using the real
     spatial neighbours: the ground truth is physical;
  2. score both windows with a detector fitted on the clean train window and
     flag the top-k valid cells (k = number of truly anomalous valid cells);
     the flagged cells that are true anomalies are what Stage C classifies;
  3. build the Stage C features (Role 5's build_feature_matrix) and fit
     Naive Bayes / logistic regression on train, predict test.

Conditions:
  all features                  - the full model
  drop <feature>                - one feature removed at a time
  only neighbour_agreement      - the spatial feature alone
  random neighbours             - all features, but neighbour_agreement is
                                  computed with random_neighbors (same k):
                                  if spatial structure matters, this is worse
Injection mixes: "standard" (10 events + 10 failures, v0 kinds) and "mixed"
(adds lagged events, stuck-at and drift failures).

Metrics: accuracy and balanced accuracy (mean of per-class recall) on the
flagged test anomalies; confusion matrices summed over folds and seeds.

    python scripts/experiments/ablation.py --detectors pca ksigma --seeds 5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common as C
from signoise.config import TEMP, VOLT
from signoise.data.neighbors import random_neighbors
from signoise.evaluation.metrics import confusion_matrix
from signoise.injection.inject import EVENT, FAILURE, inject_anomalies
from stats import mean_ci, paired_comparison

MIXES = {
    "standard": dict(n_events=10, n_failures=10),
    "mixed": dict(n_events=6, n_failures=4, n_lagged_events=4, n_stuck=3, n_drift=3),
}


def _discrimination():
    """Import Role 5's code; None if it is still a stub."""
    from signoise.discrimination import (FEATURE_NAMES, GaussianNaiveBayes,
                                         LogisticRegression, build_feature_matrix)
    from signoise.discrimination.features import fit_voltage_temperature
    try:
        fit_voltage_temperature(np.ones((5, 2)), np.arange(10.0).reshape(5, 2))
        GaussianNaiveBayes().fit(np.eye(2), np.array([1, 2]))
    except NotImplementedError:
        return None
    return dict(names=list(FEATURE_NAMES), build=build_feature_matrix,
                vfit=fit_voltage_temperature,
                classifiers={"naive_bayes": GaussianNaiveBayes,
                             "logistic": LogisticRegression})


def flag_true_anomalies(scores, labels, valid):
    """Top-k valid cells, k = #anomalous valid cells. Returns (cells, y, thr, n_fp)."""
    s = np.where(valid, scores, -np.inf).ravel()
    k = int(((labels > 0) & valid).sum())
    if k == 0:
        return np.zeros((0, 2), int), np.zeros(0, int), np.inf, 0
    top = np.argpartition(-s, k - 1)[:k]
    flagged = np.zeros(s.shape, bool)
    flagged[top] = True
    flagged = flagged.reshape(labels.shape)
    hit = flagged & (labels > 0)
    return np.argwhere(hit), labels[hit], float(s[top].min()), int((flagged & (labels == 0)).sum())


def conditions(names):
    out = [("all features", list(range(len(names))), "spatial")]
    out += [(f"drop {n}", [j for j in range(len(names)) if j != i], "spatial")
            for i, n in enumerate(names)]
    out += [("only neighbour_agreement", [names.index("neighbour_agreement")], "spatial"),
            ("random neighbours", list(range(len(names))), "random")]
    return out


def run(fb: C.FoldBench, detectors=("pca",), seeds=range(5), mixes=tuple(MIXES)):
    disc = _discrimination()
    if disc is None:
        print("  [skip] ablation: Role 5 discrimination code is not implemented yet")
        return None
    N = fb.data.shape[1]
    nbr = {"spatial": fb.neighbors,
           "random": random_neighbors(N, k=fb.neighbors.shape[1], seed=0)}
    conds = conditions(disc["names"])
    rows, cms = [], {}
    for det_name in detectors:
        for mix in mixes:
            for seed in seeds:
                print(f"  ablation {det_name} {mix} seed {seed}", flush=True)
                for fold in fb.folds:
                    det = fold.detectors[det_name]
                    vfit = disc["vfit"](fold.Xtr[:, :, VOLT], fold.Xtr[:, :, TEMP])
                    side = {}
                    for name, Z, valid, s in (
                            ("train", fold.Ztr, fold.valid_tr, seed + 100 + fold.idx),
                            ("test", fold.Zte, fold.valid_te, seed + fold.idx)):
                        Zi, lab, _ = inject_anomalies(Z, fb.neighbors, seed=s,
                                                      magnitude=C.DEFAULTS["magnitude"],
                                                      **MIXES[mix])
                        sc = det.score(Zi[:, :, C.DETECT_CH])
                        cells, y, thr, _ = flag_true_anomalies(sc, lab, valid)
                        side[name] = (fold.scaler.inverse_transform(Zi), sc, cells, y, thr)
                    feats = {}
                    for nb_kind in ("spatial", "random"):
                        for name, (raw, sc, cells, y, thr) in side.items():
                            F, _ = disc["build"](cells, raw, sc, nbr[nb_kind], vfit, thr)
                            feats[(nb_kind, name)] = (F, y)
                    for cond, cols, nb_kind in conds:
                        Ftr, ytr = feats[(nb_kind, "train")]
                        Fte, yte = feats[(nb_kind, "test")]
                        if len(np.unique(ytr)) < 2 or len(yte) == 0:
                            continue
                        for cname, cls in disc["classifiers"].items():
                            pred = cls().fit(Ftr[:, cols], ytr).predict(Fte[:, cols])
                            cm = confusion_matrix(yte, pred, (EVENT, FAILURE))
                            key = (det_name, mix, cname, cond)
                            cms[key] = cms.get(key, 0) + cm
                            rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
                            rows.append({"detector": det_name, "mix": mix, "classifier": cname,
                                         "condition": cond, "seed": seed, "fold": fold.idx,
                                         "n_test": int(cm.sum()), "n_train": len(ytr),
                                         "accuracy": np.trace(cm) / cm.sum(),
                                         "balanced_accuracy": rec.mean(),
                                         "ee": cm[0, 0], "ef": cm[0, 1],
                                         "fe": cm[1, 0], "ff": cm[1, 1]})
    return pd.DataFrame(rows), cms, [c for c, _, _ in conds]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per (detector, mix, classifier, condition): mean, CI over folds, change vs all."""
    out = []
    keys = ["detector", "mix", "classifier"]
    for k, g in df.groupby(keys):
        full = g[g.condition == "all features"]
        full_fold = full.groupby("fold")[["accuracy", "balanced_accuracy"]].mean()
        for cond, h in g.groupby("condition", sort=False):
            fm = h.groupby("fold")[["accuracy", "balanced_accuracy"]].mean()
            m, lo, hi = mean_ci(fm.balanced_accuracy)
            row = dict(zip(keys, k), condition=cond, accuracy=h.accuracy.mean(),
                       balanced_accuracy=h.balanced_accuracy.mean(), bal_ci_lo=lo,
                       bal_ci_hi=hi, n_test=int(h.n_test.sum()))
            if cond != "all features":
                j = fm.index.intersection(full_fold.index)
                pc = paired_comparison(fm.loc[j, "balanced_accuracy"].values,
                                       full_fold.loc[j, "balanced_accuracy"].values)
                row.update(delta_accuracy=h.accuracy.mean() - full.accuracy.mean(),
                           delta_balanced=pc["mean_diff"], delta_ci_lo=pc["ci_lo"],
                           delta_ci_hi=pc["ci_hi"], p_t=pc["p_t"])
            out.append(row)
    return pd.DataFrame(out)


def plot(summ: pd.DataFrame, cms: dict, order: list, detector: str):
    C.style()
    paths = []
    clf_color = {"naive_bayes": "#2a78d6", "logistic": "#eb6834"}
    clf_label = {"naive_bayes": "Naive Bayes", "logistic": "Logistic regression"}
    mixes = [m for m in MIXES if m in set(summ.mix)]
    fig, axes = C.plt.subplots(1, len(mixes), figsize=(6.2 * len(mixes), 4.4), sharey=True,
                               squeeze=False)
    conds = [c for c in order if c != "all features"]
    for ax, mix in zip(axes[0], mixes):
        s = summ[(summ.detector == detector) & (summ.mix == mix)]
        for j, cl in enumerate(clf_color):
            g = s[s.classifier == cl].set_index("condition").reindex(conds)
            full = s[(s.classifier == cl) & (s.condition == "all features")]
            lab = clf_label[cl] + (f" (full model {full.balanced_accuracy.iloc[0]:.3f})"
                                   if len(full) else "")
            y = np.arange(len(conds)) + (j - 0.5) * 0.34
            ax.barh(y, g.delta_balanced, height=0.32, color=clf_color[cl], label=lab)
            ax.errorbar(g.delta_balanced, y, xerr=[g.delta_balanced - g.delta_ci_lo,
                                                   g.delta_ci_hi - g.delta_balanced],
                        fmt="none", ecolor=C.INK, elinewidth=0.8, capsize=2)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=1, fontsize=8)
        ax.axvline(0, color=C.INK2, lw=1)
        ax.set_yticks(range(len(conds)), conds)
        ax.invert_yaxis()
        ax.set_xlabel("Change in balanced accuracy vs all features")
        ax.set_title(f"{mix} injection ({C.LABELS.get(detector, detector)} flags)", loc="left")
        ax.grid(axis="y", visible=False)
    paths.append(C.savefig(fig, f"ablation_{detector}"))

    # confusion matrices of the full model and the random-neighbour control
    fig, axes = C.plt.subplots(len(mixes), 4, figsize=(12, 3.0 * len(mixes)), squeeze=False)
    for r, mix in enumerate(mixes):
        for c, (cl, cond) in enumerate([(cl, cond) for cl in clf_color
                                        for cond in ("all features", "random neighbours")]):
            ax = axes[r][c]
            cm = cms.get((detector, mix, cl, cond))
            if cm is None:
                ax.axis("off")
                continue
            rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for k in range(2):
                    ax.text(k, i, f"{cm[i, k]}\n{rown[i, k]:.0%}", ha="center", va="center",
                            fontsize=9, color="white" if rown[i, k] > 0.6 else C.INK)
            ax.set_xticks([0, 1], ["event", "failure"])
            ax.set_yticks([0, 1], ["event", "failure"])
            ax.grid(False)
            ax.set_xlabel("predicted")
            if c == 0:
                ax.set_ylabel(f"{mix}\ntrue")
            ax.set_title(f"{clf_label[cl]}\n{cond}", fontsize=9)
    fig.suptitle(f"Event vs failure confusion matrices, summed over folds and seeds "
                 f"({C.LABELS.get(detector, detector)} flags)", y=1.01)
    fig.tight_layout()
    paths.append(C.savefig(fig, f"confusion_{detector}"))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", default=["pca", "ksigma"])
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    fb = C.FoldBench(C.load(), C.implemented_detectors(args.detectors))
    res = run(fb, fb.detectors, range(args.seeds))
    if res is None:
        return
    df, cms, order = res
    C.write_csv(df, "ablation_runs")
    summ = summarise(df)
    C.write_csv(summ, "ablation")
    print(summ.round(3).to_string(index=False))
    for d in fb.detectors:
        print("figures:", plot(summ, cms, order, d))


if __name__ == "__main__":
    main()
