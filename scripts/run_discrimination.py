"""Stage C: classify flagged anomalies as event (1) or failure (2).

Per temporal fold:
  1. impute, robust z-score on the training window (as in the Stage A bench);
  2. inject synthetic events/failures into BOTH the train and test windows
     (different seeds) so the classifiers have labelled anomalies to learn from
     and to be tested on;
  3. fit the chosen detector on the clean train window, score both windows;
  4. flag the top-k scoring valid cells (k = number of truly anomalous cells)
     and keep the flagged cells that are real injected anomalies;
  5. build Stage C features with spatial neighbours (and, as a control, with
     random neighbours), fit Naive Bayes / logistic regression on the train
     anomalies, predict the test anomalies.

Confusion matrices are summed over folds. Rows = true label, columns = predicted
label, order (event, failure).

Example:
    python scripts/run_discrimination.py --detector ksigma --folds 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signoise.config import HUMID, LIGHT, RESULTS_DIR, TEMP, VOLT  # noqa: E402
from signoise.data import load_intel  # noqa: E402
from signoise.data.neighbors import knn_neighbors, random_neighbors  # noqa: E402
from signoise.data.profile import real_anomaly_mask  # noqa: E402
from signoise.data.splits import temporal_folds  # noqa: E402
from signoise.discrimination import (FEATURE_NAMES, GaussianNaiveBayes,  # noqa: E402
                                     LogisticRegression, build_feature_matrix)
from signoise.discrimination.features import fit_voltage_temperature  # noqa: E402
from signoise.evaluation.harness import ffill_both, make_detector  # noqa: E402
from signoise.evaluation.metrics import confusion_matrix  # noqa: E402
from signoise.injection.inject import EVENT, FAILURE, inject_anomalies  # noqa: E402
from signoise.preprocess import Standardizer  # noqa: E402

CLASSIFIERS = {"naive_bayes": GaussianNaiveBayes, "logistic": LogisticRegression}
CONDITIONS = ("spatial", "random")


def flag_and_label(scores, labels, valid):
    """Flag the top-k valid cells (k = #true anomalous valid cells).

    Returns (cells (M, 2) of flagged cells that are true anomalies, their labels,
    threshold = lowest flagged score, number of flagged false positives).
    """
    s = np.where(valid, scores, -np.inf)
    k = int(((labels > 0) & valid).sum())
    flat = s.ravel()
    top = np.argpartition(-flat, k - 1)[:k]
    flagged = np.zeros(flat.shape, dtype=bool)
    flagged[top] = True
    flagged = flagged.reshape(s.shape)
    hit = flagged & (labels > 0)
    cells = np.argwhere(hit)
    return cells, labels[hit], float(flat[top].min()), int((flagged & (labels == 0)).sum())


def metrics(cm):
    n = cm.sum()
    acc = np.trace(cm) / n if n else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(cm) / cm.sum(axis=1)
    return acc, float(np.nanmean(recall))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="ksigma")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bin-epochs", type=int, default=10)
    ap.add_argument("--magnitude", type=float, default=4.0)
    ap.add_argument("--events", type=int, default=10)
    ap.add_argument("--failures", type=int, default=10)
    ap.add_argument("--neighbors", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default="2004-03-20")
    ap.add_argument("--out", default=str(RESULTS_DIR / "discrimination.csv"))
    args = ap.parse_args()

    data = load_intel(bin_epochs=args.bin_epochs).between(args.start, args.end)
    N = data.shape[1]
    print(f"data: {data.shape}, missing {data.missing_fraction():.1%}, "
          f"detector={args.detector}")
    nbr = {"spatial": knn_neighbors(data.coords, k=args.neighbors),
           "random": random_neighbors(N, k=args.neighbors, seed=args.seed)}
    real = real_anomaly_mask(data)
    observed = np.isfinite(data.values).all(axis=2)
    ch = [TEMP, HUMID, LIGHT]
    inject_kw = dict(magnitude=args.magnitude, n_events=args.events,
                     n_failures=args.failures)

    cms = {(c, m): np.zeros((2, 2), dtype=int) for c in CONDITIONS for m in CLASSIFIERS}
    total_fp = 0
    try:
        for fold, (tr, te) in enumerate(temporal_folds(data.shape[0], args.folds, 12)):
            Xtr, Xte = ffill_both(data.values[tr], data.values[te])
            scaler = Standardizer(per_node=True).fit(Xtr)
            Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
            det = make_detector(args.detector).fit(Ztr[:, :, ch])
            vfit = fit_voltage_temperature(Xtr[:, :, VOLT], Xtr[:, :, TEMP])

            side = {}
            for name, Z, idx, seed in (("train", Ztr, tr, args.seed + 100 + fold),
                                       ("test", Zte, te, args.seed + fold)):
                # always inject with the spatial neighbours: ground truth is physical
                Zi, lab, _ = inject_anomalies(Z, nbr["spatial"], seed=seed, **inject_kw)
                valid = observed[idx] & ~real[idx]
                scores = det.score(Zi[:, :, ch])
                cells, y, thr, fp = flag_and_label(scores, lab, valid)
                total_fp += fp if name == "test" else 0
                side[name] = (scaler.inverse_transform(Zi), scores, cells, y, thr)

            for cond in CONDITIONS:
                feats = {}
                for name, (raw, scores, cells, y, thr) in side.items():
                    F, _ = build_feature_matrix(cells, raw, scores, nbr[cond], vfit, thr)
                    feats[name] = (F, y)
                (Ftr, ytr), (Fte, yte) = feats["train"], feats["test"]
                if len(np.unique(ytr)) < 2 or len(yte) == 0:
                    print(f"fold {fold}: skipped ({cond}), too few flagged anomalies")
                    continue
                for mname, cls in CLASSIFIERS.items():
                    pred = cls().fit(Ftr, ytr).predict(Fte)
                    cms[(cond, mname)] += confusion_matrix(yte, pred, (EVENT, FAILURE))
    except NotImplementedError:
        sys.exit(f"detector '{args.detector}' is not implemented yet")

    rows = []
    print(f"\nfeatures: {FEATURE_NAMES}")
    print(f"flagged non-anomalous cells in test windows (false positives): {total_fp}")
    for (cond, mname), cm in cms.items():
        acc, bal = metrics(cm)
        print(f"\n[{mname}] neighbours={cond}  accuracy={acc:.3f}  "
              f"balanced accuracy={bal:.3f}")
        print("            pred event  pred failure")
        print(f"true event  {cm[0, 0]:>10d}  {cm[0, 1]:>12d}")
        print(f"true failure{cm[1, 0]:>10d}  {cm[1, 1]:>12d}")
        rows.append(f"{args.detector},{mname},{cond},{cm[0,0]},{cm[0,1]},{cm[1,0]},"
                    f"{cm[1,1]},{acc:.4f},{bal:.4f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("detector,classifier,neighbours,ee,ef,fe,ff,accuracy,balanced_accuracy\n"
                   + "\n".join(rows) + "\n")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
