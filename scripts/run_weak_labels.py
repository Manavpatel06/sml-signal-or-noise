"""Weak-label evaluation on REAL sensor failures (proposal Sec. 4.5).

Fit every detector on the healthy period (before ``--split``, default
2004-03-20) and score the late period, when most nodes are dying. Positives
are observed cells flagged by ``data.profile.real_anomaly_mask`` (voltage has
collapsed below 2.3 V, or a physically impossible reading); negatives are the
remaining observed cells. These labels are imperfect, so the numbers are
reported separately from the injected-anomaly bench.

    python scripts/run_weak_labels.py
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from signoise.config import HUMID, LIGHT, RESULTS_DIR, TEMP  # noqa: E402
from signoise.data import load_intel  # noqa: E402
from signoise.data.profile import out_of_range_cells, real_anomaly_mask  # noqa: E402
from signoise.detectors import REGISTRY  # noqa: E402
from signoise.evaluation.metrics import precision_recall_at_k, roc_auc  # noqa: E402
from signoise.imputation.simple import ffill_impute  # noqa: E402
from signoise.preprocess import Standardizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    ap.add_argument("--split", default="2004-03-20")
    ap.add_argument("--bin-epochs", type=int, default=10)
    ap.add_argument("--out", default=str(RESULTS_DIR / "weak_labels.csv"))
    args = ap.parse_args()

    data = load_intel(bin_epochs=args.bin_epochs)
    split = np.datetime64(args.split)
    tr = np.flatnonzero(data.times < split)
    te = np.flatnonzero(data.times >= split)
    real = real_anomaly_mask(data)
    impossible = out_of_range_cells(data)
    observed = np.isfinite(data.values).all(axis=2)

    # train on the healthy cells of the early period only
    Xtr_raw = data.values[tr].copy()
    Xtr_raw[real[tr]] = np.nan
    Xtr, Xte = ffill_impute(Xtr_raw), ffill_impute(data.values[te])
    scaler = Standardizer(per_node=True).fit(Xtr)
    ch = [TEMP, HUMID, LIGHT]
    Ztr, Zte = scaler.transform(Xtr)[:, :, ch], scaler.transform(Xte)[:, :, ch]

    valid = observed[te]
    y = real[te][valid]
    print(f"late period: {len(te)} steps, {valid.sum()} observed cells, "
          f"{y.mean():.1%} weak-failure positives "
          f"({impossible[te][valid].mean():.1%} physically impossible)")

    rows = []
    for name in args.detectors:
        t0 = time.perf_counter()
        try:
            s = REGISTRY[name]().fit(Ztr).score(Zte)[valid]
        except NotImplementedError:
            continue
        k = int(y.sum())
        p, r = precision_recall_at_k(y, s, k)
        rows.append({"detector": name, "auc_weak": roc_auc(y, s), "precision_at_k": p,
                     "auc_impossible_only": roc_auc(impossible[te][valid], s),
                     "n_pos": k, "n_cells": int(valid.sum()),
                     "seconds": round(time.perf_counter() - t0, 1)})
        print(f"{name:>8}: weak AUC={rows[-1]['auc_weak']:.3f}  "
              f"AUC(impossible)={rows[-1]['auc_impossible_only']:.3f}  P@k={p:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
