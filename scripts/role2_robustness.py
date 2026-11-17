"""Robustness check for the Role 2 detectors: other injection seeds and magnitudes.

Compares k-sigma, PCA, KNN (channels) and KNN (channels+diff), both KNN variants
with clip 6, on the same folds as scripts/run_bench.py but with different
injection seeds and magnitudes. Writes to results/role2/:
  robustness.csv          one row per (magnitude, seed, detector, fold)
  robustness_summary.csv  mean and std over folds per (magnitude, seed, detector)

Settings run in parallel processes (--jobs); results are deterministic.

Example:
    python scripts/role2_robustness.py --jobs 4
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from signoise.config import RESULTS_DIR  # noqa: E402
from signoise.data import load_intel  # noqa: E402
from signoise.detectors import KNNDetector  # noqa: E402
from signoise.evaluation import run_detection_bench  # noqa: E402

METRICS = ["auc", "auc_event", "auc_failure", "precision_at_k"]


def detectors():
    return {
        "ksigma": "ksigma",
        "pca": "pca",
        "knn channels clip6": KNNDetector(k=5, clip=6.0),
        "knn channels+diff clip6": KNNDetector(k=5, features="channels+diff", clip=6.0),
    }


def run_setting(args_tuple):
    """All detectors for one (magnitude, seed); returns per-fold rows."""
    mag, seed, folds, end = args_tuple
    data = load_intel().between(None, end)
    rows = []
    for label, det in detectors().items():
        r = run_detection_bench(
            data, detectors=[det], n_folds=folds, seed=seed, verbose=False,
            inject_kwargs={"magnitude": mag, "n_events": 10, "n_failures": 10})
        r = r.assign(magnitude=mag, seed=seed, detector=label)
        rows.append(r)
        print(f"mag={mag} seed={seed} {label}: auc={r['auc'].mean():.3f}", flush=True)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--end", default="2004-03-20")
    ap.add_argument("--magnitudes", type=float, nargs="+", default=[4.0, 2.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS_DIR / "role2"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = [(m, s, args.folds, args.end) for m in args.magnitudes for s in args.seeds]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        frames = list(pool.map(run_setting, settings))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "robustness.csv", index=False)
    cols = ["magnitude", "seed", "detector"]
    g = df.groupby(cols, sort=False)[METRICS]
    summary = g.mean().round(3).join(g.std(ddof=1).round(3).add_suffix("_std")).reset_index()
    summary.to_csv(out / "robustness_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + g.mean().round(3).to_string())
    print(f"\nwrote {out / 'robustness.csv'} and robustness_summary.csv")


if __name__ == "__main__":
    main()
