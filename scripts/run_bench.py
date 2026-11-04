"""Run the Stage A detector bench on injected anomalies.

Example:
    python scripts/run_bench.py --detectors ksigma knn kmeans gmm pca hmm --folds 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signoise.config import RESULTS_DIR  # noqa: E402
from signoise.data import load_intel  # noqa: E402
from signoise.detectors import REGISTRY  # noqa: E402
from signoise.evaluation import run_detection_bench, summarise  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", default=list(REGISTRY),
                    choices=list(REGISTRY))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bin-epochs", type=int, default=10)
    ap.add_argument("--magnitude", type=float, default=4.0)
    ap.add_argument("--events", type=int, default=10)
    ap.add_argument("--failures", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default=None, help="first date to use, e.g. 2004-02-28")
    ap.add_argument("--end", default="2004-03-20",
                    help="stop before this date; after ~20 March most nodes are "
                         "failing, so injected-anomaly results are not meaningful")
    ap.add_argument("--out", default=str(RESULTS_DIR / "bench.csv"))
    args = ap.parse_args()

    data = load_intel(bin_epochs=args.bin_epochs).between(args.start, args.end)
    print(f"data: {data.shape}, missing {data.missing_fraction():.1%}")
    res = run_detection_bench(
        data,
        detectors=args.detectors,
        n_folds=args.folds,
        seed=args.seed,
        inject_kwargs={"magnitude": args.magnitude, "n_events": args.events,
                       "n_failures": args.failures},
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print("\nsummary (mean, std over folds):")
    print(summarise(res).to_string())
    print(f"\nper-fold results: {args.out}")


if __name__ == "__main__":
    main()
