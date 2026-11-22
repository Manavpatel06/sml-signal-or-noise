"""Shared setup for the Role 6 experiments: data, fitted folds, figure style.

The expensive part of the bench is fitting the detectors (the per-node HMM
takes ~30 s a fold). Fitting uses only the clean TRAINING window, which does
not depend on where anomalies are injected, so ``FoldBench`` fits every
detector once per fold and then scores as many injected test windows as an
experiment needs (magnitudes, seeds, anomaly kinds). With the same seed and
injection settings the numbers are identical to ``run_detection_bench``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from signoise.config import HUMID, LIGHT, RESULTS_DIR, TEMP  # noqa: E402
from signoise.data import load_intel  # noqa: E402
from signoise.data.neighbors import knn_neighbors  # noqa: E402
from signoise.data.profile import real_anomaly_mask  # noqa: E402
from signoise.data.splits import temporal_folds  # noqa: E402
from signoise.detectors import REGISTRY  # noqa: E402
from signoise.evaluation.harness import ffill_both, make_detector, score_detection  # noqa: E402
from signoise.injection.inject import inject_anomalies  # noqa: E402
from signoise.preprocess import Standardizer  # noqa: E402

OUT = RESULTS_DIR / "experiments"
DETECT_CH = [TEMP, HUMID, LIGHT]

# ---- defaults shared by every experiment (the report states these once) ----
DEFAULTS = dict(bin_epochs=10, end="2004-03-20", n_folds=5, gap=12, k_neighbors=4,
                magnitude=4.0, n_events=10, n_failures=10)

# ---- figure style ------------------------------------------------------------
# One fixed colour per detector, used in every figure. The baseline is a
# neutral dashed grey so the learned detectors stand out against it.
DETECTOR_ORDER = ["ksigma", "knn", "kmeans", "gmm", "pca", "hmm"]
COLORS = {"ksigma": "#7a7974", "knn": "#2a78d6", "kmeans": "#eb6834",
          "gmm": "#1baf7a", "pca": "#6250d6", "hmm": "#e34948"}
LABELS = {"ksigma": "k-sigma (baseline)", "knn": "KNN distance", "kmeans": "K-means",
          "gmm": "GMM / EM", "pca": "PCA / SVD", "hmm": "HMM"}
MARKERS = {"ksigma": "o", "knn": "s", "kmeans": "D", "gmm": "^", "pca": "v", "hmm": "P"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def line_kw(det: str) -> dict:
    return dict(color=COLORS.get(det, INK2), marker=MARKERS.get(det, "o"),
                linestyle="--" if det == "ksigma" else "-", label=LABELS.get(det, det))


def savefig(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---- data and detectors --------------------------------------------------------

def load(bin_epochs: int = DEFAULTS["bin_epochs"], start=None, end=DEFAULTS["end"]):
    return load_intel(bin_epochs=bin_epochs).between(start, end)


def implemented_detectors(names=None) -> list[str]:
    """Registry names whose fit/score run (unfinished ones raise NotImplementedError)."""
    names = list(names or [d for d in DETECTOR_ORDER if d in REGISTRY])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4, 3))
    ok = []
    for n in names:
        try:
            make_detector(n).fit(X).score(X)
            ok.append(n)
        except NotImplementedError:
            print(f"  [skip] {n}: not implemented yet")
    return ok


@dataclass
class Fold:
    idx: int
    tr: np.ndarray
    te: np.ndarray
    Xtr: np.ndarray       # raw, imputed train (T, N, 4)
    Xte: np.ndarray       # raw, imputed test
    Ztr: np.ndarray       # standardised train
    Zte: np.ndarray       # standardised test (before injection)
    scaler: Standardizer
    valid_tr: np.ndarray  # (T, N) observed and not a real anomaly
    valid_te: np.ndarray
    detectors: dict       # name -> fitted Detector


class FoldBench:
    """Temporal folds with detectors fitted once per fold."""

    def __init__(self, data, detectors, n_folds=DEFAULTS["n_folds"], gap=DEFAULTS["gap"],
                 k_neighbors=DEFAULTS["k_neighbors"], verbose=True):
        self.data = data
        self.neighbors = knn_neighbors(data.coords, k=k_neighbors)
        real = real_anomaly_mask(data)
        observed = np.isfinite(data.values).all(axis=2)
        self.folds: list[Fold] = []
        self.fit_seconds = {d: 0.0 for d in detectors}
        for f, (tr, te) in enumerate(temporal_folds(data.shape[0], n_folds, gap)):
            Xtr, Xte = ffill_both(data.values[tr], data.values[te])
            scaler = Standardizer(per_node=True).fit(Xtr)
            Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
            dets = {}
            for name in detectors:
                t0 = time.perf_counter()
                dets[name] = make_detector(name).fit(Ztr[:, :, DETECT_CH])
                self.fit_seconds[name] += time.perf_counter() - t0
            self.folds.append(Fold(f, tr, te, Xtr, Xte, Ztr, Zte, scaler,
                                   observed[tr] & ~real[tr], observed[te] & ~real[te],
                                   dets))
            if verbose:
                print(f"  fold {f}: fitted {', '.join(detectors)}", flush=True)
        self.detectors = list(detectors)

    def inject(self, fold: Fold, seed: int, **kw):
        """Inject into the fold's test window with the harness seed convention."""
        return inject_anomalies(fold.Zte, self.neighbors, seed=seed + fold.idx,
                                valid=fold.valid_te, **kw)

    def evaluate(self, seed: int = 0, tag: dict | None = None, **inject_kw) -> pd.DataFrame:
        """One row per (fold, detector) with AUC / precision@k, like the harness."""
        kw = {"magnitude": DEFAULTS["magnitude"], "n_events": DEFAULTS["n_events"],
              "n_failures": DEFAULTS["n_failures"], **inject_kw}
        rows = []
        for fold in self.folds:
            Zi, labels, _ = self.inject(fold, seed, **kw)
            Zd = Zi[:, :, DETECT_CH]
            for name, det in fold.detectors.items():
                row = {"fold": fold.idx, "seed": seed, "detector": name, **(tag or {})}
                row.update(score_detection(labels, det.score(Zd), fold.valid_te))
                rows.append(row)
        return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
