"""Baseline: k-standard-deviation threshold (proposal Sec. 4.5).

Input is already z-scored per node and channel on the training window, so the
score is simply the largest absolute z-score across channels. Thresholding at
k (e.g. 3) gives the classic "more than k sigma" rule. Every learned detector
must beat this to justify its complexity.
"""

from __future__ import annotations

import numpy as np

from signoise.detectors.base import Detector


class KSigmaDetector(Detector):
    name = "ksigma"

    def __init__(self, k: float = 3.0):
        self.k = k

    def fit(self, X: np.ndarray) -> "KSigmaDetector":
        # Re-centre on the training window in case the caller did not.
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True) + 1e-6
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((X - self.mean_) / self.std_)
        return z.max(axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.k
