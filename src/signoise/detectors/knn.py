"""KNN-distance anomaly score.

OWNER: Role 2 (Rashi). Status: done.

Score of a point = Euclidean distance to its k-th nearest neighbour among the
TRAINING points. Far from everything seen in training -> anomalous.

Notes:
  * Training has tens of thousands of points (T*N). Subsample the reference
    set (e.g. 5,000-20,000 points, fixed seed) and compute distances in
    chunks so memory stays bounded. Plain NumPy is enough.
  * Report the effect of k (e.g. 1, 5, 10, 20) in the results.
  * No sklearn / scipy.spatial.KDTree for the core algorithm (own code).
  * ``clip`` (optional, default None = off) caps every feature at +-clip before
    distances are taken. The robust z-scores of this data have huge natural
    tails (|z| in the hundreds for light), which would otherwise dominate
    Euclidean distance and bury moderate injected anomalies.
"""

from __future__ import annotations

import numpy as np

from signoise.detectors.base import PointwiseDetector


class KNNDetector(PointwiseDetector):
    name = "knn"

    def __init__(self, k: int = 5, max_ref: int = 10000, chunk: int = 2048,
                 seed: int = 0, features: str = "channels", clip: float | None = None):
        super().__init__(features)
        self.k, self.max_ref, self.chunk, self.seed = k, max_ref, chunk, seed
        self.clip = clip

    def _clip(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        return P if self.clip is None else np.clip(P, -self.clip, self.clip)

    def _fit_points(self, P: np.ndarray) -> None:
        """Keep a seeded random subsample of at most ``max_ref`` training rows."""
        P = self._clip(P)
        if len(P) > self.max_ref:
            rng = np.random.default_rng(self.seed)
            P = P[rng.choice(len(P), size=self.max_ref, replace=False)]
        self.ref_ = P
        self._ref_sq = (P ** 2).sum(axis=1)  # ||b||^2, reused for every chunk

    def _score_points(self, P: np.ndarray) -> np.ndarray:
        """Distance from each row of P (n, d) to its k-th nearest reference point."""
        P = self._clip(P)
        n_ref = len(self.ref_)
        if n_ref == 0:
            raise ValueError("KNNDetector.score called before fit")
        k = min(self.k, n_ref)
        out = np.empty(len(P))
        for lo in range(0, len(P), self.chunk):
            A = P[lo:lo + self.chunk]
            # squared Euclidean distance via ||a||^2 + ||b||^2 - 2 a.b  (chunk, n_ref)
            d2 = (A ** 2).sum(axis=1)[:, None] + self._ref_sq[None, :] - 2.0 * A @ self.ref_.T
            # k-th smallest in each row (index k-1 after partition)
            kth = np.partition(d2, k - 1, axis=1)[:, k - 1]
            out[lo:lo + self.chunk] = np.sqrt(np.maximum(kth, 0.0))  # clip rounding error
        return out
