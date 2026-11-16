"""K-means anomaly score.

OWNER: Role 2 (Rashi). Status: done.

Fit K-means (Lloyd's algorithm, own implementation, k-means++ initialisation)
on the training points. Score = distance to the nearest centroid.

Notes:
  * Several restarts (n_init), keep the lowest within-cluster sum of squares.
  * Stop when assignments stop changing or after max_iter.
  * Report the effect of K and an elbow plot of inertia vs K.
  * ``clip`` (optional, default None = off) caps every feature at +-clip before
    clustering and scoring, so the huge natural tails of the robust z-scores
    neither capture centroids nor dominate the distances.
"""

from __future__ import annotations

import numpy as np

from signoise.detectors.base import PointwiseDetector


class KMeansDetector(PointwiseDetector):
    name = "kmeans"

    def __init__(self, n_clusters: int = 8, n_init: int = 5, max_iter: int = 100,
                 seed: int = 0, features: str = "channels", clip: float | None = None):
        super().__init__(features)
        self.n_clusters, self.n_init, self.max_iter, self.seed = (
            n_clusters, n_init, max_iter, seed)
        self.clip = clip

    def _clip(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        return P if self.clip is None else np.clip(P, -self.clip, self.clip)

    @staticmethod
    def _sq_dist(P: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Squared distances (n, K) between rows of P and rows of C."""
        d2 = (P ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2.0 * P @ C.T
        return np.maximum(d2, 0.0)

    def _kmeanspp(self, P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """k-means++: each new centre is drawn with probability ~ D(x)^2."""
        n = len(P)
        centres = [P[rng.integers(n)]]
        d2 = self._sq_dist(P, np.array(centres))[:, 0]
        for _ in range(1, self.n_clusters):
            total = d2.sum()
            if total <= 0:  # all points coincide with a centre already
                idx = rng.integers(n)
            else:
                idx = rng.choice(n, p=d2 / total)
            centres.append(P[idx])
            d2 = np.minimum(d2, self._sq_dist(P, P[idx][None, :])[:, 0])
        return np.array(centres)

    def _lloyd(self, P: np.ndarray, C: np.ndarray):
        """Lloyd iterations from centres C. Returns (centres, inertia)."""
        labels = None
        for _ in range(self.max_iter):
            d2 = self._sq_dist(P, C)
            new_labels = d2.argmin(axis=1)
            if labels is not None and np.array_equal(new_labels, labels):
                break  # assignments stopped changing
            labels = new_labels
            nearest = d2[np.arange(len(P)), labels]
            for j in range(len(C)):
                members = labels == j
                if members.any():
                    C[j] = P[members].mean(axis=0)
                else:  # empty cluster: re-seed at the point farthest from its centre
                    far = int(nearest.argmax())
                    C[j] = P[far]
                    nearest[far] = 0.0
        d2 = self._sq_dist(P, C)
        return C, float(d2.min(axis=1).sum())

    def _fit_points(self, P: np.ndarray) -> None:
        """Run n_init k-means++/Lloyd restarts, keep the lowest inertia."""
        P = self._clip(P)
        if len(P) < self.n_clusters:
            raise ValueError("need at least n_clusters training points")
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            C, inertia = self._lloyd(P, self._kmeanspp(P, rng))
            if best is None or inertia < best[1]:
                best = (C, inertia)
        self.centroids_, self.inertia_ = best

    def _score_points(self, P: np.ndarray) -> np.ndarray:
        """Distance from each row of P to its nearest centroid."""
        d2 = self._sq_dist(self._clip(P), self.centroids_)
        return np.sqrt(d2.min(axis=1))
