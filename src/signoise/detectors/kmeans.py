"""K-means anomaly score.

OWNER: Role 2 (Rashi). Status: TODO.

Fit K-means (Lloyd's algorithm, own implementation, k-means++ initialisation)
on the training points. Score = distance to the nearest centroid.

Notes:
  * Several restarts (n_init), keep the lowest within-cluster sum of squares.
  * Stop when assignments stop changing or after max_iter.
  * Report the effect of K and an elbow plot of inertia vs K.
"""

from __future__ import annotations

import numpy as np

from signoise.detectors.base import PointwiseDetector


class KMeansDetector(PointwiseDetector):
    name = "kmeans"

    def __init__(self, n_clusters: int = 8, n_init: int = 5, max_iter: int = 100,
                 seed: int = 0, features: str = "channels"):
        super().__init__(features)
        self.n_clusters, self.n_init, self.max_iter, self.seed = (
            n_clusters, n_init, max_iter, seed)

    def _fit_points(self, P: np.ndarray) -> None:
        raise NotImplementedError("Role 2: Lloyd's algorithm with k-means++ init")

    def _score_points(self, P: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Role 2: distance to nearest centroid")
