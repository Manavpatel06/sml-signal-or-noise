"""KNN-distance anomaly score.

OWNER: Role 2 (Rashi). Status: TODO.

Score of a point = Euclidean distance to its k-th nearest neighbour among the
TRAINING points. Far from everything seen in training -> anomalous.

Notes:
  * Training has tens of thousands of points (T*N). Subsample the reference
    set (e.g. 5,000-20,000 points, fixed seed) and compute distances in
    chunks so memory stays bounded. Plain NumPy is enough.
  * Report the effect of k (e.g. 1, 5, 10, 20) in the results.
  * No sklearn / scipy.spatial.KDTree for the core algorithm (own code).
"""

from __future__ import annotations

import numpy as np

from signoise.detectors.base import PointwiseDetector


class KNNDetector(PointwiseDetector):
    name = "knn"

    def __init__(self, k: int = 5, max_ref: int = 10000, chunk: int = 2048,
                 seed: int = 0, features: str = "channels"):
        super().__init__(features)
        self.k, self.max_ref, self.chunk, self.seed = k, max_ref, chunk, seed

    def _fit_points(self, P: np.ndarray) -> None:
        raise NotImplementedError("Role 2: store (subsampled) reference points")

    def _score_points(self, P: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Role 2: distance to k-th nearest reference point")
