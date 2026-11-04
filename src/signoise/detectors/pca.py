"""PCA / SVD reconstruction-error anomaly score.

OWNER: Role 3 (Karthik). Status: TODO.

Recommended formulation (uses cross-node correlation, which is the point of
the project): for each time step t, form the vector of ONE channel across all
nodes, x_t in R^N (or all channels, R^{N*C}). Fit PCA by SVD of the centred
training matrix; keep the top-k components (choose k by explained variance,
e.g. 90-95%). Per-node score at step t = squared residual for that node,
(x_t - x_hat_t)_n^2, summed over channels.

A node that disagrees with what its correlated neighbours imply gets a large
residual; a building-wide change is explained by the top components.

Notes:
  * numpy.linalg.svd only (own PCA, no sklearn).
  * Report score vs k and the explained-variance curve.
"""

from __future__ import annotations

import numpy as np

from signoise.detectors.base import Detector


class PCADetector(Detector):
    name = "pca"

    def __init__(self, n_components: int | None = None, var_explained: float = 0.95):
        self.n_components = n_components
        self.var_explained = var_explained

    def fit(self, X: np.ndarray) -> "PCADetector":
        raise NotImplementedError("Role 3: SVD of centred (T, N*C) training matrix")

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Role 3: per-node squared reconstruction error, (T, N)")
