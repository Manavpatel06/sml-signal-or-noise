"""GMM / EM anomaly score.

OWNER: Role 4 (Harsh). Status: TODO.

Fit a Gaussian mixture with K components by EM (own implementation) on the
training points. Score = negative log-likelihood -log sum_k pi_k N(x | mu_k, Sigma_k).

Notes:
  * Full covariances with a small ridge (e.g. 1e-6 * I) to avoid singularity.
  * Log-sum-exp in the E-step for stability.
  * Initialise from K-means (Role 2's KMeansDetector) or random points; several
    restarts, keep the best log-likelihood.
  * Pick K by BIC on the training window and report the curve.
"""

from __future__ import annotations

import numpy as np

from signoise.detectors.base import PointwiseDetector


class GMMDetector(PointwiseDetector):
    name = "gmm"

    def __init__(self, n_components: int = 5, max_iter: int = 100, tol: float = 1e-4,
                 n_init: int = 3, seed: int = 0, features: str = "channels"):
        super().__init__(features)
        self.n_components, self.max_iter, self.tol, self.n_init, self.seed = (
            n_components, max_iter, tol, n_init, seed)

    def _fit_points(self, P: np.ndarray) -> None:
        raise NotImplementedError("Role 4: EM for a Gaussian mixture")

    def _score_points(self, P: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Role 4: negative log-likelihood under the mixture")
