"""EM imputation over a multivariate Gaussian of correlated node series.

OWNER: Role 4 (Harsh). Status: TODO.

Idea (proposal Sec. 4.2): at each time step t, stack one channel across all
nodes into a vector x_t in R^N (or all channels: R^{N*C}). Model x_t ~ N(mu, Sigma).
A missing entry is estimated from the observed entries at the same step,
because neighbouring nodes are correlated.

EM for missing data (Bishop 9.x / standard Gaussian EM with missing values):
  E-step, for each t with observed set o and missing set m:
      E[x_m | x_o]   = mu_m + Sigma_mo Sigma_oo^{-1} (x_o - mu_o)
      Cov[x_m | x_o] = Sigma_mm - Sigma_mo Sigma_oo^{-1} Sigma_om
  M-step: mu = mean of completed x_t;
      Sigma = mean of (x_t - mu)(x_t - mu)^T + the conditional covariances.
  Iterate until the observed-data log-likelihood stops improving.

Requirements:
  * NumPy/SciPy linear algebra only (no sklearn).
  * Add a small ridge (e.g. 1e-3 * I) to Sigma for numerical stability.
  * Fit on the TRAINING window only; transform() may be applied to test.
  * Evaluate with signoise.imputation.masking.masking_benchmark and report
    RMSE against ffill_impute (the baseline).
"""

from __future__ import annotations

import numpy as np


class EMGaussianImputer:
    def __init__(self, max_iter: int = 50, tol: float = 1e-4, ridge: float = 1e-3,
                 per_channel: bool = True):
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge
        self.per_channel = per_channel

    def fit(self, X: np.ndarray) -> "EMGaussianImputer":
        """X: (T, N, C) with NaN. Learn mu, Sigma (per channel if per_channel)."""
        raise NotImplementedError("Role 4: implement EM fit")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return a NaN-free copy of X using conditional means E[x_m | x_o]."""
        raise NotImplementedError("Role 4: implement conditional-mean fill")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
