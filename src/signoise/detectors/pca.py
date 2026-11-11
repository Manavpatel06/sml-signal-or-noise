"""PCA / SVD reconstruction-error anomaly score.

OWNER: Role 3 (Karthik).

Formulation (uses cross-node correlation, which is the point of the project):
for each time step t, stack all channels of all nodes into one vector
x_t in R^{N*C}. Fit PCA by SVD of the centred training matrix and keep the
top-k components (k given, or the smallest k reaching ``var_explained``).
Per-node score at step t = squared residual for that node, summed over
channels:

    r_t = (x_t - mu) - V_k^T V_k (x_t - mu)          (D = N*C)
    score[t, n] = sum_c r_t[n, c]^2 / s2[n, c]

where s2[n, c] is the mean squared training residual of that node and
channel (floored). Without this, nodes that are noisy in normal operation
dominate the ranking; with it, the score asks "how unusual is this residual
for THIS sensor".

Why this separates the two anomaly types: a building-wide or regional change
(an EVENT) moves many correlated nodes together, so it lies largely in the
span of the top components and is reconstructed well. A single node that
disagrees with what its correlated neighbours imply (a FAILURE) cannot be
explained by the shared components, so its residual is large.

Choosing k: the training window still contains real failing sensors with
readings hundreds of (robust) standard deviations out. They dominate the
total variance, so the "95% variance explained" rule keeps only 1-2
components, which mostly describe those outlier nodes. On the bench
(data before 20 March, 5 folds) a fixed k = 5 with normalised residuals
gave AUC 0.86 vs 0.71 for the 95% rule, so k = 5 is the default;
``n_components=None`` restores the variance rule. k was chosen on the same
folds that are reported, so report it as tuned. The curve is not smooth:
k = 3 gives 0.82, k = 5 gives 0.86, k = 8 drops to 0.79 and k = 10-30 sit
around 0.79-0.82, all above the k-sigma baseline (0.75). See
scripts/role3_experiments.py and results/role3/pca_auc_vs_k.png.

Only numpy.linalg.svd is used (own PCA, no sklearn).
"""

from __future__ import annotations

import numpy as np

from signoise.detectors.base import Detector


class PCADetector(Detector):
    name = "pca"

    def __init__(self, n_components: int | None = 5, var_explained: float = 0.95,
                 normalize: bool = True, var_floor: float = 1e-3):
        self.n_components = n_components
        self.var_explained = var_explained
        self.normalize = normalize
        self.var_floor = var_floor

    def fit(self, X: np.ndarray) -> "PCADetector":
        """X: (T, N, C) training tensor, NaN-free and standardised."""
        T, N, C = X.shape
        M = X.reshape(T, N * C)
        self.shape_ = (N, C)
        self.mean_ = M.mean(axis=0)

        # Rows of Vt are the principal directions; S**2 / (T - 1) the variances.
        _, S, Vt = np.linalg.svd(M - self.mean_, full_matrices=False)
        var = S**2 / max(T - 1, 1)
        total = var.sum()
        ratio = var / total if total > 0 else np.zeros_like(var)

        self.explained_variance_ = var
        self.explained_variance_ratio_ = ratio
        self.n_components_ = self._choose_k(ratio)
        self.components_ = Vt[: self.n_components_]  # (k, N*C)

        # Typical squared residual of each (node, channel) on training data.
        resid = X - self.reconstruct(X)
        self.resid_var_ = np.maximum((resid**2).mean(axis=0), self.var_floor)  # (N, C)
        return self

    def _choose_k(self, ratio: np.ndarray) -> int:
        """Fixed n_components, or the smallest k whose cumulative ratio reaches
        var_explained. Capped below the full rank so a residual always remains."""
        k_max = max(len(ratio) - 1, 1)
        if self.n_components is not None:
            k = self.n_components
        else:
            k = int(np.searchsorted(np.cumsum(ratio), self.var_explained) + 1)
        return int(np.clip(k, 1, k_max))

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Project X (T, N, C) onto the top-k subspace and map back."""
        T = X.shape[0]
        Mc = X.reshape(T, -1) - self.mean_
        Mhat = (Mc @ self.components_.T) @ self.components_ + self.mean_
        return Mhat.reshape(X.shape)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-node squared reconstruction error summed over channels, (T, N).
        With normalize=True each channel's residual is divided by its
        training residual variance first."""
        if X.shape[1:] != self.shape_:
            raise ValueError(f"expected (T, {self.shape_[0]}, {self.shape_[1]}), got {X.shape}")
        sq = (X - self.reconstruct(X)) ** 2
        if self.normalize:
            sq = sq / self.resid_var_
        return sq.sum(axis=2)
