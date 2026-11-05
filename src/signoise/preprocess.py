"""Standardisation fitted on training data only."""

from __future__ import annotations

import numpy as np

MAD_TO_STD = 1.4826  # scales the median absolute deviation to a Gaussian std


class Standardizer:
    """Z-score each (node, channel) using training-window statistics.

    per_node=True uses separate statistics for every node (nodes differ in
    baseline, e.g. a node by a window reads warmer); per_node=False pools nodes.

    robust=True (default) uses median and MAD instead of mean and std. The
    training window itself contains failing sensors with absurd readings
    (e.g. 120 C), which would otherwise inflate the scale and hide anomalies.

    A node whose training scale is ~0 (dead or constant in that window) falls
    back to the scale pooled over all nodes, so it does not get huge z-scores.
    """

    def __init__(self, per_node: bool = True, robust: bool = True, eps: float = 1e-6):
        self.per_node = per_node
        self.robust = robust
        self.eps = eps

    def _center_scale(self, X, axis):
        if self.robust:
            c = np.nanmedian(X, axis=axis, keepdims=True)
            s = MAD_TO_STD * np.nanmedian(np.abs(X - c), axis=axis, keepdims=True)
        else:
            c = np.nanmean(X, axis=axis, keepdims=True)
            s = np.nanstd(X, axis=axis, keepdims=True)
        return c, s

    def fit(self, X: np.ndarray) -> "Standardizer":
        pooled_c, pooled_s = self._center_scale(X, (0, 1))
        if self.per_node:
            c, s = self._center_scale(X, 0)
            tiny = ~(s > 1e-3 * pooled_s)
            s = np.where(tiny, pooled_s, s)
            c = np.where(np.isfinite(c), c, pooled_c)
        else:
            c, s = pooled_c, pooled_s
        self.mean_, self.std_ = c, s + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std_ + self.mean_
