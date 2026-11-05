"""Standardisation fitted on training data only."""
from __future__ import annotations
import numpy as np
MAD_TO_STD = 1.4826

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

    def __init__(self, per_node: bool=True, robust: bool=True, eps: float=1e-06):
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def _center_scale(self, X, axis):
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def fit(self, X: np.ndarray) -> 'Standardizer':
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')
