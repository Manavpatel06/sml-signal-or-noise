"""Measure imputation quality by hiding known values and scoring recovery."""
from __future__ import annotations
from typing import Callable
import numpy as np

def mask_observed(X: np.ndarray, frac: float=0.1, block: int=1, seed: int=0):
    """Hide a fraction of OBSERVED entries.

    block > 1 hides runs of ``block`` consecutive steps for the same node and
    channel, which mimics real outages better than isolated holes.
    Returns (X_masked, mask) where mask marks the hidden entries.
    """
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def masking_benchmark(X: np.ndarray, impute: Callable[[np.ndarray], np.ndarray], frac: float=0.1, block: int=1, seed: int=0) -> dict:
    """RMSE / MAE per channel of ``impute`` on hidden entries."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')
