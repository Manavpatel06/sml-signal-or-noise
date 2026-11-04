"""Measure imputation quality by hiding known values and scoring recovery."""

from __future__ import annotations

from typing import Callable

import numpy as np


def mask_observed(X: np.ndarray, frac: float = 0.1, block: int = 1, seed: int = 0):
    """Hide a fraction of OBSERVED entries.

    block > 1 hides runs of ``block`` consecutive steps for the same node and
    channel, which mimics real outages better than isolated holes.
    Returns (X_masked, mask) where mask marks the hidden entries.
    """
    rng = np.random.default_rng(seed)
    T, N, C = X.shape
    obs = np.isfinite(X)
    mask = np.zeros_like(obs)
    n_target = int(frac * obs.sum())
    while mask.sum() < n_target:
        t = rng.integers(0, max(1, T - block + 1))
        n, c = rng.integers(0, N), rng.integers(0, C)
        mask[t:t + block, n, c] = True
    mask &= obs
    Xm = X.copy()
    Xm[mask] = np.nan
    return Xm, mask


def masking_benchmark(
    X: np.ndarray,
    impute: Callable[[np.ndarray], np.ndarray],
    frac: float = 0.1,
    block: int = 1,
    seed: int = 0,
) -> dict:
    """RMSE / MAE per channel of ``impute`` on hidden entries."""
    Xm, mask = mask_observed(X, frac=frac, block=block, seed=seed)
    Xhat = impute(Xm)
    out = {}
    for c in range(X.shape[2]):
        m = mask[:, :, c]
        err = Xhat[:, :, c][m] - X[:, :, c][m]
        out[c] = {"rmse": float(np.sqrt(np.mean(err ** 2))),
                  "mae": float(np.mean(np.abs(err))),
                  "n": int(m.sum())}
    return out
