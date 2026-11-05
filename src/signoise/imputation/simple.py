"""Placeholder imputation so the pipeline runs before EM imputation is ready.

Forward-fill each (node, channel) series in time, then back-fill, then fall
back to the channel's global median. This is the baseline that the EM
imputer (imputation/em_gaussian.py) must beat on the masking benchmark.
"""

from __future__ import annotations

import numpy as np


def ffill_impute(X: np.ndarray) -> np.ndarray:
    """X: (T, N, C) with NaN. Returns a NaN-free copy."""
    out = X.copy()
    T = out.shape[0]
    idx = np.arange(T)[:, None, None]
    valid = np.isfinite(out)

    last = np.where(valid, idx, 0)
    np.maximum.accumulate(last, axis=0, out=last)
    fwd = np.take_along_axis(out, last, axis=0)
    seen = np.maximum.accumulate(valid, axis=0)
    out = np.where(valid, out, np.where(seen, fwd, np.nan))

    # back-fill the leading gap
    rev = out[::-1]
    rvalid = np.isfinite(rev)
    rlast = np.where(rvalid, idx, 0)
    np.maximum.accumulate(rlast, axis=0, out=rlast)
    bwd = np.take_along_axis(rev, rlast, axis=0)
    rseen = np.maximum.accumulate(rvalid, axis=0)
    out = np.where(rvalid, rev, np.where(rseen, bwd, np.nan))[::-1].copy()

    # nodes with no data at all in this window: global channel median
    med = np.nanmedian(X.reshape(-1, X.shape[2]), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    nan = np.isnan(out)
    out[nan] = np.broadcast_to(med, out.shape)[nan]
    return out
