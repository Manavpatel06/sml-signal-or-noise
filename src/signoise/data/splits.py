"""Temporally disjoint cross-validation folds.

Adjacent readings are strongly correlated, so random splits leak information.
Each fold tests on one contiguous block of time and trains on the rest, with a
``gap`` of steps removed on both sides of the test block.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def temporal_folds(
    n_steps: int, n_folds: int = 5, gap: int = 12
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) index arrays over the time axis."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    edges = np.linspace(0, n_steps, n_folds + 1).astype(int)
    all_idx = np.arange(n_steps)
    for f in range(n_folds):
        lo, hi = edges[f], edges[f + 1]
        test = all_idx[lo:hi]
        train_mask = np.ones(n_steps, dtype=bool)
        train_mask[max(0, lo - gap): min(n_steps, hi + gap)] = False
        yield all_idx[train_mask], test


def train_test_split_time(n_steps: int, test_frac: float = 0.2, gap: int = 12):
    """Single split: train on the earliest steps, test on the last ``test_frac``."""
    cut = int(n_steps * (1 - test_frac))
    return np.arange(0, max(0, cut - gap)), np.arange(cut, n_steps)
