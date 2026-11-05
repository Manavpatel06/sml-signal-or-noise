"""Temporally disjoint cross-validation folds.

Adjacent readings are strongly correlated, so random splits leak information.
Each fold tests on one contiguous block of time and trains on the rest, with a
``gap`` of steps removed on both sides of the test block.
"""
from __future__ import annotations
from typing import Iterator
import numpy as np

def temporal_folds(n_steps: int, n_folds: int=5, gap: int=12) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) index arrays over the time axis."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def train_test_split_time(n_steps: int, test_frac: float=0.2, gap: int=12):
    """Single split: train on the earliest steps, test on the last ``test_frac``."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')
