"""Placeholder imputation so the pipeline runs before EM imputation is ready.

Forward-fill each (node, channel) series in time, then back-fill, then fall
back to the channel's global median. This is the baseline that the EM
imputer (imputation/em_gaussian.py) must beat on the masking benchmark.
"""
from __future__ import annotations
import numpy as np

def ffill_impute(X: np.ndarray) -> np.ndarray:
    """X: (T, N, C) with NaN. Returns a NaN-free copy."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')
