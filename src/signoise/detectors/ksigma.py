"""Baseline: k-standard-deviation threshold (proposal Sec. 4.5).

Input is already z-scored per node and channel on the training window, so the
score is simply the largest absolute z-score across channels. Thresholding at
k (e.g. 3) gives the classic "more than k sigma" rule. Every learned detector
must beat this to justify its complexity.
"""
from __future__ import annotations
import numpy as np
from signoise.detectors.base import Detector

class KSigmaDetector(Detector):
    name = 'ksigma'

    def __init__(self, k: float=3.0):
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def fit(self, X: np.ndarray) -> 'KSigmaDetector':
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')
