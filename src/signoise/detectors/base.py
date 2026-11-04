"""Common interface every anomaly detector implements.

Input to fit/score is a NaN-free, standardised tensor X of shape (T, N, C):
T time steps, N nodes, C channels. score() returns a (T, N) array where a
HIGHER value means MORE anomalous for that node at that step.

Two ways to implement a detector:

* Subclass ``Detector`` and implement ``fit`` and ``score`` directly. Use this
  when the model needs time order (HMM) or cross-node vectors (PCA over nodes).
* Subclass ``PointwiseDetector`` and implement ``_fit_points`` /
  ``_score_points`` on a 2-D matrix P of shape (n_samples, d). Each sample is
  one node at one step; ``features`` controls what the row contains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    name: str = "detector"

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Detector":
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_score(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        return self.fit(X_train).score(X_test)


def point_features(X: np.ndarray, mode: str = "channels") -> np.ndarray:
    """Turn (T, N, C) into rows of (T*N, d), row index = t*N + n.

    mode="channels": d = C, the node's own readings.
    mode="channels+diff": d = 2C, readings plus first difference in time
        (first step's difference is 0). Helps catch sudden jumps.
    """
    T, N, C = X.shape
    if mode == "channels":
        F = X
    elif mode == "channels+diff":
        d = np.diff(X, axis=0, prepend=X[:1])
        F = np.concatenate([X, d], axis=2)
    else:
        raise ValueError(f"unknown feature mode {mode!r}")
    return F.reshape(T * N, -1)


class PointwiseDetector(Detector):
    """Detector that scores each (node, step) as an independent point."""

    def __init__(self, features: str = "channels"):
        self.features = features

    def fit(self, X: np.ndarray) -> "PointwiseDetector":
        self._fit_points(point_features(X, self.features))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        T, N, _ = X.shape
        s = self._score_points(point_features(X, self.features))
        return np.asarray(s, dtype=float).reshape(T, N)

    @abstractmethod
    def _fit_points(self, P: np.ndarray) -> None:
        ...

    @abstractmethod
    def _score_points(self, P: np.ndarray) -> np.ndarray:
        ...
