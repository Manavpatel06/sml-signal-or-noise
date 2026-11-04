"""Event-vs-failure classifiers (proposal Sec. 4.3).

OWNER: Role 5 (Tejas). Status: TODO.

Both are own implementations with a sklearn-like interface so the harness can
swap them. Labels: 1 = event, 2 = failure (same codes as injection/).

GaussianNaiveBayes
  fit: class priors + per-class, per-feature mean and variance (add a small
  var_smoothing to every variance). predict_proba: Bayes' rule with the
  product of 1-D Gaussians, computed in log space.

LogisticRegression
  Binary (failure vs event). Minimise mean cross-entropy + (l2/2)||w||^2 by
  batch gradient descent or Newton/IRLS. Standardise features using training
  statistics. Gradient: X^T (sigmoid(Xw) - y) / n + l2 * w.
"""

from __future__ import annotations

import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        raise NotImplementedError("Role 5")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Role 5")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class LogisticRegression:
    def __init__(self, l2: float = 1e-3, lr: float = 0.1, max_iter: int = 2000,
                 tol: float = 1e-6):
        self.l2, self.lr, self.max_iter, self.tol = l2, lr, max_iter, tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        raise NotImplementedError("Role 5")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Role 5")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
