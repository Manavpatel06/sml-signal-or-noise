"""Event-vs-failure classifiers (proposal Sec. 4.3).

OWNER: Role 5 (Tejas).

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


def _check_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be (n, d) and y must be (n,) with matching n")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf")
    return X, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class GaussianNaiveBayes:
    """Gaussian Naive Bayes: features independent given the class.

    Attributes after fit: classes_ (K,), class_prior_ (K,), theta_ (K, d)
    class means, var_ (K, d) class variances (smoothed).
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        X, y = _check_xy(X, y)
        self.classes_ = np.unique(y)
        K, d = len(self.classes_), X.shape[1]
        self.class_prior_ = np.empty(K)
        self.theta_ = np.empty((K, d))
        self.var_ = np.empty((K, d))
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.class_prior_[k] = len(Xc) / len(X)
            self.theta_[k] = Xc.mean(axis=0)
            self.var_[k] = Xc.var(axis=0) + self.var_smoothing  # MLE variance
        # Guard against a constant feature in a one-sample class.
        self.var_ = np.maximum(self.var_, 1e-12)
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """log P(c) + sum_j log N(x_j; mu_cj, var_cj), shape (n, K)."""
        X = np.asarray(X, dtype=float)
        # (n, 1, d) against (1, K, d)
        diff2 = (X[:, None, :] - self.theta_[None, :, :]) ** 2
        ll = -0.5 * (np.log(2 * np.pi * self.var_)[None] + diff2 / self.var_[None])
        return np.log(self.class_prior_)[None, :] + ll.sum(axis=2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)          # log-sum-exp trick
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class LogisticRegression:
    """Binary L2-regularised logistic regression trained by gradient descent.

    The positive class is ``classes_[1]`` (the larger label, i.e. failure=2).
    Features are standardised with training mean/std. The bias is not
    penalised. ``loss_curve_`` holds the regularised loss after every step.
    predict_proba returns (n, 2) with columns in ``classes_`` order.
    """

    def __init__(self, l2: float = 1e-3, lr: float = 0.1, max_iter: int = 2000,
                 tol: float = 1e-6):
        self.l2, self.lr, self.max_iter, self.tol = l2, lr, max_iter, tol

    def _loss(self, Z, y01, w, b) -> float:
        z = Z @ w + b
        # mean cross-entropy: log(1 + e^z) - y z, computed stably
        ce = np.mean(np.logaddexp(0.0, z) - y01 * z)
        return float(ce + 0.5 * self.l2 * (w @ w))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X, y = _check_xy(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("LogisticRegression needs exactly two classes in y")
        y01 = (y == self.classes_[1]).astype(float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
        Z = (X - self.mean_) / self.scale_
        n, d = Z.shape
        w, b = np.zeros(d), 0.0
        self.loss_curve_ = [self._loss(Z, y01, w, b)]
        for _ in range(self.max_iter):
            r = _sigmoid(Z @ w + b) - y01
            w = w - self.lr * (Z.T @ r / n + self.l2 * w)
            b = b - self.lr * r.mean()
            self.loss_curve_.append(self._loss(Z, y01, w, b))
            if abs(self.loss_curve_[-2] - self.loss_curve_[-1]) < self.tol:
                break
        self.coef_, self.intercept_ = w, b   # in standardised feature space
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return Z @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
