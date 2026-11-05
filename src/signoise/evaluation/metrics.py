"""Evaluation metrics, implemented with NumPy only."""

from __future__ import annotations

import numpy as np


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties averaged).

    y_true: binary {0,1}; scores: higher = more positive. NaN if one class is empty.
    """
    y = np.asarray(y_true).ravel().astype(bool)
    s = np.asarray(scores, dtype=float).ravel()
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # 1-based ranks with ties averaged
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    _, first, counts = np.unique(sorted_s, return_index=True, return_counts=True)
    avg_rank = first + (counts + 1) / 2.0
    ranks = np.empty(len(s))
    ranks[order] = np.repeat(avg_rank, counts)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def precision_recall_at(y_true: np.ndarray, flagged: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true).ravel().astype(bool)
    f = np.asarray(flagged).ravel().astype(bool)
    tp = (y & f).sum()
    prec = tp / f.sum() if f.sum() else float("nan")
    rec = tp / y.sum() if y.sum() else float("nan")
    return float(prec), float(rec)


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int):
    """Flag the k highest scores; k is usually the number of true anomalies."""
    s = np.asarray(scores, dtype=float).ravel()
    flagged = np.zeros(len(s), dtype=bool)
    if k > 0:
        flagged[np.argpartition(-s, min(k, len(s)) - 1)[:k]] = True
    return precision_recall_at(y_true, flagged)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels=(1, 2)) -> np.ndarray:
    """Rows = true label, columns = predicted label, in the order of ``labels``."""
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(np.ravel(y_true), np.ravel(y_pred)):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm
