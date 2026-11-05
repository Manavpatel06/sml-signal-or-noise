"""Evaluation metrics, implemented with NumPy only."""
from __future__ import annotations
import numpy as np

def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties averaged).

    y_true: binary {0,1}; scores: higher = more positive. NaN if one class is empty.
    """
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def precision_recall_at(y_true: np.ndarray, flagged: np.ndarray) -> tuple[float, float]:
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int):
    """Flag the k highest scores; k is usually the number of true anomalies."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels=(1, 2)) -> np.ndarray:
    """Rows = true label, columns = predicted label, in the order of ``labels``."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')
