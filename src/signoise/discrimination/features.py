"""Stage C features for each flagged anomaly (proposal Sec. 4.3).

OWNER: Role 5 (Tejas). Status: TODO.

An anomaly is a (t, n) cell flagged by a Stage A detector. For each one,
compute a feature vector from information the detectors ignore:

  1. neighbour_agreement  - do spatial neighbours deviate at the same step?
     e.g. fraction of neighbours whose detector score (or |z|) also exceeds a
     threshold in [t - w, t + w]. Events -> high, failures -> low.
  2. battery_level, battery_slope - the node's voltage at t and its least-
     squares slope over the previous W steps (raw volts, not z-scores).
  3. residual_voltage - regress voltage on the node's own temperature over the
     training window (voltage depends on temperature, per the dataset docs);
     feature = observed minus predicted voltage at t.
  4. cross_channel_consistency - correlation between the temperature and
     humidity changes for the node over [t - w, t]. Real events move them
     together (negatively); failures look uncorrelated.

Also build the same features with ``random_neighbors`` (data/neighbors.py) as
the random-neighbour control.

Output: build_feature_matrix returns (n_anomalies, d) floats plus the
feature names, ready for the classifiers in classifiers.py.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "neighbour_agreement",
    "battery_level",
    "battery_slope",
    "residual_voltage",
    "cross_channel_consistency",
]


def neighbour_agreement(scores: np.ndarray, neighbors: np.ndarray, t: int, n: int,
                        threshold: float, window: int = 2) -> float:
    raise NotImplementedError("Role 5")


def battery_level_and_slope(volts: np.ndarray, t: int, n: int, window: int = 12):
    raise NotImplementedError("Role 5")


def fit_voltage_temperature(volts_train: np.ndarray, temp_train: np.ndarray):
    """Per-node least-squares fit volt = a_n + b_n * temp. Returns (a, b), each (N,)."""
    raise NotImplementedError("Role 5")


def cross_channel_consistency(temp: np.ndarray, humid: np.ndarray, t: int, n: int,
                              window: int = 6) -> float:
    raise NotImplementedError("Role 5")


def build_feature_matrix(cells: np.ndarray, raw_values: np.ndarray, scores: np.ndarray,
                         neighbors: np.ndarray, voltage_fit, threshold: float):
    """cells: (M, 2) array of flagged (t, n). Returns (F (M, d), FEATURE_NAMES)."""
    raise NotImplementedError("Role 5")
