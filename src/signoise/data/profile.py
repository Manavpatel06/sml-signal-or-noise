"""Descriptive profiles of the data: missingness, voltage, out-of-range readings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signoise.config import VALID_RANGE
from signoise.data.load import SensorData


def missingness_by_node(data: SensorData) -> pd.DataFrame:
    """Fraction of missing steps per node and channel."""
    miss = np.isnan(data.values).mean(axis=0)  # (N, C)
    return pd.DataFrame(miss, index=data.node_ids, columns=data.channels)


def out_of_range_by_node(data: SensorData) -> pd.DataFrame:
    """Fraction of observed readings outside VALID_RANGE per node and channel."""
    out = {}
    for c, name in enumerate(data.channels):
        lo, hi = VALID_RANGE[name]
        v = data.values[:, :, c]
        obs = np.isfinite(v)
        bad = obs & ((v < lo) | (v > hi))
        out[name] = bad.sum(0) / np.maximum(obs.sum(0), 1)
    return pd.DataFrame(out, index=data.node_ids)


def voltage_summary(data: SensorData) -> pd.DataFrame:
    """Per-node voltage: first/last observed value, min, and the first step it
    drops below ``low`` volts (a candidate weak failure label)."""
    v = data.channel("voltage")
    rows = []
    for j, node in enumerate(data.node_ids):
        col = v[:, j]
        obs = np.flatnonzero(np.isfinite(col))
        if len(obs) == 0:
            rows.append((node, np.nan, np.nan, np.nan, -1))
            continue
        low_idx = np.flatnonzero(np.isfinite(col) & (col < 2.3))
        rows.append((node, col[obs[0]], col[obs[-1]], np.nanmin(col),
                     int(low_idx[0]) if len(low_idx) else -1))
    return pd.DataFrame(
        rows, columns=["node", "v_first", "v_last", "v_min", "first_step_below_2.3V"]
    ).set_index("node")


def weak_failure_labels(data: SensorData, v_low: float = 2.3) -> np.ndarray:
    """(T, N) bool: True once a node's voltage has collapsed below ``v_low``.

    Imperfect by design (proposal Sec. 4.5); report results on these separately
    from results on injected anomalies.
    """
    v = data.channel("voltage")
    below = np.nan_to_num(v, nan=np.inf) < v_low
    return np.maximum.accumulate(below, axis=0)


def out_of_range_cells(data: SensorData) -> np.ndarray:
    """(T, N) bool: any channel outside VALID_RANGE at that step."""
    bad = np.zeros(data.shape[:2], dtype=bool)
    for c, name in enumerate(data.channels):
        lo, hi = VALID_RANGE[name]
        v = data.values[:, :, c]
        bad |= np.isfinite(v) & ((v < lo) | (v > hi))
    return bad


def real_anomaly_mask(data: SensorData, v_low: float = 2.3) -> np.ndarray:
    """(T, N) bool: cells that are probably real, unlabelled anomalies
    (voltage already collapsed, or a physically impossible reading).

    The harness leaves these out when scoring injected anomalies; they are the
    material for the weak-label evaluation instead.
    """
    return weak_failure_labels(data, v_low) | out_of_range_cells(data)
