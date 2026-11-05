"""Descriptive profiles of the data: missingness, voltage, out-of-range readings."""
from __future__ import annotations
import numpy as np
import pandas as pd
from signoise.config import VALID_RANGE
from signoise.data.load import SensorData

def missingness_by_node(data: SensorData) -> pd.DataFrame:
    """Fraction of missing steps per node and channel."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def out_of_range_by_node(data: SensorData) -> pd.DataFrame:
    """Fraction of observed readings outside VALID_RANGE per node and channel."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def voltage_summary(data: SensorData) -> pd.DataFrame:
    """Per-node voltage: first/last observed value, min, and the first step it
    drops below ``low`` volts (a candidate weak failure label)."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def weak_failure_labels(data: SensorData, v_low: float=2.3) -> np.ndarray:
    """(T, N) bool: True once a node's voltage has collapsed below ``v_low``.

    Imperfect by design (proposal Sec. 4.5); report results on these separately
    from results on injected anomalies.
    """
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def out_of_range_cells(data: SensorData) -> np.ndarray:
    """(T, N) bool: any channel outside VALID_RANGE at that step."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def real_anomaly_mask(data: SensorData, v_low: float=2.3) -> np.ndarray:
    """(T, N) bool: cells that are probably real, unlabelled anomalies
    (voltage already collapsed, or a physically impossible reading).

    The harness leaves these out when scoring injected anomalies; they are the
    material for the weak-label evaluation instead.
    """
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')
