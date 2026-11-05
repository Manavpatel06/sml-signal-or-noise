"""Synthetic anomaly injection with known ground truth (proposal Sec. 4.5).

OWNER: Role 6 (Ziyad). Status: v0 working, extend.

Works on a STANDARDISED test tensor Z (T, N, C), so magnitudes are in units of
each node's training standard deviation.

Two families:
  * event   - coherent across a node and its spatial neighbours, smooth ramp
              up and down, temperature and humidity move together (temp up,
              humidity down), as when sunlight or HVAC hits a region.
  * failure - confined to ONE node, incoherent noise across channels, paired
              with a voltage drop, as when a dying node emits garbage.

Labels: 0 = normal, 1 = event, 2 = failure.

TODO (Role 6):
  * choose injection times inside "quiet" periods (low variance, all nodes
    agree) instead of uniformly at random;
  * graded difficulty: sweep ``magnitude`` and report detection AUC as a curve;
  * optional harder variants (stuck-at sensor, slow drift, events with lag
    between neighbours).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from signoise.config import HUMID, LIGHT, TEMP, VOLT
NORMAL, EVENT, FAILURE = (0, 1, 2)

@dataclass
class Episode:
    kind: int
    t0: int
    length: int
    nodes: list
    magnitude: float

def _ramp(length: int) -> np.ndarray:
    """Smooth 0 -> 1 -> 0 profile (half sine)."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def inject_anomalies(Z: np.ndarray, neighbors: np.ndarray, n_events: int=10, n_failures: int=10, magnitude: float=4.0, length: tuple=(6, 24), neighbor_decay: float=0.7, voltage_drop: float=2.0, seed: int=0):
    """Return (Z_injected, labels (T, N) int8, episodes).

    Episodes never overlap in (time, node). ``neighbors`` is the (N, k) array
    from data.neighbors.knn_neighbors.
    """
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')
