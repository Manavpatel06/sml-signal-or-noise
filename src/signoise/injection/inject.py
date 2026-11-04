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

NORMAL, EVENT, FAILURE = 0, 1, 2


@dataclass
class Episode:
    kind: int          # EVENT or FAILURE
    t0: int
    length: int
    nodes: list        # affected node positions
    magnitude: float


def _ramp(length: int) -> np.ndarray:
    """Smooth 0 -> 1 -> 0 profile (half sine)."""
    return np.sin(np.linspace(0, np.pi, length + 2)[1:-1])


def inject_anomalies(
    Z: np.ndarray,
    neighbors: np.ndarray,
    n_events: int = 10,
    n_failures: int = 10,
    magnitude: float = 4.0,
    length: tuple = (6, 24),
    neighbor_decay: float = 0.7,
    voltage_drop: float = 2.0,
    seed: int = 0,
):
    """Return (Z_injected, labels (T, N) int8, episodes).

    Episodes never overlap in (time, node). ``neighbors`` is the (N, k) array
    from data.neighbors.knn_neighbors.
    """
    rng = np.random.default_rng(seed)
    Zi = Z.copy()
    T, N, _ = Z.shape
    labels = np.zeros((T, N), dtype=np.int8)
    episodes: list[Episode] = []

    def place(kind: int, tries: int = 200) -> None:
        for _ in range(tries):
            L = int(rng.integers(length[0], length[1] + 1))
            if L >= T:
                return
            t0 = int(rng.integers(0, T - L))
            center = int(rng.integers(0, N))
            nodes = [center] + (list(neighbors[center]) if kind == EVENT else [])
            if labels[t0:t0 + L, nodes].any():
                continue
            prof = _ramp(L) if kind == EVENT else np.ones(L)
            for rank, n in enumerate(nodes):
                if kind == EVENT:
                    a = magnitude * (1.0 if rank == 0 else neighbor_decay)
                    Zi[t0:t0 + L, n, TEMP] += a * prof
                    Zi[t0:t0 + L, n, HUMID] -= 0.6 * a * prof
                    Zi[t0:t0 + L, n, LIGHT] += 0.5 * a * prof
                else:
                    Zi[t0:t0 + L, n, TEMP] += magnitude * rng.standard_normal(L)
                    Zi[t0:t0 + L, n, HUMID] += magnitude * rng.standard_normal(L)
                    Zi[t0:t0 + L, n, VOLT] -= voltage_drop
                labels[t0:t0 + L, n] = kind
            episodes.append(Episode(kind, t0, L, nodes, magnitude))
            return

    for _ in range(n_events):
        place(EVENT)
    for _ in range(n_failures):
        place(FAILURE)
    return Zi, labels, episodes
