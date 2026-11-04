"""Spatial neighbourhoods between nodes.

All functions return an (N, k) integer array: row i lists the k neighbours of
node i (never i itself), using positions in the SensorData node order.
"""

from __future__ import annotations

import numpy as np


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def knn_neighbors(coords: np.ndarray, k: int = 4) -> np.ndarray:
    """k spatially nearest nodes for every node."""
    d = pairwise_distances(coords)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def radius_neighbors(coords: np.ndarray, radius: float) -> list[np.ndarray]:
    """Variable-size neighbourhoods: every node within ``radius`` metres."""
    d = pairwise_distances(coords)
    np.fill_diagonal(d, np.inf)
    return [np.flatnonzero(row <= radius) for row in d]


def random_neighbors(n_nodes: int, k: int = 4, seed: int = 0) -> np.ndarray:
    """Control: k random non-self nodes per node (same k as the real graph).

    Used to show that *spatial* structure, not just extra features, separates
    events from failures.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_nodes, k), dtype=int)
    for i in range(n_nodes):
        others = np.delete(np.arange(n_nodes), i)
        out[i] = rng.choice(others, size=k, replace=False)
    return out
