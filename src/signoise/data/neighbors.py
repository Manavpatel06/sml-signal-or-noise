"""Spatial neighbourhoods between nodes.

All functions return an (N, k) integer array: row i lists the k neighbours of
node i (never i itself), using positions in the SensorData node order.
"""
from __future__ import annotations
import numpy as np

def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def knn_neighbors(coords: np.ndarray, k: int=4) -> np.ndarray:
    """k spatially nearest nodes for every node."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def radius_neighbors(coords: np.ndarray, radius: float) -> list[np.ndarray]:
    """Variable-size neighbourhoods: every node within ``radius`` metres."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def random_neighbors(n_nodes: int, k: int=4, seed: int=0) -> np.ndarray:
    """Control: k random non-self nodes per node (same k as the real graph).

    Used to show that *spatial* structure, not just extra features, separates
    events from failures.
    """
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')
