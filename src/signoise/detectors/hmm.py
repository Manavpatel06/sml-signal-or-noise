"""HMM likelihood anomaly score.

OWNER: Role 3 (Karthik). Status: TODO.

Fit a Gaussian-emission HMM (S hidden states, diagonal covariances) on each
node's training sequence, or one shared HMM pooled over nodes. Use
Baum-Welch (forward-backward in log space or with scaling) for training.

Score at step t for node n = -log p(x_t | x_{1:t-1}) under the model, i.e.
the negative log of the one-step predictive likelihood from the forward
recursion. Readings the model finds surprising given recent history score high.

Notes:
  * Own implementation of forward, backward and Baum-Welch (no hmmlearn).
  * Work in log space or use scaling factors to avoid underflow.
  * Start with S = 3-5 states; report sensitivity to S.
"""

from __future__ import annotations

import numpy as np

from signoise.detectors.base import Detector


class HMMDetector(Detector):
    name = "hmm"

    def __init__(self, n_states: int = 4, n_iter: int = 30, shared: bool = True,
                 seed: int = 0):
        self.n_states, self.n_iter, self.shared, self.seed = n_states, n_iter, shared, seed

    def fit(self, X: np.ndarray) -> "HMMDetector":
        raise NotImplementedError("Role 3: Baum-Welch on (T, C) sequences per node")

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Role 3: -log one-step predictive likelihood, (T, N)")
