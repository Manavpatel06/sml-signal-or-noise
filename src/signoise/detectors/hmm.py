"""HMM likelihood anomaly score.

OWNER: Role 3 (Karthik).

Model: S hidden states, Gaussian emissions with diagonal covariance,
    z_1 ~ pi,  z_t | z_{t-1} ~ A[z_{t-1}, :],  x_t | z_t = s ~ N(mu_s, diag(var_s)).
Each node's channel sequence x_{1:T} in R^C is one observation sequence.
shared=True trains ONE model on all nodes (sufficient statistics summed over
nodes); shared=False (default) trains a separate model per node.

Why per-node is the default: nodes differ in their normal behaviour (a node
by a window sees large light and temperature swings every day). A shared
model treats those swings as surprising for every node, so the noisiest
nodes fill the top of the ranking. On the bench (data before 20 March,
5 folds, S = 4) per-node models gave AUC 0.86 vs 0.80 shared and precision@k
0.30 vs 0.09. They take ~17 s per fold to train vs ~1 s.

Training: Baum-Welch (EM) with the scaled forward-backward recursion.
Scaling: alpha_t is renormalised to sum to 1 at every step and the
normaliser c_t = p(x_t | x_{1:t-1}) is kept, so
    log p(x_{1:T}) = sum_t log c_t
without underflow. Emission likelihoods are computed in log space and shifted
by their per-step maximum before exponentiating; the shift is added back to
log c_t.

Score at step t for node n = -log p(x_t | x_{1:t-1}) = -log c_t, the negative
one-step predictive log-likelihood from the forward pass. Readings that the
model finds surprising given recent history score high.

Own implementation of forward, backward and Baum-Welch (no hmmlearn).
Loops run over time steps and EM iterations; nodes and states are vectorised.
"""

from __future__ import annotations

import numpy as np

from signoise.detectors.base import Detector

LOG_2PI = np.log(2.0 * np.pi)


def _kmeans_init(P: np.ndarray, k: int, rng: np.random.Generator,
                 n_iter: int = 10, max_points: int = 20000) -> np.ndarray:
    """Quick Lloyd's k-means on (a subsample of) points P (n, C) -> (k, C) means."""
    if len(P) > max_points:
        P = P[rng.choice(len(P), max_points, replace=False)]
    centres = P[rng.choice(len(P), k, replace=False)].copy()
    for _ in range(n_iter):
        d2 = ((P[:, None, :] - centres[None]) ** 2).sum(axis=2)  # (n, k)
        assign = d2.argmin(axis=1)
        for j in range(k):
            members = P[assign == j]
            if len(members):
                centres[j] = members.mean(axis=0)
    return centres


class GaussianHMM:
    """Diagonal-covariance Gaussian HMM trained on a batch of equal-length
    sequences X of shape (T, B, C): T steps, B sequences, C channels."""

    def __init__(self, n_states: int = 4, n_iter: int = 30, tol: float = 1e-4,
                 self_prob: float = 0.9, var_floor: float = 1e-3, seed: int = 0):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.self_prob = self_prob
        self.var_floor = var_floor
        self.seed = seed

    # ---- parameters -----------------------------------------------------
    def _init_params(self, X: np.ndarray) -> None:
        S, C = self.n_states, X.shape[2]
        rng = np.random.default_rng(self.seed)
        P = X.reshape(-1, C)
        self.means_ = _kmeans_init(P, S, rng)
        self.vars_ = np.tile(np.maximum(P.var(axis=0), self.var_floor), (S, 1))
        self.startprob_ = np.full(S, 1.0 / S)
        if S == 1:
            self.transmat_ = np.ones((1, 1))
        else:
            # uniform transitions plus a bias towards staying in the same state
            A = np.full((S, S), (1.0 - self.self_prob) / (S - 1))
            np.fill_diagonal(A, self.self_prob)
            self.transmat_ = A

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """log N(x_tb | mu_s, diag var_s) for every step, sequence and state: (T, B, S)."""
        diff2 = (X[:, :, None, :] - self.means_) ** 2           # (T, B, S, C)
        return -0.5 * (diff2 / self.vars_ + np.log(self.vars_) + LOG_2PI).sum(axis=3)

    # ---- forward / backward --------------------------------------------
    def _forward(self, logB: np.ndarray, startprob=None, transmat=None):
        """Scaled forward pass (uses the fitted parameters unless given).

        Returns alpha (T, B, S) with each alpha_t summing to 1, the scaled
        emissions Bs = exp(logB - shift) (T, B, S), and log c (T, B) where
        c_t = p(x_t | x_{1:t-1}).
        """
        pi = self.startprob_ if startprob is None else startprob
        A = self.transmat_ if transmat is None else transmat
        T, Bn, S = logB.shape
        shift = logB.max(axis=2, keepdims=True)
        Bs = np.exp(logB - shift)
        alpha = np.empty((T, Bn, S))
        log_c = np.empty((T, Bn))
        a = pi * Bs[0]
        for t in range(T):
            if t > 0:
                a = (alpha[t - 1] @ A) * Bs[t]
            c = a.sum(axis=1, keepdims=True)
            alpha[t] = a / c
            log_c[t] = np.log(c[:, 0])
        log_c += shift[:, :, 0]
        return alpha, Bs, log_c

    def _backward(self, Bs: np.ndarray, log_c: np.ndarray, logB_shift: np.ndarray):
        """Scaled backward pass using the same normalisers as the forward pass.

        beta_t = A (Bs_{t+1} * beta_{t+1}) / c'_{t+1}, where c' is the
        normaliser for the shifted emissions (c' = c * exp(-shift)).
        """
        T, Bn, S = Bs.shape
        c_scaled = np.exp(log_c - logB_shift)          # (T, B)
        beta = np.empty((T, Bn, S))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = ((Bs[t + 1] * beta[t + 1]) @ self.transmat_.T) / c_scaled[t + 1, :, None]
        return beta, c_scaled

    # ---- training -------------------------------------------------------
    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """Baum-Welch on X (T, B, C). Stores log-likelihood per iteration in
        ``loglik_`` (computed with the parameters before each M-step)."""
        T = X.shape[0]
        self._init_params(X)
        self.loglik_ = []
        for _ in range(self.n_iter):
            logB = self._log_emission(X)
            shift = logB.max(axis=2)
            alpha, Bs, log_c = self._forward(logB)
            beta, c_scaled = self._backward(Bs, log_c, shift)
            ll = float(log_c.sum())
            self.loglik_.append(ll)

            # E-step: state posteriors gamma_t(s) and summed pair posteriors xi
            gamma = alpha * beta
            gamma /= gamma.sum(axis=2, keepdims=True)                  # (T, B, S)
            xi = np.zeros_like(self.transmat_)
            for t in range(T - 1):
                nxt = Bs[t + 1] * beta[t + 1] / c_scaled[t + 1, :, None]  # (B, S)
                xi += alpha[t].T @ nxt
            xi *= self.transmat_

            # M-step (statistics summed over all sequences in the batch)
            self.startprob_ = gamma[0].mean(axis=0)
            self.transmat_ = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
            w = gamma.reshape(-1, self.n_states)                       # (T*B, S)
            P = X.reshape(-1, X.shape[2])                              # (T*B, C)
            nk = np.maximum(w.sum(axis=0), 1e-10)[:, None]             # (S, 1)
            self.means_ = (w.T @ P) / nk
            ex2 = (w.T @ P**2) / nk
            self.vars_ = np.maximum(ex2 - self.means_**2, self.var_floor)

            if len(self.loglik_) > 1 and abs(ll - self.loglik_[-2]) < self.tol * abs(ll):
                break
        return self

    def predictive_logprob(self, X: np.ndarray, prob_floor: float = 1e-6) -> np.ndarray:
        """log p(x_t | x_{1:t-1}) for each step and sequence, (T, B).

        EM can drive start or transition probabilities to exactly 0 (e.g. a
        per-node model sees a single start state). A test sequence that needs
        such a move would get log 0 = -inf, so probabilities are floored at
        ``prob_floor`` and renormalised here, at scoring time only.
        """
        pi = np.maximum(self.startprob_, prob_floor)
        A = np.maximum(self.transmat_, prob_floor)
        _, _, log_c = self._forward(self._log_emission(X), pi / pi.sum(),
                                    A / A.sum(axis=1, keepdims=True))
        return log_c

    def loglik(self, X: np.ndarray) -> np.ndarray:
        """Total log-likelihood of each sequence, (B,)."""
        return self.predictive_logprob(X).sum(axis=0)


class HMMDetector(Detector):
    name = "hmm"

    def __init__(self, n_states: int = 4, n_iter: int = 30, shared: bool = False,
                 seed: int = 0, var_floor: float = 1e-3, self_prob: float = 0.9):
        self.n_states, self.n_iter, self.shared, self.seed = n_states, n_iter, shared, seed
        self.var_floor, self.self_prob = var_floor, self_prob

    def _new_model(self) -> GaussianHMM:
        return GaussianHMM(self.n_states, self.n_iter, var_floor=self.var_floor,
                           self_prob=self.self_prob, seed=self.seed)

    def fit(self, X: np.ndarray) -> "HMMDetector":
        """X: (T, N, C). Each node's (T, C) series is one sequence."""
        if self.shared:
            self.models_ = [self._new_model().fit(X)]
        else:
            self.models_ = [self._new_model().fit(X[:, n:n + 1]) for n in range(X.shape[1])]
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """-log p(x_t | x_{1:t-1}) per node, (T, N)."""
        if self.shared:
            return -self.models_[0].predictive_logprob(X)
        return np.concatenate(
            [-m.predictive_logprob(X[:, n:n + 1]) for n, m in enumerate(self.models_)], axis=1)
