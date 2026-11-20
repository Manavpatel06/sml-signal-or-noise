"""GMM / EM anomaly score.

OWNER: Role 4 (Harsh). Status: done.

Fit a Gaussian mixture with K components by EM (own implementation) on the
training points. Score = negative log-likelihood -log sum_k pi_k N(x | mu_k, Sigma_k).

Notes:
  * Full covariances with a small ridge (e.g. 1e-6 * I) to avoid singularity.
  * Log-sum-exp in the E-step for stability.
  * Initialise from K-means (Role 2's KMeansDetector) or random points; several
    restarts, keep the best log-likelihood.
  * Pick K by BIC on the training window and report the curve.

EM recap (P is n x d, K components, theta = {pi_k, mu_k, Sigma_k}):
  E-step  r_ik = pi_k N(x_i | mu_k, Sigma_k) / sum_j pi_j N(x_i | mu_j, Sigma_j)
          computed in log space: log r_ik = a_ik - logsumexp_j a_ij.
  M-step  N_k = sum_i r_ik;  pi_k = N_k / n;  mu_k = sum_i r_ik x_i / N_k
          Sigma_k = sum_i r_ik (x_i - mu_k)(x_i - mu_k)^T / N_k + ridge * I
  The mean log-likelihood never decreases (up to the tiny ridge), and EM stops
  when its gain falls below ``tol``.

Fitted attributes: weights_ (K,), means_ (K, d), covariances_ (K, d, d),
lower_bound_ (best mean log-likelihood per point), log_likelihood_trace_
(mean log-likelihood at each EM iteration of the best restart).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from signoise.detectors.base import PointwiseDetector
from signoise.detectors.kmeans import KMeansDetector

_LOG_2PI = np.log(2.0 * np.pi)


class GMMDetector(PointwiseDetector):
    name = "gmm"

    def __init__(self, n_components: int = 5, max_iter: int = 100, tol: float = 1e-4,
                 n_init: int = 3, seed: int = 0, features: str = "channels",
                 ridge: float = 1e-6, max_points: int | None = 20000,
                 clip: float | None = None):
        """
        ridge: added to every covariance diagonal at each M-step.
        max_points: if the training set is larger, fit on a seeded random
            subsample of this size (scoring always uses every point). None = all.
        clip: if set, cap every feature at +-clip before fitting and scoring
            (same idea as KMeansDetector.clip: tames the heavy z-score tails).
        """
        super().__init__(features)
        self.n_components, self.max_iter, self.tol, self.n_init, self.seed = (
            n_components, max_iter, tol, n_init, seed)
        self.ridge, self.max_points, self.clip = ridge, max_points, clip

    # ----- density -------------------------------------------------------
    def _clip(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        return P if self.clip is None else np.clip(P, -self.clip, self.clip)

    @staticmethod
    def _log_component_density(P, means, covs) -> np.ndarray:
        """log N(x_i | mu_k, Sigma_k) for every point and component, (n, K).

        Uses the Cholesky factor Sigma = L L^T: the Mahalanobis term is
        ||L^-1 (x - mu)||^2 and log|Sigma| = 2 sum log diag(L).
        """
        n, d = P.shape
        out = np.empty((n, len(means)))
        for k in range(len(means)):
            L = np.linalg.cholesky(covs[k])
            y = solve_triangular(L, (P - means[k]).T, lower=True)   # (d, n)
            maha = np.einsum("ij,ij->j", y, y)
            logdet = 2.0 * np.log(np.diag(L)).sum()
            out[:, k] = -0.5 * (maha + logdet + d * _LOG_2PI)
        return out

    def _weighted_log_prob(self, P, weights, means, covs) -> np.ndarray:
        """log pi_k + log N(x_i | mu_k, Sigma_k), shape (n, K)."""
        return np.log(weights) + self._log_component_density(P, means, covs)

    # ----- EM ------------------------------------------------------------
    def _m_step(self, P, resp):
        """Closed-form updates of (pi, mu, Sigma) from responsibilities."""
        n, d = P.shape
        nk = resp.sum(axis=0) + 10 * np.finfo(float).eps
        weights = nk / n
        means = (resp.T @ P) / nk[:, None]
        covs = np.empty((len(nk), d, d))
        for k in range(len(nk)):
            R = P - means[k]
            covs[k] = (resp[:, k, None] * R).T @ R / nk[k] + self.ridge * np.eye(d)
        return weights, means, covs, nk

    def _init_params(self, P, rng):
        """Responsibilities from one k-means run (hard assignments)."""
        K = self.n_components
        km = KMeansDetector(n_clusters=K, n_init=1,
                            seed=int(rng.integers(2 ** 31 - 1)))
        km._fit_points(P)
        labels = km._sq_dist(P, km.centroids_).argmin(axis=1)
        resp = np.zeros((len(P), K))
        resp[np.arange(len(P)), labels] = 1.0
        return self._m_step(P, resp)[:3]

    def _reseed_degenerate(self, P, resp_nk, weights, means, covs, logp):
        """Re-seed components with < 1 effective point at the worst-fit points."""
        bad = np.flatnonzero(resp_nk < 1.0)
        if bad.size == 0:
            return weights, means, covs
        global_cov = np.cov(P.T).reshape(P.shape[1], P.shape[1]) \
            + self.ridge * np.eye(P.shape[1])
        worst = np.argsort(logsumexp(logp, axis=1))  # least likely points first
        for j, k in enumerate(bad):
            means[k] = P[worst[j]]
            covs[k] = global_cov
            weights[k] = 1.0 / len(P)
        return weights / weights.sum(), means, covs

    def _run_em(self, P, rng):
        """One EM run from a k-means start. Returns (params, trace)."""
        weights, means, covs = self._init_params(P, rng)
        trace: list[float] = []
        for _ in range(self.max_iter):
            logp = self._weighted_log_prob(P, weights, means, covs)
            lse = logsumexp(logp, axis=1)
            trace.append(float(lse.mean()))
            resp = np.exp(logp - lse[:, None])
            weights, means, covs, nk = self._m_step(P, resp)
            if nk.min() < 1.0:
                weights, means, covs = self._reseed_degenerate(
                    P, nk, weights, means, covs, logp)
                trace.clear()  # re-seeding changes the model; restart the trace
                continue
            if len(trace) > 1 and 0 <= trace[-1] - trace[-2] < self.tol:
                break
        # likelihood of the final parameters
        final = float(logsumexp(self._weighted_log_prob(P, weights, means, covs),
                                axis=1).mean())
        trace.append(final)
        return (weights, means, covs, final), trace

    def _fit_points(self, P: np.ndarray) -> None:
        P = self._clip(P)
        n, d = P.shape
        if n < self.n_components:
            raise ValueError("need at least n_components training points")
        rng = np.random.default_rng(self.seed)
        if self.max_points is not None and n > self.max_points:
            P = P[rng.choice(n, size=self.max_points, replace=False)]
        best = None
        for _ in range(self.n_init):
            params, trace = self._run_em(P, rng)
            if best is None or params[3] > best[0][3]:
                best = (params, trace)
        (self.weights_, self.means_, self.covariances_, self.lower_bound_), \
            self.log_likelihood_trace_ = best
        self.n_features_ = d
        self.n_train_ = len(P)

    # ----- scoring and model selection ------------------------------------
    def _log_likelihood(self, P: np.ndarray) -> np.ndarray:
        """log p(x_i) under the fitted mixture, shape (n,)."""
        logp = self._weighted_log_prob(self._clip(P), self.weights_, self.means_,
                                       self.covariances_)
        return logsumexp(logp, axis=1)

    def _score_points(self, P: np.ndarray) -> np.ndarray:
        """Negative log-likelihood (higher = more anomalous)."""
        return -self._log_likelihood(P)

    def n_parameters(self) -> int:
        """Free parameters: K-1 weights, K*d means, K*d(d+1)/2 covariances."""
        K, d = self.n_components, self.n_features_
        return (K - 1) + K * d + K * d * (d + 1) // 2

    def bic(self, P: np.ndarray) -> float:
        """Bayesian information criterion on P (lower is better):
        BIC = -2 * sum_i log p(x_i) + n_parameters * log(n)."""
        P = np.asarray(P, dtype=float)
        return float(-2.0 * self._log_likelihood(P).sum()
                     + self.n_parameters() * np.log(len(P)))


def bic_curve(P: np.ndarray, ks=range(1, 9), **kwargs) -> dict[int, float]:
    """BIC of a GMM fitted for each K in ``ks`` (kwargs go to GMMDetector).
    The BIC is computed on the points the model was fitted on (P)."""
    out = {}
    for k in ks:
        det = GMMDetector(n_components=k, **kwargs)
        det._fit_points(P)
        out[int(k)] = det.bic(P)
    return out
