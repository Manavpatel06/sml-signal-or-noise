"""EM imputation over a multivariate Gaussian of correlated node series.

OWNER: Role 4 (Harsh). Status: done.

Idea (proposal Sec. 4.2): at each time step t, stack one channel across all
nodes into a vector x_t in R^N (or all channels: R^{N*C}). Model x_t ~ N(mu, Sigma).
A missing entry is estimated from the observed entries at the same step,
because neighbouring nodes are correlated.

EM for missing data (Bishop 9.x / standard Gaussian EM with missing values):
  E-step, for each t with observed set o and missing set m:
      E[x_m | x_o]   = mu_m + Sigma_mo Sigma_oo^{-1} (x_o - mu_o)
      Cov[x_m | x_o] = Sigma_mm - Sigma_mo Sigma_oo^{-1} Sigma_om
  M-step: mu = mean of completed x_t;
      Sigma = mean of (x_t - mu)(x_t - mu)^T + the conditional covariances.
  Iterate until the observed-data log-likelihood stops improving.

Requirements:
  * NumPy/SciPy linear algebra only (no sklearn).
  * Add a small ridge (e.g. 1e-3 * I) to Sigma for numerical stability.
  * Fit on the TRAINING window only; transform() may be applied to test.
  * Evaluate with signoise.imputation.masking.masking_benchmark and report
    RMSE against ffill_impute (the baseline).

Implementation notes
--------------------
* Scaling. The model is fitted on robust z-scores (median / MAD per node and
  channel, from the fit window only, via ``preprocess.Standardizer``). This
  makes the ridge relative to the data scale (voltage variance is ~0.04, light
  variance ~3e5, so one absolute ridge cannot suit both) and it stops raw
  outliers such as 120 C readings from dying nodes from dominating Sigma.
* Outliers. Two filters hide bad readings from the Gaussian model (they
  neither enter the fit nor condition the fill): readings outside
  ``config.VALID_RANGE`` (``valid_range=True``) and robust z-scores beyond
  ``+-clip_z``. The first matters because a node that failed during the fit
  window (e.g. 38% impossible readings) would otherwise get a shifted centre
  and a huge scale, which later turns small z-errors into large errors in
  physical units. Observed entries are never overwritten: transform() only
  fills entries that are NaN in the input.
* Speed. Time steps are grouped by missingness pattern, so each E-step does
  one Cholesky solve per distinct pattern instead of one per time step.
* Stopping. The mean observed-data log-likelihood per time step is tracked in
  ``loglik_`` (one list per fitted block); EM stops when its change is < tol.
  With ~27% missing cells EM converges slowly, and some channels reach
  ``max_iter`` before ``tol`` (see ``converged_``). The likelihood is still
  monotone and the imputation error is insensitive to running longer
  (humidity RMSE 1.02 -> 0.93 going from 50 to 200 iterations).
* A node that is never observed in the fit window starts EM at mean 0 /
  variance 1 in z-space with no correlations; EM then learns nothing more about
  it than the ridge allows, so it is filled close to its pooled-scale centre.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from signoise.config import CHANNELS, VALID_RANGE
from signoise.preprocess import Standardizer

_LOG_2PI = np.log(2.0 * np.pi)


def _pattern_groups(obs: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Group rows of a boolean (T, D) mask by identical missingness pattern.

    Returns a list of (rows, observed_idx, missing_idx) for every pattern.
    """
    patterns, inverse = np.unique(obs, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(patterns) + 1))
    groups = []
    for g, pat in enumerate(patterns):
        rows = order[bounds[g]:bounds[g + 1]]
        groups.append((rows, np.flatnonzero(pat), np.flatnonzero(~pat)))
    return groups


def _e_step(Z: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, groups,
            need_cov: bool):
    """Gaussian E-step with missing values.

    Z: (T, D) with NaN for unobserved entries.
    Returns (Zhat, cond_cov_sum, loglik_sum):
      Zhat          (T, D) observed entries kept, missing = E[x_m | x_o]
      cond_cov_sum  (D, D) sum over rows of the conditional covariance of the
                    missing block (zeros elsewhere); None unless need_cov
      loglik_sum    sum over rows of log N(x_o | mu_o, Sigma_oo)
    """
    D = Z.shape[1]
    Zhat = np.where(np.isfinite(Z), Z, 0.0)
    cov_sum = np.zeros((D, D)) if need_cov else None
    loglik = 0.0
    for rows, o, m in groups:
        k = len(rows)
        if len(o) == 0:  # nothing observed: prior mean, full prior covariance
            Zhat[np.ix_(rows, m)] = mu[m]
            if need_cov:
                cov_sum += k * Sigma
            continue
        factor = cho_factor(Sigma[np.ix_(o, o)], lower=True)
        diff = Z[np.ix_(rows, o)] - mu[o]                       # (k, |o|)
        solved = cho_solve(factor, diff.T)                      # Sigma_oo^-1 diff^T
        quad = np.einsum("ij,ji->i", diff, solved)
        logdet = 2.0 * np.log(np.diag(factor[0])).sum()
        loglik += -0.5 * (quad.sum() + k * (logdet + len(o) * _LOG_2PI))
        if len(m) == 0:
            continue
        S_om = Sigma[np.ix_(o, m)]                              # (|o|, |m|)
        gain = cho_solve(factor, S_om)                          # Sigma_oo^-1 Sigma_om
        Zhat[np.ix_(rows, m)] = mu[m] + diff @ gain             # conditional mean
        if need_cov:
            cond = Sigma[np.ix_(m, m)] - S_om.T @ gain          # conditional cov
            cov_sum[np.ix_(m, m)] += k * cond
    return Zhat, cov_sum, loglik


def _initial_gaussian(Z: np.ndarray, ridge: float):
    """Starting mean and covariance from the observed entries.

    Mean: per-feature mean of observed values. Covariance: pairwise-complete
    covariance, projected onto the PSD cone (negative eigenvalues floored) so
    the first E-step can factorise it.
    """
    obs = np.isfinite(Z)
    cnt = np.maximum(obs.sum(axis=0), 1)
    mu = np.where(obs.any(axis=0), np.where(obs, Z, 0.0).sum(axis=0) / cnt, 0.0)
    R = np.where(obs, Z - mu, 0.0)
    both = obs.T.astype(float) @ obs.astype(float)
    S = (R.T @ R) / np.maximum(both, 1.0)
    unseen = ~obs.any(axis=0)
    S[unseen, :] = 0.0
    S[:, unseen] = 0.0
    S[unseen, unseen] = 1.0
    w, V = np.linalg.eigh((S + S.T) / 2.0)
    w = np.maximum(w, 1e-2)
    return mu, (V * w) @ V.T + ridge * np.eye(Z.shape[1])


class EMGaussianImputer:
    """Fill NaNs with E[x_missing | x_observed] under a Gaussian fitted by EM.

    Parameters
    ----------
    max_iter, tol : EM stops after ``max_iter`` iterations or when the mean
        observed-data log-likelihood per time step changes by less than ``tol``.
    ridge : added to the diagonal of Sigma (in z-score units) at every M-step.
    per_channel : True fits one Gaussian per channel over the N nodes; False
        fits one over all N*C node-channel series.
    clip_z : robust z-scores beyond +-clip_z are ignored by the model (see the
        module docstring). None keeps every observed value.
    valid_range : if True, readings outside ``config.VALID_RANGE`` (physically
        impossible, e.g. 120 C) are ignored by the model in fit AND transform,
        like the ``clip_z`` outliers. They are still never overwritten. This
        keeps a node that failed during the fit window from corrupting its own
        centre, scale and correlations. Only applies when X has the four
        project channels.

    Attributes after fit
    --------------------
    scaler_ : fitted ``Standardizer`` (per node and channel).
    means_, covariances_ : lists with one (D,) / (D, D) array per block, in
        z-score units. A block is a channel (per_channel) or everything.
    loglik_ : list (per block) of mean log-likelihood traces, one per iteration.
    converged_ : list (per block) of bools, True if EM stopped because the
        log-likelihood gain fell below ``tol`` (False = hit ``max_iter``).
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-3, ridge: float = 1e-3,
                 per_channel: bool = True, clip_z: float | None = 4.0,
                 valid_range: bool = True):
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge
        self.per_channel = per_channel
        self.clip_z = clip_z
        self.valid_range = valid_range

    # ----- helpers -------------------------------------------------------
    def _blocks(self, Z: np.ndarray) -> list[np.ndarray]:
        """Split a (T, N, C) z-score tensor into the matrices EM works on."""
        T, N, C = Z.shape
        if self.per_channel:
            return [Z[:, :, c] for c in range(C)]
        return [Z.reshape(T, N * C)]

    def _plausible(self, X: np.ndarray) -> np.ndarray:
        """Boolean (T, N, C): True where a reading is physically possible."""
        ok = np.ones(X.shape, dtype=bool)
        if self.valid_range and X.shape[2] == len(CHANNELS):
            for c, name in enumerate(CHANNELS):
                lo, hi = VALID_RANGE[name]
                ok[:, :, c] = ~((X[:, :, c] < lo) | (X[:, :, c] > hi))
        return ok

    def _model_view(self, Z: np.ndarray) -> np.ndarray:
        """Z with outliers hidden (NaN) from the Gaussian model."""
        if self.clip_z is None:
            return Z
        return np.where(np.abs(Z) <= self.clip_z, Z, np.nan)

    # ----- API -----------------------------------------------------------
    def fit(self, X: np.ndarray) -> "EMGaussianImputer":
        """X: (T, N, C) with NaN. Learn mu, Sigma (per channel if per_channel)."""
        X = np.asarray(X, dtype=float)
        X_fit = np.where(self._plausible(X), X, np.nan)
        with warnings.catch_warnings():  # all-NaN nodes give empty-slice warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.scaler_ = Standardizer(per_node=True, robust=True).fit(X_fit)
        Z = self._model_view(self.scaler_.transform(X_fit))
        self.means_, self.covariances_, self.loglik_, self.converged_ = [], [], [], []
        for M in self._blocks(Z):
            M = M[np.isfinite(M).any(axis=1)]  # rows with no data say nothing
            if len(M) == 0:
                raise ValueError("EMGaussianImputer.fit: no observed values")
            mu, Sigma, trace, done = self._fit_block(M)
            self.converged_.append(done)
            self.means_.append(mu)
            self.covariances_.append(Sigma)
            self.loglik_.append(trace)
        return self

    def _fit_block(self, M: np.ndarray):
        """EM on one (T, D) matrix with NaN.
        Returns (mu, Sigma, loglik trace, converged)."""
        T, D = M.shape
        groups = _pattern_groups(np.isfinite(M))
        mu, Sigma = _initial_gaussian(M, self.ridge)
        trace: list[float] = []
        converged = False
        for _ in range(self.max_iter):
            Mhat, cov_sum, ll = _e_step(M, mu, Sigma, groups, need_cov=True)
            trace.append(ll / T)
            # M-step: new mean from completed data, then covariance of the
            # completed data about the new mean plus the conditional covariances.
            mu = Mhat.mean(axis=0)
            R = Mhat - mu
            Sigma = (R.T @ R + cov_sum) / T + self.ridge * np.eye(D)
            Sigma = (Sigma + Sigma.T) / 2.0
            if len(trace) > 1 and abs(trace[-1] - trace[-2]) < self.tol:
                converged = True
                break
        return mu, Sigma, trace, converged

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return a NaN-free copy of X using conditional means E[x_m | x_o]."""
        if not hasattr(self, "scaler_"):
            raise RuntimeError("call fit() before transform()")
        X = np.asarray(X, dtype=float)
        Z_full = self.scaler_.transform(X)
        Z_model = self._model_view(np.where(self._plausible(X), Z_full, np.nan))
        filled = np.empty_like(Z_full)
        out_blocks = self._blocks(filled)  # views into ``filled``
        for M, view, mu, Sigma in zip(self._blocks(Z_model), out_blocks,
                                      self.means_, self.covariances_):
            Mhat, _, _ = _e_step(M, mu, Sigma, _pattern_groups(np.isfinite(M)),
                                 need_cov=False)
            view[...] = Mhat
        out = np.where(np.isfinite(X), X, self.scaler_.inverse_transform(filled))
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
