"""Small statistics helpers for the results tables (Role 6).

Only the Student-t quantile and tail come from SciPy; everything else is NumPy.

Unit of replication. Repeated temporal CV gives one number per (fold, seed).
Runs that share a fold share the same training and test data and differ only
in where the synthetic anomalies were injected, so they are not independent.
The experiments therefore average over seeds inside each fold first and
compute confidence intervals and paired tests over the fold means (n = number
of folds). Run-level win counts are reported alongside as a descriptive
number.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy import stats as _st


def mean_ci(x, level: float = 0.95):
    """Return (mean, lo, hi): Student-t interval for the mean of x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = float(x.mean())
    if n == 1:
        return m, np.nan, np.nan
    half = _st.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - half, m + half


def sign_flip_pvalue(d, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided paired permutation test of mean(d) = 0 by random sign flips.

    Exact (all 2^n sign patterns) when n <= 16, Monte Carlo otherwise.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return np.nan
    obs = abs(d.mean())
    if n <= 16:
        signs = np.array(list(product((-1.0, 1.0), repeat=n)))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(n_perm, n))
    null = np.abs(signs @ d / n)
    return float(np.mean(null >= obs - 1e-12))


def paired_comparison(a, b, level: float = 0.95, n_perm: int = 10000, seed: int = 0):
    """Compare paired samples a vs b (e.g. detector AUC vs k-sigma AUC per fold).

    Returns dict: mean_diff, ci_lo, ci_hi (t interval of a - b), p_t (paired
    two-sided t-test), p_perm (sign-flip permutation), wins (a > b), n.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    m, lo, hi = mean_ci(d, level)
    sd = d.std(ddof=1) if n > 1 else np.nan
    if n > 1 and sd > 0:
        p_t = float(2 * _st.t.sf(abs(m) / (sd / np.sqrt(n)), n - 1))
    else:
        p_t = 1.0 if (n > 0 and m == 0) else np.nan
    return {"mean_diff": m, "ci_lo": lo, "ci_hi": hi, "p_t": p_t,
            "p_perm": sign_flip_pvalue(d, n_perm, seed), "wins": int((d > 0).sum()),
            "n": n}
