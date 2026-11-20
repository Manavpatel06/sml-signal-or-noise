"""Unit tests for Role 4: GMM detector (EM) and EM Gaussian imputation."""
import numpy as np
import pytest

from signoise.detectors.gmm import GMMDetector, bic_curve
from signoise.imputation import EMGaussianImputer, ffill_impute, masking_benchmark


# ---------------------------------------------------------------- GMM
def _two_gaussians(n=600, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal([-5.0, 0.0], [1.0, 0.5], size=(n, 2))
    b = rng.normal([5.0, 3.0], [0.7, 1.0], size=(n, 2))
    return np.vstack([a, b])


def test_gmm_log_likelihood_is_non_decreasing():
    P = _two_gaussians()
    det = GMMDetector(n_components=3, n_init=1, tol=0.0, max_iter=40)
    det._fit_points(P)
    trace = np.array(det.log_likelihood_trace_)
    assert len(trace) > 5
    assert np.all(np.diff(trace) >= -1e-9)


def test_gmm_recovers_means_of_two_separated_gaussians():
    P = _two_gaussians()
    det = GMMDetector(n_components=2, seed=0)
    det._fit_points(P)
    order = np.argsort(det.means_[:, 0])
    assert np.allclose(det.means_[order], [[-5, 0], [5, 3]], atol=0.2)
    assert np.allclose(det.weights_, 0.5, atol=0.05)
    assert np.allclose(det.covariances_[order][0], np.diag([1.0, 0.25]), atol=0.2)


def test_gmm_parameters_are_valid_and_scores_are_nll():
    P = _two_gaussians()
    det = GMMDetector(n_components=2)
    det._fit_points(P)
    assert np.isclose(det.weights_.sum(), 1.0)
    for cov in det.covariances_:
        assert np.allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() > 0
    s = det._score_points(P)
    assert np.isclose(-s.mean(), det.lower_bound_, atol=1e-6)  # mean log-lik
    far = det._score_points(np.array([[40.0, 40.0]]))
    assert far[0] > s.max()


def test_gmm_density_matches_scipy_multivariate_normal():
    from scipy.stats import multivariate_normal  # reference check only
    P = _two_gaussians(100)
    det = GMMDetector(n_components=2)
    det._fit_points(P)
    ref = sum(w * multivariate_normal(m, c).pdf(P)
              for w, m, c in zip(det.weights_, det.means_, det.covariances_))
    assert np.allclose(np.exp(det._log_likelihood(P)), ref)


def test_gmm_is_deterministic_and_handles_extreme_values():
    P = _two_gaussians()
    P[0] = [1e6, -1e6]  # a huge outlier must not produce inf/nan scores
    a = GMMDetector(seed=3)
    b = GMMDetector(seed=3)
    a._fit_points(P)
    b._fit_points(P)
    assert np.allclose(a.means_, b.means_)
    assert np.isfinite(a._score_points(P)).all()


def test_gmm_subsampling_limits_fit_points():
    det = GMMDetector(n_components=2, max_points=300)
    det._fit_points(_two_gaussians())
    assert det.n_train_ == 300


def test_gmm_bic_prefers_true_number_of_components():
    P = _two_gaussians()
    curve = bic_curve(P, ks=[1, 2, 3, 4], seed=0)
    assert min(curve, key=curve.get) == 2


# ---------------------------------------------------------- EM imputer
def _correlated_tensor(T=600, N=8, C=2, seed=0, noise=0.2):
    """Nodes share one slow common factor, so neighbours predict each other."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((T, 1, C)) * 2.0
    load = rng.uniform(0.8, 1.2, size=(1, N, C))
    return 20.0 + common * load + noise * rng.standard_normal((T, N, C))


def _punch_holes(X, frac=0.2, seed=1):
    rng = np.random.default_rng(seed)
    hole = rng.random(X.shape) < frac
    Xm = X.copy()
    Xm[hole] = np.nan
    return Xm, hole


def test_em_imputer_beats_ffill_on_correlated_data_with_random_holes():
    X = _correlated_tensor()
    Xm, hole = _punch_holes(X)
    em = EMGaussianImputer().fit(Xm[:400]).transform(Xm[400:])
    ff = ffill_impute(Xm[400:])
    truth, h = X[400:], hole[400:]
    rmse_em = np.sqrt(np.mean((em[h] - truth[h]) ** 2))
    rmse_ff = np.sqrt(np.mean((ff[h] - truth[h]) ** 2))
    assert rmse_em < 0.5 * rmse_ff


def test_em_imputer_via_masking_benchmark():
    X = _correlated_tensor()
    imp = EMGaussianImputer().fit(X[:400])
    em = masking_benchmark(X[400:], imp.transform, frac=0.1, block=1)
    ff = masking_benchmark(X[400:], ffill_impute, frac=0.1, block=1)
    for c in em:
        assert em[c]["rmse"] < ff[c]["rmse"]


def test_em_imputer_keeps_observed_values_and_fills_everything():
    X = _correlated_tensor()
    Xm, hole = _punch_holes(X, frac=0.3)
    Xm[:, 2, :] = np.nan  # one node completely missing in this window
    out = EMGaussianImputer().fit(Xm[:400]).transform(Xm[400:])
    assert np.isfinite(out).all()
    seen = np.isfinite(Xm[400:])
    assert np.array_equal(out[seen], Xm[400:][seen])


def test_em_imputer_recovers_mean_and_covariance():
    rng = np.random.default_rng(0)
    N = 4
    A = rng.standard_normal((N, N))
    cov = A @ A.T + np.eye(N)
    mu = np.array([1.0, -2.0, 3.0, 0.5])
    X = rng.multivariate_normal(mu, cov, size=4000)[:, :, None]
    Xm, _ = _punch_holes(X, frac=0.3, seed=5)
    imp = EMGaussianImputer(ridge=1e-6, clip_z=None, valid_range=False, max_iter=200,
                            tol=1e-8)
    imp.fit(Xm)
    # undo the (per node) z-scaling to compare with the generating parameters
    s = imp.scaler_.std_[0, :, 0]
    c = imp.scaler_.mean_[0, :, 0]
    mu_hat = imp.means_[0] * s + c
    cov_hat = imp.covariances_[0] * np.outer(s, s)
    assert np.allclose(mu_hat, mu, atol=0.2)
    assert np.allclose(cov_hat, cov, atol=0.5)


def test_em_imputer_log_likelihood_increases():
    X = _correlated_tensor()
    Xm, _ = _punch_holes(X)
    imp = EMGaussianImputer(tol=0.0, max_iter=15, ridge=1e-9).fit(Xm)
    for trace in imp.loglik_:
        assert np.all(np.diff(trace) >= -1e-6)


def test_em_imputer_ignores_impossible_readings():
    X = _correlated_tensor(N=6, C=4)
    X[:, :, 0] += 20.0  # temperature around 40 C, still plausible
    X[:, :, 3] = 2.7 + 0.01 * (X[:, :, 3] - 20.0)  # voltage inside 2.0-3.2 V
    X[:50, 0, 0] = 120.0  # a failing node reports impossible values
    Xm, hole = _punch_holes(X, frac=0.1)
    imp = EMGaussianImputer().fit(Xm)
    out = imp.transform(Xm)
    assert out[:50, 0, 0][np.isfinite(Xm[:50, 0, 0])].min() == 120.0  # kept
    filled = hole[:, :, 0] & (np.arange(len(X))[:, None] >= 50)
    assert np.sqrt(np.mean((out[:, :, 0][filled] - X[:, :, 0][filled]) ** 2)) < 1.0


def test_em_imputer_requires_fit_and_supports_joint_channels():
    X = _correlated_tensor()
    Xm, _ = _punch_holes(X)
    with pytest.raises(RuntimeError):
        EMGaussianImputer().transform(Xm)
    out = EMGaussianImputer(per_channel=False).fit_transform(Xm)
    assert out.shape == X.shape and np.isfinite(out).all()
