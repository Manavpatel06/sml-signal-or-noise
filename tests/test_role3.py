"""Unit tests for the Role 3 detectors: PCA reconstruction and Gaussian HMM."""

import itertools

import numpy as np
import pytest

from signoise.detectors.hmm import GaussianHMM, HMMDetector
from signoise.detectors.pca import PCADetector


# ---------------------------------------------------------------- PCA

def _rank_k_tensor(T=200, N=6, C=3, k=2, seed=0):
    """(T, N, C) tensor whose (T, N*C) matrix has rank exactly k (plus a mean)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((T, k)) @ rng.standard_normal((k, N * C)) + rng.standard_normal(N * C)
    return M.reshape(T, N, C)


def test_pca_reconstruction_exact_on_rank_k_data():
    X = _rank_k_tensor(k=2)
    det = PCADetector(n_components=2, normalize=False).fit(X)
    assert np.allclose(det.reconstruct(X), X, atol=1e-8)
    assert det.score(X).max() < 1e-12


def test_pca_variance_rule_finds_true_rank():
    X = _rank_k_tensor(k=3)
    det = PCADetector(n_components=None, var_explained=0.999).fit(X)
    assert det.n_components_ == 3
    assert det.components_.shape == (3, X.shape[1] * X.shape[2])
    assert det.explained_variance_ratio_[:3].sum() == pytest.approx(1.0)


def test_pca_components_are_orthonormal():
    rng = np.random.default_rng(1)
    det = PCADetector(n_components=4).fit(rng.standard_normal((300, 5, 3)))
    V = det.components_
    assert np.allclose(V @ V.T, np.eye(4), atol=1e-10)


def test_pca_flags_single_node_break_not_shared_shift():
    """A shift shared by all nodes lies in the top component and is explained;
    the same shift on one node only is not."""
    rng = np.random.default_rng(2)
    T, N, C = 400, 8, 2
    common = rng.standard_normal((T, 1, C)) * 2.0
    X = common + 0.1 * rng.standard_normal((T, N, C))
    det = PCADetector(n_components=2).fit(X)
    Y = X[:50].copy()
    Y[10:20] += 3.0             # whole building moves together
    Y[30:40, 4] += 3.0          # one node breaks away
    s = det.score(Y)
    assert s[30:40, 4].min() > 10 * np.median(s[10:20])


def test_pca_rejects_wrong_node_count():
    det = PCADetector(n_components=1).fit(np.random.default_rng(0).standard_normal((50, 4, 3)))
    with pytest.raises(ValueError):
        det.score(np.zeros((10, 5, 3)))


# ---------------------------------------------------------------- HMM

def _two_state_sequences(T=600, B=4, seed=0, p_stay=0.95):
    """Sample B sequences from a 2-state HMM with well-separated means."""
    rng = np.random.default_rng(seed)
    means = np.array([[-4.0, 0.0], [4.0, 3.0]])
    X = np.empty((T, B, 2))
    z = rng.integers(0, 2, B)
    for t in range(T):
        flip = rng.random(B) > p_stay
        z = np.where(flip, 1 - z, z)
        X[t] = means[z] + 0.5 * rng.standard_normal((B, 2))
    return X, means


def test_hmm_forward_matches_brute_force():
    """Scaled forward log-likelihood equals the sum over every state path."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 2, 2))
    m = GaussianHMM(n_states=3, seed=0)
    m._init_params(X)
    m.means_ = rng.standard_normal((3, 2))
    m.vars_ = rng.uniform(0.5, 2.0, (3, 2))
    m.startprob_ = rng.dirichlet(np.ones(3))
    m.transmat_ = rng.dirichlet(np.ones(3), size=3)
    logB = m._log_emission(X)
    for b in range(X.shape[1]):
        total = 0.0
        for path in itertools.product(range(3), repeat=X.shape[0]):
            p = m.startprob_[path[0]] * np.exp(logB[0, b, path[0]])
            for t in range(1, X.shape[0]):
                p *= m.transmat_[path[t - 1], path[t]] * np.exp(logB[t, b, path[t]])
            total += p
        assert m.loglik(X)[b] == pytest.approx(np.log(total), rel=1e-10)


def test_hmm_loglik_never_decreases():
    X, _ = _two_state_sequences()
    m = GaussianHMM(n_states=3, n_iter=40, tol=0.0, seed=0).fit(X)
    ll = np.array(m.loglik_)
    assert len(ll) == 40
    assert np.all(np.diff(ll) >= -1e-8 * np.abs(ll[1:]))


def test_hmm_recovers_two_separated_states():
    X, means = _two_state_sequences()
    m = GaussianHMM(n_states=2, n_iter=50, seed=0).fit(X)
    order = np.argsort(m.means_[:, 0])
    assert np.allclose(m.means_[order], means, atol=0.2)
    assert np.allclose(m.vars_, 0.25, atol=0.1)
    assert np.diag(m.transmat_[np.ix_(order, order)]).min() > 0.9


def test_hmm_parameters_are_valid_distributions():
    X, _ = _two_state_sequences(seed=3)
    m = GaussianHMM(n_states=4, n_iter=20, seed=0).fit(X)
    assert m.startprob_.sum() == pytest.approx(1.0)
    assert np.allclose(m.transmat_.sum(axis=1), 1.0)
    assert (m.vars_ >= m.var_floor).all()


def test_hmm_score_finite_on_unseen_transitions():
    """A per-node model can learn exactly-zero start/transition probabilities;
    a test sequence that needs those moves must still get finite scores."""
    rng = np.random.default_rng(0)
    # one switch from regime A to B in training: P(B -> A) and P(start in B)
    # underflow to exactly 0 because the regimes are far apart
    regime = np.where(np.arange(300)[:, None, None] < 150, -40.0, 40.0)
    X = regime + 0.5 * rng.standard_normal((300, 1, 2))
    det = HMMDetector(n_states=2, shared=False).fit(X)
    m = det.models_[0]
    assert (m.transmat_ == 0).any() or (m.startprob_ == 0).any()
    Y = X[::-1][:40].copy()                         # regime order reversed
    with np.errstate(divide="ignore", invalid="ignore"):
        assert not np.isfinite(m.predictive_logprob(Y, prob_floor=0.0)).all()
    assert np.isfinite(det.score(Y)).all()


def test_hmm_flags_outlier_step():
    X, _ = _two_state_sequences(B=3, seed=4)
    for shared in (True, False):
        det = HMMDetector(n_states=2, shared=shared).fit(X)
        Y = X[:100].copy()
        Y[50, 1] += 15.0
        s = det.score(Y)
        assert s.shape == (100, 3)
        assert s[50, 1] == s.max()


def test_hmm_is_reproducible():
    X, _ = _two_state_sequences(B=2, seed=5)
    a = HMMDetector(n_states=3, seed=7).fit(X).score(X)
    b = HMMDetector(n_states=3, seed=7).fit(X).score(X)
    assert np.array_equal(a, b)
