"""Tests for Role 6: injection variants and the statistics helpers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from signoise.config import HUMID, LIGHT, TEMP, VOLT
from signoise.data.neighbors import knn_neighbors
from signoise.injection import EVENT, FAILURE, inject_anomalies
from signoise.injection.inject import quiet_steps

ROOT = Path(__file__).resolve().parents[1]


def _load_stats():
    path = ROOT / "scripts" / "experiments" / "stats.py"
    spec = importlib.util.spec_from_file_location("exp_stats", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp_stats"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((400, 12, 4))
    coords = np.stack([np.arange(12.0), np.zeros(12)], 1)
    return Z, knn_neighbors(coords, k=3)


HARD = dict(n_events=0, n_failures=0)


@pytest.mark.parametrize("kw", [
    dict(n_lagged_events=4, **HARD),
    dict(n_stuck=4, **HARD),
    dict(n_drift=4, **HARD),
    dict(n_events=2, n_failures=2, n_lagged_events=2, n_stuck=2, n_drift=2),
    dict(n_events=3, n_failures=3, quiet_only=True),
])
def test_labels_match_changed_cells(setup, kw):
    Z, nb = setup
    Zi, labels, eps = inject_anomalies(Z, nb, seed=1, **kw)
    changed = np.any(Zi != Z, axis=2)
    assert labels.any()
    assert np.array_equal(changed, labels > 0)
    # every episode's own cells carry its label
    for e in eps:
        lags = e.lags or [0] * len(e.nodes)
        for n, d in zip(e.nodes, lags):
            assert (labels[e.t0 + d:e.t0 + d + e.length, n] == e.kind).all()


def test_lagged_event_offsets(setup):
    Z, nb = setup
    _, labels, eps = inject_anomalies(Z, nb, seed=2, max_lag=3, n_lagged_events=3, **HARD)
    for e in eps:
        assert e.variant == "lagged_event" and e.kind == EVENT
        assert e.lags[0] == 0 and all(1 <= d <= 3 for d in e.lags[1:])
        assert len(e.nodes) == 4
        # each neighbour's labelled block starts exactly at t0 + lag
        for n, d in zip(e.nodes[1:], e.lags[1:]):
            assert labels[e.t0 + d, n] == EVENT
            assert labels[e.t0 + d + e.length - 1, n] == EVENT


def test_stuck_freezes_reading(setup):
    Z, nb = setup
    Zi, _, eps = inject_anomalies(Z, nb, seed=3, n_stuck=3, **HARD)
    for e in eps:
        n, s = e.nodes[0], slice(e.t0, e.t0 + e.length)
        assert e.kind == FAILURE and len(e.nodes) == 1
        for c in (TEMP, HUMID, LIGHT):
            assert np.all(Zi[s, n, c] == Z[max(e.t0 - 1, 0), n, c])
        assert np.allclose(Zi[s, n, VOLT], Z[s, n, VOLT] - 2.0)


def test_drift_is_linear_and_reaches_magnitude(setup):
    Z, nb = setup
    Zi, _, eps = inject_anomalies(Z, nb, seed=4, n_drift=3, magnitude=3.0, **HARD)
    for e in eps:
        n, s = e.nodes[0], slice(e.t0, e.t0 + e.length)
        d = Zi[s, n, TEMP] - Z[s, n, TEMP]
        assert np.isclose(abs(d[-1]), 3.0)
        assert np.allclose(np.diff(d), d[-1] / e.length)
        assert np.array_equal(Zi[s, n, HUMID], Z[s, n, HUMID])


def test_quiet_only_picks_calm_windows():
    rng = np.random.default_rng(5)
    T, N = 600, 10
    Z = 0.1 * rng.standard_normal((T, N, 4))
    Z[300:, :, :3] += 3.0 * rng.standard_normal((300, N, 3))   # second half is noisy
    coords = np.stack([np.arange(N, dtype=float), np.zeros(N)], 1)
    nb = knn_neighbors(coords, k=3)
    calm = quiet_steps(Z, quantile=0.5, window=5)
    assert calm[:290].mean() > 0.95 and calm[310:].mean() < 0.05
    _, labels, eps = inject_anomalies(Z, nb, n_events=5, n_failures=5, seed=0,
                                      quiet_only=True, quiet_window=5)
    assert len(eps) == 10
    assert all(e.t0 + e.length <= 300 for e in eps)


def test_quiet_only_respects_observed_mask():
    rng = np.random.default_rng(6)
    T, N = 500, 10
    Z = rng.standard_normal((T, N, 4))
    valid = np.ones((T, N), dtype=bool)
    valid[:, :5] = False                     # nodes 0-4 never observed
    nb = knn_neighbors(np.stack([np.arange(N, dtype=float), np.zeros(N)], 1), k=2)
    _, _, eps = inject_anomalies(Z, nb, n_events=0, n_failures=8, seed=0,
                                 quiet_only=True, valid=valid, quiet_quantile=1.0)
    assert eps and all(e.nodes[0] >= 5 for e in eps)


def test_valid_ignored_without_quiet_only(setup):
    Z, nb = setup
    a = inject_anomalies(Z, nb, seed=7)
    b = inject_anomalies(Z, nb, seed=7, valid=np.zeros(Z.shape[:2], dtype=bool))
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[0], b[0])


# ---- statistics helpers ---------------------------------------------------

def test_mean_ci_matches_t_interval():
    st = _load_stats()
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m, lo, hi = st.mean_ci(x)
    # t_{0.975, 4} = 2.776; s = 1.5811; half = 2.776 * 1.5811 / sqrt(5)
    assert m == 3.0 and np.isclose(hi - m, 2.7764 * 1.5811 / np.sqrt(5), atol=1e-3)
    assert np.isclose(m - lo, hi - m)


def test_paired_tests():
    st = _load_stats()
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.02, 25)
    b = a - 0.05 + rng.normal(0, 0.01, 25)    # clearly worse
    res = st.paired_comparison(a, b, n_perm=2000, seed=0)
    assert res["mean_diff"] > 0.04 and res["p_t"] < 1e-6 and res["p_perm"] < 0.01
    same = st.paired_comparison(a, a.copy(), n_perm=500)
    assert same["mean_diff"] == 0 and same["p_perm"] == 1.0
