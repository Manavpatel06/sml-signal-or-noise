import numpy as np
import pytest

from signoise.config import HUMID, TEMP, VOLT
from signoise.discrimination import (FEATURE_NAMES, GaussianNaiveBayes,
                                     LogisticRegression, build_feature_matrix)
from signoise.discrimination.features import (battery_level_and_slope,
                                              cross_channel_consistency,
                                              fit_voltage_temperature,
                                              neighbour_agreement)


# ---------------------------------------------------------------- classifiers
def test_naive_bayes_matches_hand_computation():
    # one feature; class 1: {0, 2} (mu=1, var=1), class 2: {4, 6} (mu=5, var=1)
    X = np.array([[0.0], [2.0], [4.0], [6.0]])
    y = np.array([1, 1, 2, 2])
    nb = GaussianNaiveBayes(var_smoothing=0.0).fit(X, y)
    assert np.allclose(nb.theta_.ravel(), [1, 5])
    assert np.allclose(nb.var_.ravel(), [1, 1])
    # at x=2: likelihoods exp(-0.5*1) vs exp(-0.5*9); equal priors
    l1, l2 = np.exp(-0.5), np.exp(-4.5)
    p = nb.predict_proba(np.array([[2.0]]))[0]
    assert np.allclose(p, [l1 / (l1 + l2), l2 / (l1 + l2)])
    assert nb.predict(np.array([[0.5], [5.5]])).tolist() == [1, 2]


def test_naive_bayes_uses_priors():
    X = np.array([[0.0], [0.0], [0.0], [2.0]])
    y = np.array([1, 1, 1, 2])
    nb = GaussianNaiveBayes(var_smoothing=1.0).fit(X, y)
    assert np.allclose(nb.class_prior_, [0.75, 0.25])
    assert np.allclose(nb.predict_proba(X).sum(axis=1), 1.0)


def _separable(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = np.where(X @ np.array([2.0, -1.0, 0.0]) + 0.3 > 0, 2, 1)
    return X, y


def test_logistic_regression_separates_and_loss_decreases():
    X, y = _separable()
    lr = LogisticRegression().fit(X, y)
    assert (lr.predict(X) == y).mean() > 0.97
    assert np.all(np.diff(lr.loss_curve_) <= 1e-12)
    assert lr.loss_curve_[-1] < lr.loss_curve_[0]
    p = lr.predict_proba(X)
    assert p.shape == (len(X), 2) and np.allclose(p.sum(axis=1), 1)


def test_logistic_regression_handles_unscaled_features_and_labels():
    X, y = _separable()
    X = X * np.array([1000.0, 1e-3, 1.0])
    lr = LogisticRegression().fit(X, y)
    assert (lr.predict(X) == y).mean() > 0.97
    assert set(lr.classes_) == {1, 2}


def test_logistic_regression_needs_two_classes():
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.zeros((4, 2)), np.ones(4))


# ------------------------------------------------------------------- features
def test_neighbour_agreement_event_vs_failure():
    T, N = 30, 6
    neighbors = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
    scores = np.zeros((T, N))
    scores[10:14, [0, 1, 2, 3]] = 5.0     # event over a 4-node group
    scores[20:23, 4] = 5.0                # failure confined to node 4
    assert neighbour_agreement(scores, neighbors, 11, 0, threshold=3.0) == pytest.approx(1.0)
    assert neighbour_agreement(scores, neighbors, 21, 4, threshold=3.0) == pytest.approx(0.0)
    # window: neighbours fire 2 steps later, still counted
    assert neighbour_agreement(scores, neighbors, 8, 0, 3.0, window=2) == pytest.approx(1.0)
    assert neighbour_agreement(scores, neighbors, 5, 0, 3.0, window=2) == 0.0
    # NaN scores never count
    scores[10:14, 1] = np.nan
    assert neighbour_agreement(scores, neighbors, 11, 0, 3.0) == pytest.approx(2 / 3)


def test_battery_level_and_slope():
    volts = np.full((40, 2), np.nan)
    volts[:, 0] = 2.8 - 0.01 * np.arange(40)
    volts[5:9, 0] = np.nan                     # gaps are ignored
    level, slope = battery_level_and_slope(volts, 30, 0, window=12)
    assert level == pytest.approx(2.8 - 0.3)
    assert slope == pytest.approx(-0.01)
    assert battery_level_and_slope(volts, 30, 1)[1] == 0.0   # no data -> slope 0
    # missing reading at t falls back to last observed one
    volts[30, 0] = np.nan
    assert battery_level_and_slope(volts, 30, 0)[0] == pytest.approx(2.8 - 0.29)


def test_fit_voltage_temperature_recovers_line_and_ignores_outliers():
    rng = np.random.default_rng(1)
    temp = rng.uniform(15, 30, size=(300, 3))
    volt = 2.5 + 0.004 * temp
    volt[0, 0], temp[0, 0] = 0.5, 120.0        # dying-sensor outlier
    temp[:, 2] = np.nan                         # node 2 has no temperature
    a, b = fit_voltage_temperature(volt, temp)
    assert np.allclose(a[:2], 2.5, atol=1e-6) and np.allclose(b[:2], 0.004, atol=1e-6)
    assert b[2] == 0.0 and np.isfinite(a[2])


def test_cross_channel_consistency():
    t = np.arange(20, dtype=float)
    temp = np.tile(np.sin(t)[:, None], (1, 2))
    humid = -0.6 * temp                          # moves oppositely
    assert cross_channel_consistency(temp, humid, 15, 0) == pytest.approx(-1.0)
    assert cross_channel_consistency(temp, 2 * temp, 15, 1) == pytest.approx(1.0)
    flat = np.ones((20, 2))
    assert cross_channel_consistency(temp, flat, 15, 0) == 0.0     # zero variance
    assert cross_channel_consistency(temp, humid, 1, 0) == 0.0     # too little history


def test_build_feature_matrix_event_vs_failure():
    """Hand-built tensor: an event on nodes 0-2 and a failure on node 4."""
    T, N = 80, 5
    t = np.arange(T)
    raw = np.zeros((T, N, 4))
    raw[:, :, TEMP] = 22.0
    raw[:, :, HUMID] = 40.0
    raw[:, :, VOLT] = 2.7
    rng = np.random.default_rng(0)
    scores = np.zeros((T, N))
    ev = np.sin(np.linspace(0, np.pi, 10))
    for n in (0, 1, 2):                                   # coherent event
        raw[40:50, n, TEMP] += 5 * ev
        raw[40:50, n, HUMID] -= 3 * ev
        scores[40:50, n] = 5.0
    raw[60:70, 4, TEMP] += 5 * rng.standard_normal(10)    # incoherent failure
    raw[60:70, 4, HUMID] += 5 * rng.standard_normal(10)
    raw[60:70, 4, VOLT] = 2.0
    scores[60:70, 4] = 5.0
    neighbors = np.array([[1, 2], [0, 2], [0, 1], [0, 1], [3, 0]])
    fit = fit_voltage_temperature(raw[:30, :, VOLT], raw[:30, :, TEMP])
    F, names = build_feature_matrix(np.array([[45, 0], [65, 4]]), raw, scores,
                                    neighbors, fit, threshold=3.0)
    assert names == FEATURE_NAMES and F.shape == (2, len(FEATURE_NAMES))
    agree = names.index("neighbour_agreement")
    assert F[0, agree] == pytest.approx(1.0) and F[1, agree] == 0.0
    # event: temp/humidity strongly anti-correlated; failure: voltage collapsed
    assert F[0, names.index("cross_channel_consistency")] < -0.9
    assert F[1, names.index("battery_level")] == pytest.approx(2.0)
    assert F[1, names.index("residual_voltage")] < -0.5
    assert np.isfinite(F).all()
