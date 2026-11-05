"""Stage C features for each flagged anomaly (proposal Sec. 4.3).

OWNER: Role 5 (Tejas).

An anomaly is a (t, n) cell flagged by a Stage A detector. For each one,
compute a feature vector from information the detectors ignore:

  1. neighbour_agreement  - do spatial neighbours deviate at the same step?
     Fraction of neighbours whose detector score exceeds a threshold in
     [t - w, t + w]. Events -> high, failures -> low.
  2. battery_level, battery_slope - the node's voltage at t and its least-
     squares slope over the previous W steps (raw volts, not z-scores).
  3. residual_voltage - regress voltage on the node's own temperature over the
     training window (voltage depends on temperature, per the dataset docs);
     feature = observed minus predicted voltage at t.
  4. cross_channel_consistency - correlation between the temperature and
     humidity changes for the node over [t - w, t]. Real events move them
     together (negatively); failures look uncorrelated.

The same builder is used for the random-neighbour control: pass the output of
``data.neighbors.random_neighbors`` as ``neighbors``.

Output: build_feature_matrix returns (n_anomalies, d) floats plus the
feature names, ready for the classifiers in classifiers.py.
"""

from __future__ import annotations

import numpy as np

from signoise.config import HUMID, TEMP, VALID_RANGE, VOLT

FEATURE_NAMES = [
    "neighbour_agreement",
    "battery_level",
    "battery_slope",
    "residual_voltage",
    "cross_channel_consistency",
]


def neighbour_agreement(scores: np.ndarray, neighbors: np.ndarray, t: int, n: int,
                        threshold: float, window: int = 2) -> float:
    """Fraction of node n's neighbours with score > threshold at any step in
    [t-window, t+window]. scores: (T, N); neighbors: (N, k). NaN scores never
    count as exceeding the threshold."""
    lo, hi = max(0, t - window), min(scores.shape[0], t + window + 1)
    nb = np.asarray(neighbors[n])
    if nb.size == 0:
        return 0.0
    block = scores[lo:hi][:, nb]                    # (steps, k)
    with np.errstate(invalid="ignore"):
        hit = (block > threshold).any(axis=0)       # (k,)
    return float(hit.mean())


def battery_level_and_slope(volts: np.ndarray, t: int, n: int, window: int = 12):
    """Return (voltage at t, least-squares slope in V/step over [t-window, t-1]).

    volts: (T, N) raw volts. Only observed (finite) steps are used. If the
    reading at t is missing, the most recent observed value in the window is
    used (NaN if there is none). The slope is 0.0 with fewer than two
    observed points.
    """
    level = volts[t, n]
    lo = max(0, t - window)
    past = volts[lo:t, n]
    steps = np.arange(lo, t, dtype=float)
    ok = np.isfinite(past)
    if not np.isfinite(level):
        level = past[ok][-1] if ok.any() else np.nan
    if ok.sum() < 2:
        return float(level), 0.0
    x, y = steps[ok], past[ok]
    xc = x - x.mean()
    denom = float(xc @ xc)
    slope = float(xc @ (y - y.mean()) / denom) if denom > 0 else 0.0
    return float(level), slope


def fit_voltage_temperature(volts_train: np.ndarray, temp_train: np.ndarray):
    """Per-node least-squares fit volt = a_n + b_n * temp. Returns (a, b), each (N,).

    Inputs are (T, N) raw values from the TRAINING window. Only steps where
    both readings are finite and physically plausible (config.VALID_RANGE) are
    used, so dying-sensor outliers do not distort the line. A node with fewer
    than 3 usable steps or no temperature variation gets b = 0 and a = the
    mean voltage (or the pooled mean if it has none).
    """
    vlo, vhi = VALID_RANGE["voltage"]
    tlo, thi = VALID_RANGE["temperature"]
    ok = (np.isfinite(volts_train) & np.isfinite(temp_train)
          & (volts_train >= vlo) & (volts_train <= vhi)
          & (temp_train >= tlo) & (temp_train <= thi))
    cnt = ok.sum(axis=0).astype(float)
    safe = np.maximum(cnt, 1.0)
    V = np.where(ok, volts_train, 0.0)
    Tm = np.where(ok, temp_train, 0.0)
    mv, mt = V.sum(0) / safe, Tm.sum(0) / safe
    dv, dt = np.where(ok, V - mv, 0.0), np.where(ok, Tm - mt, 0.0)
    var_t = (dt ** 2).sum(0)
    cov = (dt * dv).sum(0)
    good = (cnt >= 3) & (var_t > 1e-9)
    b = np.where(good, cov / np.where(good, var_t, 1.0), 0.0)
    pooled = V.sum() / max(ok.sum(), 1)
    a = np.where(cnt > 0, mv - b * mt, pooled)
    return a, b


def cross_channel_consistency(temp: np.ndarray, humid: np.ndarray, t: int, n: int,
                              window: int = 6) -> float:
    """Pearson correlation of first differences of temperature and humidity for
    node n over [t-window, t]. Returns 0.0 when undefined (fewer than 3 usable
    pairs, or zero variance)."""
    lo = max(0, t - window)
    dT = np.diff(temp[lo:t + 1, n])
    dH = np.diff(humid[lo:t + 1, n])
    ok = np.isfinite(dT) & np.isfinite(dH)
    if ok.sum() < 3:
        return 0.0
    a, b = dT[ok] - dT[ok].mean(), dH[ok] - dH[ok].mean()
    denom = np.sqrt((a @ a) * (b @ b))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(a @ b / denom, -1.0, 1.0))


def build_feature_matrix(cells: np.ndarray, raw_values: np.ndarray, scores: np.ndarray,
                         neighbors: np.ndarray, voltage_fit, threshold: float,
                         agreement_window: int = 2, battery_window: int = 12,
                         consistency_window: int = 6):
    """cells: (M, 2) array of flagged (t, n). Returns (F (M, d), FEATURE_NAMES).

    raw_values: (T, N, 4) raw (unstandardised, imputed) temperature, humidity,
    light, voltage; scores: (T, N) detector scores on the same time axis;
    voltage_fit: (a, b) from fit_voltage_temperature on the training window.
    Any feature still NaN (e.g. no voltage history at all) is set to the
    column median of the finite values, or 0 if the column has none.
    """
    cells = np.asarray(cells, dtype=int).reshape(-1, 2)
    volts, temp, humid = raw_values[:, :, VOLT], raw_values[:, :, TEMP], raw_values[:, :, HUMID]
    a, b = voltage_fit
    F = np.zeros((len(cells), len(FEATURE_NAMES)))
    for i, (t, n) in enumerate(cells):
        level, slope = battery_level_and_slope(volts, t, n, battery_window)
        F[i, 0] = neighbour_agreement(scores, neighbors, t, n, threshold, agreement_window)
        F[i, 1] = level
        F[i, 2] = slope
        F[i, 3] = level - (a[n] + b[n] * temp[t, n]) if np.isfinite(temp[t, n]) else np.nan
        F[i, 4] = cross_channel_consistency(temp, humid, t, n, consistency_window)
    for j in range(F.shape[1]):
        bad = ~np.isfinite(F[:, j])
        if bad.any():
            good = F[~bad, j]
            F[bad, j] = np.median(good) if good.size else 0.0
    return F, list(FEATURE_NAMES)
