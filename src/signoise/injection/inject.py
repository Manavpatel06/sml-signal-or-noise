"""Synthetic anomaly injection with known ground truth (proposal Sec. 4.5).

OWNER: Role 6 (Ziyad).

Works on a STANDARDISED test tensor Z (T, N, C), so magnitudes are in units of
each node's training standard deviation.

Two families, five kinds:
  * event   - coherent across a node and its spatial neighbours, smooth ramp
              up and down, temperature and humidity move together (temp up,
              humidity down), as when sunlight or HVAC hits a region.
      - "event"         all affected nodes start at the same step.
      - "lagged_event"  (harder) each neighbour starts 1..max_lag steps after
                        the centre node, as when warm air spreads.
  * failure - confined to ONE node and paired with a voltage drop.
      - "failure"       incoherent noise on temperature and humidity.
      - "stuck"         (harder) temperature, humidity and light freeze at the
                        last reading before onset. Magnitude is not used: the
                        value is plausible, only its lack of change is wrong.
      - "drift"         (harder) temperature drifts linearly from 0 to
                        +/- magnitude over a longer window, as a degrading
                        sensor does.

Labels: 0 = normal, 1 = event, 2 = failure. Every failure kind lowers voltage
by ``voltage_drop``, so with voltage_drop != 0 the set of changed cells equals
the set of labelled cells exactly (the tests check this).

Quiet periods (``quiet_only=True``): onsets are drawn only from windows where
the network is calm, so the injected anomaly is the only thing happening.
Calm means the rolling mean (over ``quiet_window`` steps) of the cross-node
variance of temperature, humidity and light is at or below its
``quiet_quantile`` quantile. When a ``valid`` (T, N) mask is given, the centre
node must also be observed (valid) for at least ``min_observed`` of the
window, because unobserved cells are never scored by the harness.

With the defaults (quiet_only=False, no harder kinds) the random stream and
the output are exactly those of v0, so earlier bench numbers still hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from signoise.config import HUMID, LIGHT, TEMP, VOLT

NORMAL, EVENT, FAILURE = 0, 1, 2

# variant name -> label code
KINDS = {"event": EVENT, "lagged_event": EVENT,
         "failure": FAILURE, "stuck": FAILURE, "drift": FAILURE}


@dataclass
class Episode:
    kind: int          # EVENT or FAILURE
    t0: int
    length: int
    nodes: list        # affected node positions
    magnitude: float
    variant: str = ""  # one of KINDS; "" is read as the plain kind
    lags: list = field(default_factory=list)  # per-node onset lag (lagged_event)


def _ramp(length: int) -> np.ndarray:
    """Smooth 0 -> 1 -> 0 profile (half sine), strictly positive."""
    return np.sin(np.linspace(0, np.pi, length + 2)[1:-1])


def quiet_steps(Z: np.ndarray, valid: np.ndarray | None = None,
                quantile: float = 0.5, window: int = 12,
                channels: tuple = (TEMP, HUMID, LIGHT)) -> np.ndarray:
    """(T,) bool: True where the network is calm.

    activity_t = mean over ``channels`` of the variance across nodes of Z[t]
    (only valid nodes count when ``valid`` is given), smoothed by a centred
    rolling mean of ``window`` steps. Calm = activity <= its ``quantile``.
    """
    X = Z[:, :, list(channels)]                               # (T, N, c)
    w = np.ones(X.shape[:2]) if valid is None else valid.astype(float)
    cnt = np.maximum(w.sum(1, keepdims=True), 1.0)            # (T, 1)
    Xw = np.where(w[:, :, None] > 0, X, 0.0)
    mean = Xw.sum(1) / cnt                                    # (T, c)
    var = ((Xw - mean[:, None, :]) ** 2 * w[:, :, None]).sum(1) / cnt
    act = var.mean(1)                                         # (T,)
    act[w.sum(1) < 2] = np.inf                                # too few nodes to judge
    if window > 1:
        fin = np.isfinite(act)
        k = np.ones(window)
        num = np.convolve(np.where(fin, act, 0.0), k, mode="same")
        den = np.convolve(fin.astype(float), k, mode="same")
        act = np.where(den > 0, num / np.maximum(den, 1.0), np.inf)
    finite = act[np.isfinite(act)]
    if finite.size == 0:
        return np.zeros(len(act), dtype=bool)
    return act <= np.quantile(finite, quantile)


def inject_anomalies(
    Z: np.ndarray,
    neighbors: np.ndarray,
    n_events: int = 10,
    n_failures: int = 10,
    magnitude: float = 4.0,
    length: tuple = (6, 24),
    neighbor_decay: float = 0.7,
    voltage_drop: float = 2.0,
    seed: int = 0,
    *,
    quiet_only: bool = False,
    valid: np.ndarray | None = None,
    quiet_quantile: float = 0.5,
    quiet_window: int = 12,
    min_observed: float = 0.8,
    n_lagged_events: int = 0,
    n_stuck: int = 0,
    n_drift: int = 0,
    max_lag: int = 3,
    drift_length: tuple = (24, 72),
):
    """Return (Z_injected, labels (T, N) int8, episodes).

    Episodes never overlap in (time, node). ``neighbors`` is the (N, k) array
    from data.neighbors.knn_neighbors. ``valid`` is the (T, N) mask of scored
    cells; it is only used when ``quiet_only`` is True. The harder kinds are
    placed after the plain ones, so turning them on does not move the plain
    episodes.
    """
    rng = np.random.default_rng(seed)
    Zi = Z.copy()
    T, N, _ = Z.shape
    labels = np.zeros((T, N), dtype=np.int8)
    episodes: list[Episode] = []

    if quiet_only:
        calm = quiet_steps(Z, valid, quiet_quantile, quiet_window)
        not_calm = np.concatenate([[0], np.cumsum(~calm)])   # prefix count

    def draw_t0(span: int):
        """Onset so that [t0, t0 + span) fits (and is calm if quiet_only)."""
        if not quiet_only:
            return int(rng.integers(0, T - span))
        starts = np.arange(T - span)
        ok = not_calm[starts + span] - not_calm[starts] == 0
        cand = starts[ok]
        return int(rng.choice(cand)) if cand.size else None

    def observed_enough(t0: int, L: int, n: int) -> bool:
        if not quiet_only or valid is None:
            return True
        return valid[t0:t0 + L, n].mean() >= min_observed

    def place(variant: str, tries: int = 200) -> None:
        kind = KINDS[variant]
        lo, hi = drift_length if variant == "drift" else length
        for _ in range(tries):
            L = int(rng.integers(lo, hi + 1))
            extra = max_lag if variant == "lagged_event" else 0
            if L + extra >= T:
                return
            t0 = draw_t0(L + extra)
            if t0 is None:
                return
            center = int(rng.integers(0, N))
            nodes = [center] + (list(neighbors[center]) if kind == EVENT else [])
            lags = [0] * len(nodes)
            if variant == "lagged_event":
                lags = [0] + [int(x) for x in rng.integers(1, max_lag + 1, len(nodes) - 1)]
            if any(labels[t0 + d:t0 + d + L, n].any() for n, d in zip(nodes, lags)):
                continue
            if not observed_enough(t0, L, center):
                continue
            _apply(variant, Zi, Z, t0, L, nodes, lags, magnitude, neighbor_decay,
                   voltage_drop, rng)
            for n, d in zip(nodes, lags):
                labels[t0 + d:t0 + d + L, n] = kind
            episodes.append(Episode(kind, t0, L, nodes, magnitude, variant,
                                    lags if variant == "lagged_event" else []))
            return

    for _ in range(n_events):
        place("event")
    for _ in range(n_failures):
        place("failure")
    for _ in range(n_lagged_events):
        place("lagged_event")
    for _ in range(n_stuck):
        place("stuck")
    for _ in range(n_drift):
        place("drift")
    return Zi, labels, episodes


def _apply(variant, Zi, Z, t0, L, nodes, lags, magnitude, decay, vdrop, rng):
    """Write one episode into Zi in place (Z is the untouched original)."""
    if variant in ("event", "lagged_event"):
        prof = _ramp(L)
        for rank, (n, d) in enumerate(zip(nodes, lags)):
            a = magnitude * (1.0 if rank == 0 else decay)
            s = slice(t0 + d, t0 + d + L)
            Zi[s, n, TEMP] += a * prof
            Zi[s, n, HUMID] -= 0.6 * a * prof
            Zi[s, n, LIGHT] += 0.5 * a * prof
        return

    n, s = nodes[0], slice(t0, t0 + L)
    if variant == "failure":
        Zi[s, n, TEMP] += magnitude * rng.standard_normal(L)
        Zi[s, n, HUMID] += magnitude * rng.standard_normal(L)
    elif variant == "stuck":
        ref = Z[max(t0 - 1, 0), n]                    # last reading before onset
        for c in (TEMP, HUMID, LIGHT):
            Zi[s, n, c] = ref[c]
    elif variant == "drift":
        sign = 1.0 if rng.random() < 0.5 else -1.0
        Zi[s, n, TEMP] += sign * magnitude * np.arange(1, L + 1) / L
    Zi[s, n, VOLT] -= vdrop
