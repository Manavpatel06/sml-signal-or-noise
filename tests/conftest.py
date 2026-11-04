import numpy as np
import pytest

from signoise.data.load import SensorData


@pytest.fixture
def tiny_data():
    """Synthetic 200-step, 8-node, 4-channel dataset with smooth daily cycles,
    shared across nodes, plus a little noise and some missing values."""
    rng = np.random.default_rng(0)
    T, N = 200, 8
    t = np.arange(T)
    base = np.sin(2 * np.pi * t / 48)[:, None]
    temp = 22 + 3 * base + 0.3 * rng.standard_normal((T, N))
    humid = 40 - 5 * base + 0.5 * rng.standard_normal((T, N))
    light = 300 + 200 * np.clip(base, 0, None) + 5 * rng.standard_normal((T, N))
    volt = 2.7 - 0.0005 * t[:, None] + 0.005 * rng.standard_normal((T, N))
    values = np.stack([temp, humid, light, volt], axis=2)
    values[rng.random(values.shape) < 0.05] = np.nan
    coords = np.stack([np.arange(N) * 3.0, np.zeros(N)], axis=1)
    return SensorData(
        values=values,
        times=np.arange(T).astype("datetime64[s]"),
        node_ids=np.arange(1, N + 1),
        coords=coords,
        link_prob=np.full((N, N), np.nan),
    )
