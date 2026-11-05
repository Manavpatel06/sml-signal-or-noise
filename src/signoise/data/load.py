"""Load the Intel Berkeley Lab data into a dense (time, node, channel) tensor.

Raw format (data.txt): ``date time epoch moteid temperature humidity light voltage``.
Rows can be truncated, mote ids outside 1..54 exist, and many (epoch, mote)
pairs are missing entirely. Missing readings become NaN in the tensor.

Outlier values are deliberately kept: readings from failing, low-battery nodes
are the "failure" class this project tries to recognise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from signoise.config import CHANNELS, EPOCH_SECONDS, N_NODES, PROCESSED_DIR, RAW_DIR

RAW_COLUMNS = ["date", "time", "epoch", "moteid", *CHANNELS]


@dataclass
class SensorData:
    """Dense sensor tensor plus metadata.

    values : (T, N, C) float array, NaN where a reading is missing.
    times  : (T,) datetime64[s], start time of each time step.
    node_ids : (N,) int, original mote ids (1..54).
    coords : (N, 2) float, node positions in metres.
    link_prob : (N, N) float, message-delivery probability sender->receiver
        (NaN when unknown).
    bin_epochs : number of raw 31-second epochs averaged into one time step.
    """

    values: np.ndarray
    times: np.ndarray
    node_ids: np.ndarray
    coords: np.ndarray
    link_prob: np.ndarray
    channels: tuple = CHANNELS
    bin_epochs: int = 1
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.values.shape

    def channel(self, name: str) -> np.ndarray:
        """(T, N) slice for one channel name."""
        return self.values[:, :, self.channels.index(name)]

    def window(self, idx) -> "SensorData":
        """Copy restricted to time indices ``idx`` (slice or index array)."""
        return SensorData(
            values=self.values[idx].copy(),
            times=self.times[idx],
            node_ids=self.node_ids,
            coords=self.coords,
            link_prob=self.link_prob,
            channels=self.channels,
            bin_epochs=self.bin_epochs,
            meta=dict(self.meta),
        )

    def between(self, start=None, end=None) -> "SensorData":
        """Copy restricted to start <= time < end (strings like "2004-03-20")."""
        keep = np.ones(len(self.times), dtype=bool)
        if start is not None:
            keep &= self.times >= np.datetime64(start)
        if end is not None:
            keep &= self.times < np.datetime64(end)
        return self.window(np.flatnonzero(keep))

    def missing_fraction(self) -> float:
        return float(np.isnan(self.values).mean())


def read_raw_readings(path: Path) -> pd.DataFrame:
    """Parse data.txt(.gz) into a DataFrame; malformed numbers become NaN."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
        engine="c",
        on_bad_lines="skip",
        dtype=str,
    )
    for col in ["epoch", "moteid", *CHANNELS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["epoch", "moteid"])
    df["epoch"] = df["epoch"].astype(np.int64)
    df["moteid"] = df["moteid"].astype(np.int64)
    df = df[(df["moteid"] >= 1) & (df["moteid"] <= N_NODES)]
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"], errors="coerce", format="mixed"
    )
    return df.drop(columns=["date", "time"])


def read_coords(path: Path) -> np.ndarray:
    """Return (54, 2) coordinates indexed by mote id - 1."""
    locs = np.loadtxt(path)
    coords = np.full((N_NODES, 2), np.nan)
    for mote, x, y in locs:
        coords[int(mote) - 1] = (x, y)
    return coords


def read_connectivity(path: Path) -> np.ndarray:
    """Return (54, 54) delivery probability, NaN where unknown."""
    prob = np.full((N_NODES, N_NODES), np.nan)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:  # the file has a few truncated rows
                continue
            s, r, p = int(parts[0]), int(parts[1]), float(parts[2])
            if 1 <= s <= N_NODES and 1 <= r <= N_NODES:
                prob[s - 1, r - 1] = p
    return prob


def to_tensor(df: pd.DataFrame, bin_epochs: int = 1):
    """Pivot readings to (T, N, C) on a regular time grid.

    Steps are ``bin_epochs * 31`` seconds long and are assigned from the
    timestamp, NOT the epoch counter: the epoch counter is 16-bit and resets
    partway through the deployment (around 21 March), so it is not monotonic.
    Readings falling in the same (step, node) are averaged.
    """
    df = df.dropna(subset=["timestamp"])
    dt = np.timedelta64(EPOCH_SECONDS * bin_epochs, "s")
    t0 = df["timestamp"].min().floor("s").to_datetime64()
    ts = df["timestamp"].to_numpy()
    step = ((ts - t0) // dt).astype(np.int64)
    node = df["moteid"].to_numpy() - 1
    T = int(step.max()) + 1
    C = len(CHANNELS)

    sums = np.zeros((T, N_NODES, C))
    counts = np.zeros((T, N_NODES, C))
    for c, name in enumerate(CHANNELS):
        v = df[name].to_numpy(dtype=float)
        ok = np.isfinite(v)
        np.add.at(sums[:, :, c], (step[ok], node[ok]), v[ok])
        np.add.at(counts[:, :, c], (step[ok], node[ok]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = sums / counts
    values[counts == 0] = np.nan

    times = (t0 + np.arange(T) * dt).astype("datetime64[s]")
    return values, times


def load_intel(
    raw_dir: Path = RAW_DIR,
    bin_epochs: int = 10,
    use_cache: bool = True,
    drop_empty_nodes: bool = True,
) -> SensorData:
    """Load the dataset as a SensorData tensor.

    bin_epochs=10 averages ten 31 s epochs (~5 min) into one step, which keeps
    the tensor small for development. Use bin_epochs=1 for full resolution.
    The processed tensor is cached in data/processed/.
    """
    raw_dir = Path(raw_dir)
    cache = PROCESSED_DIR / f"intel_t{bin_epochs}.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        values, times = z["values"], z["times"]
    else:
        df = read_raw_readings(raw_dir / "data.txt.gz")
        values, times = to_tensor(df, bin_epochs=bin_epochs)
        if use_cache:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, values=values, times=times)

    coords = read_coords(raw_dir / "mote_locs.txt")
    link_prob = read_connectivity(raw_dir / "connectivity.txt")
    node_ids = np.arange(1, N_NODES + 1)

    if drop_empty_nodes:
        keep = ~np.all(np.isnan(values), axis=(0, 2)) & np.isfinite(coords).all(axis=1)
        values = values[:, keep]
        coords, link_prob = coords[keep], link_prob[np.ix_(keep, keep)]
        node_ids = node_ids[keep]

    return SensorData(
        values=values,
        times=times,
        node_ids=node_ids,
        coords=coords,
        link_prob=link_prob,
        bin_epochs=bin_epochs,
    )
