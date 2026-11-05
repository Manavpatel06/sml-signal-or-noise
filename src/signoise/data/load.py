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
RAW_COLUMNS = ['date', 'time', 'epoch', 'moteid', *CHANNELS]

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
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def channel(self, name: str) -> np.ndarray:
        """(T, N) slice for one channel name."""
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def window(self, idx) -> 'SensorData':
        """Copy restricted to time indices ``idx`` (slice or index array)."""
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def between(self, start=None, end=None) -> 'SensorData':
        """Copy restricted to start <= time < end (strings like "2004-03-20")."""
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

    def missing_fraction(self) -> float:
        raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def read_raw_readings(path: Path) -> pd.DataFrame:
    """Parse data.txt(.gz) into a DataFrame; malformed numbers become NaN."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def read_coords(path: Path) -> np.ndarray:
    """Return (54, 2) coordinates indexed by mote id - 1."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def read_connectivity(path: Path) -> np.ndarray:
    """Return (54, 54) delivery probability, NaN where unknown."""
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def to_tensor(df: pd.DataFrame, bin_epochs: int=1):
    """Pivot readings to (T, N, C) on a regular time grid.

    Steps are ``bin_epochs * 31`` seconds long and are assigned from the
    timestamp, NOT the epoch counter: the epoch counter is 16-bit and resets
    partway through the deployment (around 21 March), so it is not monotonic.
    Readings falling in the same (step, node) are averaged.
    """
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')

def load_intel(raw_dir: Path=RAW_DIR, bin_epochs: int=10, use_cache: bool=True, drop_empty_nodes: bool=True) -> SensorData:
    """Load the dataset as a SensorData tensor.

    bin_epochs=10 averages ten 31 s epochs (~5 min) into one step, which keeps
    the tensor small for development. Use bin_epochs=1 for full resolution.
    The processed tensor is cached in data/processed/.
    """
    raise NotImplementedError('Role 1: implemented on branch role1-data-harness')
