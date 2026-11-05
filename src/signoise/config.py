"""Project-wide paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"

DATA_URL = "http://db.csail.mit.edu/labdata/"
RAW_FILES = ("data.txt.gz", "mote_locs.txt", "connectivity.txt")

CHANNELS = ("temperature", "humidity", "light", "voltage")
TEMP, HUMID, LIGHT, VOLT = range(4)

N_NODES = 54
EPOCH_SECONDS = 31

# Plausible physical ranges. NOT applied when loading: out-of-range readings
# from dying sensors are exactly what this project studies. Use these only for
# profiling or for the weak "failure" labels.
VALID_RANGE = {
    "temperature": (-10.0, 60.0),   # deg C
    "humidity": (0.0, 100.0),       # %
    "light": (0.0, 100000.0),       # lux
    "voltage": (2.0, 3.2),          # V
}
