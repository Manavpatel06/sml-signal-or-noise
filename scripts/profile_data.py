"""Characterise the data: shape, missingness, voltage and out-of-range readings.

Writes CSV tables and figures to results/profile/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from signoise.config import RESULTS_DIR  # noqa: E402
from signoise.data import load_intel  # noqa: E402
from signoise.data.profile import (  # noqa: E402
    missingness_by_node,
    out_of_range_by_node,
    voltage_summary,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-epochs", type=int, default=10)
    args = ap.parse_args()

    data = load_intel(bin_epochs=args.bin_epochs)
    out = RESULTS_DIR / "profile"
    out.mkdir(parents=True, exist_ok=True)

    T, N, C = data.shape
    print(f"tensor: {T} steps x {N} nodes x {C} channels "
          f"({data.bin_epochs} epoch(s) per step)")
    print(f"time span: {data.times[0]} -> {data.times[-1]}")
    print(f"missing: {data.missing_fraction():.1%} of cells")

    miss = missingness_by_node(data)
    oor = out_of_range_by_node(data)
    volt = voltage_summary(data)
    miss.to_csv(out / "missingness_by_node.csv")
    oor.to_csv(out / "out_of_range_by_node.csv")
    volt.to_csv(out / "voltage_summary.csv")
    print("\nnodes whose voltage drops below 2.3 V:",
          int((volt["first_step_below_2.3V"] >= 0).sum()))
    print("mean out-of-range fraction per channel:")
    print(oor.mean().round(4).to_string())

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].imshow(np.isnan(data.channel("temperature")).T, aspect="auto",
                 interpolation="nearest", cmap="Greys")
    ax[0].set_ylabel("node")
    ax[0].set_title("Missing temperature readings (black = missing)")
    ax[1].plot(data.channel("voltage"), lw=0.4)
    ax[1].set_ylabel("voltage (V)")
    ax[1].set_xlabel("time step")
    ax[1].set_ylim(1.8, 3.1)
    ax[1].set_title("Battery voltage per node")
    fig.tight_layout()
    fig.savefig(out / "missing_and_voltage.png", dpi=150)
    print(f"\nwrote tables and figure to {out}")


if __name__ == "__main__":
    main()
