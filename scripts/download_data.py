"""Download the Intel Berkeley Lab dataset into data/raw/ (about 34 MB)."""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signoise.config import DATA_URL, RAW_DIR, RAW_FILES  # noqa: E402


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip {name} (already present)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(DATA_URL + name, dest)
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
