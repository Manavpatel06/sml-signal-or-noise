import gzip

import numpy as np

from signoise.data.load import read_raw_readings, to_tensor
from signoise.data.neighbors import knn_neighbors, random_neighbors
from signoise.data.splits import temporal_folds
from signoise.imputation import ffill_impute, masking_benchmark

RAW = """2004-02-28 00:59:16.02785 3 1 19.9884 37.0933 45.08 2.69964
2004-02-28 01:03:16.33393 11 1 19.3024 38.4629 45.08 2.68742
2004-02-28 01:06:16.013453 17 1 19.1652 38.8039 45.08 2.68742
2004-02-28 00:59:16.02785 3 2 20.1 36.0 50.0 2.7
2004-02-28 00:59:16.02785 3 99 20.1 36.0 50.0 2.7
2004-02-28 01:06:16.013453 17 2 19.0
"""


def test_read_and_pivot(tmp_path):
    p = tmp_path / "data.txt.gz"
    with gzip.open(p, "wt") as f:
        f.write(RAW)
    df = read_raw_readings(p)
    assert set(df["moteid"]) == {1, 2}          # mote 99 dropped
    values, times = to_tensor(df, bin_epochs=1)
    assert values.shape == (14, 54, 4)            # 31 s grid from 00:59:16
    assert np.isclose(values[0, 0, 0], 19.9884)   # first reading, mote 1
    assert np.isnan(values[13, 1, 1])             # truncated row -> NaN humidity
    assert np.isclose(values[13, 1, 0], 19.0)
    assert len(times) == 14


def test_knn_neighbors_excludes_self():
    coords = np.array([[0, 0], [1, 0], [5, 0], [6, 0]], float)
    nb = knn_neighbors(coords, k=1)
    assert nb.tolist() == [[1], [0], [3], [2]]
    rnd = random_neighbors(4, k=2, seed=1)
    assert all(i not in rnd[i] for i in range(4))


def test_temporal_folds_disjoint_with_gap():
    for tr, te in temporal_folds(100, n_folds=4, gap=5):
        assert len(np.intersect1d(tr, te)) == 0
        lo, hi = te.min(), te.max()
        assert not np.any((tr >= lo - 5) & (tr <= hi + 5))


def test_ffill_impute_has_no_nan(tiny_data):
    out = ffill_impute(tiny_data.values)
    assert not np.isnan(out).any()
    obs = np.isfinite(tiny_data.values)
    assert np.allclose(out[obs], tiny_data.values[obs])


def test_masking_benchmark_runs(tiny_data):
    res = masking_benchmark(tiny_data.values, ffill_impute, frac=0.05, block=3)
    assert set(res) == {0, 1, 2, 3}
    assert all(r["n"] > 0 and r["rmse"] >= 0 for r in res.values())
