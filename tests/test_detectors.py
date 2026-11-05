"""Contract tests every detector must pass once implemented.

Unimplemented detectors are skipped, so these turn green one by one as each
owner finishes their part. Do not weaken these tests to make them pass.
"""

import numpy as np
import pytest

from signoise.detectors import REGISTRY
from signoise.evaluation.harness import run_detection_bench
from signoise.evaluation.metrics import roc_auc
from signoise.imputation import ffill_impute
from signoise.injection import inject_anomalies
from signoise.data.neighbors import knn_neighbors
from signoise.preprocess import Standardizer


def _fit_score(name, tiny_data):
    try:
        return _fit_score_impl(name, tiny_data)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


def _fit_score_impl(name, tiny_data):
    X = ffill_impute(tiny_data.values)
    Z = Standardizer().fit_transform(X)
    tr, te = Z[:120], Z[120:]
    nb = knn_neighbors(tiny_data.coords, k=3)
    Zi, labels, _ = inject_anomalies(te, nb, n_events=2, n_failures=2,
                                     magnitude=6.0, length=(4, 8), seed=0)
    det = REGISTRY[name]()
    det.fit(tr)
    return det.score(Zi), labels


@pytest.mark.parametrize("name", list(REGISTRY))
def test_score_shape_and_finite(name, tiny_data):
    s, labels = _fit_score(name, tiny_data)
    assert s.shape == labels.shape
    assert np.isfinite(s).all()


@pytest.mark.parametrize("name", list(REGISTRY))
def test_beats_chance_on_easy_injections(name, tiny_data):
    s, labels = _fit_score(name, tiny_data)
    assert roc_auc(labels > 0, s) > 0.7


def test_harness_runs_with_todo_detectors(tiny_data):
    try:
        _ = run_detection_bench(tiny_data, detectors=["ksigma"], n_folds=2, gap=2,
                                k_neighbors=3, verbose=False,
                                inject_kwargs={"n_events": 1, "n_failures": 1,
                                               "length": (4, 8)})
    except NotImplementedError as e:
        pytest.skip(f"harness not implemented yet: {e}")
    res = run_detection_bench(tiny_data, detectors=list(REGISTRY), n_folds=2, gap=2,
                              k_neighbors=3,
                              inject_kwargs={"n_events": 2, "n_failures": 2,
                                             "length": (4, 8)},
                              verbose=False)
    assert set(res["status"]) <= {"ok", "todo"}
    assert (res.loc[res.detector == "ksigma", "status"] == "ok").all()
