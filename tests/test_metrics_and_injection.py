import numpy as np

from signoise.data.neighbors import knn_neighbors
from signoise.evaluation.metrics import confusion_matrix, precision_recall_at_k, roc_auc
from signoise.injection import EVENT, FAILURE, inject_anomalies


def test_roc_auc_known_values():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5
    # classic example: 0.75
    assert np.isclose(roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)


def test_precision_recall_at_k():
    p, r = precision_recall_at_k([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.0], k=2)
    assert (p, r) == (0.5, 0.5)


def test_confusion_matrix():
    cm = confusion_matrix([1, 1, 2, 2], [1, 2, 2, 2])
    assert cm.tolist() == [[1, 1], [0, 2]]


def test_injection_labels_match_changes():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((300, 10, 4))
    coords = np.stack([np.arange(10.0), np.zeros(10)], 1)
    nb = knn_neighbors(coords, k=3)
    Zi, labels, eps = inject_anomalies(Z, nb, n_events=3, n_failures=3, seed=1)
    assert (labels == EVENT).any() and (labels == FAILURE).any()
    changed = np.any(Zi != Z, axis=2)
    assert np.array_equal(changed, labels > 0)
    for e in eps:
        if e.kind == EVENT:
            assert len(e.nodes) == 4
        else:
            assert len(e.nodes) == 1
