"""Unit tests for Role 2: KNN and K-means detectors."""

import numpy as np

from signoise.detectors import KMeansDetector, KNNDetector


def test_knn_toy_distances_by_hand():
    # Reference points on a line: 0, 1, 3, 10 (d=1 so P is (n, 1)).
    ref = np.array([[0.0], [1.0], [3.0], [10.0]])
    q = np.array([[2.0], [0.0], [9.0]])
    # query 2.0: distances 2,1,1,8 -> sorted 1,1,2,8
    # query 0.0: distances 0,1,3,10 -> sorted 0,1,3,10
    # query 9.0: distances 9,8,6,1  -> sorted 1,6,8,9
    expected = {1: [1, 0, 1], 2: [1, 1, 6], 3: [2, 3, 8]}
    for k, want in expected.items():
        det = KNNDetector(k=k)
        det._fit_points(ref)
        np.testing.assert_allclose(det._score_points(q), want, atol=1e-9)


def test_knn_chunking_and_subsample_are_consistent():
    rng = np.random.default_rng(1)
    P = rng.standard_normal((300, 3))
    Q = rng.standard_normal((50, 3))
    a = KNNDetector(k=5, chunk=7)
    b = KNNDetector(k=5, chunk=1000)
    a._fit_points(P), b._fit_points(P)
    np.testing.assert_allclose(a._score_points(Q), b._score_points(Q), atol=1e-9)
    sub = KNNDetector(k=5, max_ref=100, seed=3)
    sub._fit_points(P)
    assert sub.ref_.shape == (100, 3)
    same = KNNDetector(k=5, max_ref=100, seed=3)
    same._fit_points(P)
    np.testing.assert_array_equal(sub.ref_, same.ref_)


def test_knn_outlier_scores_higher():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((500, 3))
    det = KNNDetector(k=5)
    det._fit_points(P)
    s = det._score_points(np.array([[0.0, 0, 0], [8.0, 8, 8]]))
    assert s[1] > 5 * s[0]


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    P = np.concatenate([c + 0.3 * rng.standard_normal((100, 2)) for c in centres])
    return P, centres


def test_kmeans_recovers_three_blobs():
    P, centres = _blobs()
    det = KMeansDetector(n_clusters=3, seed=0)
    det._fit_points(P)
    # every true centre has a fitted centroid within 0.2
    d = np.linalg.norm(centres[:, None, :] - det.centroids_[None, :, :], axis=2)
    assert (d.min(axis=1) < 0.2).all()
    assert det.centroids_.shape == (3, 2)
    # inertia ~ n * 2 * sigma^2 = 300 * 2 * 0.09 = 54
    assert 30 < det.inertia_ < 80


def test_kmeans_scores_far_point_high_and_is_seeded():
    P, _ = _blobs()
    a = KMeansDetector(n_clusters=3, seed=2)
    b = KMeansDetector(n_clusters=3, seed=2)
    a._fit_points(P), b._fit_points(P)
    np.testing.assert_array_equal(a.centroids_, b.centroids_)
    s = a._score_points(np.array([[0.0, 0.0], [5.0, 5.0]]))
    assert s[1] > 10 * s[0]


def test_kmeans_inertia_decreases_with_K():
    P, _ = _blobs()
    inertias = []
    for K in (1, 2, 3, 6):
        d = KMeansDetector(n_clusters=K, seed=0)
        d._fit_points(P)
        inertias.append(d.inertia_)
    assert all(x >= y - 1e-9 for x, y in zip(inertias, inertias[1:]))


def test_kmeans_handles_duplicate_points_and_empty_clusters():
    P = np.zeros((20, 2))
    P[10:] = 1.0  # only two distinct locations but ask for 4 clusters
    det = KMeansDetector(n_clusters=4, seed=0)
    det._fit_points(P)
    assert np.isfinite(det.centroids_).all()
    assert det.inertia_ < 1e-9


def test_clip_default_is_off_and_clip_caps_features():
    P = np.array([[0.0, 0.0], [1.0, 1.0], [100.0, 0.0]])
    assert KNNDetector().clip is None and KMeansDetector().clip is None
    # no clip: far point keeps its true distance; clip=5: it is pulled to (5, 0)
    off, on = KNNDetector(k=1), KNNDetector(k=1, clip=5.0)
    off._fit_points(P[:2]), on._fit_points(P[:2])
    q = P[2:]
    assert off._score_points(q)[0] == np.sqrt((100 - 1) ** 2 + 1)
    assert np.isclose(on._score_points(q)[0], np.sqrt((5 - 1) ** 2 + 1))
    # clipping is also applied to the stored reference points
    on._fit_points(np.array([[100.0, 0.0]]))
    np.testing.assert_array_equal(on.ref_, [[5.0, 0.0]])
    km = KMeansDetector(n_clusters=1, clip=5.0)
    km._fit_points(P)
    np.testing.assert_allclose(km.centroids_, [[(0 + 1 + 5) / 3, (0 + 1 + 0) / 3]])


def test_clip_makes_moderate_anomaly_outrank_natural_outlier():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((1000, 3))
    P[:5, 2] = 500.0                        # natural heavy-tail readings in training
    moderate = np.array([[4.0, 4.0, 0.0]])  # injected-style anomaly
    natural = np.array([[0.0, 0.0, 400.0]])
    plain = KNNDetector(k=5)
    plain._fit_points(P)
    clipped = KNNDetector(k=5, clip=8.0)
    clipped._fit_points(P)
    assert plain._score_points(natural)[0] > plain._score_points(moderate)[0]
    assert clipped._score_points(moderate)[0] > clipped._score_points(natural)[0]
