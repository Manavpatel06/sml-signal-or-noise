# Role 2 — Detectors I: KNN and K-means (Rashi)

Files: `src/signoise/detectors/knn.py`, `src/signoise/detectors/kmeans.py`.
Both subclass `PointwiseDetector`: implement `_fit_points(P)` and
`_score_points(P)` on `P` of shape `(n_points, d)`. Read the docstrings at the
top of each file first; they are the spec.

## Tasks

1. **KNNDetector**
   - `_fit_points`: keep a random subsample of at most `max_ref` training
     points (seeded).
   - `_score_points`: Euclidean distance from each point to its k-th nearest
     reference point. Compute in chunks of `chunk` rows using
     `||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b`; `np.partition` for the k-th smallest.
   - Clip tiny negative squared distances to 0 before `sqrt`.
2. **KMeansDetector**
   - k-means++ initialisation, Lloyd iterations until assignments stop
     changing or `max_iter`; `n_init` restarts, keep lowest inertia.
   - Handle empty clusters (re-seed to the farthest point).
   - `_score_points`: distance to the nearest centroid.
   - Store `inertia_` and `centroids_`.
3. Add unit tests in `tests/test_role2.py`: KNN on a toy set where the answer
   is known by hand; K-means recovers three well-separated blobs.
4. Experiments (with Ziyad): `run_bench.py --detectors ksigma knn kmeans`;
   sweep k in {1, 5, 10, 20} and K in {2, 4, 8, 16}; elbow plot of inertia vs K;
   try `features="channels+diff"`. Save figures to `results/role2/`.

Done when `pytest -q` runs (not skips) the knn and kmeans contract tests and
they pass.
