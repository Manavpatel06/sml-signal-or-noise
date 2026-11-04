# Role 3 — Detectors II: PCA/SVD and HMM (Karthik)

Files: `src/signoise/detectors/pca.py`, `src/signoise/detectors/hmm.py`.
Both subclass `Detector` directly (they need cross-node vectors or time
order). Read each file's docstring first; it is the spec.

## Tasks

1. **PCADetector**
   - Reshape train X `(T, N, C)` to `(T, N*C)`; centre with the training mean.
   - `np.linalg.svd(..., full_matrices=False)`; choose `n_components` or the
     smallest k reaching `var_explained`.
   - `score`: project test rows, reconstruct, squared residual reshaped back
     to `(T, N, C)`, summed over channels -> `(T, N)`.
   - Store `components_`, `explained_variance_ratio_`.
2. **HMMDetector** (Gaussian emissions, diagonal covariance)
   - Initialise means from a quick k-means on the training points (or random
     training points, seeded); uniform transitions plus self-transition bias.
   - Baum-Welch with scaled forward-backward (or log space). `shared=True`:
     one model trained on all nodes' sequences (sum the sufficient statistics
     over nodes).
   - `score`: forward pass on each node's test sequence; score at t =
     `-log p(x_t | x_{1:t-1})` = `-log` of the scaling factor c_t.
   - Variance floor (e.g. 1e-3) on every state.
3. Tests in `tests/test_role3.py`: PCA reconstruction error ~0 on data that is
   exactly rank k; HMM log-likelihood never decreases across Baum-Welch
   iterations; HMM recovers two well-separated states on synthetic data.
4. Experiments: explained-variance curve; score vs k; HMM with S in {2,3,4,6}.
   Figures to `results/role3/`.

Note: PCA across nodes is the detector most likely to separate events (shared
across nodes, explained by top components) from failures (one node off) — make
sure the write-up explains why.

Done when the pca and hmm contract tests run and pass.
