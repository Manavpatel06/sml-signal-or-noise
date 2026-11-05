# Role 4 — EM: GMM detector and EM imputation (Harsh)

Files: `src/signoise/detectors/gmm.py`, `src/signoise/imputation/em_gaussian.py`.
Read the docstrings at the top of both files; they are the spec.

## Tasks

1. **GMMDetector** (`PointwiseDetector`)
   - EM with full covariances, ridge `1e-6 * I`, log-sum-exp responsibilities.
   - Initialise from k-means (`KMeansDetector` once Role 2 merges; until then,
     random training points), `n_init` restarts, keep best log-likelihood.
   - Subsample training points (e.g. 20,000, seeded) if fitting is slow.
   - `_score_points`: negative log-likelihood. Store `weights_`, `means_`,
     `covariances_`, `lower_bound_`.
   - Helper `bic(P)` so K can be chosen by BIC.
2. **EMGaussianImputer** (first priority: Role 1's pipeline waits on it)
   - Per channel (default), vectors x_t in R^N across nodes; Gaussian EM with
     missing values (conditional mean and covariance in the E-step).
   - `fit` on train window, `transform` fills NaN in any window.
   - Ridge on Sigma; stop on log-likelihood change < tol.
3. **Masking study:** `imputation.masking_benchmark` with `frac=0.1` and
   `block` in {1, 6, 24}; table of RMSE per channel, EM vs `ffill_impute`.
   Save to `results/role4/`.
4. Tests in `tests/test_role4.py`: GMM log-likelihood is non-decreasing over
   EM iterations; GMM recovers means of two separated Gaussians; EM imputer
   beats ffill on synthetic correlated data with random holes.

Done when the gmm contract tests pass and the imputer test passes.
