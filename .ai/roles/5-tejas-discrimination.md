# Role 5 — Discrimination: features, Naive Bayes, logistic regression (Tejas)

Files: `src/signoise/discrimination/features.py`,
`src/signoise/discrimination/classifiers.py`. Read both docstrings; they are
the spec. This stage is the project's contribution.

## Tasks

1. **Features** for each flagged cell `(t, n)`:
   - `neighbour_agreement`: fraction of `neighbors[n]` whose score exceeds
     `threshold` at any step in `[t-window, t+window]`.
   - `battery_level_and_slope`: raw voltage at t and least-squares slope over
     the previous `window` steps (NaN-safe; use observed steps only).
   - `fit_voltage_temperature` + residual: per-node regression of voltage on
     temperature over the train window; feature = observed - predicted.
   - `cross_channel_consistency`: Pearson correlation of the first differences
     of temperature and humidity over `[t-window, t]` (0 if undefined).
   - `build_feature_matrix` returns `(M, 5)` and `FEATURE_NAMES`.
   Use RAW (unstandardised, imputed) values for battery features and the
   detector scores for neighbour agreement.
2. **GaussianNaiveBayes** and **LogisticRegression** as specified; both expose
   `classes_`, `fit`, `predict_proba`, `predict`. Logistic regression
   standardises features with train statistics and records the loss curve.
3. Tests in `tests/test_role5.py`: NB matches hand-computed posteriors on a
   tiny dataset; logistic regression separates linearly separable data and its
   loss decreases; features on a hand-built tensor where one event (3 nodes)
   and one failure (1 node) give agreement ~1 vs ~0.
4. With Manav: the end-to-end script (`scripts/run_discrimination.py`),
   confusion matrices, and the random-neighbour control (same features built
   with `data.neighbors.random_neighbors`).

Done when tests pass and the discrimination script prints confusion matrices
for both classifiers.
