# Brief for AI coding assistants working on this repo

Give this file to your coding assistant at the start of every session, followed
by your role file from `.ai/roles/`. Example opening message:

> Read `.ai/AGENTS.md` and `.ai/roles/<your-role>.md`, then `README.md`.
> Implement my role's tasks in order. Run `pytest -q` after each step.

---

## Git

- Work only on the role branch you were given (see START_HERE.md); it is
  created from an up-to-date `main`. Never commit to `main` directly.
- Do not run `git push --force`, rewrite history, or touch other branches.
- Pull `main` and rebase/merge it into your branch when shared code changes.

## Project in one paragraph

CSE 575 (Statistical Machine Learning) course project. Data: Intel Berkeley Lab
sensor network (54 nodes; temperature, humidity, light, voltage; 31 s epochs;
28 Feb - 5 Apr 2004). Goal: detect anomalies with five self-implemented detectors
(KNN distance, K-means, GMM/EM, PCA/SVD, HMM), then classify each detected
anomaly as a real environmental EVENT (shows up at neighbouring nodes, temperature
and humidity move together) or a sensor FAILURE (confined to one node,
incoherent, low battery), using Naive Bayes and logistic regression on spatial and
battery features. Evaluation uses synthetic injected anomalies (known labels)
plus weak labels from collapsed voltage.

## Hard rules

1. **Own implementations only.** Every learning algorithm (KNN, K-means, GMM/EM,
   PCA, HMM, EM imputation, Naive Bayes, logistic regression, metrics) is written
   with NumPy/SciPy linear algebra. Do NOT import scikit-learn, hmmlearn,
   statsmodels, pomegranate, PyTorch or similar for the methods. This is graded.
2. **Do not change shared interfaces** without the integration owner (Manav):
   - `Detector.fit(X)` / `Detector.score(X)`: X is `(T, N, C)` float, NaN-free,
     standardised; `score` returns `(T, N)`, higher = more anomalous.
   - `SensorData` fields, `inject_anomalies` return values, label codes
     (0 normal, 1 event, 2 failure), `run_detection_bench` signature.
   If you need something new, ADD an optional argument with a default that keeps
   current behaviour.
3. **No leakage.** Anything fitted (means, scales, centroids, mixture
   parameters, PCA components, HMM parameters, imputation model, classifier
   weights) is fitted on the TRAINING window of a fold only. Folds are
   contiguous in time (`data.splits.temporal_folds`); never shuffle time steps.
4. **Do not weaken or delete tests** to make them pass. `tests/test_detectors.py`
   is the contract every detector must meet. Add tests for your own code.
5. **Do not commit data or results.** `data/` and `results/` are git-ignored.
6. **Reproducible:** every random choice takes a `seed` argument (default 0) and
   uses `np.random.default_rng(seed)`.
7. **Vectorise.** The default tensor is ~9,900 x 54 x 3; a Python loop over
   cells is too slow. Loop over iterations/chunks, not over samples.
8. **Numerical care:** log-sum-exp for mixture and HMM likelihoods, a small
   ridge on covariance matrices, never divide by a variance without a floor.
9. **Keep it readable.** Short functions, docstrings stating shapes, comments
   for the maths. The team must be able to explain every line in the
   presentation Q&A.
10. Commit messages: short, imperative, describe the change
    (e.g. "Add k-means++ initialisation"). No tool or assistant attribution
    lines in commits, comments or files.

## Map of the code

```
src/signoise/
  config.py              paths, channel indices (TEMP, HUMID, LIGHT, VOLT), VALID_RANGE
  data/load.py           load_intel() -> SensorData; timestamps define the time grid
  data/neighbors.py      knn_neighbors, radius_neighbors, random_neighbors (control)
  data/splits.py         temporal_folds (blocked CV with gap)
  data/profile.py        missingness, voltage summary, weak_failure_labels, real_anomaly_mask
  preprocess.py          Standardizer (robust median/MAD per node, train-only)
  imputation/simple.py   ffill_impute (baseline imputer)
  imputation/em_gaussian.py   EMGaussianImputer  <- Role 4
  imputation/masking.py  masking_benchmark (hide observed values, score RMSE)
  detectors/base.py      Detector, PointwiseDetector, point_features
  detectors/ksigma.py    baseline (done)
  detectors/knn.py, kmeans.py   <- Role 2
  detectors/pca.py, hmm.py      <- Role 3
  detectors/gmm.py              <- Role 4
  injection/inject.py    inject_anomalies (v0 done)   <- Role 6 extends
  evaluation/metrics.py  roc_auc, precision/recall, confusion_matrix
  evaluation/harness.py  run_detection_bench, summarise
  discrimination/features.py, classifiers.py   <- Role 5
scripts/  download_data.py, profile_data.py, run_bench.py
tests/    contract + unit tests
```

## Facts about the data the base already handles

- The `epoch` column is a 16-bit counter that resets around 21 March. The time
  grid is built from timestamps. Do not go back to epoch numbers.
- Raw outliers (e.g. temperature 120 C) are KEPT on load. They are the failure
  class. `VALID_RANGE` is only for profiling and weak labels.
- About 27% of (step, node) cells are missing at bin_epochs=10. After ~20 March
  most nodes are failing; the injected-anomaly bench uses data before
  2004-03-20 by default.
- Detectors receive temperature, humidity and light only. Voltage drifts with
  battery drain and is a Stage C feature.
- Only observed, non-real-anomaly cells are scored (`real_anomaly_mask`).

## Definition of done for any task

- Implementation matches the docstring spec in the stub file.
- `pytest -q` passes (your detector's contract tests now run instead of skip).
- `python scripts/run_bench.py --detectors ksigma <yours>` runs; paste the
  summary table into your pull request.
- A short note in the PR: what you implemented, hyper-parameters chosen and why,
  and anything the report should mention.

## Timeline

| Dates (2026) | Milestone |
| --- | --- |
| now - Nov 8 | detectors, EM imputation, discrimination features + classifiers |
| Nov 9-15 | baseline comparison, confusion matrices, ablation |
| Nov 16-22 | cross-validation, consolidated results, slides, dry run |
| Nov 23 - Dec 2 | 15-minute presentation |
| Dec 3-7 | final report (due Dec 7) |
