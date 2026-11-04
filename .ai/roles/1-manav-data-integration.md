# Role 1 — Data, harness & integration (Manav)

Done in the base: loader, neighbours, temporal folds, profiling, robust
standardiser, ffill imputer, masking benchmark, k-sigma baseline, metrics,
detector bench, CLI scripts, tests.

## Remaining tasks, in order

1. **Wire EM imputation into the bench** once Role 4 lands `EMGaussianImputer`:
   add an imputer option to `run_detection_bench` and `scripts/run_bench.py`
   (`--imputer ffill|em`). Fit the EM model on the train window, apply to test.
2. **End-to-end Stage C pipeline** `scripts/run_discrimination.py`:
   for each fold, run a chosen detector, flag cells above a threshold (top-q%
   of train scores), call Role 5's `build_feature_matrix` on the flagged
   injected cells, train NB and logistic regression on the train-fold flags,
   evaluate on the test fold, print a confusion matrix (event vs failure) and
   the false-alarm reduction (alerts removed vs events kept).
   Run once with `knn_neighbors` and once with `random_neighbors` (control).
3. **Weak-label evaluation** on the late period (after 2004-03-20): score
   cells with each detector, use `real_anomaly_mask` as positives, report AUC
   separately from the injected-anomaly results.
4. **Stage D (if time):** predict failure from voltage history before bad
   readings appear: features = voltage level and slope over the last W steps;
   label = node enters `real_anomaly_mask` within the next H steps; logistic
   regression (Role 5's implementation); report AUC vs horizon H.
5. **Integration duty:** review and merge PRs, keep `pytest -q` green on
   `main`, keep the README results table current.

Acceptance: each script runs end to end from a fresh clone after
`download_data.py`, and writes CSVs to `results/`.
