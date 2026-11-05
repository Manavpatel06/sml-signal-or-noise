# Role 5: Discrimination (event vs failure)

Owner: Tejas. Stage C of the project. Given a cell `(t, n)` flagged by a Stage A
detector, decide whether it is a real environmental **event** (label 1) or a
sensor **failure** (label 2).

All code is our own (NumPy only). No scikit-learn or similar.

## Files

| File | Contents |
| --- | --- |
| `src/signoise/discrimination/features.py` | The five features and `build_feature_matrix` |
| `src/signoise/discrimination/classifiers.py` | `GaussianNaiveBayes`, `LogisticRegression` |
| `tests/test_role5.py` | Unit tests for both files |
| `scripts/run_discrimination.py` | End-to-end run with confusion matrices |

## Features

| Feature | Definition | Expected |
| --- | --- | --- |
| `neighbour_agreement` | Fraction of the node's neighbours whose detector score exceeds the threshold at any step in `[t-2, t+2]` | high for events, low for failures |
| `battery_level` | Raw voltage at `t` (last observed value in the window if missing) | low for failures |
| `battery_slope` | Least-squares slope (V/step) over the previous 12 steps, observed steps only | negative for failing nodes |
| `residual_voltage` | Observed voltage minus the voltage predicted from the node's own temperature (per-node line fitted on the training window) | strongly negative for failures |
| `cross_channel_consistency` | Pearson correlation of first differences of temperature and humidity over `[t-6, t]` (0 if undefined) | events move them together, failures look uncorrelated |

Notes:
- Battery features use raw (un-standardised, imputed) values. Neighbour agreement uses detector scores.
- The voltage-temperature fit ignores physically impossible readings (`config.VALID_RANGE`), so dying-sensor outliers do not distort it.
- Passing `random_neighbors(...)` instead of `knn_neighbors(...)` gives the random-neighbour control.

## Classifiers

Both expose `classes_`, `fit(X, y)`, `predict_proba(X)`, `predict(X)`.

- **GaussianNaiveBayes**: class priors plus per-class, per-feature mean and variance (`var_smoothing` added to each variance). Posteriors are computed in log space with the log-sum-exp trick.
- **LogisticRegression**: binary, positive class is the larger label (failure). L2-regularised cross-entropy, batch gradient descent, features standardised with training statistics, bias not penalised. `loss_curve_` holds the loss per iteration. `predict_proba` returns `(n, 2)` in `classes_` order.

## Usage

```python
from signoise.config import TEMP, VOLT
from signoise.discrimination import build_feature_matrix, GaussianNaiveBayes, LogisticRegression
from signoise.discrimination.features import fit_voltage_temperature

vfit = fit_voltage_temperature(raw_train[:, :, VOLT], raw_train[:, :, TEMP])
F, names = build_feature_matrix(cells, raw_values, scores, neighbors, vfit, threshold)
clf = LogisticRegression().fit(F_train, y_train)
pred = clf.predict(F_test)
```

`cells` is an `(M, 2)` array of flagged `(t, n)`, `raw_values` is `(T, N, 4)` raw
temperature, humidity, light, voltage, and `scores` is the `(T, N)` detector output.

## Run

```bash
pytest -q tests/test_role5.py
python scripts/run_discrimination.py --detector ksigma --folds 5
```

The script, per temporal fold:
1. imputes and robust z-scores using the training window only;
2. injects synthetic events and failures into the train and test windows (different seeds);
3. fits the detector on the clean train window and scores both windows;
4. flags the top-k valid cells (k = number of injected anomalous cells) and keeps the flagged cells that are true anomalies;
5. builds features (spatial and random neighbours), trains on the train anomalies, predicts the test anomalies.

Confusion matrices are summed over folds and saved to `results/discrimination.csv`
(not committed). Other detectors work through `--detector` once implemented.

## Results (k-sigma detector, 5 folds, data before 20 March, magnitude 4)

| Classifier | Neighbours | Accuracy | Balanced accuracy |
| --- | --- | --- | --- |
| Naive Bayes | spatial | 0.942 | 0.931 |
| Logistic regression | spatial | 0.956 | 0.945 |
| Naive Bayes | random | 0.781 | 0.791 |
| Logistic regression | random | 0.946 | 0.923 |

Confusion matrix, logistic regression with spatial neighbours (rows true, columns predicted):

| | pred event | pred failure |
| --- | --- | --- |
| true event | 731 | 16 |
| true failure | 34 | 351 |

## Caveats for the report

- **Oracle flagging.** The flag count k comes from the injected ground truth, as in the bench's precision@k. A deployed system would need a fixed threshold.
- **Voltage shortcut.** Injected failures include a voltage drop, so battery features alone separate them. This is why logistic regression barely drops with random neighbours while Naive Bayes does. A feature ablation (Stage D) would quantify it.
- **Synthetic labels only.** Results are on injected anomalies. Real failing nodes (weak voltage-collapse labels) have not been evaluated.
- **Single detector.** Only k-sigma has been run so far. Rerun with the other detectors when they land.
