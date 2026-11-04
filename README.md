# Signal or Noise

**Fault-versus-event anomaly discrimination in correlated sensor streams**
CSE 575 Statistical Machine Learning, Fall 2026 · Topic P12: Anomaly Detection · Team *Signal or Noise*

An unusual reading from a sensor network has two very different causes: a real
environmental **event** (a door opens, HVAC cycles, sunlight hits a window) or an
instrument **failure** (a degrading node emitting garbage). Standard detectors
can't tell them apart. We detect anomalies with five detectors we implement
ourselves, then classify each one as event or failure using spatial agreement
between neighbouring nodes and the node's battery telemetry.

Data: [Intel Berkeley Research Lab](http://db.csail.mit.edu/labdata/labdata.html),
54 nodes, temperature / humidity / light / voltage every 31 s, 28 Feb – 5 Apr 2004.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/download_data.py      # ~34 MB into data/raw/ (not committed)
python scripts/profile_data.py       # missingness + voltage tables and plot -> results/profile/
python scripts/run_bench.py          # detector bench on injected anomalies -> results/bench.csv
pytest -q                            # tests; unimplemented detectors are skipped
```

## Team and ownership

| Role | Owner | Code | Status |
| --- | --- | --- | --- |
| 1 Data, harness & integration | Manav | `data/`, `preprocess.py`, `evaluation/`, `imputation/simple.py`, `detectors/ksigma.py`, `scripts/` | base done |
| 2 Detectors I | Rashi | `detectors/knn.py`, `detectors/kmeans.py` | TODO |
| 3 Detectors II | Karthik | `detectors/pca.py`, `detectors/hmm.py` | TODO |
| 4 EM & imputation | Harsh | `detectors/gmm.py`, `imputation/em_gaussian.py` | TODO |
| 5 Discrimination | Tejas | `discrimination/features.py`, `discrimination/classifiers.py` | TODO |
| 6 Evaluation & results | Ziyad | `injection/inject.py` (v0 works), experiments, figures, results table | extend |

Every stub file starts with a docstring that specifies exactly what to build.
Cross-review pairs: Manav ↔ Ziyad, Rashi ↔ Karthik, Harsh ↔ Tejas.

## How the pieces fit

```
data/raw  ──load_intel──▶  SensorData (T steps × N nodes × C channels, NaN = missing)
                                │
              temporal_folds ◀──┤ (contiguous test blocks, gap on each side)
                                ▼
          impute (ffill now, EM later) ─▶ robust z-score on train ─▶ inject events/failures into test
                                                                         │
                         detectors: fit(train) → score(test) → (T, N) anomaly scores
                                                                         │
                       metrics: AUC overall / events / failures, precision@k, recall@k
                                                                         │
            Stage C (Role 5): features per flagged cell → Naive Bayes / logistic regression → event vs failure
```

### The detector contract

```python
class MyDetector(Detector):          # or PointwiseDetector for per-point models
    name = "mine"
    def fit(self, X):   ...          # X: (T, N, C) float, no NaN, z-scored
    def score(self, X): ...          # returns (T, N); higher = more anomalous
```

Register it in `src/signoise/detectors/__init__.py` (`REGISTRY`). The bench and the
contract tests in `tests/test_detectors.py` pick it up automatically.

## Things the base already handles (read before changing)

- **Time axis from timestamps, not `epoch`.** The epoch counter is 16-bit and resets
  around 21 March. Steps are `bin_epochs × 31 s`; default `bin_epochs=10` (~5 min,
  9,940 steps). Use `bin_epochs=1` for full resolution.
- **Outliers are kept on load.** Readings like 120 °C from dying nodes are the
  failure class we study. `config.VALID_RANGE` is only for profiling and weak labels.
- **After ~20 March most nodes are failing** (voltage < 2.3 V, impossible values).
  `run_bench.py` uses data before 2004-03-20 by default; the late period is for the
  weak-label evaluation.
- **Scored cells** are only those actually observed (not imputed) and not already
  real anomalies (`data.profile.real_anomaly_mask`).
- **Detectors see temperature, humidity, light only.** Voltage drifts as batteries
  drain and would dominate every z-score; it is a Stage C feature instead.
- **Robust standardisation** (median/MAD per node, fitted on train only).

## Current baseline (k-sigma, 5 folds, data before 20 March, magnitude 4σ)

| AUC | AUC events | AUC failures | precision@k |
| --- | --- | --- | --- |
| 0.75 ± 0.08 | 0.72 ± 0.10 | 0.91 ± 0.02 | 0.31 ± 0.09 |

Every learned detector should beat this. Events are the hard class for a
per-node threshold, which is the point of the project.

## Rules from the course

- All detectors and classifiers are **our own implementations**. NumPy / SciPy for
  arrays and linear algebra, pandas for loading, matplotlib for figures. No
  scikit-learn, hmmlearn, or similar for the methods themselves.
- The final report must say which code is ours and which is third-party.
- Key dates: results consolidated Nov 16–22, presentation Nov 23 – Dec 2, report due Dec 7.

## Git workflow

- Branch per role: `role2-knn-kmeans`, `role3-pca-hmm`, `role4-gmm-em`, `role5-discrimination`, `role6-evaluation`.
- Open a pull request into `main`; your cross-review partner approves before merge.
- `pytest -q` must pass. Never commit anything under `data/` or `results/`.
