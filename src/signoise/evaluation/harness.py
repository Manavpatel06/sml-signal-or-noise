"""Shared evaluation harness for the Stage A detector bench.

For every temporal fold:
  1. impute train and test windows separately (no information crosses folds);
  2. z-score with robust statistics from the training window only;
  3. inject synthetic events and failures into the TEST window;
  4. fit each detector on train, score the injected test window;
  5. compute AUC / precision@k / recall@k against the injected labels.

Only cells that were actually observed (not imputed) are scored, and cells
that are probably REAL anomalies (voltage already collapsed, or a physically
impossible reading; see data.profile.real_anomaly_mask) are left out, because
they are unlabelled and would otherwise count as false positives.

Detectors see only ``detect_channels`` (temperature, humidity, light by
default). Battery voltage drifts steadily as batteries drain, so it would
dominate any z-score; it is used in Stage C (discrimination) instead.

Detectors that still raise NotImplementedError are reported with status
"todo" instead of crashing the run, so the bench works while the team builds.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from signoise.config import HUMID, LIGHT, TEMP
from signoise.data.load import SensorData
from signoise.data.neighbors import knn_neighbors
from signoise.data.profile import real_anomaly_mask
from signoise.data.splits import temporal_folds
from signoise.detectors import REGISTRY, Detector
from signoise.evaluation.metrics import precision_recall_at_k, roc_auc
from signoise.imputation.simple import ffill_impute
from signoise.injection.inject import EVENT, FAILURE, inject_anomalies
from signoise.preprocess import Standardizer

Imputer = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def ffill_both(train: np.ndarray, test: np.ndarray):
    return ffill_impute(train), ffill_impute(test)


def make_detector(spec) -> Detector:
    """spec: registry name, (name, kwargs) tuple, or a Detector instance."""
    if isinstance(spec, Detector):
        return spec
    if isinstance(spec, str):
        return REGISTRY[spec]()
    name, kwargs = spec
    return REGISTRY[name](**kwargs)


def score_detection(labels: np.ndarray, scores: np.ndarray, valid: np.ndarray) -> dict:
    y = labels[valid]
    s = scores[valid]
    any_anom = y > 0
    k = int(any_anom.sum())
    p, r = precision_recall_at_k(any_anom, s, k)
    ev = (y == EVENT) | (y == 0)
    fa = (y == FAILURE) | (y == 0)
    return {
        "auc": roc_auc(any_anom, s),
        "auc_event": roc_auc(y[ev] == EVENT, s[ev]),
        "auc_failure": roc_auc(y[fa] == FAILURE, s[fa]),
        "precision_at_k": p,
        "recall_at_k": r,
        "n_anomalous": k,
        "n_cells": int(len(y)),
    }


def run_detection_bench(
    data: SensorData,
    detectors=("ksigma",),
    n_folds: int = 5,
    gap: int = 12,
    k_neighbors: int = 4,
    imputer: Imputer = ffill_both,
    inject_kwargs: dict | None = None,
    seed: int = 0,
    exclude_real_failures: bool = True,
    detect_channels: tuple = (TEMP, HUMID, LIGHT),
    verbose: bool = True,
) -> pd.DataFrame:
    inject_kwargs = dict(inject_kwargs or {})
    neighbors = knn_neighbors(data.coords, k=k_neighbors)
    real = real_anomaly_mask(data) if exclude_real_failures else None
    observed = np.isfinite(data.values).all(axis=2)
    rows = []

    for fold, (tr, te) in enumerate(temporal_folds(data.shape[0], n_folds, gap)):
        Xtr, Xte = imputer(data.values[tr], data.values[te])
        scaler = Standardizer(per_node=True).fit(Xtr)
        Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
        Zinj, labels, episodes = inject_anomalies(
            Zte, neighbors, seed=seed + fold, **inject_kwargs)
        ch = list(detect_channels)
        Ztr_d, Zinj_d = Ztr[:, :, ch], Zinj[:, :, ch]
        valid = observed[te].copy()
        if real is not None:
            valid &= ~real[te]

        for spec in detectors:
            det = make_detector(spec)
            row = {"fold": fold, "detector": det.name}
            t0 = time.perf_counter()
            try:
                scores = det.fit(Ztr_d).score(Zinj_d)
                row.update(score_detection(labels, scores, valid))
                row["status"] = "ok"
            except NotImplementedError:
                row["status"] = "todo"
            row["seconds"] = round(time.perf_counter() - t0, 2)
            rows.append(row)
            if verbose:
                msg = (f"auc={row['auc']:.3f} event={row['auc_event']:.3f} "
                       f"failure={row['auc_failure']:.3f}" if row["status"] == "ok"
                       else "not implemented yet")
                print(f"fold {fold} {det.name:>8}: {msg}")

    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across folds per detector (ok rows only)."""
    ok = results[results["status"] == "ok"]
    if ok.empty:
        return ok
    cols = ["auc", "auc_event", "auc_failure", "precision_at_k", "recall_at_k"]
    return ok.groupby("detector")[cols].agg(["mean", "std"]).round(3)
