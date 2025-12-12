"""Metric utilities for CheXpert training."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def compute_auroc(targets: np.ndarray, predictions: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Compute AUROC for each label plus the valid mean."""
    aurocs: Dict[str, float] = {}
    valid_aurocs = []

    for idx, label in enumerate(labels):
        unique_vals = np.unique(targets[:, idx])
        if len(unique_vals) < 2:
            aurocs[label] = float("nan")
            continue
        auc = roc_auc_score(targets[:, idx], predictions[:, idx])
        aurocs[label] = auc
        valid_aurocs.append(auc)

    aurocs["mean"] = np.mean(valid_aurocs) if valid_aurocs else float("nan")
    return aurocs


def compute_all_metrics(
    targets: np.ndarray, predictions: np.ndarray, labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute AUROC/AP metrics for each class and aggregate means."""
    metrics = {"auroc": {}, "ap": {}}
    valid_aurocs = []
    valid_aps = []

    for idx, label in enumerate(labels):
        unique_vals = np.unique(targets[:, idx])
        if len(unique_vals) < 2:
            metrics["auroc"][label] = float("nan")
            metrics["ap"][label] = float("nan")
            continue
        auc = roc_auc_score(targets[:, idx], predictions[:, idx])
        ap = average_precision_score(targets[:, idx], predictions[:, idx])
        metrics["auroc"][label] = auc
        metrics["ap"][label] = ap
        valid_aurocs.append(auc)
        valid_aps.append(ap)

    metrics["auroc"]["mean"] = np.mean(valid_aurocs) if valid_aurocs else float("nan")
    metrics["ap"]["mean"] = np.mean(valid_aps) if valid_aps else float("nan")
    return metrics


__all__ = ["compute_auroc", "compute_all_metrics"]
