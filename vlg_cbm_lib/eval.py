from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


def evaluate(
    concept_activations: torch.Tensor,
    labels: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    label_names: List[str]
) -> Dict:
    """Evaluate model and return metrics."""

    c_norm = (concept_activations - mean) / std
    logits = c_norm @ W.T + b
    probs = torch.sigmoid(logits).numpy()
    targets = labels.numpy()

    metrics = {}
    valid_aurocs = []
    valid_aps = []

    for i, name in enumerate(label_names):
        if len(np.unique(targets[:, i])) > 1:
            auroc = roc_auc_score(targets[:, i], probs[:, i])
            ap = average_precision_score(targets[:, i], probs[:, i])
            metrics[f"auroc_{name}"] = auroc
            metrics[f"ap_{name}"] = ap
            valid_aurocs.append(auroc)
            valid_aps.append(ap)

    metrics["mean_auroc"] = np.mean(valid_aurocs) if valid_aurocs else 0.0
    metrics["mean_ap"] = np.mean(valid_aps) if valid_aps else 0.0

    return metrics


def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str]
) -> Dict:
    """Compute AUROC/AP for raw predictions."""
    metrics = {}
    valid_aurocs = []
    valid_aps = []

    for i, name in enumerate(label_names):
        if len(np.unique(y_true[:, i])) > 1:
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            metrics[f"auroc_{name}"] = auroc
            metrics[f"ap_{name}"] = ap
            valid_aurocs.append(auroc)
            valid_aps.append(ap)

    metrics["mean_auroc"] = np.mean(valid_aurocs) if valid_aurocs else 0.0
    metrics["mean_ap"] = np.mean(valid_aps) if valid_aps else 0.0
    return metrics


def evaluate_baseline_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    label_names: List[str]
) -> Dict:
    """Evaluate the backbone classifier before CBM training."""
    model = model.to(device)
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to(device)
            logits = model(images)
            preds.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels_batch.numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    return evaluate_multilabel(targets, preds, label_names)
