"""Plotting helpers for CheXpert training."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_training_curves(history: List[Dict], output_dir: str) -> str:
    """Plot training loss, validation AUROC, and learning-rate curves."""
    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    val_aurocs = [entry["val_mean_auroc"] for entry in history]
    lrs = [entry["lr"] for entry in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_losses, "b-", linewidth=2, marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_aurocs, "g-", linewidth=2, marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean AUROC")
    axes[1].set_title("Validation Mean AUROC")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    axes[2].plot(epochs, lrs, "r-", linewidth=2, marker="o")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale("log")

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_per_class_auroc(aurocs: Dict[str, float], labels: List[str], output_dir: str, epoch: int | None = None) -> str:
    """Plot a per-class AUROC bar chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    values = [aurocs.get(label, 0) for label in labels]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim([0, 1])
    ax.axhline(
        y=aurocs.get("mean", 0),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {aurocs.get("mean", 0):.4f}',
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    title = "Per-Class Validation AUROC"
    if epoch is not None:
        title += f" (Epoch {epoch})"
    ax.set_title(title)

    for bar, val in zip(bars, values):
        if np.isnan(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    filename = f"per_class_auroc_epoch{epoch}.png" if epoch else "per_class_auroc.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_roc_curves(targets: np.ndarray, predictions: np.ndarray, labels: List[str], output_dir: str) -> str:
    """Plot ROC curves for all labels."""
    num_labels = len(labels)
    cols = min(4, num_labels)
    rows = (num_labels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)

    for idx, label in enumerate(labels):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        unique_vals = np.unique(targets[:, idx])
        if len(unique_vals) >= 2:
            fpr, tpr, _ = roc_curve(targets[:, idx], predictions[:, idx])
            auc = roc_auc_score(targets[:, idx], predictions[:, idx])
            ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(label)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    for idx in range(num_labels, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_pr_curves(targets: np.ndarray, predictions: np.ndarray, labels: List[str], output_dir: str) -> str:
    """Plot precision-recall curves."""
    num_labels = len(labels)
    cols = min(4, num_labels)
    rows = (num_labels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)

    for idx, label in enumerate(labels):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        unique_vals = np.unique(targets[:, idx])
        if len(unique_vals) >= 2:
            precision, recall, _ = precision_recall_curve(targets[:, idx], predictions[:, idx])
            ap = average_precision_score(targets[:, idx], predictions[:, idx])
            ax.plot(recall, precision, "b-", linewidth=2, label=f"AP={ap:.3f}")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(label)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

    for idx in range(num_labels, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_confusion_heatmap(
    targets: np.ndarray,
    predictions: np.ndarray,
    labels: List[str],
    output_dir: str,
    threshold: float = 0.5,
) -> str:
    """Plot heatmap summarizing per-class classification metrics."""
    preds_binary = (predictions >= threshold).astype(int)
    tp = ((preds_binary == 1) & (targets == 1)).sum(axis=0)
    fp = ((preds_binary == 1) & (targets == 0)).sum(axis=0)
    tn = ((preds_binary == 0) & (targets == 0)).sum(axis=0)
    fn = ((preds_binary == 0) & (targets == 1)).sum(axis=0)

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision_vals = tp / (tp + fp + 1e-8)
    f1 = 2 * precision_vals * sensitivity / (precision_vals + sensitivity + 1e-8)
    data = np.array([sensitivity, specificity, precision_vals, f1])

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Sensitivity", "Specificity", "Precision", "F1"])

    for i in range(4):
        for j in range(len(labels)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label="Score")
    ax.set_title(f"Classification Metrics (threshold={threshold})")
    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_label_distribution(targets: np.ndarray, labels: List[str], output_dir: str, split: str = "train") -> str:
    """Plot class distribution histograms for a dataset split."""
    pos_counts = targets.sum(axis=0)
    total = len(targets)
    pos_ratios = pos_counts / total

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels)))
    axes[0].bar(range(len(labels)), pos_counts, color=colors)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{split.capitalize()} Set: Positive Label Counts")
    axes[0].grid(axis="y", alpha=0.3)

    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(labels)))
    axes[1].bar(range(len(labels)), pos_ratios * 100, color=colors)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("Positive Rate (%)")
    axes[1].set_title(f"{split.capitalize()} Set: Class Imbalance")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Balanced")

    plt.tight_layout()
    path = os.path.join(output_dir, f"{split}_label_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


__all__ = [
    "plot_training_curves",
    "plot_per_class_auroc",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_confusion_heatmap",
    "plot_label_distribution",
]
