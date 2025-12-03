"""
Training script for CheXpert multi-label classification.

Usage:
    python train.py --data_dir /workspace/CheXpert-v1.0-small --output checkpoints/exp1
    
    # With competition labels only (5 classes)
    python train.py --data_dir /workspace/CheXpert-v1.0-small --competition_labels --output checkpoints/exp_comp
    
    # With different uncertain handling
    python train.py --data_dir /workspace/CheXpert-v1.0-small --uncertain_strategy zeros --output checkpoints/exp_uzero
    
    # With wandb logging
    python train.py --data_dir /workspace/CheXpert-v1.0-small --use_wandb --wandb_project chexpert --output checkpoints/exp1
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from dataset import (
    CheXpertDataset, 
    CHEXPERT_LABELS, 
    CHEXPERT_COMPETITION_LABELS,
    CHEXPERT_PATHOLOGY_LABELS,
    get_transforms
)
from models import get_model

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CheXpert multi-label classifier")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to CheXpert-v1.0-small directory")
    parser.add_argument("--competition_labels", action="store_true",
                        help="Use only 5 competition labels instead of all 14")
    parser.add_argument("--pathology_labels", action="store_true",
                        help="Use 12 pathology labels (excludes 'No Finding' and 'Support Devices')")
    parser.add_argument("--uncertain_strategy", type=str, default="ones",
                        choices=["ones", "zeros", "ignore"],
                        help="How to handle uncertain labels (-1)")
    parser.add_argument("--frontal_only", action="store_true", default=True,
                        help="Use only frontal views")
    parser.add_argument("--limit_samples", type=int, default=None,
                        help="Limit training samples (for debugging/testing)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model
    parser.add_argument("--model", type=str, default="densenet121",
                        help="Model architecture: densenet121, resnet50, or xrv-* (e.g., xrv-all, xrv-chex)")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights (ImageNet for standard, chest X-ray for XRV)")
    parser.add_argument("--use_xrv_head", action="store_true",
                        help="For XRV models: use pretrained classifier head instead of new one")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout before classifier")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone, only train classifier")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_pos_weight", action="store_true",
                        help="Use class-balanced positive weights in BCE loss")
    
    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="chexpert-cbm",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (team/username)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not specified)")
    
    # Plotting
    parser.add_argument("--save_plots", action="store_true", default=True,
                        help="Save training plots locally")
    parser.add_argument("--plot_every", type=int, default=1,
                        help="Generate plots every N epochs")
    
    return parser.parse_args()


def compute_auroc(targets: np.ndarray, predictions: np.ndarray, labels: list) -> dict:
    """
    Compute AUROC for each label and mean AUROC.
    
    Args:
        targets: Ground truth labels (N, num_classes)
        predictions: Predicted probabilities (N, num_classes)
        labels: List of label names
        
    Returns:
        Dictionary with per-class and mean AUROC
    """
    aurocs = {}
    valid_aurocs = []
    
    for i, label in enumerate(labels):
        # Check if there are both positive and negative samples
        unique_vals = np.unique(targets[:, i])
        if len(unique_vals) < 2:
            aurocs[label] = float('nan')
        else:
            auc = roc_auc_score(targets[:, i], predictions[:, i])
            aurocs[label] = auc
            valid_aurocs.append(auc)
    
    aurocs['mean'] = np.mean(valid_aurocs) if valid_aurocs else float('nan')
    return aurocs


def compute_all_metrics(targets: np.ndarray, predictions: np.ndarray, labels: list) -> dict:
    """
    Compute comprehensive metrics for each label.
    
    Args:
        targets: Ground truth labels (N, num_classes)
        predictions: Predicted probabilities (N, num_classes)
        labels: List of label names
        
    Returns:
        Dictionary with per-class AUROC, AP, and means
    """
    metrics = {'auroc': {}, 'ap': {}}
    valid_aurocs = []
    valid_aps = []
    
    for i, label in enumerate(labels):
        unique_vals = np.unique(targets[:, i])
        if len(unique_vals) < 2:
            metrics['auroc'][label] = float('nan')
            metrics['ap'][label] = float('nan')
        else:
            auc = roc_auc_score(targets[:, i], predictions[:, i])
            ap = average_precision_score(targets[:, i], predictions[:, i])
            metrics['auroc'][label] = auc
            metrics['ap'][label] = ap
            valid_aurocs.append(auc)
            valid_aps.append(ap)
    
    metrics['auroc']['mean'] = np.mean(valid_aurocs) if valid_aurocs else float('nan')
    metrics['ap']['mean'] = np.mean(valid_aps) if valid_aps else float('nan')
    return metrics


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_training_curves(history: list, output_dir: str, labels: list):
    """Plot training loss and validation AUROC curves."""
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_aurocs = [h['val_mean_auroc'] for h in history]
    lrs = [h['lr'] for h in history]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training loss
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Validation AUROC
    axes[1].plot(epochs, val_aurocs, 'g-', linewidth=2, marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean AUROC')
    axes[1].set_title('Validation Mean AUROC')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Learning rate
    axes[2].plot(epochs, lrs, 'r-', linewidth=2, marker='o')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return os.path.join(output_dir, 'training_curves.png')


def plot_per_class_auroc(aurocs: dict, labels: list, output_dir: str, epoch: int = None):
    """Plot bar chart of per-class AUROC scores."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Get values for each label
    values = [aurocs.get(label, 0) for label in labels]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('AUROC')
    ax.set_ylim([0, 1])
    ax.axhline(y=aurocs.get('mean', 0), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {aurocs.get("mean", 0):.4f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    title = 'Per-Class Validation AUROC'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    ax.set_title(title)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    filename = f'per_class_auroc_epoch{epoch}.png' if epoch else 'per_class_auroc.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return os.path.join(output_dir, filename)


def plot_roc_curves(targets: np.ndarray, predictions: np.ndarray, labels: list, output_dir: str):
    """Plot ROC curves for all classes."""
    num_labels = len(labels)
    cols = min(4, num_labels)
    rows = (num_labels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_2d(axes)
    
    for i, label in enumerate(labels):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        unique_vals = np.unique(targets[:, i])
        if len(unique_vals) >= 2:
            fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
            auc = roc_auc_score(targets[:, i], predictions[:, i])
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(label)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_labels, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return os.path.join(output_dir, 'roc_curves.png')


def plot_pr_curves(targets: np.ndarray, predictions: np.ndarray, labels: list, output_dir: str):
    """Plot Precision-Recall curves for all classes."""
    num_labels = len(labels)
    cols = min(4, num_labels)
    rows = (num_labels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_2d(axes)
    
    for i, label in enumerate(labels):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        unique_vals = np.unique(targets[:, i])
        if len(unique_vals) >= 2:
            precision, recall, _ = precision_recall_curve(targets[:, i], predictions[:, i])
            ap = average_precision_score(targets[:, i], predictions[:, i])
            ax.plot(recall, precision, 'b-', linewidth=2, label=f'AP={ap:.3f}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(label)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_labels, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return os.path.join(output_dir, 'pr_curves.png')


def plot_confusion_heatmap(targets: np.ndarray, predictions: np.ndarray, labels: list, 
                           output_dir: str, threshold: float = 0.5):
    """Plot a heatmap showing prediction quality for each class."""
    preds_binary = (predictions >= threshold).astype(int)
    
    # Calculate TP, FP, TN, FN for each class
    tp = ((preds_binary == 1) & (targets == 1)).sum(axis=0)
    fp = ((preds_binary == 1) & (targets == 0)).sum(axis=0)
    tn = ((preds_binary == 0) & (targets == 0)).sum(axis=0)
    fn = ((preds_binary == 0) & (targets == 1)).sum(axis=0)
    
    # Calculate metrics
    sensitivity = tp / (tp + fn + 1e-8)  # Recall
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
    
    # Create heatmap data
    data = np.array([sensitivity, specificity, precision, f1])
    
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(4))
    ax.set_yticklabels(['Sensitivity', 'Specificity', 'Precision', 'F1'])
    
    # Add text annotations
    for i in range(4):
        for j in range(len(labels)):
            val = data[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Score')
    ax.set_title(f'Classification Metrics (threshold={threshold})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return os.path.join(output_dir, 'metrics_heatmap.png')


def plot_label_distribution(targets: np.ndarray, labels: list, output_dir: str, split: str = 'train'):
    """Plot the distribution of positive labels in the dataset."""
    pos_counts = targets.sum(axis=0)
    total = len(targets)
    pos_ratios = pos_counts / total
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute counts
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels)))
    bars = axes[0].bar(range(len(labels)), pos_counts, color=colors)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{split.capitalize()} Set: Positive Label Counts')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Positive ratio
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(labels)))
    bars = axes[1].bar(range(len(labels)), pos_ratios * 100, color=colors)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('Positive Rate (%)')
    axes[1].set_title(f'{split.capitalize()} Set: Class Imbalance')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Balanced')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{split}_label_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return os.path.join(output_dir, f'{split}_label_distribution.png')


# ============================================================================
# Wandb Utility Functions
# ============================================================================

def init_wandb(args, config):
    """Initialize wandb run."""
    if not args.use_wandb:
        return None
    
    if not WANDB_AVAILABLE:
        print("Warning: wandb requested but not installed. Skipping wandb logging.")
        return None
    
    run_name = args.wandb_run_name
    if run_name is None:
        run_name = f"{args.model}_{args.uncertain_strategy}"
        if args.competition_labels:
            run_name += "_comp"
        run_name += f"_lr{args.lr}_bs{args.batch_size}"
        if args.limit_samples:
            run_name += f"_n{args.limit_samples}"
    
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
        tags=[args.model, args.uncertain_strategy, 
              "competition" if args.competition_labels else "all_labels"],
        reinit=True
    )
    
    # Log code
    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    
    return run


def log_to_wandb(epoch, train_loss, val_aurocs, val_aps, lr, labels, best_auroc):
    """Log metrics to wandb."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    log_dict = {
        'epoch': epoch,
        'train/loss': train_loss,
        'val/mean_auroc': val_aurocs['mean'],
        'val/mean_ap': val_aps['mean'],
        'train/learning_rate': lr,
        'val/best_auroc': best_auroc,
    }
    
    # Per-class metrics
    for label in labels:
        log_dict[f'val/auroc/{label}'] = val_aurocs.get(label, float('nan'))
        log_dict[f'val/ap/{label}'] = val_aps.get(label, float('nan'))
    
    wandb.log(log_dict)


def log_plots_to_wandb(history, aurocs, labels, targets, predictions, output_dir):
    """Log plots to wandb."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    # Training curves
    training_curves_path = plot_training_curves(history, output_dir, labels)
    wandb.log({"charts/training_curves": wandb.Image(training_curves_path)})
    
    # Per-class AUROC bar chart
    auroc_bar_path = plot_per_class_auroc(aurocs, labels, output_dir)
    wandb.log({"charts/per_class_auroc": wandb.Image(auroc_bar_path)})
    
    # ROC curves
    roc_path = plot_roc_curves(targets, predictions, labels, output_dir)
    wandb.log({"charts/roc_curves": wandb.Image(roc_path)})
    
    # PR curves
    pr_path = plot_pr_curves(targets, predictions, labels, output_dir)
    wandb.log({"charts/pr_curves": wandb.Image(pr_path)})
    
    # Confusion heatmap
    heatmap_path = plot_confusion_heatmap(targets, predictions, labels, output_dir)
    wandb.log({"charts/metrics_heatmap": wandb.Image(heatmap_path)})


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)
        pbar.set_postfix({'loss': total_loss / num_samples})
    
    return total_loss / num_samples


def evaluate(model, loader, device, label_names):
    """Evaluate model and compute metrics."""
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(labels)
    
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Convert logits to probabilities
    probs = torch.sigmoid(logits).numpy()
    targets = targets.numpy()
    
    # Compute all metrics
    metrics = compute_all_metrics(targets, probs, label_names)
    
    return metrics, targets, probs


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Select labels
    if args.competition_labels:
        labels = CHEXPERT_COMPETITION_LABELS
        print("Using 5 competition labels")
    elif args.pathology_labels:
        labels = CHEXPERT_PATHOLOGY_LABELS
        print("Using 12 pathology labels (excluding 'No Finding' and 'Support Devices')")
    else:
        labels = CHEXPERT_LABELS
        print("Using all 14 labels")
    
    num_classes = len(labels)
    
    # Paths
    train_csv = os.path.join(args.data_dir, "train.csv")
    val_csv = os.path.join(args.data_dir, "valid.csv")
    img_root = os.path.dirname(args.data_dir)  # Parent directory
    
    # Transforms
    train_transform = get_transforms(args.img_size, is_training=True)
    val_transform = get_transforms(args.img_size, is_training=False)
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = CheXpertDataset(
        csv_path=train_csv,
        img_root=img_root,
        transform=train_transform,
        labels=labels,
        uncertain_strategy=args.uncertain_strategy,
        frontal_only=args.frontal_only
    )
    
    # Limit training samples if specified (for testing/debugging)
    if args.limit_samples is not None and args.limit_samples < len(train_dataset):
        print(f"Limiting training to {args.limit_samples} samples (seed={args.seed})")
        # Use seeded random permutation for consistent subset
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(len(train_dataset))[:args.limit_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        # Recompute targets for the subset (for pos_weight calculation)
        subset_targets = torch.stack([train_dataset.dataset.targets[i] for i in indices])
        train_dataset.targets = subset_targets
    
    val_dataset = CheXpertDataset(
        csv_path=val_csv,
        img_root=img_root,
        transform=val_transform,
        labels=labels,
        uncertain_strategy=args.uncertain_strategy,
        frontal_only=args.frontal_only
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    print(f"\nCreating model: {args.model}")
    
    # For XRV models, pass target labels for output mapping
    if args.model.startswith('xrv'):
        from models import XRV_WEIGHTS
        if args.model not in XRV_WEIGHTS and not args.model.startswith('densenet121-res224'):
            print(f"Available XRV models: {list(XRV_WEIGHTS.keys())}")
        
        model = get_model(
            model_name=args.model,
            num_classes=num_classes,
            pretrained=args.pretrained,
            dropout=args.dropout,
            target_labels=labels
        )
        # Set use_pretrained_head based on flag
        if hasattr(model, 'use_pretrained_head'):
            model.use_pretrained_head = args.use_xrv_head
    else:
        model = get_model(
            model_name=args.model,
            num_classes=num_classes,
            pretrained=args.pretrained,
            dropout=args.dropout
        )
    model = model.to(device)
    
    if args.freeze_backbone:
        model.freeze_backbone()
        print("Backbone frozen, only training classifier")
    
    # Loss function
    if args.use_pos_weight:
        # Get targets - handle both Dataset and Subset
        if hasattr(train_dataset, 'targets'):
            targets = train_dataset.targets
        else:
            targets = train_dataset.dataset.targets
        pos_counts = targets.sum(dim=0)
        neg_counts = len(targets) - pos_counts
        pos_counts = torch.clamp(pos_counts, min=1.0)
        pos_weight = (neg_counts / pos_counts).to(device)
        print(f"Using pos_weight: {pos_weight.tolist()}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Save config
    config = vars(args)
    config['labels'] = labels
    config['num_classes'] = num_classes
    config['train_samples'] = len(train_dataset)
    config['val_samples'] = len(val_dataset)
    config['start_time'] = datetime.now().isoformat()
    
    with open(os.path.join(args.output, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize wandb
    wandb_run = init_wandb(args, config)
    
    # Plot and log initial data distribution
    if args.save_plots:
        os.makedirs(os.path.join(args.output, 'plots'), exist_ok=True)
        plots_dir = os.path.join(args.output, 'plots')
        
        # Plot label distribution
        if hasattr(train_dataset, 'targets'):
            train_targets = train_dataset.targets.numpy()
        else:
            train_targets = train_dataset.dataset.targets.numpy()
        plot_label_distribution(train_targets, labels, plots_dir, 'train')
        
        if WANDB_AVAILABLE and wandb_run:
            wandb.log({"data/train_distribution": wandb.Image(
                os.path.join(plots_dir, 'train_label_distribution.png'))})
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_auroc = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        metrics, val_targets, val_preds = evaluate(model, val_loader, device, labels)
        val_aurocs = metrics['auroc']
        val_aps = metrics['ap']
        mean_auroc = val_aurocs['mean']
        mean_ap = val_aps['mean']
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Log
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Mean AUROC: {mean_auroc:.4f} | Mean AP: {mean_ap:.4f}")
        print("Per-class AUROC:")
        for label in labels:
            print(f"  {label}: AUROC={val_aurocs[label]:.4f}, AP={val_aps[label]:.4f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mean_auroc': mean_auroc,
            'val_mean_ap': mean_ap,
            'val_aurocs': val_aurocs,
            'val_aps': val_aps,
            'lr': current_lr
        })
        
        # Log to wandb
        if WANDB_AVAILABLE and wandb_run:
            log_to_wandb(epoch, train_loss, val_aurocs, val_aps, current_lr, labels, best_auroc)
        
        # Generate plots periodically
        if args.save_plots and epoch % args.plot_every == 0:
            plot_training_curves(history, plots_dir, labels)
            plot_per_class_auroc(val_aurocs, labels, plots_dir, epoch)
        
        # Save best model
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auroc': best_auroc,
                'config': config
            }, os.path.join(args.output, "best_model.pth"))
            print(f"*** New best model saved with AUROC: {best_auroc:.4f} ***")
            
            # Save best predictions
            np.save(os.path.join(args.output, "best_val_predictions.npy"), val_preds)
            np.save(os.path.join(args.output, "best_val_targets.npy"), val_targets)
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auroc': best_auroc,
            'config': config
        }, os.path.join(args.output, "latest_model.pth"))
    
    # Save training history
    with open(os.path.join(args.output, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate final plots
    if args.save_plots:
        print("\nGenerating final plots...")
        
        # Training curves
        plot_training_curves(history, plots_dir, labels)
        
        # Load best predictions for final plots
        best_preds = np.load(os.path.join(args.output, "best_val_predictions.npy"))
        best_targets = np.load(os.path.join(args.output, "best_val_targets.npy"))
        
        # Best model metrics
        best_metrics = compute_all_metrics(best_targets, best_preds, labels)
        
        # Final AUROC bar chart
        plot_per_class_auroc(best_metrics['auroc'], labels, plots_dir)
        
        # ROC curves
        plot_roc_curves(best_targets, best_preds, labels, plots_dir)
        
        # PR curves
        plot_pr_curves(best_targets, best_preds, labels, plots_dir)
        
        # Confusion heatmap
        plot_confusion_heatmap(best_targets, best_preds, labels, plots_dir)
        
        # Log final plots to wandb
        if WANDB_AVAILABLE and wandb_run:
            log_plots_to_wandb(history, best_metrics['auroc'], labels, 
                             best_targets, best_preds, plots_dir)
    
    # Finish wandb run
    if WANDB_AVAILABLE and wandb_run:
        # Log final summary
        wandb.run.summary['best_auroc'] = best_auroc
        wandb.run.summary['best_epoch'] = history[np.argmax([h['val_mean_auroc'] for h in history])]['epoch']
        
        # Save model artifact
        artifact = wandb.Artifact(f'model-{wandb.run.id}', type='model')
        artifact.add_file(os.path.join(args.output, "best_model.pth"))
        wandb.log_artifact(artifact)
        
        wandb.finish()
    
    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Best Val AUROC: {best_auroc:.4f}")
    print(f"Checkpoints saved to: {args.output}")
    if args.save_plots:
        print(f"Plots saved to: {plots_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
