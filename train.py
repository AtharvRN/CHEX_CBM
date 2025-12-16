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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    CheXpertDataset, 
    CHEXPERT_LABELS, 
    CHEXPERT_COMPETITION_LABELS,
    CHEXPERT_PATHOLOGY_LABELS,
    get_transforms
)
from models import get_model
from utils.metrics import compute_all_metrics
from utils.plotting import (
    plot_confusion_heatmap,
    plot_label_distribution,
    plot_per_class_auroc,
    plot_pr_curves,
    plot_roc_curves,
    plot_training_curves,
)
from utils.wandb_utils import WANDB_AVAILABLE, init_wandb, log_plots_to_wandb, log_to_wandb, wandb


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


def load_history(path):
    """Load history from disk if it exists."""
    if not os.path.exists(path):
        return [], 0.0
    try:
        with open(path, 'r') as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Warning: failed to load history ({path}); starting fresh.")
        return [], 0.0
    best_auroc = max(((entry.get('val_mean_auroc') or 0.0) for entry in history), default=0.0)
    print(f"Loaded {len(history)} history entries, best AUROC so far {best_auroc:.4f}")
    return history, best_auroc


def save_history(history, path):
    """Persist history to disk."""
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


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
    
    history_path = os.path.join(args.output, "history.json")
    history, best_auroc = load_history(history_path)
    
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
        save_history(history, history_path)
        
        # Log to wandb
        if WANDB_AVAILABLE and wandb_run:
            log_to_wandb(epoch, train_loss, val_aurocs, val_aps, current_lr, labels, best_auroc)
        
        # Generate plots periodically
        if args.save_plots and epoch % args.plot_every == 0:
            plot_training_curves(history, plots_dir)
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
        plot_training_curves(history, plots_dir)
        
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
