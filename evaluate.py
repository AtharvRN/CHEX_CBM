"""
Evaluation script for trained CheXpert models.

Usage:
    python evaluate.py --checkpoint checkpoints/exp1/best_model.pth --data_dir /workspace/CheXpert-v1.0-small
"""

import argparse
import json
import os

import contextlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    classification_report
)
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import CheXpertDataset, get_transforms
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CheXpert classifier")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to CheXpert-v1.0-small directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results (default: same as checkpoint)")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save raw predictions to file")
    parser.add_argument("--plot_curves", action="store_true",
                        help="Generate ROC and PR curves")
    parser.add_argument("--split", type=str, default="valid",
                        choices=["valid", "test"],
                        help="Which CheXpert split to evaluate (valid/test)")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Explicit path to a CSV file containing image paths/labels (overrides --split)")
    
    return parser.parse_args()


def compute_metrics(targets: np.ndarray, predictions: np.ndarray, labels: list) -> dict:
    """Compute comprehensive metrics for multi-label classification."""
    results = {
        'per_class': {},
        'macro': {},
        'micro': {}
    }
    
    valid_aurocs = []
    valid_aps = []
    
    for i, label in enumerate(labels):
        unique_vals = np.unique(targets[:, i])
        
        if len(unique_vals) < 2:
            results['per_class'][label] = {
                'auroc': float('nan'),
                'ap': float('nan'),
                'positive_samples': int((targets[:, i] == 1).sum()),
                'negative_samples': int((targets[:, i] == 0).sum())
            }
        else:
            auroc = roc_auc_score(targets[:, i], predictions[:, i])
            ap = average_precision_score(targets[:, i], predictions[:, i])
            
            results['per_class'][label] = {
                'auroc': auroc,
                'ap': ap,
                'positive_samples': int((targets[:, i] == 1).sum()),
                'negative_samples': int((targets[:, i] == 0).sum())
            }
            valid_aurocs.append(auroc)
            valid_aps.append(ap)
    
    # Macro averages (average of per-class metrics)
    results['macro']['auroc'] = np.mean(valid_aurocs) if valid_aurocs else float('nan')
    results['macro']['ap'] = np.mean(valid_aps) if valid_aps else float('nan')
    
    # Micro averages (flatten all predictions)
    try:
        results['micro']['auroc'] = roc_auc_score(targets.ravel(), predictions.ravel())
        results['micro']['ap'] = average_precision_score(targets.ravel(), predictions.ravel())
    except:
        results['micro']['auroc'] = float('nan')
        results['micro']['ap'] = float('nan')
    
    return results


def plot_roc_curves(targets, predictions, labels, output_dir):
    """Plot ROC curves for each class."""
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.ravel()
    
    for i, label in enumerate(labels):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
            
        unique_vals = np.unique(targets[:, i])
        if len(unique_vals) >= 2:
            fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
            auroc = roc_auc_score(targets[:, i], predictions[:, i])
            ax.plot(fpr, tpr, label=f'AUC = {auroc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
        
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    # Hide unused subplots
    for i in range(len(labels), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150)
    plt.close()
    print(f"ROC curves saved to {output_dir}/roc_curves.png")


def plot_pr_curves(targets, predictions, labels, output_dir):
    """Plot Precision-Recall curves for each class."""
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.ravel()
    
    for i, label in enumerate(labels):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
            
        unique_vals = np.unique(targets[:, i])
        if len(unique_vals) >= 2:
            precision, recall, _ = precision_recall_curve(targets[:, i], predictions[:, i])
            ap = average_precision_score(targets[:, i], predictions[:, i])
            ax.plot(recall, precision, label=f'AP = {ap:.3f}')
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
        
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='lower left')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    # Hide unused subplots
    for i in range(len(labels), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=150)
    plt.close()
    print(f"PR curves saved to {output_dir}/pr_curves.png")


def main():
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_kwargs = {"map_location": "cpu", "weights_only": False}
    safe_ctx = contextlib.nullcontext()
    if hasattr(torch.serialization, "safe_globals"):
        safe_ctx = torch.serialization.safe_globals(
            ["numpy.core.multiarray.scalar"])
    with safe_ctx:
        checkpoint = torch.load(args.checkpoint, **load_kwargs)
    config = checkpoint['config']
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = args.output or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get labels from config
    labels = config['labels']
    num_classes = len(labels)
    print(f"Evaluating on {num_classes} classes: {labels}")
    
    # Paths
    img_root = os.path.dirname(args.data_dir)
    eval_csv = args.csv_path or os.path.join(
        args.data_dir, f"{args.split}.csv")
    print(f"Evaluating split '{args.split}' with CSV: {eval_csv}")
    
    # Dataset
    val_transform = get_transforms(config.get('img_size', 224), is_training=False)
    
    eval_dataset = CheXpertDataset(
        csv_path=eval_csv,
        img_root=img_root,
        transform=val_transform,
        labels=labels,
        uncertain_strategy=config.get('uncertain_strategy', 'ones'),
        frontal_only=config.get('frontal_only', True)
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    model = get_model(
        model_name=config['model'],
        num_classes=num_classes,
        pretrained=False,
        dropout=config.get('dropout', 0.0)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Inference
    print("\nRunning inference...")
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels_batch in tqdm(eval_loader):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(labels_batch)
    
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Convert to numpy
    probs = torch.sigmoid(logits).numpy()
    targets_np = targets.numpy()
    
    # Compute metrics
    print("\nComputing metrics...")
    results = compute_metrics(targets_np, probs, labels)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nMacro AUROC: {results['macro']['auroc']:.4f}")
    print(f"Macro AP: {results['macro']['ap']:.4f}")
    print(f"Micro AUROC: {results['micro']['auroc']:.4f}")
    print(f"Micro AP: {results['micro']['ap']:.4f}")
    
    print("\nPer-class Results:")
    print("-"*60)
    print(f"{'Label':<30} {'AUROC':<10} {'AP':<10} {'Pos':<8} {'Neg':<8}")
    print("-"*60)
    
    for label in labels:
        r = results['per_class'][label]
        auroc_str = f"{r['auroc']:.4f}" if not np.isnan(r['auroc']) else "N/A"
        ap_str = f"{r['ap']:.4f}" if not np.isnan(r['ap']) else "N/A"
        print(f"{label:<30} {auroc_str:<10} {ap_str:<10} {r['positive_samples']:<8} {r['negative_samples']:<8}")
    
    # Save results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Save predictions
    if args.save_predictions:
        np.savez(
            os.path.join(output_dir, 'predictions.npz'),
            probabilities=probs,
            targets=targets_np,
            labels=labels
        )
        print(f"Predictions saved to: {output_dir}/predictions.npz")
    
    # Plot curves
    if args.plot_curves:
        print("\nGenerating plots...")
        plot_roc_curves(targets_np, probs, labels, output_dir)
        plot_pr_curves(targets_np, probs, labels, output_dir)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
