"""
Evaluation script for CheXpert multi-label classification models.
Supports standard, XRV, and CBM models.

Usage:
    python eval.py \
        --data_dir /workspace/CheXpert-v1.0-small \
        --model xrv-all \
        --pathology_labels \
        --use_xrv_head \
        --checkpoint checkpoints/xrv_all_finetune/best_model.pth \
        --output eval_results/xrv_all_finetune

    # For XRV pretrained head (no checkpoint needed)
    python eval.py \
        --data_dir /workspace/CheXpert-v1.0-small \
        --model xrv-all \
        --pathology_labels \
        --use_xrv_head \
        --output eval_results/xrv_all_pretrained
"""
import argparse
import os
import json
import numpy as np
import torch
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
from train import plot_per_class_auroc

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CheXpert model")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="densenet121")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (not needed for XRV pretrained head)")
    parser.add_argument("--competition_labels", action="store_true")
    parser.add_argument("--pathology_labels", action="store_true")
    parser.add_argument("--use_xrv_head", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--nec_metrics",
        type=str,
        nargs="*",
        help="Optional list of ModelName=path/to/nec_metrics.csv to print NEC comparison table"
    )
    return parser.parse_args()

def get_labels(args):
    if args.competition_labels:
        return CHEXPERT_COMPETITION_LABELS
    elif args.pathology_labels:
        return CHEXPERT_PATHOLOGY_LABELS
    else:
        return CHEXPERT_LABELS

def compute_metrics(targets, preds, labels):
    from sklearn.metrics import roc_auc_score, average_precision_score
    metrics = {'auroc': {}, 'ap': {}}
    valid_aurocs, valid_aps = [], []
    for i, label in enumerate(labels):
        y_true, y_pred = targets[:, i], preds[:, i]
        if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
            metrics['auroc'][label] = float('nan')
        else:
            auc = roc_auc_score(y_true, y_pred)
            metrics['auroc'][label] = auc
            valid_aurocs.append(auc)
        ap = average_precision_score(y_true, y_pred)
        metrics['ap'][label] = ap
        valid_aps.append(ap)
    metrics['auroc']['mean'] = np.nanmean(valid_aurocs)
    metrics['ap']['mean'] = np.nanmean(valid_aps)
    return metrics


def load_nec_csv(path):
    data = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"NEC metrics file not found: {path}")
    with open(path, "r") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            nec_str, auroc_str, _ = line.split(",")
            data[int(nec_str)] = float(auroc_str)
    return data


def print_nec_table(nec_sources):
    """Print NEC comparison table for multiple models."""
    tables = {}
    for entry in nec_sources:
        if "=" not in entry:
            raise ValueError("NEC metrics must be in the form ModelName=path/to/file.csv")
        name, path = entry.split("=", 1)
        tables[name] = load_nec_csv(path)
    all_nec = sorted({nec for model in tables.values() for nec in model.keys()})
    header = ["NEC"] + list(tables.keys())
    print("\nNEC Comparison (Mean AUROC):")
    print("  " + "  ".join(f"{h:>10}" for h in header))
    for nec in all_nec:
        row = [f"{nec:>10}"]
        for name in tables.keys():
            val = tables[name].get(nec, float('nan'))
            row.append(f"{val:>10.4f}" if not np.isnan(val) else f"{np.nan:>10}")
        print("  " + "  ".join(row))

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    labels = get_labels(args)
    num_classes = len(labels)

    # Load validation set
    val_csv = os.path.join(args.data_dir, "valid.csv")
    img_root = os.path.dirname(args.data_dir)
    transform = get_transforms(args.img_size, is_training=False)
    val_dataset = CheXpertDataset(val_csv, img_root, labels=labels, transform=transform, frontal_only=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        target_labels=labels
    )
    model = model.to(device)
    model.eval()
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        load_kwargs = {"map_location": device}
        import inspect
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        state = torch.load(args.checkpoint, **load_kwargs)
        state_dict = state
        if isinstance(state, dict):
            if 'model' in state:
                state_dict = state['model']
            elif 'model_state_dict' in state:
                state_dict = state['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    if hasattr(model, 'use_pretrained_head'):
        model.use_pretrained_head = args.use_xrv_head

    # Evaluation
    all_targets, all_preds = [], []
    with torch.no_grad():
        for images, labels_batch in tqdm(val_loader, desc="Evaluating", ncols=80):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels_batch.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Metrics
    metrics = compute_metrics(all_targets, all_preds, labels)
    print("\nValidation metrics:")
    for k, v in metrics['auroc'].items():
        print(f"  AUROC {k:25s}: {v:.4f}")
    print(f"  Mean AUROC: {metrics['auroc']['mean']:.4f}")
    print(f"  Mean AP:    {metrics['ap']['mean']:.4f}")

    # Save
    np.save(os.path.join(args.output, "val_predictions.npy"), all_preds)
    np.save(os.path.join(args.output, "val_targets.npy"), all_targets)
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot per-class AUROC
    plots_dir = os.path.join(args.output, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_per_class_auroc(metrics['auroc'], labels, plots_dir)

    if args.nec_metrics:
        print_nec_table(args.nec_metrics)

if __name__ == "__main__":
    main()
