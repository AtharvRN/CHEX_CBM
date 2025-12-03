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
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state['model'] if 'model' in state else state)
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

if __name__ == "__main__":
    main()
