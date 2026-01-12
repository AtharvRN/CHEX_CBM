#!/usr/bin/env python3
"""
Sweep classification thresholds on saved predictions/targets to maximize F1/accuracy.

Typical use (after running eval.py):
python scripts/threshold_sweep.py \
  --predictions outputs/chexpert_pos_weight_densenet121/eval/valid_predictions.npy \
  --targets outputs/chexpert_pos_weight_densenet121/eval/valid_targets.npy \
  --metrics outputs/chexpert_pos_weight_densenet121/eval/valid_metrics.json \
  --output outputs/chexpert_pos_weight_densenet121/eval/threshold_sweep.json
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep thresholds to maximize F1/accuracy.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions .npy")
    parser.add_argument("--targets", type=str, required=True, help="Path to targets .npy")
    parser.add_argument("--metrics", type=str, default=None, help="Optional metrics.json (to infer label order)")
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit label names (overrides metrics.json inference)",
    )
    parser.add_argument("--output", type=str, default=None, help="Where to save JSON results")
    parser.add_argument("--step", type=float, default=0.01, help="Threshold step size (0-1]")
    parser.add_argument(
        "--min_pos",
        type=int,
        default=1,
        help="Skip labels with fewer than this many positives when reporting per-label sweeps",
    )
    return parser.parse_args()


def load_labels(args: argparse.Namespace) -> List[str]:
    if args.labels:
        return args.labels
    if args.metrics and os.path.exists(args.metrics):
        try:
            with open(args.metrics, "r") as f:
                metrics = json.load(f)
            if "auroc" in metrics:
                return [k for k in metrics["auroc"].keys() if k != "mean"]
        except Exception:
            pass
    raise ValueError("Could not determine labels; provide --labels or a metrics JSON with 'auroc' keys.")


def binarize(preds: np.ndarray, threshold: float) -> np.ndarray:
    return (preds >= threshold).astype(np.int32)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    tn = np.logical_and(y_true == 0, y_pred == 0).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "accuracy": float(acc)}


def sweep_label(y_true: np.ndarray, y_scores: np.ndarray, thresholds: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_f1 = -1.0
    best_t = 0.5
    best = {}
    for t in thresholds:
        metrics = binary_metrics(y_true, binarize(y_scores, t))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = float(t)
            best = metrics
    return best_t, best


def sweep_global(y_true: np.ndarray, y_scores: np.ndarray, thresholds: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_macro_f1 = -1.0
    best_t = 0.5
    best = {}
    for t in thresholds:
        preds = binarize(y_scores, t)
        per_label = []
        for i in range(y_true.shape[1]):
            per_label.append(binary_metrics(y_true[:, i], preds[:, i])["f1"])
        macro_f1 = float(np.mean(per_label))
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_t = float(t)
            best = {"macro_f1": macro_f1}
    return best_t, best


def main():
    args = parse_args()
    preds = np.load(args.predictions)
    targets = np.load(args.targets)

    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}")

    # Single-label (softmax) fallback: just report accuracy
    if preds.ndim == 2 and np.allclose(preds.sum(axis=1), 1.0, atol=1e-3):
        acc = float((preds.argmax(axis=1) == targets.argmax(axis=1)).mean())
        results = {"mode": "single_label", "accuracy": acc}
        out_path = args.output or "threshold_sweep.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved single-label accuracy to {out_path} (accuracy={acc:.4f})")
        return

    labels = load_labels(args)
    if len(labels) != preds.shape[1]:
        raise ValueError(f"Number of labels ({len(labels)}) does not match prediction dims ({preds.shape[1]})")

    thresholds = np.arange(0.0, 1.0 + 1e-8, args.step)
    results = {"mode": "multilabel", "per_label": {}, "global": {}}

    # Global sweep (single threshold for all labels)
    g_t, g_metrics = sweep_global(targets, preds, thresholds)
    results["global"]["best_threshold_macro_f1"] = g_t
    results["global"].update(g_metrics)

    # Per-label sweeps
    for idx, name in enumerate(labels):
        y_true = targets[:, idx]
        if y_true.sum() < args.min_pos:
            results["per_label"][name] = {"skipped": True, "reason": "insufficient positives"}
            continue
        best_t, best_metrics = sweep_label(y_true, preds[:, idx], thresholds)
        base_metrics = binary_metrics(y_true, binarize(preds[:, idx], 0.5))
        results["per_label"][name] = {
            "best_threshold_f1": best_t,
            "best": best_metrics,
            "at_0_5": base_metrics,
            "support_pos": int(y_true.sum()),
            "support_total": int(len(y_true)),
        }

    out_path = args.output or "threshold_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved threshold sweep to {out_path}")
    print(f"Global macro-F1 best @ {g_t:.3f}: {g_metrics.get('macro_f1', 0):.4f}")
    for name, stats in results["per_label"].items():
        if "skipped" in stats:
            print(f"{name}: skipped ({stats['reason']})")
            continue
        print(
            f"{name}: best F1 {stats['best']['f1']:.4f} @ {stats['best_threshold_f1']:.2f} "
            f"(P {stats['best']['precision']:.3f}, R {stats['best']['recall']:.3f}); "
            f"@0.50 F1 {stats['at_0_5']['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
