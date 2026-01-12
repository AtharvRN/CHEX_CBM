#!/usr/bin/env python3
"""
Evaluate VinDr-CXR bounding-box detectors (CheX, CheXagent, MedGemma, etc.).

Ground truth: annotations CSV with columns
    image_id,class_name,x_min,y_min,x_max,y_max
Predictions: CSV per method with columns
    image_id,class_name,x_min,y_min,x_max,y_max,score
If score is missing, it is treated as 1.0.

Example:
python scripts/compare_vindr_det.py \
  --gt /workspace/vindr/annotations/annotations_test.csv \
  --pred chex:/workspace/vindr/preds_chex.csv \
  --pred chexagent:/workspace/vindr/preds_chexagent.csv \
  --pred medgemma:/workspace/vindr/preds_medgemma.csv \
  --iou 0.4 0.5 \
  --output_json /workspace/vindr/vindr_det_eval.json \
  --output_table /workspace/vindr/vindr_det_eval.tsv
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="VinDr-CXR detection evaluation (mAP / per-class AP).")
    ap.add_argument("--gt", required=True, help="Ground-truth annotations CSV")
    ap.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Prediction file in the form name:path/to/preds.csv (repeatable)",
    )
    ap.add_argument("--iou", type=float, nargs="+", default=[0.5], help="IoU thresholds, e.g., 0.4 0.5")
    ap.add_argument("--top_k", type=int, default=None, help="Optional cap on predictions per class (after sorting by score)")
    ap.add_argument("--output_json", type=str, default=None, help="Optional JSON output with full metrics")
    ap.add_argument("--output_table", type=str, default=None, help="Optional TSV summary table (per IoU)")
    return ap.parse_args()


def iou(box1: List[float], box2: List[float]) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / max(area1 + area2 - inter, 1e-8)


def voc_ap(tp: np.ndarray, fp: np.ndarray, npos: int) -> float:
    """11-point VOC-style AP with precision envelope."""
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(npos, 1e-8)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    inds = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[inds + 1] - mrec[inds]) * mpre[inds + 1]))
    return ap


def eval_single(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_thr: float,
    top_k: int | None,
) -> Tuple[Dict[str, Dict], float]:
    # ground truth: class -> image_id -> list of boxes + flags
    gt = defaultdict(lambda: defaultdict(list))
    for _, r in gt_df.iterrows():
        gt[r.class_name][r.image_id].append({"box": [r.x_min, r.y_min, r.x_max, r.y_max], "used": False})
    classes = sorted(gt.keys())
    results = {}
    for cls in classes:
        preds = pred_df[pred_df.class_name == cls].copy()
        preds = preds.sort_values("score", ascending=False)
        if top_k:
            preds = preds.head(top_k)
        npos = sum(len(v) for v in gt[cls].values())
        tp: List[int] = []
        fp: List[int] = []
        for _, p in preds.iterrows():
            gts = gt[cls].get(p.image_id, [])
            bb_pred = [p.x_min, p.y_min, p.x_max, p.y_max]
            ious = [iou(bb_pred, g["box"]) for g in gts]
            if ious:
                m = int(np.argmax(ious))
                if ious[m] >= iou_thr and not gts[m]["used"]:
                    tp.append(1)
                    fp.append(0)
                    gts[m]["used"] = True
                else:
                    tp.append(0)
                    fp.append(1)
            else:
                tp.append(0)
                fp.append(1)
        ap = voc_ap(np.array(tp), np.array(fp), npos)
        results[cls] = {
            "ap": ap,
            "npos": npos,
            "tp": int(np.sum(tp)),
            "fp": int(np.sum(fp)),
        }
        # reset used flags for next iteration
        for imgs in gt[cls].values():
            for g in imgs:
                g["used"] = False
    mAP = float(np.mean([r["ap"] for r in results.values()])) if results else 0.0
    return results, mAP


def load_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"{path} missing columns: {missing}")
    if "score" not in df.columns:
        df["score"] = 1.0
    return df


def main():
    args = parse_args()
    gt_df = pd.read_csv(args.gt)
    for col in ["image_id", "class_name", "x_min", "y_min", "x_max", "y_max"]:
        if col not in gt_df.columns:
            raise ValueError(f"GT missing column {col}")

    pred_specs = {}
    for spec in args.pred:
        if ":" not in spec:
            raise ValueError(f"--pred must be name:path, got {spec}")
        name, path = spec.split(":", 1)
        pred_specs[name] = Path(path)

    summary = {}
    for name, path in pred_specs.items():
        pred_df = load_pred_csv(path)
        model_res = {}
        for thr in args.iou:
            per_class, mAP = eval_single(gt_df, pred_df, thr, args.top_k)
            model_res[f"iou_{thr}"] = {"mAP": mAP, "per_class": per_class}
            print(f"{name} | IoU {thr:.2f} mAP={mAP:.4f}")
        summary[name] = model_res

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)

    if args.output_table:
        rows = []
        for name, res in summary.items():
            for thr_key, vals in res.items():
                rows.append({"model": name, "iou": thr_key.replace("iou_", ""), "mAP": vals["mAP"]})
        df_out = pd.DataFrame(rows)
        Path(args.output_table).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.output_table, sep="\t", index=False)


if __name__ == "__main__":
    main()
