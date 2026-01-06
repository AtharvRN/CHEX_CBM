#!/usr/bin/env python3
"""
Evaluate CheXagent for CheXpert or COVID-QU classification.

Examples:
  python scripts/eval_chexagent.py --label_set chexpert --data_dir /path/to/CheXpert-v1.0-small
  python scripts/eval_chexagent.py --label_set covidqu --data_dir /path/to/COVIDQU --covidqu_variant infection
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import (
    CheXpertDataset,
    CovidQUDataset,
    CHEXPERT_LABELS,
    CHEXPERT_COMPETITION_LABELS,
    CHEXPERT_PATHOLOGY_LABELS,
    COVIDQU_LABELS,
)


DEFAULT_MODEL_ID = "StanfordAIMI/CheXagent-2-3b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CheXagent for classification")
    parser.add_argument("--label_set", type=str, default="chexpert",
                        choices=["chexpert", "covidqu"])
    parser.add_argument("--data_dir", type=str, required=True,
                        help="CheXpert-v1.0-small directory or COVID-QU root directory")
    parser.add_argument("--split", type=str, default="valid",
                        help="CheXpert: train/valid/test, COVID-QU: Train/Val/Test")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="CheXpert: explicit CSV path (overrides --split)")
    parser.add_argument("--competition_labels", action="store_true",
                        help="CheXpert: use 5 competition labels")
    parser.add_argument("--pathology_labels", action="store_true",
                        help="CheXpert: use 12 pathology labels")
    parser.add_argument("--covidqu_variant", type=str, default="infection",
                        choices=["infection", "lung"])
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/models/chexagent"),
                        help="Local CheXagent path; falls back to HF model ID if missing")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="auto",
                        help="Device or 'auto' to use device_map=auto")
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle samples before evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="chexagent_eval",
                        help="Output directory for metrics/predictions")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save per-sample predictions to CSV")
    parser.add_argument("--unknown_policy", type=str, default="skip",
                        choices=["skip", "negative", "positive", "count_wrong"],
                        help="How to handle unparseable responses")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--prompt_mode", type=str, default="single_prompt",
                        choices=["single_prompt", "one_vs_rest"],
                        help="COVID-QU only: single_prompt or one_vs_rest")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_chexagent(model_path: str, dtype: torch.dtype, device: str):
    model_name = model_path if Path(model_path).exists() else DEFAULT_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=None, trust_remote_code=True
        )
        model = model.to(device)
    model = model.to(dtype)
    model.eval()
    return tokenizer, model


def chexagent_answer(
    tokenizer,
    model,
    image_path: str,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> str:
    query = tokenizer.from_list_format([{"image": image_path}, {"text": prompt}])
    conv = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": query},
    ]
    input_ids = tokenizer.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt"
    )
    if device != "auto":
        input_ids = input_ids.to(device)
    output = model.generate(
        input_ids,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )[0]
    response = tokenizer.decode(output[input_ids.size(1):-1])
    return response.strip()


def parse_yes_no(text: str) -> Optional[bool]:
    t = text.strip().lower()
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    positive = any(tok in t for tok in (" yes", "present", "positive", "shows"))
    negative = any(tok in t for tok in (" no", "absent", "negative", "not present"))
    if positive and not negative:
        return True
    if negative and not positive:
        return False
    return None


def chexpert_prompt(label: str) -> str:
    return f"Does this chest X-ray show {label}? Answer yes or no."


def covidqu_single_prompt(labels: List[str]) -> str:
    options = ", ".join(labels)
    return (
        f"Classify this chest X-ray as one of: {options}. "
        "Respond with exactly one label."
    )


def parse_covidqu_label(text: str, labels: List[str]) -> Optional[str]:
    t = text.strip().lower()
    matches = []
    for label in labels:
        key = label.lower()
        if key in t:
            matches.append(label)
        if label.lower().replace("-", " ") in t:
            matches.append(label)
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    return None


def compute_binary_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


def normalize_chexpert_split(split: str) -> str:
    s = split.strip().lower()
    if s in ("val", "valid", "validation"):
        return "valid"
    if s in ("train", "training"):
        return "train"
    if s in ("test",):
        return "test"
    return split


def normalize_covidqu_split(split: str) -> str:
    s = split.strip().lower()
    if s in ("val", "valid", "validation"):
        return "Val"
    if s in ("train", "training"):
        return "Train"
    if s in ("test",):
        return "Test"
    return split


def eval_chexpert(args, tokenizer, model, output_dir: Path) -> Dict[str, Dict]:
    csv_path = args.csv_path
    if csv_path is None:
        split = normalize_chexpert_split(args.split)
        csv_path = os.path.join(args.data_dir, f"{split}.csv")
    img_root = str(Path(args.data_dir).parent)
    if args.competition_labels and args.pathology_labels:
        raise ValueError("Use only one of --competition_labels or --pathology_labels")
    if args.competition_labels:
        labels = CHEXPERT_COMPETITION_LABELS
    elif args.pathology_labels:
        labels = CHEXPERT_PATHOLOGY_LABELS
    else:
        labels = CHEXPERT_LABELS
    dataset = CheXpertDataset(
        csv_path=csv_path,
        img_root=img_root,
        transform=None,
        labels=labels,
        uncertain_strategy="ones",
        frontal_only=True,
    )

    indices = list(range(len(dataset)))
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(indices)
    if args.limit_samples:
        indices = indices[: args.limit_samples]

    preds: List[List[float]] = []
    targets: List[np.ndarray] = []
    responses: List[List[str]] = []

    for idx in tqdm(indices, desc="CheXpert eval", unit="image"):
        img_path = dataset.get_image_path(idx)
        target = dataset.targets[idx].numpy().astype(int)
        image_preds = []
        image_responses = []
        for label_idx, label in enumerate(labels):
            prompt = chexpert_prompt(label)
            response = chexagent_answer(
                tokenizer, model, img_path, prompt, args.device, args.max_new_tokens
            )
            pred_raw = parse_yes_no(response)
            if pred_raw is None:
                if args.unknown_policy == "skip":
                    pred = np.nan
                elif args.unknown_policy == "negative":
                    pred = 0.0
                elif args.unknown_policy == "positive":
                    pred = 1.0
                else:
                    pred = float(1 - target[label_idx])
            else:
                pred = float(pred_raw)
            image_preds.append(pred)
            image_responses.append(response)
        preds.append(image_preds)
        targets.append(target)
        responses.append(image_responses)

    preds_arr = np.array(preds, dtype=float)
    targets_arr = np.array(targets, dtype=int)

    per_label = {}
    macro = {"precision": [], "recall": [], "f1": [], "accuracy": []}
    for i, label in enumerate(labels):
        pred_col = preds_arr[:, i]
        target_col = targets_arr[:, i]
        if args.unknown_policy == "skip":
            mask = ~np.isnan(pred_col)
            pred_col = pred_col[mask].astype(int)
            target_col = target_col[mask]
        else:
            pred_col = pred_col.astype(int)
        metrics = compute_binary_metrics(target_col, pred_col)
        per_label[label] = metrics
        for key in macro.keys():
            macro[key].append(metrics[key])

    macro_avg = {k: float(np.nanmean(v)) if v else float("nan") for k, v in macro.items()}

    output = {
        "task": "chexpert",
        "num_samples": len(indices),
        "unknown_policy": args.unknown_policy,
        "labels": labels,
        "per_label": per_label,
        "macro_avg": macro_avg,
    }

    if args.save_predictions:
        pred_path = output_dir / "predictions_chexpert.csv"
        with pred_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label", "target", "pred", "response"])
            for row_idx, idx in enumerate(indices):
                img_path = dataset.get_image_path(idx)
                for col_idx, label in enumerate(labels):
                    writer.writerow([
                        img_path,
                        label,
                        int(targets_arr[row_idx, col_idx]),
                        preds_arr[row_idx, col_idx],
                        responses[row_idx][col_idx],
                    ])

    return output


def eval_covidqu(args, tokenizer, model, output_dir: Path) -> Dict[str, Dict]:
    dataset = CovidQUDataset(
        root=args.data_dir,
        split=normalize_covidqu_split(args.split),
        transform=None,
        variant=args.covidqu_variant,
    )
    indices = list(range(len(dataset)))
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(indices)
    if args.limit_samples:
        indices = indices[: args.limit_samples]

    preds = []
    targets = []
    responses = []

    for idx in tqdm(indices, desc="COVID-QU eval", unit="image"):
        img_path = dataset.get_image_path(idx)
        target_idx = dataset.targets[idx]
        if args.prompt_mode == "single_prompt":
            prompt = covidqu_single_prompt(COVIDQU_LABELS)
            response = chexagent_answer(
                tokenizer, model, img_path, prompt, args.device, args.max_new_tokens
            )
            label = parse_covidqu_label(response, COVIDQU_LABELS)
            if label is None:
                if args.unknown_policy == "skip":
                    pred_idx = None
                else:
                    pred_idx = (target_idx + 1) % len(COVIDQU_LABELS)
            else:
                pred_idx = COVIDQU_LABELS.index(label)
            preds.append(pred_idx)
            targets.append(target_idx)
            responses.append({"mode": "single_prompt", "response": response})
        else:
            image_preds = []
            image_responses = []
            for label in COVIDQU_LABELS:
                prompt = chexpert_prompt(label)
                response = chexagent_answer(
                    tokenizer, model, img_path, prompt, args.device, args.max_new_tokens
                )
                pred_raw = parse_yes_no(response)
                if pred_raw is None:
                    if args.unknown_policy == "skip":
                        pred = None
                    elif args.unknown_policy == "negative":
                        pred = 0
                    elif args.unknown_policy == "positive":
                        pred = 1
                    else:
                        pred = 0
                else:
                    pred = int(pred_raw)
                image_preds.append(pred)
                image_responses.append(response)
            pred_idx = None
            positives = [i for i, p in enumerate(image_preds) if p == 1]
            if len(positives) == 1:
                pred_idx = positives[0]
            if pred_idx is None and args.unknown_policy != "skip":
                pred_idx = (target_idx + 1) % len(COVIDQU_LABELS)
            preds.append(pred_idx)
            targets.append(target_idx)
            responses.append({"mode": "one_vs_rest", "responses": image_responses})

    preds_arr = np.array([p if p is not None else -1 for p in preds], dtype=int)
    targets_arr = np.array(targets, dtype=int)

    valid_mask = preds_arr != -1
    if args.unknown_policy == "skip":
        eval_preds = preds_arr[valid_mask]
        eval_targets = targets_arr[valid_mask]
    else:
        eval_preds = preds_arr.copy()
        eval_preds[eval_preds == -1] = (targets_arr[eval_preds == -1] + 1) % len(COVIDQU_LABELS)
        eval_targets = targets_arr

    accuracy = float((eval_preds == eval_targets).mean()) if len(eval_preds) else float("nan")
    per_class = {}
    for cls_idx, cls_name in enumerate(COVIDQU_LABELS):
        cls_targets = (eval_targets == cls_idx).astype(int)
        cls_preds = (eval_preds == cls_idx).astype(int)
        per_class[cls_name] = compute_binary_metrics(cls_targets, cls_preds)

    output = {
        "task": "covidqu",
        "num_samples": len(indices),
        "unknown_policy": args.unknown_policy,
        "prompt_mode": args.prompt_mode,
        "accuracy": accuracy,
        "per_class": per_class,
        "unknown_count": int((preds_arr == -1).sum()),
    }

    if args.save_predictions:
        pred_path = output_dir / "predictions_covidqu.csv"
        with pred_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "target", "pred", "responses"])
            for row_idx, idx in enumerate(indices):
                img_path = dataset.get_image_path(idx)
                writer.writerow([
                    img_path,
                    COVIDQU_LABELS[targets_arr[row_idx]],
                    COVIDQU_LABELS[preds_arr[row_idx]] if preds_arr[row_idx] != -1 else "unknown",
                    json.dumps(responses[row_idx]),
                ])

    return output


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.dtype)
    tokenizer, model = load_chexagent(args.model_path, dtype, args.device)

    if args.label_set == "chexpert":
        metrics = eval_chexpert(args, tokenizer, model, output_dir)
    else:
        metrics = eval_covidqu(args, tokenizer, model, output_dir)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
