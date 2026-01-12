#!/usr/bin/env python3
"""
Phrase grounding on VinDr-CXR annotations using CheXagent or MedGemma.

Inputs:
  - annotations CSV with columns: image_id,class_name,x_min,y_min,x_max,y_max
  - image_root: folder containing the images (e.g., /workspace/vindr-cxr/1.0.0/test)
  - model backend: chexagent or medgemma

Outputs:
  - JSON with per-sample reference/candidate boxes, IoU, and raw model response.

Example:
python scripts/phrase_grounding_vindr.py \
  --annotations /workspace/vindr/annotations/annotations_test.csv \
  --image_root /workspace/vindr/test \
  --model_backend chexagent \
  --model_id StanfordAIMI/CheXagent-2-3b \
  --output_json /workspace/vindr/results/phrase_grounding_chexagent.json \
  --limit 200
"""
import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1, min=0
    )
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phrase grounding on VinDr-CXR.")
    ap.add_argument("--annotations", required=True, help="CSV with image_id,class_name,x_min,y_min,x_max,y_max")
    ap.add_argument("--image_root", required=True, help="Folder containing images")
    ap.add_argument("--image_suffix", default=".dicom", help="Suffix/extension to append to image_id (default: .dicom)")
    ap.add_argument("--model_backend", choices=["chexagent", "medgemma"], required=True)
    ap.add_argument("--model_id", required=True, help="HF model id/path")
    ap.add_argument("--output_json", required=True, help="Where to save predictions")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="Optional number of samples to evaluate")
    ap.add_argument(
        "--question_template",
        default="Please locate the following phrase: {label} is present.",
        help="Template to form the grounding question",
    )
    return ap.parse_args()


def load_chexagent(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True).eval()
    return tok, mdl


def load_medgemma(model_id: str):
    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    return pipe


_BOX_RE = re.compile(
    r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?"
)


def extract_boxes(text: str) -> List[List[int]]:
    boxes = []
    for m in _BOX_RE.finditer(text):
        boxes.append([int(float(m.group(i))) for i in range(1, 5)])
    return boxes


def list_to_box_tuple(box_str: str) -> List[int]:
    clean = box_str.replace("(", "").replace(")", "")
    return [int(float(x)) for x in clean.split(",")]


def run_chexagent(samples: List[Dict[str, Any]], tok, mdl, max_new_tokens: int) -> List[Dict[str, Any]]:
    last = []
    for sample in samples:
        paths = sample["image_path"]
        prompt = sample["question"]
        query = tok.from_list_format([*[{ "image": p } for p in paths], {"text": prompt}])
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        input_ids = tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
        gen = mdl.generate(
            input_ids=input_ids,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )[0]
        response = tok.decode(gen[input_ids.size(1) : -1])

        ref_box = sample["reference_box"]
        cand_boxes = extract_boxes(response)
        if not cand_boxes:
            cand_boxes = last if last else [[0, 0, 0, 0]]
        else:
            last = copy.deepcopy(cand_boxes)
        cand_box = cand_boxes[0]
        sample["candidate_box"] = cand_box
        sample["raw_response"] = response
        iou = bbox_iou(torch.tensor([ref_box]), torch.tensor([cand_box])).item()
        sample["iou"] = iou
    return samples


def run_medgemma(samples: List[Dict[str, Any]], pipe, max_new_tokens: int) -> List[Dict[str, Any]]:
    last = []
    for sample in samples:
        prompt = sample["question"]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                          + [{"type": "image", "image": p} for p in sample["image_path"]],
            },
        ]
        out = pipe(text=messages, max_new_tokens=max_new_tokens)
        response = out[0]["generated_text"] if isinstance(out, list) else out

        ref_box = sample["reference_box"]
        cand_boxes = extract_boxes(response) if isinstance(response, str) else []
        if not cand_boxes:
            cand_boxes = last if last else [[0, 0, 0, 0]]
        else:
            last = copy.deepcopy(cand_boxes)
        cand_box = cand_boxes[0]
        sample["candidate_box"] = cand_box
        sample["raw_response"] = response
        iou = bbox_iou(torch.tensor([ref_box]), torch.tensor([cand_box])).item()
        sample["iou"] = iou
    return samples


def build_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    df = pd.read_csv(args.annotations)
    required = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Annotations CSV missing columns: {missing}")
    samples = []
    for _, r in df.iterrows():
        img_path = os.path.join(args.image_root, f"{r.image_id}{args.image_suffix}")
        if not os.path.exists(img_path):
            # fallback: try without suffix
            img_path = os.path.join(args.image_root, str(r.image_id))
        ref_box = [int(r.x_min), int(r.y_min), int(r.x_max), int(r.y_max)]
        question = args.question_template.format(label=str(r.class_name))
        samples.append(
            {
                "image_path": [img_path],
                "question": question,
                "reference_box": ref_box,
                "class_name": r.class_name,
                "image_id": r.image_id,
            }
        )
    return samples


def main():
    args = parse_args()
    samples = build_samples(args)
    if args.limit:
        samples = samples[: args.limit]

    if args.model_backend == "chexagent":
        tok, mdl = load_chexagent(args.model_id)
        processed = run_chexagent(samples, tok, mdl, args.max_new_tokens)
    else:
        pipe = load_medgemma(args.model_id)
        processed = run_medgemma(samples, pipe, args.max_new_tokens)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    print(f"Saved predictions to {args.output_json}")


if __name__ == "__main__":
    main()
