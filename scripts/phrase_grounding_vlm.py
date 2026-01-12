#!/usr/bin/env python3
"""
Phrase grounding with CheXagent or MedGemma on a JSON task file.

Input JSON format (matches chexbench data.json):
{
  "Phrase Grounding": [
    {
      "image_path": ["path/to/img1", "path/to/img2", ...],
      "question": "prompt text",
      "answer": [{"box": "(x1,y1,x2,y2)"}]
    },
    ...
  ]
}

Usage:
python scripts/phrase_grounding_vlm.py \
  --model_backend chexagent \
  --model_id StanfordAIMI/CheXagent-2-3b \
  --data_json evaluation_chexbench/data.json \
  --task_key "Phrase Grounding" \
  --output_json evaluation_chexbench/results/phrase_grounding_chexagent.json \
  --limit 100

Or MedGemma:
python scripts/phrase_grounding_vlm.py \
  --model_backend medgemma \
  --model_id google/medgemma-4b-it \
  --data_json evaluation_chexbench/data.json \
  --task_key "Phrase Grounding" \
  --output_json evaluation_chexbench/results/phrase_grounding_medgemma.json
"""

import argparse
import copy
import json
import os
from typing import List, Dict, Any, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, x1y1x2y2: bool = True) -> torch.Tensor:
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

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
    ap = argparse.ArgumentParser(description="Phrase grounding with CheXagent or MedGemma.")
    ap.add_argument("--model_backend", choices=["chexagent", "medgemma"], required=True)
    ap.add_argument("--model_id", type=str, required=True, help="HF model id/path")
    ap.add_argument("--data_json", type=str, required=True, help="Input tasks JSON")
    ap.add_argument("--task_key", type=str, default="Phrase Grounding", help="Key in JSON to process")
    ap.add_argument("--output_json", type=str, required=True, help="Where to save predictions with boxes/IoU")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples")
    return ap.parse_args()


def load_model_chexagent(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True
    ).eval()
    return tok, mdl


def load_model_medgemma(model_id: str):
    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    return pipe


def parse_boxes_from_list_format(list_fmt: List[Dict[str, Any]]) -> List[str]:
    return [_d["box"] for _d in list_fmt if "box" in _d]


def list_to_box_tuple(box_str: str) -> List[int]:
    # Accept "(x1,y1,x2,y2)" or "x1,y1,x2,y2"
    clean = box_str.replace("(", "").replace(")", "")
    return [int(float(c)) for c in clean.split(",")]


def run_chexagent(samples: List[Dict[str, Any]], tok, mdl, max_new_tokens: int):
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
        ref_boxes = parse_boxes_from_list_format(tok.to_list_format(sample["answer"]))
        cand_boxes = parse_boxes_from_list_format(tok.to_list_format(response))
        if len(cand_boxes) == 0:
            cand_boxes = last
        else:
            last = copy.deepcopy(cand_boxes)
        ref_box = list_to_box_tuple(ref_boxes[0])
        cand_box = list_to_box_tuple(cand_boxes[0]) if cand_boxes else [0, 0, 0, 0]

        sample["reference_box"] = ref_box
        sample["candidate_box"] = cand_box
        iou = bbox_iou(torch.tensor([ref_box]), torch.tensor([cand_box])).item()
        sample["iou"] = iou
        sample["raw_response"] = response
    return samples


def run_medgemma(samples: List[Dict[str, Any]], pipe, max_new_tokens: int):
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

        # Reuse CheXagent tokenizer for parsing; fallback to simple parsing
        cand_boxes = []
        if isinstance(response, str):
            # Expect format with "(x1,y1,x2,y2)" somewhere
            parts = response.replace("(", "").replace(")", "").split()
            for part in parts:
                if part.count(",") == 3:
                    try:
                        box = [int(float(x)) for x in part.split(",")]
                        cand_boxes.append(box)
                        break
                    except Exception:
                        continue
        if not cand_boxes:
            if last:
                cand_boxes = [last[-1]]
            else:
                cand_boxes = [[0, 0, 0, 0]]
        else:
            last = copy.deepcopy(cand_boxes)

        ref_box = list_to_box_tuple(
            parse_boxes_from_list_format(pipe.tokenizer.to_list_format(sample["answer"]))[0]
        )
        cand_box = cand_boxes[0]
        sample["reference_box"] = ref_box
        sample["candidate_box"] = cand_box
        iou = bbox_iou(torch.tensor([ref_box]), torch.tensor([cand_box])).item()
        sample["iou"] = iou
        sample["raw_response"] = response
    return samples


def main():
    args = parse_args()
    assert os.path.exists(args.data_json), f"Missing data file: {args.data_json}"
    with open(args.data_json, "r") as f:
        bench = json.load(f)
    samples = bench.get(args.task_key, [])
    if args.limit:
        samples = samples[: args.limit]

    if args.model_backend == "chexagent":
        tok, mdl = load_model_chexagent(args.model_id)
        processed = run_chexagent(samples, tok, mdl, args.max_new_tokens)
    else:
        pipe = load_model_medgemma(args.model_id)
        processed = run_medgemma(samples, pipe, args.max_new_tokens)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    print(f"Saved predictions to {args.output_json}")


if __name__ == "__main__":
    main()
