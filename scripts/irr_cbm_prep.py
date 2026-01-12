#!/usr/bin/env python3
"""
Prepare IRR-style concept similarity vectors using CheXagent vision embeddings and Mistral embeddings.

Outputs per-image pickle files with:
    {"image_path": <abs path>, "label": <str>, "e": <list[float] similarities>}

Assumes COVID-QU folder layout:
  <dataset_root>/<variant dir>/<variant dir>/<Split>/{COVID-19,Non-COVID,Normal}/images/*.png|jpg
Variant dirs:
  infection -> "Infection Segmentation Data/Infection Segmentation Data"
  lung      -> "Lung Segmentation Data/Lung Segmentation Data"

Requirements:
  - MISTRAL_API_KEY in env (or pass via --mistral_api_key)
  - transformers >= 4.38 with trust_remote_code support
  - requests, tqdm, torch, pillow
"""

import argparse
import os
import pickle
from typing import List, Optional

import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


CLASS_NAMES = ["COVID-19", "Non-COVID", "Normal"]
VARIANT_DIRS = {
    "infection": os.path.join("Infection Segmentation Data", "Infection Segmentation Data"),
    "lung": os.path.join("Lung Segmentation Data", "Lung Segmentation Data"),
}


def load_concepts(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_mistral_embeddings(texts: List[str], api_key: str, model: str = "mistral-embed") -> torch.Tensor:
    """Call Mistral embeddings API and return a tensor [n, d]."""
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "input": texts}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    embs = torch.tensor([item["embedding"] for item in data], dtype=torch.float32)
    return embs


def collect_images(root: str, variant: str, split: str):
    base_dir = os.path.join(root, VARIANT_DIRS[variant], split)
    samples = []
    for cls in CLASS_NAMES:
        img_dir = os.path.join(base_dir, cls, "images")
        if not os.path.isdir(img_dir):
            print(f"Warning: missing directory {img_dir}, skipping.")
            continue
        for fname in os.listdir(img_dir):
            if fname.startswith("."):
                continue
            path = os.path.join(img_dir, fname)
            samples.append((path, cls))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare IRR-style CBM data with CheXagent vision + Mistral embeddings")
    parser.add_argument("--dataset_root", required=True, help="COVID-QU dataset root")
    parser.add_argument("--concepts", required=True, help="Path to concepts TXT")
    parser.add_argument("--variant", choices=["infection", "lung"], default="infection", help="COVID-QU variant")
    parser.add_argument("--splits", nargs="+", default=["Train", "Val", "Test"], help="Splits to process")
    parser.add_argument("--output_dir", required=True, help="Output base directory (will create subfolders per split)")
    parser.add_argument("--mistral_api_key", default=None, help="Mistral API key (fallback to env MISTRAL_API_KEY)")
    parser.add_argument("--chex_model", default="StanfordAIMI/CheXagent-8b", help="HF model id for CheXagent")
    parser.add_argument("--device", default="cuda", help="Device for vision model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for vision encoding")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Dtype for vision model")
    args = parser.parse_args()

    api_key = args.mistral_api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Mistral API key not provided. Set --mistral_api_key or MISTRAL_API_KEY env var.")

    concepts = load_concepts(args.concepts)
    print(f"Loaded {len(concepts)} concepts")
    concept_embs = get_mistral_embeddings(concepts, api_key)
    concept_embs = torch.nn.functional.normalize(concept_embs, dim=1)
    concept_dim = concept_embs.shape[1]

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.chex_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.chex_model,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device).eval()

    # Infer vision hidden size
    sample_inputs = processor(images=[Image.new("RGB", (224, 224))], return_tensors="pt").to(device)
    with torch.no_grad():
        vision_out = model.model.vision_model(pixel_values=sample_inputs["pixel_values"])
    vision_dim = vision_out.last_hidden_state.shape[-1]
    if vision_dim != concept_dim:
        raise RuntimeError(f"Dimension mismatch: vision dim {vision_dim} vs concept dim {concept_dim}. "
                           "Cannot compute cosine similarity without a projection.")

    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        samples = collect_images(args.dataset_root, args.variant, split)
        out_split = os.path.join(args.output_dir, split.lower())
        os.makedirs(out_split, exist_ok=True)
        print(f"{split}: {len(samples)} images")

        # Batch processing
        batch_paths, batch_labels = [], []
        counter = 0
        for path, lbl in tqdm(samples, desc=f"Encoding {split}"):
            batch_paths.append(path)
            batch_labels.append(lbl)
            if len(batch_paths) == args.batch_size or path == samples[-1][0]:
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = processor(images=images, return_tensors="pt").to(device)
                with torch.no_grad():
                    vision_out = model.model.vision_model(pixel_values=inputs["pixel_values"])
                    feats = vision_out.last_hidden_state.mean(dim=1)
                    feats = torch.nn.functional.normalize(feats, dim=1)
                    sims = torch.matmul(feats, concept_embs.to(device).T)  # [B, n_concepts]
                    sims = sims.cpu()

                for i, (p, lbl) in enumerate(zip(batch_paths, batch_labels)):
                    data = {"image_path": p, "label": lbl.lower().replace("-", "_"), "e": sims[i]}
                    counter += 1
                    out_path = os.path.join(out_split, f"{counter}.pkl")
                    with open(out_path, "wb") as f:
                        pickle.dump(data, f)

                batch_paths, batch_labels = [], []

        print(f"Wrote {counter} embeddings to {out_split}")


if __name__ == "__main__":
    main()
