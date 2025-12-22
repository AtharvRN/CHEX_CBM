#!/usr/bin/env python3
"""
Generate Concept Annotations using ChEX for VLG-CBM on CheXpert

This script uses ChEX (a chest X-ray grounded object detector) to annotate
images with concept bounding boxes. This replaces GroundingDINO in the 
original VLG-CBM paper with a domain-specific model.

ChEX paper: https://arxiv.org/pdf/2404.15770
VLG-CBM paper: https://arxiv.org/pdf/2408.01432

For each image, we:
1. Load the image
2. Query ChEX with each concept
3. Get bounding boxes and confidence scores (regions)
4. Save per-image annotation JSON

Output format (same as VLG-CBM):
[
    {"img_path": "/path/to/image.jpg"},
    {"label": "concept1", "box": [x1, y1, x2, y2], "logit": 0.45},
    {"label": "concept2", "box": [x1, y1, x2, y2], "logit": 0.32},
    ...
]

Usage:
------
python generate_annotations.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concepts concepts/chexpert_concepts.json \
    --output_dir annotations/train \
    --limit_samples 10000 \
    --threshold 0.15
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import functools

# Add project paths
PROJECT_ROOT = Path(__file__).parent

# Add ChEX to path
CHEX_SRC = Path("/workspace/chex/src")
if CHEX_SRC.exists():
    sys.path.insert(0, str(CHEX_SRC))
    CHEX_AVAILABLE = True
else:
    CHEX_AVAILABLE = False
    print("Warning: ChEX not found at /workspace/chex/src")

# Set environment for ChEX
os.environ.setdefault('LOG_DIR', '/workspace/chex_models')


def load_concepts(
    concepts_path: str,
    subset_labels: Optional[List[str]] = None
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Load concepts from JSON or TXT file."""
    if concepts_path.endswith('.json'):
        with open(concepts_path, 'r') as f:
            data = json.load(f)
    
    if "concepts" in data:
        concepts_dict = data["concepts"]
    else:
        concepts_dict = data
    if subset_labels:
        concepts_dict = {
            label: concepts_dict[label]
            for label in subset_labels
            if label in concepts_dict
        }

    # Flatten to unique list
    all_concepts = []
    for cls_concepts in concepts_dict.values():
        all_concepts.extend(cls_concepts)
    unique_concepts = list(set(all_concepts))
    if not concepts_dict and subset_labels:
        print(f"Warning: no concepts found for labels {subset_labels}")
    print(f"Loaded {len(unique_concepts)} unique concepts {'from subset' if subset_labels else ''}")
    return concepts_dict, unique_concepts


def preprocess_image_for_chex(image_path: str, target_size: int = 224):
    """Load and preprocess image for ChEX."""
    img = Image.open(image_path).convert('L')  # Grayscale
    orig_size = img.size
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
    mean, std = 0.505, 0.248
    img_tensor = (img_tensor - mean) / std
    
    # ChEX expects 3 channels
    img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
    
    return img_tensor, img, orig_size


def annotate_image_with_chex(
    model,
    image_path: str,
    concepts: List[str],
    device: str,
    threshold: float = 0.15,
    return_all_regions: bool = True,
    concept_batch_size: int = 16,
    concept_features: Optional[torch.Tensor] = None,
) -> List[Dict]:
    """
    Annotate a single image using ChEX model.
    
    Returns list of annotations with label, box, and logit.
    """
    # Import here to avoid issues if ChEX not available
    from model.detector.token_decoder_detector import TokenDetectorOutput
    
    img_tensor, img_pil, orig_size = preprocess_image_for_chex(image_path)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dim
    
    annotations = []
    w, h = orig_size
    
    with torch.no_grad():
        # Encode image once
        img_output = model.img_encoder(img_tensor)
        
        # Process concepts in batches
        for start in range(0, len(concepts), concept_batch_size):
            end = min(start + concept_batch_size, len(concepts))
            batch_concepts = concepts[start:end]
            
            # Use precomputed features if available
            if concept_features is not None:
                batch_features = concept_features[start:end]
                text_features = batch_features.unsqueeze(0)
            else:
                text_features = model.txt_encoder.encode_sentences(batch_concepts)
                if text_features.dim() == 2:
                    text_features = text_features.unsqueeze(0)
            
            # Run detection
            detector_output: TokenDetectorOutput = model.detect_prompts(
                x=img_output,
                box_prompts_emb=text_features,
                clip_boxes=True,
            )
            
            multiboxes = detector_output.multiboxes  # (N, Q, R, 4)
            multiboxes_weights = detector_output.multiboxes_weights  # (N, Q, R)
            
            if multiboxes is None:
                continue
            
            boxes_np = multiboxes[0].cpu().numpy()  # (Q, R, 4)
            weights_np = multiboxes_weights[0].cpu().numpy()  # (Q, R)
            
            for local_idx, concept in enumerate(batch_concepts):
                boxes = boxes_np[local_idx]  # (R, 4)
                weights = weights_np[local_idx]  # (R,)
                
                if return_all_regions:
                    for box, weight in zip(boxes, weights):
                        if weight > threshold:
                            cx, cy, bw, bh = box
                            x1 = (cx - bw/2) * w
                            y1 = (cy - bh/2) * h
                            x2 = (cx + bw/2) * w
                            y2 = (cy + bh/2) * h
                            
                            annotations.append({
                                "label": concept,
                                "box": [float(x1), float(y1), float(x2), float(y2)],
                                "logit": float(weight),
                            })
                else:
                    # Only top region per concept
                    top_idx = weights.argmax()
                    weight = weights[top_idx]
                    
                    if weight > threshold:
                        box = boxes[top_idx]
                        cx, cy, bw, bh = box
                        x1 = (cx - bw/2) * w
                        y1 = (cy - bh/2) * h
                        x2 = (cx + bw/2) * w
                        y2 = (cy + bh/2) * h
                        
                        annotations.append({
                            "label": concept,
                            "box": [float(x1), float(y1), float(x2), float(y2)],
                            "logit": float(weight),
                        })
    
    return annotations


def save_annotation(output_dir: Path, idx: int, image_path: str, annotations: List[Dict]):
    """Save annotations to JSON file."""
    json_data = [{"img_path": image_path}] + annotations
    output_file = output_dir / f"{idx}.json"
    with open(output_file, 'w') as f:
        json.dump(json_data, f)


def process_image_task(
    task,
    dataset,
    model,
    concepts,
    device,
    threshold,
    concept_batch_size,
    concept_features,
    output_dir,
):
    """Worker that annotates a single image and writes the JSON."""
    idx, output_idx = task
    try:
        image_path = dataset.get_image_path(idx)
        annotations = annotate_image_with_chex(
            model=model,
            image_path=image_path,
            concepts=concepts,
            device=device,
            threshold=threshold,
            return_all_regions=True,
            concept_batch_size=concept_batch_size,
            concept_features=concept_features,
        )
        save_annotation(output_dir, output_idx, image_path, annotations)

        return {
            "status": "ok",
            "idx": idx,
            "annotations": len(annotations),
            "has_detections": bool(annotations),
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "status": "error",
            "idx": idx,
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate concept annotations using ChEX",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to CheXpert-v1.0-small directory")
    parser.add_argument("--concepts", type=str, 
                        default="concepts/chexpert_concepts.json",
                        help="Path to concepts file (JSON or TXT)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="Dataset split")
    parser.add_argument("--concept_subset", type=str, default="pathology",
                        choices=["all", "pathology", "competition", "custom"],
                        help="Subset of labels whose concepts should be used")
    parser.add_argument("--custom_concept_labels", type=str, default=None,
                        help="Comma-separated labels to use when concept_subset=custom")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for annotations")
    
    # Processing
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Confidence threshold for detections")
    parser.add_argument("--limit_samples", type=int, default=None,
                        help="Limit number of images")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index (for resuming)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concept_batch_size", type=int, default=16,
                        help="Concepts to process in parallel")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel annotation workers")
    
    # ChEX model
    parser.add_argument("--model_name", type=str, default="chex_stage3",
                        help="ChEX model name")
    parser.add_argument("--run_name", type=str, default="run_0",
                        help="ChEX run name")
    
    args = parser.parse_args()
    
    # =========================================
    # Setup
    # =========================================
    print("="*60)
    print("VLG-CBM Annotation Generation for CheXpert")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Concepts file: {args.concepts}")
    print(f"Output directory: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    np.random.seed(args.seed)
    
    # Bring dataset utilities into scope (needed for label subsets)
    sys.path.insert(0, str(PROJECT_ROOT))
    from dataset import (
        CheXpertDataset,
        CHEXPERT_PATHOLOGY_LABELS,
        CHEXPERT_COMPETITION_LABELS,
        get_transforms,
    )

    if args.concept_subset == "custom" and not args.custom_concept_labels:
        parser.error("--custom_concept_labels is required when using concept_subset=custom")

    subset_labels = None
    if args.concept_subset == "pathology":
        subset_labels = CHEXPERT_PATHOLOGY_LABELS
    elif args.concept_subset == "competition":
        subset_labels = CHEXPERT_COMPETITION_LABELS
    elif args.concept_subset == "custom":
        subset_labels = [
            label.strip()
            for label in args.custom_concept_labels.split(",")
            if label.strip()
        ]

    # =========================================
    # Load concepts
    # =========================================
    _, concepts = load_concepts(
        args.concepts,
        subset_labels=None if args.concept_subset == "all" else subset_labels
    )

    # =========================================
    # Load dataset
    # =========================================
    print("\nLoading CheXpert dataset...")
    
    if args.split == "train":
        csv_path = os.path.join(args.data_dir, "train.csv")
    else:
        csv_path = os.path.join(args.data_dir, "valid.csv")
    
    img_root = os.path.dirname(args.data_dir)
    transform = get_transforms(224, is_training=False)
    
    dataset = CheXpertDataset(
        csv_path=csv_path,
        img_root=img_root,
        transform=transform,
        labels=CHEXPERT_PATHOLOGY_LABELS,
        uncertain_strategy="ones",
        frontal_only=True
    )
    
    # Limit samples
    n_images = len(dataset)
    if args.limit_samples:
        n_images = min(n_images, args.limit_samples + args.start_idx)
        # Use seeded permutation for consistent subset
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(len(dataset))[:n_images]
    else:
        indices = list(range(n_images))
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Processing images {args.start_idx} to {n_images}")
    
    # =========================================
    # Load ChEX model
    # =========================================
    # Remove our local dataset module/path so ChEX can import its own dataset package
    sys.modules.pop('dataset', None)
    if str(PROJECT_ROOT) in sys.path:
        sys.path.remove(str(PROJECT_ROOT))

    if not CHEX_AVAILABLE:
        print("\nERROR: ChEX not available. Cannot generate annotations.")
        print("Please ensure ChEX is installed at /workspace/chex/src")
        return
    
    print("\nLoading ChEX model...")
    original_dir = os.getcwd()
    os.chdir(CHEX_SRC)
    
    from util.model_utils import load_model_by_name, ModelRegistry
    import model
    from model import img_encoder, txt_encoder, txt_decoder, detector
    
    ModelRegistry.init_registries([model, img_encoder, txt_encoder, txt_decoder, detector])
    
    chex_model, _ = load_model_by_name(
        args.model_name,
        run_name=args.run_name,
        load_best=False,
        return_dict=True
    )
    chex_model = chex_model.to(args.device)
    chex_model.eval()
    
    os.chdir(original_dir)
    print(f"ChEX model loaded on {args.device}")
    
    # =========================================
    # Precompute concept text features
    # =========================================
    concept_features_path = output_dir / "concept_features.pt"
    concept_features_cached = False
    concept_features = None

    if concept_features_path.exists():
        try:
            with torch.no_grad():
                cached_data = torch.load(concept_features_path, map_location="cpu")
            if (
                cached_data.get("concepts") == concepts and
                cached_data.get("model") == args.model_name and
                cached_data.get("run_name") == args.run_name
            ):
                concept_features = cached_data["features"].to(args.device)
                concept_features_cached = True
                print("\nLoaded cached concept text encodings.")
            else:
                print("\nConcept cache mismatch; recomputing encodings.")
        except Exception as exc:
            print(f"\nFailed to load cached concept features: {exc}")

    if not concept_features_cached:
        print("\nEncoding concept texts...")
        os.chdir(CHEX_SRC)
        with torch.no_grad():
            concept_features = chex_model.txt_encoder.encode_sentences(concepts)
        os.chdir(original_dir)

        # Save CPU copy for future runs
        concept_features_cpu = concept_features.cpu()
        torch.save({
            "features": concept_features_cpu,
            "concepts": concepts,
            "model": args.model_name,
            "run_name": args.run_name,
        }, concept_features_path)

        concept_features = concept_features.to(args.device)

    # =========================================
    # Generate annotations
    # =========================================
    total_annotations = 0
    images_with_detections = 0

    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "concepts_file": args.concepts,
        "split": args.split,
        "threshold": args.threshold,
        "model": f"{args.model_name}/{args.run_name}",
        "num_concepts": len(concepts),
        "workers": args.workers,
        "concepts": concepts,
        "concept_features_cached": concept_features_cached,
        "concept_features_path": str(concept_features_path)
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    tasks = []
    for i, idx in enumerate(indices[args.start_idx:n_images]):
        output_idx = args.start_idx + i
        task_file = output_dir / f"{output_idx}.json"
        if task_file.exists():
            continue
        tasks.append((idx, output_idx))

    print(f"\nAnnotating {len(tasks)} images with {len(concepts)} concepts using {args.workers} workers...")
    worker = functools.partial(
        process_image_task,
        dataset=dataset,
        model=chex_model,
        concepts=concepts,
        device=args.device,
        threshold=args.threshold,
        concept_batch_size=args.concept_batch_size,
        concept_features=concept_features,
        output_dir=output_dir,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in tqdm(
            executor.map(worker, tasks),
            total=len(tasks),
            desc="Annotating images"
        ):
            if result["status"] == "error":
                print(f"\nError processing image {result['idx']}: {result['error']}")
                continue

            total_annotations += result["annotations"]
            if result["has_detections"]:
                images_with_detections += 1
    
    processed = len(tasks)

    # =========================================
    # Summary
    # =========================================
    print("\n" + "="*60)
    print("ANNOTATION COMPLETE")
    print("="*60)
    print(f"Processed: {processed} images")
    print(f"Total annotations: {total_annotations}")
    print(f"Images with detections: {images_with_detections} ({100*images_with_detections/max(processed,1):.1f}%)")
    print(f"Avg annotations per image: {total_annotations/max(processed,1):.1f}")
    print(f"Output directory: {output_dir}")
    
    # Update metadata
    metadata["num_images"] = processed
    metadata["total_annotations"] = total_annotations
    metadata["images_with_detections"] = images_with_detections
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nNext step: Train VLG-CBM model:")
    print(f"  python vlg_cbm.py \\")
    print(f"      --data_dir {args.data_dir} \\")
    print(f"      --concepts {args.concepts} \\")
    print(f"      --annotation_dir {args.output_dir} \\")
    print(f"      --output checkpoints/vlg_cbm_exp1")


if __name__ == "__main__":
    main()
