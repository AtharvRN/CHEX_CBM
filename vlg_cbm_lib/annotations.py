import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def load_concepts(path: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """Load concepts from file. Returns (class->concepts dict, flat list)."""
    if path.endswith('.json'):
        with open(path, 'r') as f:
            data = json.load(f)
        if "concepts" in data:
            concepts_dict = data["concepts"]
        else:
            concepts_dict = data

        all_concepts = []
        for cls_concepts in concepts_dict.values():
            all_concepts.extend(cls_concepts)
        unique_concepts = list(set(all_concepts))
    else:
        with open(path, 'r') as f:
            unique_concepts = [line.strip() for line in f if line.strip()]
        concepts_dict = {}

    return concepts_dict, unique_concepts


def _annotation_cache_path(
    annotation_dir: str,
    n_images: int,
    concepts: List[str],
    confidence_threshold: float
) -> str:
    digest = hashlib.md5(("\n".join(concepts)).encode('utf-8')).hexdigest()
    filename = f"concept_cache_{n_images}_{len(concepts)}_{confidence_threshold:.2f}_{digest}.npz"
    return os.path.join(annotation_dir, filename)


def load_annotations(
    annotation_dir: str,
    n_images: int,
    concepts: List[str],
    confidence_threshold: float = 0.15,
    num_workers: int = 8,
    cache_path: Optional[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Load annotations and create concept presence matrix.

    Returns:
        concept_matrix: (n_images, n_concepts) binary matrix
        image_paths: List of image paths
    """
    default_cache_path = _annotation_cache_path(annotation_dir, n_images, concepts, confidence_threshold)
    actual_cache_path = cache_path or default_cache_path
    requested_cache = cache_path is not None

    if requested_cache and not os.path.exists(cache_path):
        print(f"Requested concept cache not found at {cache_path}, recomputing...")
        actual_cache_path = default_cache_path
        requested_cache = False

    if os.path.exists(actual_cache_path):
        try:
            cached = np.load(actual_cache_path, allow_pickle=True)
            matrix = cached["concept_matrix"]
            paths = cached["image_paths"].tolist()
            if matrix.shape == (n_images, len(concepts)):
                if requested_cache:
                    print(f"Loaded requested concept cache from {actual_cache_path}")
                else:
                    print(f"Loaded cached annotations from {actual_cache_path}")
                return matrix, paths
            else:
                print("Cached annotation shape mismatch, recomputing...")
        except Exception as exc:
            print(f"Failed to read annotation cache ({exc}), recomputing...")

    print(f"Loading annotations from {annotation_dir}...")
    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    concept_matrix = np.zeros((n_images, len(concepts)), dtype=np.float32)
    image_paths = [""] * n_images

    def process_index(idx: int):
        ann_file = os.path.join(annotation_dir, f"{idx}.json")
        if not os.path.exists(ann_file):
            return idx, "", None

        with open(ann_file, 'r') as f:
            data = json.load(f)

        img_entry = data[0]
        img_path = img_entry.get('img_path', '')
        row = np.zeros(len(concepts), dtype=np.float32)

        for ann in data[1:]:
            label = ann.get('label')
            if label not in concept_to_idx:
                continue
            logit = float(ann.get('logit', 0.0))
            if logit > confidence_threshold:
                cidx = concept_to_idx[label]
                if logit > row[cidx]:
                    row[cidx] = logit

        return idx, img_path, row

    found = 0
    worker_count = min(max(num_workers, 1), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(process_index, idx): idx for idx in range(n_images)}
        for future in tqdm(as_completed(futures), total=n_images, desc="Loading annotations"):
            idx, img_path, row = future.result()
            image_paths[idx] = img_path
            if row is not None:
                concept_matrix[idx] = row
                if row.sum() > 0:
                    found += 1

    print(f"Found annotations for {found}/{n_images} images")
    try:
        np.savez_compressed(
            actual_cache_path,
            concept_matrix=concept_matrix,
            image_paths=np.array(image_paths, dtype=object)
        )
        print(f"Cached annotations to {actual_cache_path}")
    except Exception as exc:
        print(f"Warning: failed to cache annotations ({exc})")

    return concept_matrix, image_paths


def filter_concepts_by_frequency(
    concept_matrix: np.ndarray,
    concepts: List[str],
    threshold: float,
    min_freq: float,
    max_freq: float
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Filter concepts by detection presence (keep concepts with any detections).

    Returns:
        filtered_matrix: Filtered concept matrix
        filtered_concepts: List of kept concepts
        removed_concepts: List of removed concepts
    """
    binary = (concept_matrix > threshold).astype(float)
    counts = binary.sum(axis=0)
    total = max(len(concept_matrix), 1)
    freq = counts / total
    keep_mask = (freq >= min_freq) & (freq <= max_freq)

    filtered_concepts = [c for c, keep in zip(concepts, keep_mask) if keep]
    removed_concepts = [c for c, keep in zip(concepts, keep_mask) if not keep]
    filtered_matrix = concept_matrix[:, keep_mask]

    print(f"Filtered concepts: {len(concepts)} -> {len(filtered_concepts)}")
    print(f"  Kept concepts with frequency in [{min_freq:.3f}, {max_freq:.3f}]")

    return filtered_matrix, filtered_concepts, removed_concepts
