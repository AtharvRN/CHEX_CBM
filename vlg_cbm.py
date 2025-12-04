#!/usr/bin/env python3
"""
VLG-CBM (Vision-Language Grounded Concept Bottleneck Model) for CheXpert

This implements the VLG-CBM approach from:
"VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance"
https://arxiv.org/pdf/2408.01432

Key differences from Label-Free CBM:
1. Uses grounded annotations (from ChEX) instead of just CLIP similarity
2. Learns concept layer with explicit bounding box supervision
3. Can crop to concept regions for better localization
4. Final layer can be sparse (SAGA) or dense

Pipeline:
1. Load precomputed ChEX annotations (from generate_annotations.py)
2. Filter concepts by detection frequency
3. Train Concept Bottleneck Layer (CBL): backbone features -> concepts
4. Train sparse final layer: concept activations -> pathology labels

Usage:
    python vlg_cbm.py \
        --data_dir /workspace/CheXpert-v1.0-small \
        --concepts concepts/chexpert_concepts.json \
        --annotation_dir annotations/train \
        --output checkpoints/vlg_cbm_exp1
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Import from our modules
from dataset import (
    CheXpertDataset,
    CHEXPERT_PATHOLOGY_LABELS,
    CHEXPERT_COMPETITION_LABELS,
    get_transforms
)
from models import get_model, XRV_WEIGHTS


AVAILABLE_BACKBONES = ["densenet121", "resnet50"] + list(XRV_WEIGHTS.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="VLG-CBM for CheXpert")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to CheXpert-v1.0-small directory")
    parser.add_argument("--concepts", type=str, 
                        default="concepts/chexpert_concepts.json",
                        help="Path to concepts file (JSON or TXT)")
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default="annotations/train_chex_stage3",
        help="Directory with training annotations (default: annotations/train_chex_stage3)"
    )
    parser.add_argument(
        "--val_annotation_dir",
        type=str,
        default="annotations/val_chex_stage3",
        help="Directory with validation annotations (default: annotations/val_chex_stage3)"
    )
    parser.add_argument("--competition_labels", action="store_true",
                        help="Use only 5 competition labels")
    parser.add_argument("--uncertain_strategy", type=str, default="ones",
                        choices=["ones", "zeros"])
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--limit_strategy", type=str, default="sequential",
                        choices=["random", "sequential"],
                        help="How to pick samples when limiting the training set")
    parser.add_argument("--seed", type=int, default=42)
    
    # Models
    parser.add_argument("--backbone", type=str, default="densenet121",
                        choices=AVAILABLE_BACKBONES)
    parser.add_argument("--backbone_ckpt", type=str, default=None,
                        help="Path to finetuned backbone checkpoint")
    
    # Concept filtering
    parser.add_argument("--confidence_threshold", type=float, default=0.15,
                        help="Threshold for considering a concept detected")
    parser.add_argument("--min_concept_freq", type=float, default=0.01,
                        help="Min frequency for keeping a concept")
    parser.add_argument("--max_concept_freq", type=float, default=0.95,
                        help="Max frequency for keeping a concept")
    
    # CBL training
    parser.add_argument("--cbl_epochs", type=int, default=20,
                        help="Epochs for concept layer training")
    parser.add_argument("--cbl_lr", type=float, default=5e-4,
                        help="Learning rate for CBL")
    parser.add_argument("--cbl_batch_size", type=int, default=32)
    parser.add_argument("--cbl_hidden_layers", type=int, default=1,
                        help="Hidden layers in concept projection")
    parser.add_argument("--crop_to_concept_prob", type=float, default=0.0,
                        help="Probability of cropping to concept box during training")
    parser.add_argument("--cbl_finetune_backbone", action="store_true",
                        help="Finetune backbone during CBL training")
    parser.add_argument(
        "--resume_concept_layer",
        type=str,
        default=None,
        help="Path to a pre-trained concept_layer.pt to skip CBL training"
    )
    
    # Final layer
    parser.add_argument("--use_saga", action="store_true", default=True,
                        help="Use sparse SAGA solver for final layer")
    parser.add_argument("--saga_lam", type=float, default=0.0007,
                        help="Sparsity regularization")
    parser.add_argument("--saga_iters", type=int, default=2000)
    parser.add_argument("--saga_max_lr", type=float, default=0.1,
                        help="Initial learning rate for the SAGA solver")
    parser.add_argument("--saga_batch_size", type=int, default=256)
    parser.add_argument("--dense_lr", type=float, default=1e-3,
                        help="Learning rate for dense final layer")
    parser.add_argument("--dense_epochs", type=int, default=100)
    parser.add_argument(
        "--pre_eval_backbone",
        action="store_true",
        help="Evaluate the backbone classifier before CBM training"
    )
    
    # Output
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


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
    num_workers: int = 8
) -> Tuple[np.ndarray, List[str]]:
    """
    Load annotations and create concept presence matrix.

    Returns:
        concept_matrix: (n_images, n_concepts) binary matrix
        image_paths: List of image paths
    """
    cache_path = _annotation_cache_path(annotation_dir, n_images, concepts, confidence_threshold)
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            matrix = cached["concept_matrix"]
            paths = cached["image_paths"].tolist()
            if matrix.shape == (n_images, len(concepts)):
                print(f"Loaded cached annotations from {cache_path}")
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
            cache_path,
            concept_matrix=concept_matrix,
            image_paths=np.array(image_paths, dtype=object)
        )
        print(f"Cached annotations to {cache_path}")
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
    # Binarize detections and count occurrences
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


class ConceptDataset(Dataset):
    """Dataset that returns images and concept annotations."""
    
    def __init__(
        self,
        base_dataset: CheXpertDataset,
        concept_matrix: np.ndarray,
        threshold: float = 0.15
    ):
        self.base_dataset = base_dataset
        self.concept_matrix = concept_matrix
        self.threshold = threshold
        
        # Binarize concepts
        self.concept_labels = (concept_matrix > threshold).astype(np.float32)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, pathology_labels = self.base_dataset[idx]
        concept_labels = torch.from_numpy(self.concept_labels[idx])
        return image, concept_labels, pathology_labels


class ConceptLayer(nn.Module):
    """
    Concept Bottleneck Layer: maps backbone features to concept activations.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_concepts: int,
        num_hidden: int = 1,
        hidden_dim: int = None
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        layers = []
        in_dim = input_dim
        
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, n_concepts))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class BackboneWithConcepts(nn.Module):
    """Backbone + Concept Layer for end-to-end training."""
    
    def __init__(self, backbone: nn.Module, concept_layer: ConceptLayer):
        super().__init__()
        self.backbone = backbone
        self.concept_layer = concept_layer
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        concepts = self.concept_layer(features)
        return concepts
    
    def get_features(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            if features.dim() > 2:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features


def train_concept_layer(
    model: BackboneWithConcepts,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    lr: float,
    device: str,
    finetune_backbone: bool = False
) -> BackboneWithConcepts:
    """Train the concept bottleneck layer."""
    
    model = model.to(device)
    
    if finetune_backbone:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        # Only train concept layer
        model.backbone.eval()
        for param in model.backbone.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(model.concept_layer.parameters(), lr=lr)
    
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        # Train
        model.concept_layer.train()
        train_loss = 0.0
        
        for images, concept_labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            concept_labels = concept_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, concept_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, concept_labels, _ in val_loader:
                images = images.to(device)
                concept_labels = concept_labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, concept_labels)
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def extract_concept_activations(
    model: BackboneWithConcepts,
    loader: DataLoader,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract concept activations and pathology labels."""
    
    model.eval()
    all_concepts = []
    all_labels = []
    
    with torch.no_grad():
        for images, _, pathology_labels in tqdm(loader, desc="Extracting concepts"):
            images = images.to(device)
            concepts = model(images)
            all_concepts.append(concepts.cpu())
            all_labels.append(pathology_labels)
    
    return torch.cat(all_concepts), torch.cat(all_labels)


def train_final_layer_saga(
    concept_activations: torch.Tensor,
    labels: torch.Tensor,
    val_concepts: torch.Tensor,
    val_labels: torch.Tensor,
    n_classes: int,
    lam: float,
    n_iters: int,
    batch_size: int,
    device: str,
    max_lr: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train sparse final layer using SAGA."""
    
    try:
        from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
    except ImportError:
        print("glm_saga not available, falling back to dense")
        return train_final_layer_dense(
            concept_activations, labels,
            val_concepts, val_labels,
            n_classes, batch_size, device
        )
    
    # Normalize
    mean = concept_activations.mean(dim=0, keepdim=True)
    std = concept_activations.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-6)
    
    train_c = (concept_activations - mean) / std
    val_c = (val_concepts - mean) / std
    
    # DataLoaders
    train_ds = IndexedTensorDataset(train_c, labels.float())
    val_ds = TensorDataset(val_c, val_labels.float())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Initialize
    linear = nn.Linear(train_c.shape[1], n_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    metadata = {'max_reg': {'nongrouped': lam}}
    
    output = glm_saga(
        linear, train_loader,
        max_lr=max_lr,
        nepochs=n_iters,
        alpha=0.99,
        epsilon=1.0,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_c),
        n_classes=n_classes,
        family='multilabel'
    )
    
    W = output['path'][0]['weight']
    b = output['path'][0]['bias']
    
    return W, b, mean, std


def train_final_layer_dense(
    concept_activations: torch.Tensor,
    labels: torch.Tensor,
    val_concepts: torch.Tensor,
    val_labels: torch.Tensor,
    n_classes: int,
    batch_size: int,
    device: str,
    n_epochs: int = 100,
    lr: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train dense final layer with BCE loss."""
    
    # Normalize
    mean = concept_activations.mean(dim=0, keepdim=True)
    std = concept_activations.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-6)
    
    train_c = (concept_activations - mean) / std
    val_c = (val_concepts - mean) / std
    
    # DataLoaders
    train_ds = TensorDataset(train_c, labels.float())
    val_ds = TensorDataset(val_c, val_labels.float())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Model
    linear = nn.Linear(train_c.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        linear.train()
        for batch_c, batch_y in train_loader:
            batch_c = batch_c.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            loss = criterion(linear(batch_c), batch_y)
            loss.backward()
            optimizer.step()
        
        # Validate
        linear.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_c, batch_y in val_loader:
                batch_c = batch_c.to(device)
                batch_y = batch_y.to(device)
                val_loss += criterion(linear(batch_c), batch_y).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in linear.state_dict().items()}
    
    if best_state:
        linear.load_state_dict(best_state)
    
    return linear.weight.data, linear.bias.data, mean, std


def evaluate(
    concept_activations: torch.Tensor,
    labels: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    label_names: List[str]
) -> Dict:
    """Evaluate model and return metrics."""
    
    # Normalize
    c_norm = (concept_activations - mean) / std
    
    # Predict
    logits = c_norm @ W.T + b
    probs = torch.sigmoid(logits).numpy()
    targets = labels.numpy()
    
    metrics = {}
    valid_aurocs = []
    valid_aps = []
    
    for i, name in enumerate(label_names):
        if len(np.unique(targets[:, i])) > 1:
            auroc = roc_auc_score(targets[:, i], probs[:, i])
            ap = average_precision_score(targets[:, i], probs[:, i])
            metrics[f"auroc_{name}"] = auroc
            metrics[f"ap_{name}"] = ap
            valid_aurocs.append(auroc)
            valid_aps.append(ap)
    
    metrics["mean_auroc"] = np.mean(valid_aurocs) if valid_aurocs else 0.0
    metrics["mean_ap"] = np.mean(valid_aps) if valid_aps else 0.0
    
    return metrics


def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str]
) -> Dict:
    """Compute AUROC/AP for raw predictions."""
    metrics = {}
    valid_aurocs = []
    valid_aps = []
    
    for i, name in enumerate(label_names):
        if len(np.unique(y_true[:, i])) > 1:
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            metrics[f"auroc_{name}"] = auroc
            metrics[f"ap_{name}"] = ap
            valid_aurocs.append(auroc)
            valid_aps.append(ap)
    
    metrics["mean_auroc"] = np.mean(valid_aurocs) if valid_aurocs else 0.0
    metrics["mean_ap"] = np.mean(valid_aps) if valid_aps else 0.0
    return metrics


def evaluate_baseline_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    label_names: List[str]
) -> Dict:
    """Evaluate the backbone classifier before CBM training."""
    model = model.to(device)
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for images, labels_batch in tqdm(loader, desc="Baseline evaluation"):
            images = images.to(device)
            logits = model(images)
            preds.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels_batch.numpy())
    
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    metrics = evaluate_multilabel(targets, preds, label_names)
    return metrics


def save_concept_artifacts(
    model: BackboneWithConcepts,
    concepts: List[str],
    output_dir: str,
    save_backbone: bool = False
) -> None:
    """Persist the concept bottleneck weights so training progress is not lost."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "concepts.txt"), 'w') as f:
        f.write('\n'.join(concepts))
    
    torch.save(
        model.concept_layer.state_dict(),
        os.path.join(output_dir, "concept_layer.pt")
    )
    
    if save_backbone:
        torch.save(
            model.backbone.state_dict(),
            os.path.join(output_dir, "backbone.pt")
        )


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Select labels
    if args.competition_labels:
        labels = CHEXPERT_COMPETITION_LABELS
    else:
        labels = CHEXPERT_PATHOLOGY_LABELS
    num_classes = len(labels)
    print(f"Using {num_classes} pathology labels")
    
    # Load concepts
    concepts_dict, concepts = load_concepts(args.concepts)
    print(f"Loaded {len(concepts)} concepts")
    
    # Save config
    config = vars(args).copy()
    config['labels'] = labels
    config['num_classes'] = num_classes
    config['num_concepts_initial'] = len(concepts)
    config['start_time'] = datetime.now().isoformat()
    
    with open(os.path.join(args.output, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # =========================================
    # Load datasets
    # =========================================
    print("\nLoading datasets...")
    train_csv = os.path.join(args.data_dir, "train.csv")
    val_csv = os.path.join(args.data_dir, "valid.csv")
    img_root = os.path.dirname(args.data_dir)
    
    transform = get_transforms(224, is_training=False)
    
    train_dataset = CheXpertDataset(
        csv_path=train_csv,
        img_root=img_root,
        transform=transform,
        labels=labels,
        uncertain_strategy=args.uncertain_strategy,
        frontal_only=True
    )
    
    val_dataset = CheXpertDataset(
        csv_path=val_csv,
        img_root=img_root,
        transform=transform,
        labels=labels,
        uncertain_strategy=args.uncertain_strategy,
        frontal_only=True
    )
    
    classifier_val_loader = DataLoader(
        val_dataset,
        batch_size=args.cbl_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Limit samples
    if args.limit_samples and args.limit_samples < len(train_dataset):
        print(f"Limiting to {args.limit_samples} training samples")
        if args.limit_strategy == "random":
            rng = np.random.RandomState(args.seed)
            indices = rng.permutation(len(train_dataset))[:args.limit_samples]
        else:
            indices = np.arange(args.limit_samples)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    print(f"Train: {n_train}, Val: {n_val}")
    
    # Optional baseline evaluation before loading annotations/training
    if args.pre_eval_backbone:
        print("\nRunning baseline backbone evaluation before CBM training...")
        baseline_kwargs = {}
        if args.backbone in XRV_WEIGHTS:
            baseline_kwargs["target_labels"] = labels
        baseline_model = get_model(
            args.backbone,
            num_classes=num_classes,
            pretrained=True,
            **baseline_kwargs
        )
        if args.backbone_ckpt:
            import inspect
            load_kwargs = {'map_location': device}
            if 'weights_only' in inspect.signature(torch.load).parameters:
                load_kwargs['weights_only'] = False
            checkpoint = torch.load(args.backbone_ckpt, **load_kwargs)
            state = checkpoint.get('model_state_dict', checkpoint)
            baseline_model.load_state_dict(state, strict=False)
        baseline_metrics = evaluate_baseline_model(
            baseline_model, classifier_val_loader, device, labels
        )
        print("\nBaseline Backbone Metrics:")
        print(f"  Mean AUROC: {baseline_metrics['mean_auroc']:.4f}")
        print(f"  Mean AP:    {baseline_metrics['mean_ap']:.4f}")
        for label in labels:
            auroc = baseline_metrics.get(f'auroc_{label}', float('nan'))
            ap = baseline_metrics.get(f'ap_{label}', float('nan'))
            print(f"    {label}: AUROC={auroc:.4f}, AP={ap:.4f}")
    
    # =========================================
    # Load annotations
    # =========================================
    concept_matrix, _ = load_annotations(
        args.annotation_dir,
        n_train,
        concepts,
        args.confidence_threshold,
        num_workers=max(args.num_workers, 1)
    )
    
    # Also load val annotations if available
    val_ann_dir = args.val_annotation_dir
    if val_ann_dir and os.path.exists(val_ann_dir):
        val_concept_matrix, _ = load_annotations(
            val_ann_dir,
            n_val,
            concepts,
            args.confidence_threshold,
            num_workers=max(args.num_workers, 1)
        )
    else:
        missing = val_ann_dir or "<unspecified>"
        print(f"No validation annotations found at {missing}, using zeros")
        val_concept_matrix = np.zeros((n_val, len(concepts)), dtype=np.float32)
    
    # =========================================
    # Filter concepts
    # =========================================
    concept_matrix, filtered_concepts, removed = filter_concepts_by_frequency(
        concept_matrix, concepts,
        args.confidence_threshold,
        args.min_concept_freq,
        args.max_concept_freq
    )
    
    # Filter val matrix too
    keep_indices = [i for i, c in enumerate(concepts) if c in filtered_concepts]
    val_concept_matrix = val_concept_matrix[:, keep_indices]
    concepts = filtered_concepts
    
    # Check if we have enough concepts
    if len(concepts) < 10:
        print(f"Warning: Only {len(concepts)} concepts remaining!")
    
    # =========================================
    # Create concept datasets
    # =========================================
    train_concept_ds = ConceptDataset(
        train_dataset,
        concept_matrix,
        args.confidence_threshold
    )
    
    val_concept_ds = ConceptDataset(
        val_dataset,
        val_concept_matrix,
        args.confidence_threshold
    )
    
    train_loader = DataLoader(
        train_concept_ds, batch_size=args.cbl_batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_concept_ds, batch_size=args.cbl_batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    
    # =========================================
    # Create backbone + concept layer
    # =========================================
    print("\nCreating model...")
    
    use_xrv_backbone = args.backbone in XRV_WEIGHTS
    backbone_kwargs = {}
    if use_xrv_backbone:
        backbone_kwargs["target_labels"] = labels
    backbone_model = get_model(
        args.backbone,
        num_classes=num_classes,
        pretrained=True,
        **backbone_kwargs,
    )
    
    if args.backbone_ckpt:
        import inspect
        load_kwargs = {'map_location': device}
        if 'weights_only' in inspect.signature(torch.load).parameters:
            load_kwargs['weights_only'] = False
        ckpt = torch.load(args.backbone_ckpt, **load_kwargs)
        state = ckpt.get('model_state_dict', ckpt)
        backbone_model.load_state_dict(state, strict=False)
    
    # Get feature extractor from classifier wrapper
    if args.backbone == "densenet121":
        feature_dim = 1024
        backbone = backbone_model.backbone.features
    elif args.backbone == "resnet50":
        feature_dim = 2048
        backbone = nn.Sequential(*list(backbone_model.backbone.children())[:-1])
    else:
        feature_dim = getattr(backbone_model, "feature_dim", 1024)

        class XRVDenseNetBackbone(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.get_features(x)

        backbone = XRVDenseNetBackbone(backbone_model)
    
    # Concept layer
    concept_layer = ConceptLayer(
        feature_dim, len(concepts),
        num_hidden=args.cbl_hidden_layers
    )
    
    model = BackboneWithConcepts(backbone, concept_layer)
    
    # =========================================
    # Train concept layer
    # =========================================
    if args.resume_concept_layer:
        print(f"\nLoading concept bottleneck layer from {args.resume_concept_layer}...")
        state = torch.load(args.resume_concept_layer, map_location=device)
        model.concept_layer.load_state_dict(state)
        model = model.to(device)
    else:
        print("\nTraining concept bottleneck layer...")
        model = train_concept_layer(
            model, train_loader, val_loader,
            args.cbl_epochs, args.cbl_lr, device,
            finetune_backbone=args.cbl_finetune_backbone
        )
        
        print("\nSaving concept bottleneck checkpoint...")
        save_concept_artifacts(
            model, concepts, args.output,
            save_backbone=args.cbl_finetune_backbone
        )
    
    # =========================================
    # Extract concept activations
    # =========================================
    print("\nExtracting concept activations...")
    train_concepts, train_labels = extract_concept_activations(model, train_loader, device)
    val_concepts, val_labels = extract_concept_activations(model, val_loader, device)
    
    print(f"Train concepts: {train_concepts.shape}")
    print(f"Val concepts: {val_concepts.shape}")
    
    # =========================================
    # Train final layer
    # =========================================
    if args.use_saga:
        print("\nTraining sparse final layer (SAGA)...")
        W, b, mean, std = train_final_layer_saga(
            train_concepts, train_labels,
            val_concepts, val_labels,
            num_classes,
            args.saga_lam, args.saga_iters,
            args.saga_batch_size, device,
            args.saga_max_lr
        )
    else:
        print("\nTraining dense final layer...")
        W, b, mean, std = train_final_layer_dense(
            train_concepts, train_labels,
            val_concepts, val_labels,
            num_classes,
            args.saga_batch_size, device,
            args.dense_epochs, args.dense_lr
        )
    
    # Sparsity stats
    nnz = (W.abs() > 1e-5).sum().item()
    total = W.numel()
    print(f"Sparsity: {nnz}/{total} non-zero ({100*nnz/total:.1f}%)")
    
    # =========================================
    # Evaluate
    # =========================================
    print("\nEvaluating...")
    train_metrics = evaluate(train_concepts, train_labels, W, b, mean, std, labels)
    val_metrics = evaluate(val_concepts, val_labels, W, b, mean, std, labels)
    
    print(f"\nTrain - Mean AUROC: {train_metrics['mean_auroc']:.4f}")
    print(f"Val   - Mean AUROC: {val_metrics['mean_auroc']:.4f}")
    
    print("\nPer-class Validation AUROC:")
    for label in labels:
        auroc = val_metrics.get(f'auroc_{label}', float('nan'))
        print(f"  {label}: {auroc:.4f}")
    
    # =========================================
    # Save model and results
    # =========================================
    print("\nSaving...")
    
    save_concept_artifacts(
        model, concepts, args.output,
        save_backbone=args.cbl_finetune_backbone
    )
    
    # Save final layer weights
    torch.save(W, os.path.join(args.output, "W_g.pt"))
    torch.save(b, os.path.join(args.output, "b_g.pt"))
    torch.save(mean, os.path.join(args.output, "concept_mean.pt"))
    torch.save(std, os.path.join(args.output, "concept_std.pt"))
    
    # Save metrics
    with open(os.path.join(args.output, "train_metrics.json"), 'w') as f:
        json.dump(train_metrics, f, indent=2)
    with open(os.path.join(args.output, "val_metrics.json"), 'w') as f:
        json.dump(val_metrics, f, indent=2)
    
    # Extract interpretations
    interpretations = {}
    W_np = W.cpu().numpy()
    
    for class_idx, class_name in enumerate(labels):
        weights = W_np[class_idx]
        
        top_pos_idx = np.argsort(weights)[-10:][::-1]
        top_neg_idx = np.argsort(weights)[:10]
        
        interpretations[class_name] = {
            'positive': [(concepts[i], float(weights[i])) 
                        for i in top_pos_idx if weights[i] > 0],
            'negative': [(concepts[i], float(weights[i])) 
                        for i in top_neg_idx if weights[i] < 0]
        }
    
    with open(os.path.join(args.output, "interpretations.json"), 'w') as f:
        json.dump(interpretations, f, indent=2)
    
    print("\n" + "="*60)
    print("VLG-CBM TRAINING COMPLETE")
    print("="*60)
    print(f"Final concepts: {len(concepts)}")
    print(f"Val Mean AUROC: {val_metrics['mean_auroc']:.4f}")
    print(f"Sparsity: {100*nnz/total:.1f}% non-zero")
    print(f"Saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
