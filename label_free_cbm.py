#!/usr/bin/env python3
"""
Label-Free Concept Bottleneck Model (LF-CBM) for CheXpert

This implements the Label-Free CBM approach from:
"Label-Free Concept Bottleneck Models" (Oikarinen et al., 2023)

The key idea:
1. Use CLIP to compute image-text similarity for concept "pseudo-labels"
2. Learn a projection from backbone features -> concept space
3. Train a sparse linear classifier on concept activations

No explicit concept annotations needed!

Usage:
    python label_free_cbm.py \
        --data_dir /workspace/CheXpert-v1.0-small \
        --concepts concepts/chexpert_concepts.txt \
        --output checkpoints/lf_cbm_exp1
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Import from our existing modules
from dataset import (
    CheXpertDataset,
    CHEXPERT_PATHOLOGY_LABELS,
    CHEXPERT_COMPETITION_LABELS,
    get_transforms
)
from models import get_model

# Try importing BiomedCLIP
try:
    from open_clip import create_model_from_pretrained, get_tokenizer
    BIOMEDCLIP_AVAILABLE = True
except ImportError:
    BIOMEDCLIP_AVAILABLE = False
    print("Warning: open_clip not available. Install with: pip install open_clip_torch")


def parse_args():
    parser = argparse.ArgumentParser(description="Label-Free CBM for CheXpert")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to CheXpert-v1.0-small directory")
    parser.add_argument("--concepts", type=str, default="concepts/chexpert_concepts.txt",
                        help="Path to concept set file (one concept per line)")
    parser.add_argument("--competition_labels", action="store_true",
                        help="Use only 5 competition labels")
    parser.add_argument("--uncertain_strategy", type=str, default="ones",
                        choices=["ones", "zeros"],
                        help="How to handle uncertain labels")
    parser.add_argument("--limit_samples", type=int, default=None,
                        help="Limit training samples")
    parser.add_argument("--seed", type=int, default=42)
    
    # Models
    parser.add_argument("--backbone", type=str, default="densenet121",
                        choices=["densenet121", "resnet50"],
                        help="Backbone model for feature extraction")
    parser.add_argument("--backbone_ckpt", type=str, default=None,
                        help="Path to finetuned backbone checkpoint")
    parser.add_argument("--clip_name", type=str, default="biomedclip",
                        help="CLIP model to use: biomedclip or openai")
    
    # CBM parameters
    parser.add_argument("--clip_cutoff", type=float, default=0.20,
                        help="Concepts with smaller top-k CLIP activation will be deleted")
    parser.add_argument("--interpretability_cutoff", type=float, default=0.40,
                        help="Concepts with smaller similarity will be deleted")
    parser.add_argument("--proj_steps", type=int, default=1000,
                        help="Steps to train projection layer")
    parser.add_argument("--proj_lr", type=float, default=1e-3,
                        help="Learning rate for projection layer")
    parser.add_argument("--proj_batch_size", type=int, default=512,
                        help="Batch size for projection training")
    
    # Final layer (SAGA)
    parser.add_argument("--lam", type=float, default=0.0007,
                        help="Sparsity regularization (higher = more sparse)")
    parser.add_argument("--saga_iters", type=int, default=1000,
                        help="Iterations for SAGA solver")
    parser.add_argument("--saga_batch_size", type=int, default=256)
    
    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Caching
    parser.add_argument("--activation_dir", type=str, default="saved_activations",
                        help="Directory to cache activations")
    parser.add_argument("--recompute_activations", action="store_true",
                        help="Force recompute activations even if cached")
    
    return parser.parse_args()


def load_concepts(path: str) -> list:
    """Load concepts from text file (one per line)."""
    with open(path) as f:
        concepts = [line.strip() for line in f if line.strip()]
    return concepts


def cos_similarity_cubed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cubed cosine similarity (per-column)."""
    a_norm = F.normalize(a, dim=0)
    b_norm = F.normalize(b, dim=0)
    sim = (a_norm * b_norm).sum(dim=0)
    return sim ** 3


class BiomedCLIPEncoder:
    """Wrapper for BiomedCLIP model."""
    
    def __init__(self, device: str = "cuda"):
        if not BIOMEDCLIP_AVAILABLE:
            raise RuntimeError("open_clip not installed")
        
        self.device = device
        self.model, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.model = self.model.to(device).eval()
    
    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode batch of images."""
        images = images.to(self.device)
        features = self.model.encode_image(images)
        return features.cpu()
    
    @torch.no_grad()
    def encode_texts(self, texts: list) -> torch.Tensor:
        """Encode list of text concepts."""
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        return features.cpu()
    
    def get_preprocess(self):
        """Return image preprocessing transform."""
        return self.preprocess


class BackboneEncoder(nn.Module):
    """Wrapper for backbone model feature extraction."""
    
    def __init__(self, model_name: str, checkpoint: str = None, device: str = "cuda"):
        super().__init__()
        self.device = device
        
        # Get the model without the final classifier
        self.model = get_model(model_name, num_classes=1, pretrained=True)
        # Underlying torchvision backbone (DenseNet/ResNet) may hang off .backbone
        self.backbone = getattr(self.model, 'backbone', self.model)
        
        if checkpoint is not None:
            import inspect
            load_kwargs = {'map_location': device}
            if 'weights_only' in inspect.signature(torch.load).parameters:
                load_kwargs['weights_only'] = False
            ckpt = torch.load(checkpoint, **load_kwargs)
            state_dict = ckpt.get('model_state_dict', ckpt)
            # Remove classifier weights regardless of prefix (e.g., backbone.classifier)
            filtered = {}
            for k, v in state_dict.items():
                if k.startswith('classifier') or '.classifier' in k:
                    continue
                filtered[k] = v
            self.model.load_state_dict(filtered, strict=False)
        
        self.model = self.model.to(device).eval()
        
        # Get feature dimension
        if model_name == "densenet121":
            self.feature_dim = 1024
        else:  # resnet50
            self.feature_dim = 2048
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        backbone = self.backbone
        if hasattr(backbone, 'features'):
            # DenseNet
            features = backbone.features(x)
            features = F.relu(features, inplace=True)
            features = F.adaptive_avg_pool2d(features, (1, 1))
        else:
            # ResNet - extract before final layer
            x = backbone.conv1(x)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x)
            x = backbone.layer2(x)
            x = backbone.layer3(x)
            x = backbone.layer4(x)
            features = backbone.avgpool(x)
        return features.view(features.size(0), -1).cpu()


def compute_and_cache_activations(
    dataset,
    backbone: BackboneEncoder,
    clip_encoder: BiomedCLIPEncoder,
    concepts: list,
    cache_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cuda",
    force_recompute: bool = False
):
    """
    Compute backbone features, CLIP image features, and CLIP text features.
    Cache results for efficiency.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    backbone_path = cache_path + "_backbone.pt"
    clip_img_path = cache_path + "_clip_img.pt"
    clip_txt_path = cache_path + "_clip_txt.pt"
    
    # Check if all cached
    if (os.path.exists(backbone_path) and 
        os.path.exists(clip_img_path) and 
        os.path.exists(clip_txt_path) and 
        not force_recompute):
        print(f"Loading cached activations from {cache_path}")
        backbone_feats = torch.load(backbone_path)
        clip_img_feats = torch.load(clip_img_path)
        clip_txt_feats = torch.load(clip_txt_path)
        return backbone_feats, clip_img_feats, clip_txt_feats
    
    print(f"Computing activations for {len(dataset)} images...")
    
    # Create loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    backbone_features = []
    clip_img_features = []
    
    for images, _ in tqdm(loader, desc="Extracting features"):
        # Backbone features
        bb_feats = backbone(images)
        backbone_features.append(bb_feats)
        
        # CLIP image features (need to convert grayscale to RGB if needed)
        if images.shape[1] == 1:
            images_rgb = images.repeat(1, 3, 1, 1)
        else:
            images_rgb = images
        clip_feats = clip_encoder.encode_images(images_rgb)
        clip_img_features.append(clip_feats)
    
    backbone_feats = torch.cat(backbone_features, dim=0)
    clip_img_feats = torch.cat(clip_img_features, dim=0)
    
    # CLIP text features
    print(f"Encoding {len(concepts)} concepts...")
    clip_txt_feats = clip_encoder.encode_texts(concepts)
    
    # Cache
    torch.save(backbone_feats, backbone_path)
    torch.save(clip_img_feats, clip_img_path)
    torch.save(clip_txt_feats, clip_txt_path)
    print(f"Cached activations to {cache_path}")
    
    return backbone_feats, clip_img_feats, clip_txt_feats


def train_projection_layer(
    backbone_features: torch.Tensor,
    clip_features: torch.Tensor,  # image @ text.T
    val_backbone_features: torch.Tensor,
    val_clip_features: torch.Tensor,
    n_concepts: int,
    proj_steps: int = 1000,
    proj_lr: float = 1e-3,
    proj_batch_size: int = 512,
    device: str = "cuda"
) -> nn.Linear:
    """
    Train projection layer: backbone features -> concept activations
    Optimized to match CLIP concept similarities.
    """
    proj_layer = nn.Linear(backbone_features.shape[1], n_concepts, bias=False).to(device)
    optimizer = torch.optim.Adam(proj_layer.parameters(), lr=proj_lr)
    
    indices = list(range(len(backbone_features)))
    best_val_loss = float('inf')
    best_weights = proj_layer.weight.clone()
    best_step = 0
    
    batch_size = min(proj_batch_size, len(backbone_features))
    
    for step in range(proj_steps):
        batch_idx = torch.LongTensor(random.sample(indices, batch_size))
        
        proj_out = proj_layer(backbone_features[batch_idx].to(device))
        target = clip_features[batch_idx].to(device)
        
        # Loss: negative cubed cosine similarity
        loss = -cos_similarity_cubed(proj_out.T, target.T).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation every 50 steps
        if step % 50 == 0 or step == proj_steps - 1:
            with torch.no_grad():
                val_out = proj_layer(val_backbone_features.to(device))
                val_loss = -cos_similarity_cubed(val_out.T, val_clip_features.to(device).T).mean()
            
            if step == 0 or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = proj_layer.weight.clone()
                best_step = step
                
            if step % 100 == 0:
                print(f"Step {step}: train_sim={-loss.item():.4f}, val_sim={-val_loss.item():.4f}")
            
            # Early stopping
            if step > 0 and val_loss > best_val_loss:
                break
    
    proj_layer.load_state_dict({"weight": best_weights})
    print(f"Best step: {best_step}, val_similarity: {-best_val_loss.item():.4f}")
    
    return proj_layer


def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list
) -> dict:
    """Compute AUROC and AP for multi-label classification."""
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


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Select labels
    if args.competition_labels:
        labels = CHEXPERT_COMPETITION_LABELS
        print("Using 5 competition labels")
    else:
        labels = CHEXPERT_PATHOLOGY_LABELS
        print("Using 12 pathology labels")
    
    num_classes = len(labels)
    
    # Load concepts
    concepts = load_concepts(args.concepts)
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
    
    # Limit samples if specified
    if args.limit_samples is not None and args.limit_samples < len(train_dataset):
        print(f"Limiting to {args.limit_samples} training samples")
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(len(train_dataset))[:args.limit_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Get targets
    if hasattr(train_dataset, 'targets'):
        train_targets = train_dataset.targets.numpy()
    else:
        train_targets = train_dataset.dataset.targets[train_dataset.indices].numpy()
    val_targets = val_dataset.targets.numpy()
    
    # =========================================
    # Load models
    # =========================================
    print("\nLoading models...")
    
    # Backbone
    backbone = BackboneEncoder(args.backbone, args.backbone_ckpt, device)
    print(f"Backbone: {args.backbone}, feature_dim={backbone.feature_dim}")
    
    # CLIP
    clip_encoder = BiomedCLIPEncoder(device)
    print("BiomedCLIP loaded")
    
    # =========================================
    # Compute activations
    # =========================================
    cache_base = os.path.join(
        args.activation_dir,
        f"chexpert_{args.backbone}_{len(train_dataset)}"
    )
    
    train_bb, train_clip_img, clip_txt = compute_and_cache_activations(
        train_dataset, backbone, clip_encoder, concepts,
        cache_base + "_train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        force_recompute=args.recompute_activations
    )
    
    val_bb, val_clip_img, _ = compute_and_cache_activations(
        val_dataset, backbone, clip_encoder, concepts,
        cache_base + "_val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        force_recompute=args.recompute_activations
    )
    
    # Normalize CLIP features
    train_clip_img = F.normalize(train_clip_img, dim=1)
    val_clip_img = F.normalize(val_clip_img, dim=1)
    clip_txt = F.normalize(clip_txt, dim=1)
    
    # Compute image-concept similarities
    train_clip_sim = train_clip_img @ clip_txt.T
    val_clip_sim = val_clip_img @ clip_txt.T
    
    # =========================================
    # Filter concepts by CLIP activation
    # =========================================
    print("\nFiltering concepts by CLIP activation...")
    highest = torch.mean(torch.topk(train_clip_sim, dim=0, k=5)[0], dim=0)
    
    keep_mask = highest > args.clip_cutoff
    removed_clip = [concepts[i] for i in range(len(concepts)) if not keep_mask[i]]
    concepts = [concepts[i] for i in range(len(concepts)) if keep_mask[i]]
    
    print(f"Removed {len(removed_clip)} concepts (CLIP cutoff)")
    print(f"Remaining: {len(concepts)} concepts")
    
    # Filter features
    clip_txt = clip_txt[keep_mask]
    train_clip_sim = train_clip_img @ clip_txt.T
    val_clip_sim = val_clip_img @ clip_txt.T
    
    # =========================================
    # Train projection layer
    # =========================================
    print("\nTraining projection layer...")
    proj_layer = train_projection_layer(
        train_bb, train_clip_sim,
        val_bb, val_clip_sim,
        n_concepts=len(concepts),
        proj_steps=args.proj_steps,
        proj_lr=args.proj_lr,
        proj_batch_size=args.proj_batch_size,
        device=device
    )
    
    # =========================================
    # Filter concepts by interpretability
    # =========================================
    print("\nFiltering concepts by interpretability...")
    with torch.no_grad():
        val_proj = proj_layer(val_bb.to(device))
        sim = cos_similarity_cubed(val_proj, val_clip_sim.to(device))
        interpretable = sim > args.interpretability_cutoff
    
    removed_interp = [concepts[i] for i in range(len(concepts)) if not interpretable[i].item()]
    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i].item()]
    
    print(f"Removed {len(removed_interp)} concepts (interpretability)")
    print(f"Final concepts: {len(concepts)}")
    
    # Update projection layer
    W_c = proj_layer.weight[interpretable.cpu()].clone()
    proj_layer = nn.Linear(train_bb.shape[1], len(concepts), bias=False)
    proj_layer.load_state_dict({"weight": W_c})
    
    # =========================================
    # Compute concept activations
    # =========================================
    print("\nComputing concept activations...")
    with torch.no_grad():
        train_c = proj_layer(train_bb)
        val_c = proj_layer(val_bb)
        
        # Normalize
        train_mean = train_c.mean(dim=0, keepdim=True)
        train_std = train_c.std(dim=0, keepdim=True)
        train_std = torch.clamp(train_std, min=1e-6)
        
        train_c = (train_c - train_mean) / train_std
        val_c = (val_c - train_mean) / train_std
    
    # =========================================
    # Train final sparse layer (SAGA)
    # =========================================
    print("\nTraining sparse final layer (SAGA)...")
    
    # Try importing glm_saga
    try:
        from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
        USE_SAGA = True
    except ImportError:
        print("Warning: glm_saga not available, using dense layer")
        USE_SAGA = False
    
    train_y = torch.from_numpy(train_targets).float()
    val_y = torch.from_numpy(val_targets).float()
    
    if USE_SAGA:
        indexed_train_ds = IndexedTensorDataset(train_c, train_y)
        train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
        val_ds = TensorDataset(val_c, val_y)
        val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
        
        # Initialize final layer
        final_layer = nn.Linear(len(concepts), num_classes).to(device)
        final_layer.weight.data.zero_()
        final_layer.bias.data.zero_()
        
        metadata = {'max_reg': {'nongrouped': args.lam}}
        
        output = glm_saga(
            final_layer,
            train_loader,
            max_lr=0.1,
            nepochs=args.saga_iters,
            alpha=0.99,
            epsilon=1.0,
            k=1,
            val_loader=val_loader,
            do_zero=False,
            metadata=metadata,
            n_ex=len(train_c),
            n_classes=num_classes,
            family='multilabel'
        )
        
        W_g = output['path'][0]['weight']
        b_g = output['path'][0]['bias']
        final_layer.load_state_dict({'weight': W_g, 'bias': b_g})
        
        # Sparsity stats
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        print(f"Sparsity: {nnz}/{total} non-zero ({100*nnz/total:.1f}%)")
    else:
        # Dense fallback
        final_layer = nn.Linear(len(concepts), num_classes).to(device)
        optimizer = torch.optim.Adam(final_layer.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        train_ds = TensorDataset(train_c, train_y)
        train_loader = DataLoader(train_ds, batch_size=args.saga_batch_size, shuffle=True)
        
        for epoch in range(100):
            for batch_c, batch_y in train_loader:
                batch_c = batch_c.to(device)
                batch_y = batch_y.to(device)
                
                logits = final_layer(batch_c)
                loss = criterion(logits, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        W_g = final_layer.weight.data
        b_g = final_layer.bias.data
    
    # =========================================
    # Evaluate
    # =========================================
    print("\nEvaluating...")
    final_layer.eval()
    
    with torch.no_grad():
        train_logits = final_layer(train_c.to(device))
        val_logits = final_layer(val_c.to(device))
        
        train_probs = torch.sigmoid(train_logits).cpu().numpy()
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
    
    train_metrics = evaluate_multilabel(train_targets, train_probs, labels)
    val_metrics = evaluate_multilabel(val_targets, val_probs, labels)
    
    print(f"\nTrain - Mean AUROC: {train_metrics['mean_auroc']:.4f}, Mean AP: {train_metrics['mean_ap']:.4f}")
    print(f"Val   - Mean AUROC: {val_metrics['mean_auroc']:.4f}, Mean AP: {val_metrics['mean_ap']:.4f}")
    
    print("\nPer-class Validation AUROC:")
    for label in labels:
        auroc = val_metrics.get(f'auroc_{label}', float('nan'))
        print(f"  {label}: {auroc:.4f}")
    
    # =========================================
    # Save model and results
    # =========================================
    print("\nSaving model...")
    
    # Save concept set
    with open(os.path.join(args.output, "concepts.txt"), 'w') as f:
        f.write('\n'.join(concepts))
    
    # Save weights
    torch.save(W_c, os.path.join(args.output, "W_c.pt"))
    torch.save(W_g, os.path.join(args.output, "W_g.pt"))
    torch.save(b_g, os.path.join(args.output, "b_g.pt"))
    torch.save(train_mean, os.path.join(args.output, "proj_mean.pt"))
    torch.save(train_std, os.path.join(args.output, "proj_std.pt"))
    
    # Save metrics
    with open(os.path.join(args.output, "train_metrics.json"), 'w') as f:
        json.dump(train_metrics, f, indent=2)
    with open(os.path.join(args.output, "val_metrics.json"), 'w') as f:
        json.dump(val_metrics, f, indent=2)
    
    # Extract interpretations (top concepts per class)
    interpretations = {}
    W_g_np = W_g.cpu().numpy()
    
    for class_idx, class_name in enumerate(labels):
        weights = W_g_np[class_idx]
        
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
    
    # Print interpretations
    print("\n" + "="*60)
    print("MODEL INTERPRETATIONS")
    print("="*60)
    for cls, weights in interpretations.items():
        print(f"\n{cls}:")
        if weights['positive']:
            print("  Positive:")
            for concept, w in weights['positive'][:3]:
                print(f"    + {concept}: {w:.3f}")
        if weights['negative']:
            print("  Negative:")
            for concept, w in weights['negative'][:3]:
                print(f"    - {concept}: {w:.3f}")
    
    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Final concepts: {len(concepts)}")
    print(f"Val Mean AUROC: {val_metrics['mean_auroc']:.4f}")
    print(f"Saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
