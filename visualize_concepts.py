#!/usr/bin/env python3
"""
Generate concept saliency maps for trained CBM models.

This script visualizes which parts of X-ray images activate specific concepts
and how concepts contribute to disease predictions.

Usage examples:

1. Visualize top concepts for VLG-CBM:
   python visualize_concepts.py \
       --model_type vlg_cbm \
       --model_dir checkpoints/vlg_cbm_10k \
       --data_dir /workspace/CheXpert-v1.0-small \
       --concepts concepts/chexpert_concepts.txt \
       --output visualizations/vlg_cbm \
       --num_samples 20

2. Visualize specific concepts for LF-CBM:
   python visualize_concepts.py \
       --model_type lf_cbm \
       --model_dir saved_models/lf_cbm_10k \
       --data_dir /workspace/CheXpert-v1.0-small \
       --concepts concepts/chexpert_concepts.txt \
       --concept_indices 0 5 10 15 \
       --output visualizations/lf_cbm \
       --method gradcam

3. Disease-level attribution:
   python visualize_concepts.py \
       --model_type vlg_cbm \
       --model_dir checkpoints/vlg_cbm_10k \
       --data_dir /workspace/CheXpert-v1.0-small \
       --disease_attribution \
       --disease_idx 0 \
       --output visualizations/disease_attr
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    CheXpertDataset,
    CovidQUDataset,
    CHEXPERT_PATHOLOGY_LABELS,
    CHEXPERT_COMPETITION_LABELS,
    COVIDQU_LABELS,
    get_transforms
)
from models import get_model
from utils.saliency import (
    GradCAM,
    IntegratedGradients,
    SmoothGrad,
    ConceptAttributionMap,
    save_saliency_visualization,
    visualize_heatmap
)
from vlg_cbm_lib.datasets import ConceptLayer, BackboneWithConcepts


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize concept saliency maps")
    
    # Model
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["vlg_cbm", "lf_cbm"],
                        help="Type of CBM model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained model")
    parser.add_argument("--backbone", type=str, default="densenet121",
                        help="Backbone architecture")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--label_set", type=str, default="chexpert",
                        choices=["chexpert", "covidqu"])
    parser.add_argument("--covidqu_variant", type=str, default="infection",
                        choices=["infection", "lung"])
    parser.add_argument("--pathology_labels", action="store_true", default=True,
                        help="Use pathology labels subset")
    parser.add_argument("--split", type=str, default="valid",
                        choices=["train", "valid", "test"])
    
    # Concepts
    parser.add_argument("--concepts", type=str, required=True,
                        help="Path to concepts file (.txt or .json)")
    parser.add_argument("--concept_indices", type=int, nargs="+",
                        help="Specific concept indices to visualize")
    parser.add_argument("--top_k_concepts", type=int, default=10,
                        help="Number of top concepts to visualize per disease")
    
    # Visualization
    parser.add_argument("--method", type=str, default="gradcam",
                        choices=["gradcam", "integrated_gradients", "smoothgrad"],
                        help="Saliency method")
    parser.add_argument("--disease_attribution", action="store_true",
                        help="Generate disease-level attribution via concepts")
    parser.add_argument("--disease_idx", type=int, default=0,
                        help="Disease index for attribution (if disease_attribution=True)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of images to visualize")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    
    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


def load_concepts(path: str) -> list:
    """Load concepts from text or json file."""
    if path.endswith('.json'):
        with open(path) as f:
            data = json.load(f)
        if "concepts" in data:
            # Flatten per-class concepts
            all_concepts = []
            for concepts in data["concepts"].values():
                all_concepts.extend(concepts)
            return list(dict.fromkeys(all_concepts))  # Remove duplicates, preserve order
        else:
            return data
    else:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]


def load_vlg_cbm_model(args, device):
    """Load VLG-CBM model components."""
    # Load model config
    config_path = os.path.join(args.model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    feature_dim = config.get("feature_dim", 1024)
    
    # Get number of concepts - try multiple sources in order of reliability
    n_concepts = None
    
    # 1. Try to load from model directory's concepts.txt (most reliable)
    model_concepts_path = os.path.join(args.model_dir, "concepts.txt")
    if os.path.exists(model_concepts_path):
        with open(model_concepts_path) as f:
            n_concepts = sum(1 for line in f if line.strip())
        print(f"Loaded {n_concepts} concepts from {model_concepts_path}")
    
    # 2. Try to infer from W_c.pt shape
    if n_concepts is None:
        W_path = os.path.join(args.model_dir, "W_c.pt")
        if os.path.exists(W_path):
            W_temp = torch.load(W_path, map_location='cpu', weights_only=False)
            if isinstance(W_temp, torch.Tensor):
                n_concepts = W_temp.shape[1]  # (n_classes, n_concepts)
                print(f"Inferred {n_concepts} concepts from W_c.pt shape: {W_temp.shape}")
    
    # 3. Try config file keys
    if n_concepts is None:
        for key in ["n_concepts", "num_concepts_initial", "num_concepts"]:
            if key in config:
                n_concepts = config[key]
                print(f"Using {n_concepts} concepts from config['{key}']")
                break
    
    # 4. Fallback: load from input concept file
    if n_concepts is None:
        print(f"Warning: Could not determine concept count from model, loading from {args.concepts}")
        concept_names = load_concepts(args.concepts)
        n_concepts = len(concept_names)
        print(f"Using {n_concepts} concepts from input file")
    
    
    # Get labels - default to 5 competition labels unless pathology_labels is set
    if args.label_set == "covidqu":
        labels = COVIDQU_LABELS
    elif args.pathology_labels:
        labels = CHEXPERT_PATHOLOGY_LABELS  # 12 classes
    else:
        labels = CHEXPERT_COMPETITION_LABELS  # 5 classes (default)
    
    # Build backbone (following evaluate_nec.py pattern)
    use_xrv = args.backbone in ["xrv-all", "xrv-chex", "xrv-nih"]
    backbone_kwargs = {}
    if use_xrv:
        backbone_kwargs["target_labels"] = labels
    
    backbone_model = get_model(
        args.backbone,
        num_classes=len(labels),
        pretrained=True,
        **backbone_kwargs
    )
    
    # Load backbone weights if available
    backbone_path = os.path.join(args.model_dir, "backbone.pth")
    if os.path.exists(backbone_path):
        state = torch.load(backbone_path, map_location=device, weights_only=False)
        # Filter out classifier weights if there's a size mismatch
        filtered_state = {k: v for k, v in state.items() 
                         if not k.startswith('classifier') and not k.startswith('backbone.classifier')}
        if hasattr(backbone_model, "backbone"):
            backbone_model.backbone.load_state_dict(filtered_state, strict=False)
        else:
            backbone_model.load_state_dict(filtered_state, strict=False)
    elif config.get("backbone_ckpt"):
        ckpt = torch.load(config["backbone_ckpt"], map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        # Filter out classifier weights if there's a size mismatch
        filtered_state = {k: v for k, v in state.items() 
                         if not k.startswith('classifier') and not k.startswith('backbone.classifier')}
        backbone_model.load_state_dict(filtered_state, strict=False)
    
    # Extract backbone features module
    if args.backbone == "densenet121":
        feature_dim = 1024
        backbone = backbone_model.backbone.features
    elif args.backbone == "resnet50":
        feature_dim = 2048
        backbone = nn.Sequential(*list(backbone_model.backbone.children())[:-1])
    else:
        feature_dim = getattr(backbone_model, "feature_dim", 1024)
        
        class XRVBackbone(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                return self.model.get_features(x)
        
        backbone = XRVBackbone(backbone_model)
    
    # Load concept layer
    concept_layer = ConceptLayer(
        input_dim=feature_dim,
        n_concepts=n_concepts,
        num_hidden=config.get("cbl_hidden_layers", 1),
        hidden_dim=config.get("hidden_dim")
    )
    
    concept_layer_path = os.path.join(args.model_dir, "concept_layer.pt")
    if os.path.exists(concept_layer_path):
        concept_layer.load_state_dict(
            torch.load(concept_layer_path, map_location=device, weights_only=False)
        )
    else:
        raise FileNotFoundError(f"Could not find concept_layer.pt in {args.model_dir}")
    
    # Create combined model
    model = BackboneWithConcepts(backbone, concept_layer)
    
    # Load final layer weights (W_g.pt instead of W_c.pt)
    W_path = os.path.join(args.model_dir, "W_g.pt")
    if not os.path.exists(W_path):
        W_path = os.path.join(args.model_dir, "W_c.pt")  # Fallback
    
    W = torch.load(W_path, map_location=device, weights_only=False)
    
    return model, W


def load_lf_cbm_model(args, device):
    """Load Label-Free CBM model components."""
    # Get number of concepts - try multiple sources
    n_concepts = None
    
    # 1. Try to load from model directory's concepts.txt (most reliable)
    model_concepts_path = os.path.join(args.model_dir, "concepts.txt")
    if os.path.exists(model_concepts_path):
        with open(model_concepts_path) as f:
            n_concepts = sum(1 for line in f if line.strip())
        print(f"Loaded {n_concepts} concepts from {model_concepts_path}")
    
    # 2. Try to infer from W_c.pt shape
    if n_concepts is None:
        W_path = os.path.join(args.model_dir, "W_c.pt")
        if os.path.exists(W_path):
            W_temp = torch.load(W_path, map_location='cpu', weights_only=False)
            if isinstance(W_temp, torch.Tensor):
                n_concepts = W_temp.shape[1]  # (n_classes, n_concepts)
                print(f"Inferred {n_concepts} concepts from W_c.pt shape: {W_temp.shape}")
    
    # 3. Try to infer from concept_proj.pt
    if n_concepts is None:
        proj_path = os.path.join(args.model_dir, "concept_proj.pt")
        if os.path.exists(proj_path):
            proj_state = torch.load(proj_path, map_location='cpu', weights_only=False)
            if "n_concepts" in proj_state:
                n_concepts = proj_state["n_concepts"]
                print(f"Using {n_concepts} concepts from concept_proj.pt")
            else:
                # Infer from state_dict shape
                state_dict = proj_state.get("state_dict", proj_state)
                if "weight" in state_dict:
                    n_concepts = state_dict["weight"].shape[0]
                    print(f"Inferred {n_concepts} concepts from projection layer shape")
    
    # 4. Fallback: load from input concept file
    if n_concepts is None:
        print(f"Warning: Could not determine concept count from model, loading from {args.concepts}")
        concept_names = load_concepts(args.concepts)
        n_concepts = len(concept_names)
        print(f"Using {n_concepts} concepts from input file")
    
    # Load concept projection layer
    proj_path = os.path.join(args.model_dir, "concept_proj.pt")
    proj_state = torch.load(proj_path, map_location=device, weights_only=False)
    
    proj_layer = nn.Linear(1024, n_concepts)  # Assuming DenseNet
    
    # Load state dict properly
    if "state_dict" in proj_state:
        proj_layer.load_state_dict(proj_state["state_dict"])
    else:
        proj_layer.load_state_dict(proj_state)
    
    # Load backbone
    backbone_model = get_model(args.backbone, num_classes=12, pretrained=False)
    if args.backbone == "densenet121":
        backbone = backbone_model.backbone.features
    else:
        backbone = nn.Sequential(*list(backbone_model.backbone.children())[:-1])
    
    # Load final layer
    W_path = os.path.join(args.model_dir, "W_c.pt")
    W = torch.load(W_path, map_location=device, weights_only=False)
    
    # Create combined model
    class LFCBM(nn.Module):
        def __init__(self, backbone, proj_layer):
            super().__init__()
            self.backbone = backbone
            self.proj_layer = proj_layer
        
        def forward(self, x):
            features = self.backbone(x)
            if features.dim() > 2:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            return self.proj_layer(features)
    
    model = LFCBM(backbone, proj_layer)
    
    return model, W


def visualize_concept_saliency(
    model,
    concept_layer,
    dataloader,
    concept_names,
    concept_indices,
    method,
    output_dir,
    device,
    num_samples
):
    """Generate and save concept saliency maps."""
    model.eval()
    if hasattr(model, 'concept_layer'):
        concept_layer = model.concept_layer
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    if method == 'gradcam':
        if hasattr(model, 'backbone'):
            visualizer = GradCAM(model.backbone)
        else:
            visualizer = GradCAM(model)
    elif method == 'integrated_gradients':
        visualizer = IntegratedGradients(model)
    elif method == 'smoothgrad':
        visualizer = SmoothGrad(model)
    
    sample_count = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating saliency maps")):
        if sample_count >= num_samples:
            break
        
        images = batch[0].to(device)
        
        for img_idx in range(images.shape[0]):
            if sample_count >= num_samples:
                break
            
            image = images[img_idx:img_idx+1]
            
            # Get concept activations
            with torch.no_grad():
                if hasattr(model, 'backbone'):
                    features = model.backbone(image)
                    if features.dim() > 2:
                        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
                    concept_scores = concept_layer(features).squeeze().cpu().numpy()
                else:
                    concept_scores = model(image).squeeze().cpu().numpy()
            
            for concept_idx in concept_indices:
                # Generate saliency map
                if method == 'gradcam':
                    heatmap = visualizer.generate_cam(image, concept_layer, concept_idx)
                    heatmap = cv2.resize(heatmap, (image.shape[3], image.shape[2]))
                else:
                    heatmap = visualizer.generate_attribution(image, concept_idx)
                
                concept_name = concept_names[concept_idx] if concept_idx < len(concept_names) else f"Concept_{concept_idx}"
                score = concept_scores[concept_idx]
                
                output_path = os.path.join(
                    output_dir,
                    f"sample{sample_count}_concept{concept_idx}_{method}.png"
                )
                
                save_saliency_visualization(
                    image,
                    heatmap,
                    output_path,
                    title=f"Sample {sample_count} - {method.upper()}",
                    concept_name=concept_name,
                    score=float(score)
                )
            
            sample_count += 1


def visualize_disease_attribution(
    model,
    concept_layer,
    W,
    dataloader,
    concept_names,
    disease_names,
    disease_idx,
    top_k,
    output_dir,
    device,
    num_samples
):
    """Generate disease-level attribution via concepts."""
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(model, 'backbone'):
        backbone = model.backbone
    else:
        backbone = model
    
    attr_viz = ConceptAttributionMap(backbone, concept_layer, W)
    
    sample_count = 0
    
    for batch in tqdm(dataloader, desc="Generating disease attributions"):
        if sample_count >= num_samples:
            break
        
        images = batch[0].to(device)
        
        for img_idx in range(images.shape[0]):
            if sample_count >= num_samples:
                break
            
            image = images[img_idx:img_idx+1]
            
            # Generate attribution
            top_concepts, weights, heatmap = attr_viz.generate_disease_attribution(
                image, disease_idx, top_k
            )
            
            # Create comprehensive visualization
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(16, 6))
            
            # Original image
            ax1 = plt.subplot(1, 3, 1)
            img_np = image.squeeze().cpu().numpy()
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            img_display = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            ax1.imshow(img_display, cmap='gray' if img_np.ndim == 2 else None)
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            # Heatmap overlay
            ax2 = plt.subplot(1, 3, 2)
            overlay = visualize_heatmap(image, heatmap, alpha=0.5)
            ax2.imshow(overlay)
            ax2.set_title(f"Attribution for: {disease_names[disease_idx]}")
            ax2.axis('off')
            
            # Top concepts bar chart
            ax3 = plt.subplot(1, 3, 3)
            concept_labels = [concept_names[i][:40] + '...' if len(concept_names[i]) > 40 
                            else concept_names[i] for i in top_concepts]
            colors = ['red' if w < 0 else 'green' for w in weights]
            ax3.barh(range(len(weights)), weights, color=colors, alpha=0.7)
            ax3.set_yticks(range(len(weights)))
            ax3.set_yticklabels(concept_labels, fontsize=8)
            ax3.set_xlabel("Concept Weight")
            ax3.set_title(f"Top {top_k} Contributing Concepts")
            ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax3.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"sample{sample_count}_disease{disease_idx}_attribution.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            sample_count += 1
    
    print(f"\nSaved {sample_count} disease attribution visualizations to {output_dir}")


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load concepts
    print(f"Loading concepts from {args.concepts}")
    concept_names = load_concepts(args.concepts)
    print(f"Loaded {len(concept_names)} concepts")
    
    # Load model
    print(f"Loading {args.model_type} model from {args.model_dir}")
    if args.model_type == "vlg_cbm":
        model, W = load_vlg_cbm_model(args, device)
        concept_layer = model.concept_layer
    else:  # lf_cbm
        model, W = load_lf_cbm_model(args, device)
        concept_layer = model.proj_layer if hasattr(model, 'proj_layer') else None
    
    model.to(device)
    model.eval()
    
    # Get labels
    if args.label_set == "covidqu":
        labels = COVIDQU_LABELS
    elif args.pathology_labels:
        labels = CHEXPERT_PATHOLOGY_LABELS
    else:
        labels = CHEXPERT_COMPETITION_LABELS
    
    print(f"Using {len(labels)} disease labels: {labels}")
    
    # Create dataset
    transform = get_transforms(train=False, img_size=224)
    
    if args.label_set == "covidqu":
        dataset = CovidQUDataset(
            root=args.data_dir,
            split=args.split.capitalize(),
            transform=transform,
            variant=args.covidqu_variant
        )
    else:
        csv_path = os.path.join(args.data_dir, f"{args.split}.csv")
        dataset = CheXpertDataset(
            csv_path=csv_path,
            transform=transform,
            uncertain_strategy="ones",
            use_frontal_only=True,
            label_subset=labels
        )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(dataset)} samples from {args.split} split")
    
    # Determine which concepts to visualize
    if args.concept_indices:
        concept_indices = args.concept_indices
    else:
        # Use top concepts by average activation
        print("Computing concept activations to select top concepts...")
        all_activations = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing activations", total=min(100, len(dataloader))):
                if len(all_activations) >= 100:
                    break
                images = batch[0].to(device)
                activations = model(images)
                all_activations.append(torch.sigmoid(activations).cpu())
        
        all_activations = torch.cat(all_activations, dim=0)
        mean_activations = all_activations.mean(dim=0)
        concept_indices = torch.topk(mean_activations, min(20, len(concept_names))).indices.tolist()
        print(f"Selected top {len(concept_indices)} concepts by activation")
    
    print(f"Visualizing concepts: {concept_indices}")
    
    # Generate visualizations
    if args.disease_attribution:
        print(f"\nGenerating disease attribution for: {labels[args.disease_idx]}")
        visualize_disease_attribution(
            model,
            concept_layer,
            W,
            dataloader,
            concept_names,
            labels,
            args.disease_idx,
            args.top_k_concepts,
            args.output,
            device,
            args.num_samples
        )
    else:
        print(f"\nGenerating concept saliency maps using {args.method}")
        visualize_concept_saliency(
            model,
            concept_layer,
            dataloader,
            concept_names,
            concept_indices,
            args.method,
            args.output,
            device,
            args.num_samples
        )
    
    print(f"\nDone! Visualizations saved to {args.output}")


if __name__ == "__main__":
    main()
