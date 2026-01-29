#!/usr/bin/env python3
"""
Spatially-Aware Label-Free CBM (SALF-CBM)

Implements the spatial concept bottleneck idea from
"Spatially-Aware Label-Free Concept Bottleneck Models" (arXiv:2502.20134).

Pipeline (simplified):
1) CLIP text encoder -> concept embeddings.
2) For each image, draw red-circle prompts on a uniform grid to get local CLIP image
   embeddings and build a spatial image-concept similarity tensor P (H_grid x W_grid x M).
3) Train a 1x1 conv concept bottleneck on backbone feature maps to predict concept maps,
   using cubic cosine similarity to align with P.
4) Spatially pool concept maps to global concept activations, train a sparse (or dense)
   final classifier.

This mirrors label_free_cbm.py but keeps spatial structure in the bottleneck.
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from dataset import (
    CheXpertDataset,
    CHEXPERT_PATHOLOGY_LABELS,
    CHEXPERT_COMPETITION_LABELS,
    COVIDQU_LABELS,
    CovidQUDataset,
    get_transforms,
)
from models import get_model
from utils.metrics import compute_all_metrics

# CLIP encoders (reuse helpers from label_free_cbm style)
try:
    from open_clip import create_model_from_pretrained, get_tokenizer, tokenize as oc_tokenize
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: open_clip not available. Install with: pip install open_clip_torch")

try:
    from transformers import AutoTokenizer, SiglipModel, SiglipProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. MedSigLIP encoder will be disabled.")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Spatially-Aware Label-Free CBM")
    # Data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--concepts", type=str, default="concepts/chexpert_concepts.txt")
    p.add_argument("--label_set", type=str, default="chexpert", choices=["chexpert", "covidqu"])
    p.add_argument("--covidqu_variant", type=str, default="infection", choices=["infection", "lung"])
    p.add_argument("--competition_labels", action="store_true")
    p.add_argument("--uncertain_strategy", type=str, default="ones",
                   choices=["ones", "zeros", "ignore"])
    p.add_argument("--limit_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    # Models
    p.add_argument("--backbone", type=str, default="densenet121", choices=["densenet121", "resnet50"])
    p.add_argument("--backbone_ckpt", type=str, default=None)
    p.add_argument("--pretrained", action="store_true", default=True,
                   help="Use pretrained backbone weights (default True)")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                   help="Train backbone from scratch")
    p.add_argument("--clip_name", type=str, default="biomedclip",
                   choices=["biomedclip", "medsiglip"])

    # Grid / prompts
    p.add_argument("--grid_h", type=int, default=7, help="Grid cells vertically")
    p.add_argument("--grid_w", type=int, default=7, help="Grid cells horizontally")
    p.add_argument("--prompt_radius", type=int, default=6, help="Red circle radius in pixels after resize")

    # Training
    p.add_argument("--epochs", type=int, default=5, help="CBL epochs")
    p.add_argument("--batch_size", type=int, default=8, help="Backbone/CBL batch size")
    p.add_argument("--prompt_batch_size", type=int, default=64, help="Prompted image batch size for CLIP sims")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=0.0007, help="Sparsity reg for final layer")
    p.add_argument("--saga_iters", type=int, default=1000)
    p.add_argument("--saga_batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--img_size", type=int, default=224)

    # Caching
    p.add_argument("--activation_dir", type=str, default="saved_activations")
    p.add_argument("--recompute", action="store_true", help="Force recompute CLIP spatial sims")

    # Output
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_concepts(path: str):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class BiomedCLIP:
    def __init__(self, device):
        if not CLIP_AVAILABLE:
            raise RuntimeError("open_clip not installed")
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")
        self.device = device
        self.model, self.preprocess = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def encode_texts(self, texts):
        # BiomedCLIP uses a HF text tower, so use the matching HF tokenizer
        if isinstance(texts, str):
            texts = [texts]
        ctx_len = getattr(self.model, "context_length", None)
        tok = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=ctx_len,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        return self.model.encode_text(tok)

    @torch.no_grad()
    def encode_images(self, pil_batch):
        tensors = torch.stack([self.preprocess(img) for img in pil_batch]).to(self.device)
        feats = self.model.encode_image(tensors)
        return feats


class MedSigLIP:
    MODEL_ID = "google/medsiglip-448"

    def __init__(self, device):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")
        self.device = device
        self.processor = SiglipProcessor.from_pretrained(self.MODEL_ID)
        self.model = SiglipModel.from_pretrained(self.MODEL_ID).to(device).eval()

    @torch.no_grad()
    def encode_texts(self, texts):
        inputs = self.processor(text=texts, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        return feats

    @torch.no_grad()
    def encode_images(self, pil_batch):
        inputs = self.processor(images=pil_batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        feats = self.model.get_image_features(pixel_values=pixel_values)
        return feats


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Undo ImageNet norm and convert to PIL."""
    t = image_tensor.unsqueeze(0)
    t = t * IMAGENET_STD.to(t.device) + IMAGENET_MEAN.to(t.device)
    t = t.clamp(0, 1)[0]
    return TF.to_pil_image(t.cpu())


def draw_prompt(pil_img: Image.Image, center: Tuple[int, int], radius: int) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="red", width=3)
    return img


def compute_spatial_sims(dataset, clip_enc, concepts, grid_h, grid_w, radius, cache_base,
                         device, data_batch_size, prompt_batch_size, num_workers, force_recompute):
    os.makedirs(os.path.dirname(cache_base), exist_ok=True)
    cache_path = cache_base + "_P.pt"
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached spatial sims from {cache_path}")
        P = torch.load(cache_path)
        return P

    print(f"Computing spatial similarities on grid {grid_h}x{grid_w} with radius {radius}...")
    loader = DataLoader(dataset, batch_size=data_batch_size, shuffle=False, num_workers=num_workers)

    text_emb = clip_enc.encode_texts(concepts).to(device)  # (M, d)
    text_emb = F.normalize(text_emb, dim=1)

    all_P = []
    for batch_imgs, _ in tqdm(loader, desc="CLIP spatial sims"):
        B = batch_imgs.size(0)
        for b in range(B):
            img_tensor = batch_imgs[b]
            pil = tensor_to_pil(img_tensor)
            W, H = pil.size
            xs = np.linspace(radius, W - radius - 1, grid_w).astype(int)
            ys = np.linspace(radius, H - radius - 1, grid_h).astype(int)
            prompts = [draw_prompt(pil, (x, y), radius) for y in ys for x in xs]

            sims_chunks = []
            for i in range(0, len(prompts), prompt_batch_size):
                prompt_batch = prompts[i:i + prompt_batch_size]
                img_emb = clip_enc.encode_images(prompt_batch)  # (Bchunk, d)
                img_emb = F.normalize(img_emb, dim=1)
                sim = img_emb.to(device) @ text_emb.T  # (Bchunk, M)
                sims_chunks.append(sim.cpu())
            sims = torch.cat(sims_chunks, dim=0)  # (grid_h*grid_w, M)
            sim_map = sims.view(grid_h, grid_w, -1)  # (grid_h, grid_w, M)
            all_P.append(sim_map)

    P = torch.stack(all_P, dim=0)  # (N, grid_h, grid_w, M)
    torch.save(P, cache_path)
    print(f"Saved spatial sims to {cache_path}")
    return P


class BackboneSpatial(nn.Module):
    """Return spatial feature map (no global pooling)."""
    def __init__(self, model_name, checkpoint, device, pretrained=True):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = get_model(model_name, num_classes=1, pretrained=pretrained)
        self.backbone = getattr(self.model, "backbone", self.model)

        if checkpoint:
            import inspect
            load_kwargs = {"map_location": self.device}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = False
            state = torch.load(checkpoint, **load_kwargs)
            sd = state.get("model_state_dict", state)
            filtered = {k: v for k, v in sd.items() if "classifier" not in k}
            self.model.load_state_dict(filtered, strict=False)

        if model_name == "densenet121":
            self.feature_dim = 1024
            self.encoder = self.backbone.features
        else:
            self.feature_dim = 2048
            self.encoder = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            )
        self.to(self.device)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device, non_blocking=True)
        feats = self.encoder(x)
        return feats


class ConceptConv(nn.Module):
    """1x1 conv mapping feature maps -> concept maps."""
    def __init__(self, in_channels, n_concepts):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_concepts, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


def cbl_loss(pred_maps, target_maps):
    """
    pred_maps: (B, M, H, W)
    target_maps: (B, H, W, M)
    """
    B, M, H, W = pred_maps.shape
    pred = pred_maps.permute(0, 2, 3, 1).reshape(-1, M)
    tgt = target_maps.reshape(-1, M)
    pred = F.normalize(pred, dim=0)
    tgt = F.normalize(tgt, dim=0)
    sim = (pred * tgt).sum(dim=0)  # per concept
    return -(sim ** 3).mean()


def train_cbl(backbone, concept_layer, train_loader, train_P, val_loader, val_P, device, epochs, lr, grid_h, grid_w, n_concepts):
    optimizer = torch.optim.Adam(concept_layer.parameters(), lr=lr)
    concept_layer.to(device)
    backbone.eval()

    best_state = None
    best_loss = float("inf")

    for epoch in range(epochs):
        running = 0.0
        for i, (images, _) in enumerate(tqdm(train_loader, desc=f"CBL epoch {epoch+1}")):
            images = images.to(device)
            feats = backbone(images)  # B,C,Hf,Wf
            concept_maps = concept_layer(feats)  # B,M,Hf,Wf
            concept_maps = F.interpolate(concept_maps, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
            target = train_P[i * train_loader.batch_size : i * train_loader.batch_size + images.size(0)].to(device)

            loss = cbl_loss(concept_maps, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        epoch_loss = running / len(train_loader.dataset)

        # Validation loss
        with torch.no_grad():
            val_running = 0.0
            for j, (images, _) in enumerate(val_loader):
                images = images.to(device)
                feats = backbone(images)
                concept_maps = concept_layer(feats)
                concept_maps = F.interpolate(concept_maps, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
                target = val_P[j * val_loader.batch_size : j * val_loader.batch_size + images.size(0)].to(device)
                vloss = cbl_loss(concept_maps, target)
                val_running += vloss.item() * images.size(0)
            val_loss = val_running / len(val_loader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in concept_layer.state_dict().items()}
        print(f"[CBL] epoch {epoch+1}: train {epoch_loss:.4f} | val {val_loss:.4f} (best {best_loss:.4f})")

    if best_state is not None:
        concept_layer.load_state_dict(best_state)
    return concept_layer


def extract_global_concepts(concept_layer, backbone, loader, device):
    backbone.eval()
    concept_layer.eval()
    feats_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extract concepts"):
            images = images.to(device)
            feats = backbone(images)
            c_maps = concept_layer(feats)  # B,M,H,W
            global_c = F.adaptive_avg_pool2d(c_maps, 1).squeeze(-1).squeeze(-1)  # B,M
            feats_list.append(global_c.cpu())
            labels_list.append(labels)
    X = torch.cat(feats_list, dim=0)
    Y = torch.cat(labels_list, dim=0)
    return X, Y


def train_classifier(concept_layer, backbone, train_loader, val_loader, num_classes, device, lam, saga_iters, saga_batch, labels, single_label):
    X_train, Y_train = extract_global_concepts(concept_layer, backbone, train_loader, device)
    X_val, Y_val = extract_global_concepts(concept_layer, backbone, val_loader, device)

    # Standardize pooled concepts (match authors): compute train mean/std, apply to train/val
    train_mean = X_train.mean(dim=0, keepdim=True)
    train_std = X_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std

    # Determine single-label
    is_single = single_label
    if is_single and Y_train.ndim > 1:
        Y_train_cls = Y_train.argmax(dim=1)
        Y_val_cls = Y_val.argmax(dim=1)
    else:
        Y_train_cls, Y_val_cls = Y_train, Y_val

    try:
        from glm_saga.elasticnet import glm_saga, IndexedTensorDataset
        USE_SAGA = True
    except ImportError:
        USE_SAGA = False

    if USE_SAGA and is_single:
        ds = IndexedTensorDataset(X_train, Y_train_cls)
        loader_cls = DataLoader(ds, batch_size=saga_batch, shuffle=True)
        linear = nn.Linear(X_train.size(1), num_classes).to(device)
        linear.weight.data.zero_()
        linear.bias.data.zero_()
        # Wrap glm_saga with a progress bar: iterate epochs manually
        history = []
        for epoch in tqdm(range(saga_iters), desc="SAGA classifier"):
            out = glm_saga(
                linear,
                loader_cls,
                max_lr=0.1,
                nepochs=1,
                alpha=0.99,
                epsilon=1.0,
                k=1,
                val_loader=None,
                do_zero=False,
                metadata={'max_reg': {'nongrouped': lam}},
                n_ex=len(X_train),
                n_classes=num_classes,
                family='multiclass',
            )
            history.append(out)
        W = linear.weight.data.cpu()
        b = linear.bias.data.cpu()
        # Metrics
        train_logits = (X_train @ W.T) + b
        val_logits = (X_val @ W.T) + b
        train_pred = torch.softmax(train_logits, dim=1).numpy()
        val_pred = torch.softmax(val_logits, dim=1).numpy()
        train_targets = Y_train_cls.numpy()
        val_targets = Y_val_cls.numpy()
        train_acc = (train_pred.argmax(axis=1) == train_targets).mean()
        val_acc = (val_pred.argmax(axis=1) == val_targets).mean()
        train_metrics = {"accuracy": float(train_acc)}
        val_metrics = {"accuracy": float(val_acc)}
    else:
        linear = nn.Linear(X_train.size(1), num_classes, bias=True).to(device)
        if is_single:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(linear.parameters(), lr=1e-3)
        ds = TensorDataset(X_train, Y_train_cls if is_single else Y_train)
        dl = DataLoader(ds, batch_size=128, shuffle=True)
        for _ in range(50):
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = linear(xb)
                loss = criterion(logits, yb if not is_single else yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        W = linear.weight.data.cpu()
        b = linear.bias.data.cpu()

        with torch.no_grad():
            train_logits = linear(X_train.to(device)).cpu()
            val_logits = linear(X_val.to(device)).cpu()
        if is_single:
            train_pred = torch.softmax(train_logits, dim=1).numpy()
            val_pred = torch.softmax(val_logits, dim=1).numpy()
            train_targets = Y_train_cls.numpy()
            val_targets = Y_val_cls.numpy()
            train_acc = (train_pred.argmax(axis=1) == train_targets).mean()
            val_acc = (val_pred.argmax(axis=1) == val_targets).mean()
            train_metrics = {"accuracy": float(train_acc)}
            val_metrics = {"accuracy": float(val_acc)}
        else:
            train_pred = torch.sigmoid(train_logits).numpy()
            val_pred = torch.sigmoid(val_logits).numpy()
            train_targets = Y_train.numpy()
            val_targets = Y_val.numpy()
            train_metrics = compute_all_metrics(train_targets, train_pred, labels)
            val_metrics = compute_all_metrics(val_targets, val_pred, labels)

    # Save normalization stats for inference
    norm_stats = {"mean": train_mean, "std": train_std}
    return W, b, train_metrics, val_metrics, norm_stats


def main():
    print("---- Starting SALF-CBM run:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    args = parse_args()
    print(args)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # Labels
    if args.label_set == "covidqu":
        labels = COVIDQU_LABELS
        single_label = True
    elif args.competition_labels:
        labels = CHEXPERT_COMPETITION_LABELS
        single_label = False
    else:
        labels = CHEXPERT_PATHOLOGY_LABELS
        single_label = False
    num_classes = len(labels)

    # Concepts
    concepts = load_concepts(args.concepts)
    print(f"Loaded {len(concepts)} concepts")

    # Data
    transform = get_transforms(args.img_size, is_training=False)
    if args.label_set == "covidqu":
        train_ds = CovidQUDataset(root=args.data_dir, split="Train", transform=transform, variant=args.covidqu_variant)
        val_ds = CovidQUDataset(root=args.data_dir, split="Val", transform=transform, variant=args.covidqu_variant)
    else:
        train_csv = os.path.join(args.data_dir, "train.csv")
        val_csv = os.path.join(args.data_dir, "valid.csv")
        img_root = os.path.dirname(args.data_dir)
        train_ds = CheXpertDataset(train_csv, img_root, transform=transform, labels=labels,
                                   uncertain_strategy=args.uncertain_strategy, frontal_only=True)
        val_ds = CheXpertDataset(val_csv, img_root, transform=transform, labels=labels,
                                 uncertain_strategy=args.uncertain_strategy, frontal_only=True)
        if args.limit_samples is not None and args.limit_samples < len(train_ds):
            idx = np.random.RandomState(args.seed).permutation(len(train_ds))[:args.limit_samples]
            train_ds = torch.utils.data.Subset(train_ds, idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # CLIP enc + spatial sims
    if args.clip_name == "biomedclip":
        clip_enc = BiomedCLIP(device)
    else:
        clip_enc = MedSigLIP(device)
    cache_tag = f"{Path(args.output).stem}_gh{args.grid_h}_gw{args.grid_w}_r{args.prompt_radius}"
    cache_base = os.path.join(args.activation_dir, cache_tag)
    P_train = compute_spatial_sims(
        train_ds, clip_enc, concepts,
        args.grid_h, args.grid_w, args.prompt_radius,
        cache_base + "_train", device,
        data_batch_size=args.batch_size,
        prompt_batch_size=args.prompt_batch_size,
        num_workers=args.num_workers,
        force_recompute=args.recompute
    )
    P_val = compute_spatial_sims(
        val_ds, clip_enc, concepts,
        args.grid_h, args.grid_w, args.prompt_radius,
        cache_base + "_val", device,
        data_batch_size=args.batch_size,
        prompt_batch_size=args.prompt_batch_size,
        num_workers=args.num_workers,
        force_recompute=args.recompute
    )

    # Backbone + concept layer
    backbone = BackboneSpatial(args.backbone, args.backbone_ckpt, device, pretrained=args.pretrained)
    concept_layer = ConceptConv(backbone.feature_dim, len(concepts))

    # Train CBL
    concept_layer = train_cbl(backbone, concept_layer, train_loader, P_train, val_loader, P_val, device,
                              args.epochs, args.lr, args.grid_h, args.grid_w, len(concepts))

    # Train classifier + eval
    W, b, train_metrics, val_metrics, norm_stats = train_classifier(
        concept_layer, backbone, train_loader, val_loader,
        num_classes, device, args.lam, args.saga_iters, args.saga_batch_size,
        labels, single_label
    )

    # Save artifacts
    torch.save(concept_layer.state_dict(), os.path.join(args.output, "concept_layer.pt"))
    torch.save(W, os.path.join(args.output, "W_g.pt"))
    torch.save(b, os.path.join(args.output, "b_g.pt"))
    torch.save(norm_stats["mean"], os.path.join(args.output, "concept_mean.pt"))
    torch.save(norm_stats["std"], os.path.join(args.output, "concept_std.pt"))
    with open(os.path.join(args.output, "train_metrics.json"), "w") as f:
        json.dump(train_metrics, f, indent=2)
    with open(os.path.join(args.output, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(args.output, "concepts.txt"), "w") as f:
        f.write("\n".join(concepts))
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Done. Artifacts saved to", args.output)


if __name__ == "__main__":
    main()
