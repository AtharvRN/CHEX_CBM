#!/usr/bin/env python3
"""
Evaluation script for Spatially-Aware Label-Free CBM (SALF-CBM).

Loads:
- backbone (as in training)
- concept_layer.pt (1x1 conv)
- W_g.pt / b_g.pt final classifier

Runs inference on a split and reports metrics.
"""

import argparse
import json
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    CheXpertDataset,
    CHEXPERT_COMPETITION_LABELS,
    CHEXPERT_PATHOLOGY_LABELS,
    COVIDQU_LABELS,
    CovidQUDataset,
    get_transforms,
)
from models import get_model
from utils.metrics import compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SALF-CBM")
    p.add_argument("--model_dir", type=str, required=True, help="Path with concept_layer.pt, W_g.pt, b_g.pt, config.json")
    p.add_argument("--data_dir", type=str, required=True, help="Dataset root (CSV dir for CheXpert, folder root for COVID-QU)")
    p.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--nec_levels", type=int, nargs="+", default=None,
                   help="Optional list of NEC values to sweep (truncates final weights by top-|W| per class)")
    return p.parse_args()


class ConceptConv(nn.Module):
    def __init__(self, in_channels, n_concepts):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_concepts, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class BackboneSpatial(nn.Module):
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
        return self.encoder(x.to(self.device, non_blocking=True))


def get_labels(args, config):
    if config.get("label_set", args.split) == "covidqu":
        return COVIDQU_LABELS, True
    if config.get("competition_labels", False):
        return CHEXPERT_COMPETITION_LABELS, False
    # default pathology set (12)
    return CHEXPERT_PATHOLOGY_LABELS, False


def truncate_weights(weight: torch.Tensor, nec: int):
    if nec >= weight.size(1):
        return weight
    contrib = torch.sum(torch.abs(weight), dim=0)
    topk = torch.topk(contrib, nec, largest=True).indices
    mask = torch.zeros_like(contrib, dtype=torch.bool)
    mask[topk] = True
    truncated = weight.clone()
    truncated[:, ~mask] = 0.0
    return truncated


def build_dataset(args, labels, single_label):
    transform = get_transforms(args.img_size, is_training=False)
    if single_label:  # covidqu
        split_name = args.split.capitalize() if args.split != "train" else "Train"
        ds = CovidQUDataset(root=args.data_dir, split=split_name, transform=transform, variant="infection")
    else:
        csv_path = os.path.join(args.data_dir, f"{args.split}.csv")
        img_root = os.path.dirname(args.data_dir)
        ds = CheXpertDataset(csv_path, img_root, transform=transform, labels=labels, uncertain_strategy="ones", frontal_only=True)
    return ds


def evaluate(model_dir, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    labels, single_label = get_labels(args, config)
    num_classes = len(labels)

    # Backbone + concept layer
    backbone = BackboneSpatial(config.get("backbone", "densenet121"),
                               config.get("backbone_ckpt"),
                               device,
                               pretrained=config.get("pretrained", True))
    n_concepts = sum(1 for _ in open(os.path.join(model_dir, "concepts.txt")))
    concept_layer = ConceptConv(backbone.feature_dim, n_concepts)
    concept_layer.load_state_dict(torch.load(os.path.join(model_dir, "concept_layer.pt"), map_location=device))
    concept_layer.to(device).eval()

    # Final classifier + normalization
    W = torch.load(os.path.join(model_dir, "W_g.pt"), map_location=device)
    b = torch.load(os.path.join(model_dir, "b_g.pt"), map_location=device)
    mean_path = os.path.join(model_dir, "concept_mean.pt")
    std_path = os.path.join(model_dir, "concept_std.pt")
    concept_mean = torch.load(mean_path, map_location=device) if os.path.exists(mean_path) else None
    concept_std = torch.load(std_path, map_location=device) if os.path.exists(std_path) else None

    # Data
    ds = build_dataset(args, labels, single_label)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            feats = backbone(images)
            c_maps = concept_layer(feats)
            global_c = F.adaptive_avg_pool2d(c_maps, 1).squeeze(-1).squeeze(-1)  # B,M
            if concept_mean is not None and concept_std is not None:
                global_c = (global_c - concept_mean) / (concept_std + 1e-6)
            logits = global_c @ W.T + b
            all_logits.append(logits.cpu())
            all_targets.append(targets)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    if single_label:
        probs = torch.softmax(logits, dim=1).numpy()
        tgt = targets.argmax(dim=1).numpy() if targets.ndim > 1 else targets.numpy()
        acc = (probs.argmax(axis=1) == tgt).mean()
        metrics = {"accuracy": float(acc)}
    else:
        probs = torch.sigmoid(logits).numpy()
        tgt = targets.numpy()
        metrics = compute_all_metrics(tgt, probs, labels)

    # Save
    out_dir = os.path.join(model_dir, f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "predictions.npy"), probs)
    np.save(os.path.join(out_dir, "targets.npy"), tgt)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {out_dir}/metrics.json")
    if single_label:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    else:
        print(f"Mean AUROC: {metrics['auroc']['mean']:.4f} | Mean AP: {metrics['ap']['mean']:.4f}")

    # NEC sweep if requested
    if args.nec_levels:
        results = []
        for nec in args.nec_levels:
            W_trunc = truncate_weights(W, nec)
            logits_nec = concepts_norm @ W_trunc.T + b  # b unchanged
            if single_label:
                probs_nec = torch.softmax(logits_nec, dim=1).numpy()
                acc = (probs_nec.argmax(axis=1) == tgt).mean()
                results.append({"nec": nec, "accuracy": float(acc)})
                print(f"NEC={nec}: Accuracy={acc:.4f}")
            else:
                probs_nec = torch.sigmoid(logits_nec).numpy()
                metrics_nec = compute_all_metrics(tgt, probs_nec, labels)
                results.append({"nec": nec,
                                "auroc": metrics_nec['auroc']['mean'],
                                "ap": metrics_nec['ap']['mean']})
                print(f"NEC={nec}: Mean AUROC={metrics_nec['auroc']['mean']:.4f}, Mean AP={metrics_nec['ap']['mean']:.4f}")
        # save NEC metrics
        nec_path = os.path.join(out_dir, "nec_metrics.csv")
        with open(nec_path, "w") as f:
            if single_label:
                f.write("nec,accuracy\n")
                for r in results:
                    f.write(f"{r['nec']},{r['accuracy']:.6f}\n")
            else:
                f.write("nec,mean_auroc,mean_ap\n")
                for r in results:
                    f.write(f"{r['nec']},{r['auroc']:.6f},{r['ap']:.6f}\n")
        print(f"Saved NEC metrics to {nec_path}")


def main():
    args = parse_args()
    evaluate(args.model_dir, args)


if __name__ == "__main__":
    main()
