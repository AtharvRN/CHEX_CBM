#!/usr/bin/env python3
"""
Evaluate a trained VLG-CBM model under different Number of Effective Concepts (NEC).

Usage example:
    python evaluate_nec.py \
        --model_dir saved_models/vlg_cbm_10k \
        --nec_levels 5 10 15 20 \
        --batch_size 256

The script loads the saved concept layer, backbone weights, and final layer,
computes concept activations on the validation split, truncates the final layer
weights to keep only the desired number of active concepts, and reports
multi-label AUROC/AP for each NEC value.
"""
import argparse
import json
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CheXpertDataset, get_transforms
from models import get_model, XRV_WEIGHTS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLG-CBM under NEC budgets")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained CBM artifacts")
    parser.add_argument("--nec_levels", type=int, nargs="+",
                        default=[5, 10, 15, 20],
                        help="Number of effective concepts to evaluate")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for concept extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for evaluation")
    parser.add_argument("--split", type=str, default="valid",
                        choices=["valid", "test"],
                        help="Dataset split to evaluate (valid or test)")
    parser.add_argument("--split_csv", type=str, default=None,
                        help="Optional CSV path to override the split default")
    return parser.parse_args()


def build_backbone(config, labels, device):
    use_xrv = config["backbone"] in XRV_WEIGHTS
    backbone_kwargs = {}
    if use_xrv:
        backbone_kwargs["target_labels"] = labels
    model = get_model(
        config["backbone"],
        num_classes=len(labels),
        pretrained=True,
        dropout=config.get("dropout", 0.0),
        **backbone_kwargs,
    )

    # Load fine-tuned backbone if available
    backbone_path = os.path.join(config["output"], "backbone.pth")
    if os.path.exists(backbone_path):
        state = torch.load(backbone_path, map_location=device)
        model_state = getattr(model, "backbone", model)
        model_state.load_state_dict(state)
    elif config.get("backbone_ckpt"):
        import inspect

        load_kwargs = {"map_location": device}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False
        ckpt = torch.load(config["backbone_ckpt"], **load_kwargs)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    if config["backbone"] == "densenet121":
        feature_dim = 1024
        backbone = model.backbone.features
    elif config["backbone"] == "resnet50":
        feature_dim = 2048
        backbone = torch.nn.Sequential(*list(model.backbone.children())[:-1])
    else:
        feature_dim = getattr(model, "feature_dim", 1024)

        class XRVDenseNetBackbone(torch.nn.Module):
            def __init__(self, wrapper):
                super().__init__()
                self.wrapper = wrapper

            def forward(self, x):
                return self.wrapper.get_features(x)

        backbone = XRVDenseNetBackbone(model)

    return backbone, feature_dim


class ConceptLayer(torch.nn.Module):
    def __init__(self, input_dim, n_concepts, num_hidden=1, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, n_concepts))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.layers(x)


class LinearConceptLayer(torch.nn.Module):
    """Simple linear projection used by LF-CBM."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        in_dim = weight.shape[1]
        out_dim = weight.shape[0]
        linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        linear.weight.data.copy_(weight)
        self.linear = linear

    def forward(self, x):
        if x.dim() > 2:
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.linear(x)


def get_dataset(config, labels, split="valid", split_csv=None):
    data_dir = config["data_dir"]
    if split_csv:
        csv_path = split_csv
    else:
        csv_path = os.path.join(data_dir, f"{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV for split '{split}' not found at {csv_path}")
    img_root = os.path.dirname(data_dir)
    transform = get_transforms(config.get("img_size", 224), is_training=False)
    dataset = CheXpertDataset(
        csv_path=csv_path,
        img_root=img_root,
        transform=transform,
        labels=labels,
        uncertain_strategy=config.get("uncertain_strategy", "ones"),
        frontal_only=config.get("frontal_only", True)
    )
    return dataset


def extract_concepts(backbone, concept_layer, loader, device):
    backbone = backbone.to(device)
    concept_layer = concept_layer.to(device)
    backbone.eval()
    concept_layer.eval()
    all_concepts = []
    targets = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting concepts"):
            images = images.to(device)
            feats = backbone(images)
            logits = concept_layer(feats)
            all_concepts.append(logits.cpu())
            targets.append(labels)
    concepts = torch.cat(all_concepts, dim=0)
    labels = torch.cat(targets, dim=0)
    return concepts, labels


def compute_metrics(targets: np.ndarray, preds: np.ndarray, labels: List[str]):
    from sklearn.metrics import average_precision_score, roc_auc_score

    metrics = {"auroc": {}, "ap": {}}
    valid_aurocs = []
    valid_aps = []
    for idx, label in enumerate(labels):
        unique = np.unique(targets[:, idx])
        if len(unique) < 2:
            metrics["auroc"][label] = float("nan")
            metrics["ap"][label] = float("nan")
            continue
        auc = roc_auc_score(targets[:, idx], preds[:, idx])
        ap = average_precision_score(targets[:, idx], preds[:, idx])
        metrics["auroc"][label] = auc
        metrics["ap"][label] = ap
        valid_aurocs.append(auc)
        valid_aps.append(ap)
    metrics["auroc"]["mean"] = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    metrics["ap"]["mean"] = float(np.mean(valid_aps)) if valid_aps else float("nan")
    return metrics


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


def main():
    args = parse_args()
    model_dir = args.model_dir
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
        config["output"] = model_dir

    with open(os.path.join(model_dir, "concepts.txt"), "r") as f:
        concepts = [line.strip() for line in f.readlines() if line.strip()]
    labels = config["labels"]

    dataset = get_dataset(
        config,
        labels,
        split=args.split,
        split_csv=args.split_csv
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=True,
    )

    backbone, feature_dim = build_backbone(config, labels, device)
    concept_layer_path = os.path.join(model_dir, "concept_layer.pt")
    if os.path.exists(concept_layer_path):
        concept_layer = ConceptLayer(
            feature_dim,
            len(concepts),
            num_hidden=config.get("cbl_hidden_layers", 1)
        )
        concept_layer.load_state_dict(
            torch.load(concept_layer_path, map_location=device)
        )
    else:
        w_c_path = os.path.join(model_dir, "W_c.pt")
        if not os.path.exists(w_c_path):
            raise FileNotFoundError(
                "Could not find concept_layer.pt or W_c.pt in model_dir"
            )
        weight = torch.load(w_c_path, map_location=device)
        concept_layer = LinearConceptLayer(weight)

    raw_concepts, targets = extract_concepts(backbone, concept_layer, loader, device)

    mean_path = os.path.join(model_dir, "concept_mean.pt")
    std_path = os.path.join(model_dir, "concept_std.pt")
    if not os.path.exists(mean_path):
        mean_path = os.path.join(model_dir, "proj_mean.pt")
    if not os.path.exists(std_path):
        std_path = os.path.join(model_dir, "proj_std.pt")
    mean = torch.load(mean_path)
    std = torch.load(std_path)
    concepts_norm = (raw_concepts - mean) / torch.clamp(std, min=1e-6)

    W_g = torch.load(os.path.join(model_dir, "W_g.pt"))
    b_g = torch.load(os.path.join(model_dir, "b_g.pt"))

    concept_np = concepts_norm.numpy()
    target_np = targets.numpy()

    results = []
    for nec in args.nec_levels:
        truncated = truncate_weights(W_g, nec)
        logits = np.matmul(concept_np, truncated.t().numpy()) + b_g.numpy()
        probs = 1 / (1 + np.exp(-logits))
        metrics = compute_metrics(target_np, probs, labels)
        auc = metrics["auroc"]["mean"]
        ap = metrics["ap"]["mean"]
        print(f"NEC={nec}: Mean AUROC={auc:.4f}, Mean AP={ap:.4f}")
        results.append({"nec": nec, "auroc": auc, "ap": ap})

    output_csv = os.path.join(model_dir, "nec_metrics.csv")
    with open(output_csv, "w") as f:
        f.write("nec,mean_auroc,mean_ap\n")
        for row in results:
            f.write(f"{row['nec']},{row['auroc']:.6f},{row['ap']:.6f}\n")
    print(f"Saved NEC metrics to {output_csv}")


if __name__ == "__main__":
    main()
