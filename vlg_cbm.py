#!/usr/bin/env python3
"""
VLG-CBM (Vision-Language Grounded Concept Bottleneck Model) for CheXpert

This implements the VLG-CBM approach from:
"VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance"
https://arxiv.org/pdf/2408.01432

Pipeline:
1. Load precomputed ChEX annotations (from generate_annotations.py)
2. Filter concepts by detection frequency
3. Train Concept Bottleneck Layer (CBL): backbone features -> concepts
4. Train sparse final layer: concept activations -> pathology labels
"""

import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    CheXpertDataset,
    CHEXPERT_COMPETITION_LABELS,
    CHEXPERT_PATHOLOGY_LABELS,
    get_transforms
)
from models import get_model, XRV_WEIGHTS

from vlg_cbm_lib.annotations import (
    filter_concepts_by_frequency,
    load_annotations,
    load_concepts,
)
from vlg_cbm_lib.config import parse_args
from vlg_cbm_lib.datasets import BackboneWithConcepts, ConceptDataset, ConceptLayer
from vlg_cbm_lib.eval import evaluate, evaluate_baseline_model
from vlg_cbm_lib.train import (
    extract_concept_activations,
    save_concept_artifacts,
    train_concept_layer,
    train_final_layer_dense,
    train_final_layer_saga,
)


def _save_metrics(metrics: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


def _save_interpretations(labels, concepts, W, output_dir: str) -> None:
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
    _save_metrics(interpretations, os.path.join(output_dir, "interpretations.json"))


def _load_backbone(args, labels, device):
    use_xrv_backbone = args.backbone in XRV_WEIGHTS
    backbone_kwargs = {}
    if use_xrv_backbone:
        backbone_kwargs["target_labels"] = labels
    backbone_model = get_model(
        args.backbone,
        num_classes=len(labels),
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

    if args.backbone == "densenet121":
        feature_dim = 1024
        backbone = backbone_model.backbone.features
    elif args.backbone == "resnet50":
        feature_dim = 2048
        backbone = torch.nn.Sequential(*list(backbone_model.backbone.children())[:-1])
    else:
        feature_dim = getattr(backbone_model, "feature_dim", 1024)

        class XRVDenseNetBackbone(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.get_features(x)

        backbone = XRVDenseNetBackbone(backbone_model)

    return backbone, feature_dim


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.competition_labels:
        labels = CHEXPERT_COMPETITION_LABELS
    else:
        labels = CHEXPERT_PATHOLOGY_LABELS
    num_classes = len(labels)
    print(f"Using {num_classes} pathology labels")

    _, concepts = load_concepts(args.concepts)
    print(f"Loaded {len(concepts)} concepts")

    config = vars(args).copy()
    config.update({
        'labels': labels,
        'num_classes': num_classes,
        'num_concepts_initial': len(concepts),
        'start_time': datetime.now().isoformat(),
    })
    with open(os.path.join(args.output, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

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

    concept_matrix, _ = load_annotations(
        args.annotation_dir,
        n_train,
        concepts,
        args.confidence_threshold,
        num_workers=max(args.num_workers, 1),
        cache_path=args.concept_cache
    )
    if args.concept_cache:
        concept_matrix, _ = load_annotations(
            args.annotation_dir,
            n_train,
            concepts,
            args.confidence_threshold,
            num_workers=max(args.num_workers, 1),
            cache_path=args.concept_cache
        )

    if args.val_annotation_dir and os.path.exists(args.val_annotation_dir):
        val_concept_matrix, _ = load_annotations(
            args.val_annotation_dir,
            n_val,
            concepts,
            args.confidence_threshold,
            num_workers=max(args.num_workers, 1),
            cache_path=args.val_concept_cache
        )
    else:
        missing = args.val_annotation_dir or "<unspecified>"
        print(f"No validation annotations found at {missing}, using zeros")
        val_concept_matrix = np.zeros((n_val, len(concepts)), dtype=np.float32)

    concept_matrix, filtered_concepts, _ = filter_concepts_by_frequency(
        concept_matrix, concepts,
        args.confidence_threshold,
        args.min_concept_freq,
        args.max_concept_freq
    )

    keep_indices = [i for i, c in enumerate(concepts) if c in filtered_concepts]
    val_concept_matrix = val_concept_matrix[:, keep_indices]
    concepts = filtered_concepts

    if len(concepts) < 10:
        print(f"Warning: Only {len(concepts)} concepts remaining!")

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

    print("\nCreating model...")
    backbone, feature_dim = _load_backbone(args, labels, device)

    concept_layer = ConceptLayer(
        feature_dim, len(concepts),
        num_hidden=args.cbl_hidden_layers
    )
    model = BackboneWithConcepts(backbone, concept_layer)

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

    print("\nExtracting concept activations...")
    train_concepts, train_labels = extract_concept_activations(model, train_loader, device)
    val_concepts, val_labels = extract_concept_activations(model, val_loader, device)
    print(f"Train concepts: {train_concepts.shape}")
    print(f"Val concepts: {val_concepts.shape}")

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

    nnz = (W.abs() > 1e-5).sum().item()
    total = W.numel()
    print(f"Sparsity: {nnz}/{total} non-zero ({100*nnz/total:.1f}%)")

    print("\nEvaluating...")
    train_metrics = evaluate(train_concepts, train_labels, W, b, mean, std, labels)
    val_metrics = evaluate(val_concepts, val_labels, W, b, mean, std, labels)
    print(f"\nTrain - Mean AUROC: {train_metrics['mean_auroc']:.4f}")
    print(f"Val   - Mean AUROC: {val_metrics['mean_auroc']:.4f}")
    print("\nPer-class Validation AUROC:")
    for label in labels:
        auroc = val_metrics.get(f'auroc_{label}', float('nan'))
        print(f"  {label}: {auroc:.4f}")

    print("\nSaving...")
    save_concept_artifacts(
        model, concepts, args.output,
        save_backbone=args.cbl_finetune_backbone
    )
    torch.save(W, os.path.join(args.output, "W_g.pt"))
    torch.save(b, os.path.join(args.output, "b_g.pt"))
    torch.save(mean, os.path.join(args.output, "concept_mean.pt"))
    torch.save(std, os.path.join(args.output, "concept_std.pt"))
    _save_metrics(train_metrics, os.path.join(args.output, "train_metrics.json"))
    _save_metrics(val_metrics, os.path.join(args.output, "val_metrics.json"))
    _save_interpretations(labels, concepts, W, args.output)

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
