import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vlg_cbm_lib.datasets import BackboneWithConcepts, ConceptLayer


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
        model.backbone.eval()
        for param in model.backbone.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(model.concept_layer.parameters(), lr=lr)

    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.concept_layer.train()
        train_loss = 0.0
        for images, concept_labels, _ in train_loader:
            images = images.to(device)
            concept_labels = concept_labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, concept_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

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
        for images, _, pathology_labels in loader:
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

    mean = concept_activations.mean(dim=0, keepdim=True)
    std = concept_activations.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-6)

    train_c = (concept_activations - mean) / std
    val_c = (val_concepts - mean) / std

    train_ds = IndexedTensorDataset(train_c, labels.float())
    val_ds = TensorDataset(val_c, val_labels.float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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

    mean = concept_activations.mean(dim=0, keepdim=True)
    std = concept_activations.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-6)

    train_c = (concept_activations - mean) / std
    val_c = (val_concepts - mean) / std

    train_ds = TensorDataset(train_c, labels.float())
    val_ds = TensorDataset(val_c, val_labels.float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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
