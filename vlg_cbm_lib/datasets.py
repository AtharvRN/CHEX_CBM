from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ConceptDataset(Dataset):
    """Dataset that returns images and concept annotations."""

    def __init__(
        self,
        base_dataset: Dataset,
        concept_matrix: np.ndarray,
        threshold: float = 0.15
    ):
        self.base_dataset = base_dataset
        self.concept_matrix = concept_matrix
        self.threshold = threshold
        self.concept_labels = (concept_matrix > threshold).astype(np.float32)

    def __len__(self) -> int:
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
        hidden_dim: Union[int, None] = None
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BackboneWithConcepts(nn.Module):
    """Backbone + Concept Layer for end-to-end training."""

    def __init__(self, backbone: nn.Module, concept_layer: ConceptLayer):
        super().__init__()
        self.backbone = backbone
        self.concept_layer = concept_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        concepts = self.concept_layer(features)
        return concepts

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
            if features.dim() > 2:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features
