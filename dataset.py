"""
CheXpert Dataset for Multi-Label Classification
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Optional, Callable, List, Tuple


# CheXpert label columns (in order)
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum", 
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

# Competition subset (5 classes)
CHEXPERT_COMPETITION_LABELS = [
    "Atelectasis",
    "Cardiomegaly", 
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]

# Pathology-only labels (12 classes - excludes "No Finding" and "Support Devices")
CHEXPERT_PATHOLOGY_LABELS = [
    "Enlarged Cardiomediastinum", 
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture"
]


class CheXpertDataset(Dataset):
    """
    CheXpert Dataset for multi-label chest X-ray classification.
    
    Args:
        csv_path: Path to the CSV file (train.csv or valid.csv)
        img_root: Root directory for images (parent of CheXpert-v1.0-small/)
        transform: Optional transform to apply to images
        labels: List of label columns to use (default: all 14)
        uncertain_strategy: How to handle uncertain labels (-1)
            - 'ones': Map -1 to 1 (U-Ones, default)
            - 'zeros': Map -1 to 0 (U-Zeros)
            - 'ignore': Map -1 to 0 and create mask
        frontal_only: If True, only use frontal (AP/PA) views
    """
    
    def __init__(
        self,
        csv_path: str,
        img_root: str,
        transform: Optional[Callable] = None,
        labels: Optional[List[str]] = None,
        uncertain_strategy: str = 'ones',
        frontal_only: bool = True
    ):
        self.img_root = img_root
        self.transform = transform
        self.labels = labels if labels is not None else CHEXPERT_LABELS
        self.uncertain_strategy = uncertain_strategy
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Filter to frontal views only if requested
        if frontal_only:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)
        
        # Extract labels and convert
        self.targets = self._process_labels()
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(f"Labels ({len(self.labels)}): {self.labels}")
        print(f"Uncertain strategy: {uncertain_strategy}")
        
    def _process_labels(self) -> torch.Tensor:
        """Process labels according to uncertain_strategy."""
        label_matrix = self.df[self.labels].values.astype(np.float32)
        
        # Handle NaN (not mentioned) -> 0
        label_matrix = np.nan_to_num(label_matrix, nan=0.0)
        
        # Handle uncertain labels (-1)
        if self.uncertain_strategy == 'ones':
            # U-Ones: -1 -> 1
            label_matrix[label_matrix == -1.0] = 1.0
        elif self.uncertain_strategy == 'zeros':
            # U-Zeros: -1 -> 0
            label_matrix[label_matrix == -1.0] = 0.0
        elif self.uncertain_strategy == 'ignore':
            # Map to 0, will need separate mask handling in loss
            label_matrix[label_matrix == -1.0] = 0.0
        else:
            raise ValueError(f"Unknown uncertain_strategy: {self.uncertain_strategy}")
        
        return torch.from_numpy(label_matrix).float()
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get image path
        img_path = self.df.iloc[idx]['Path']
        # The path in CSV is relative like "CheXpert-v1.0-small/train/..."
        full_path = os.path.join(self.img_root, img_path)
        
        # Load image
        img = Image.open(full_path).convert('RGB')
        
        # Apply transform
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        # Get label
        label = self.targets[idx]
        
        return img, label

    def get_image_path(self, idx: int) -> str:
        """Return the absolute image path without loading the file."""
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.df)}")
        rel_path = self.df.iloc[idx]['Path']
        return os.path.join(self.img_root, rel_path)
    
    def get_pos_weights(self) -> torch.Tensor:
        """Compute positive class weights for imbalanced data (neg/pos ratio)."""
        pos_counts = self.targets.sum(dim=0)
        neg_counts = len(self.targets) - pos_counts
        # Avoid division by zero
        pos_counts = torch.clamp(pos_counts, min=1.0)
        return neg_counts / pos_counts


def get_transforms(img_size: int = 224, is_training: bool = True):
    """Get standard transforms for CheXpert images."""
    if is_training:
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def get_grayscale_transforms(img_size: int = 224, is_training: bool = True):
    """Get transforms that output single-channel grayscale (for TorchXRayVision)."""
    if is_training:
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.25])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.25])
        ])
