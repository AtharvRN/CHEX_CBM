"""
Model architectures for CheXpert classification

Supports:
1. Standard ImageNet pretrained models (DenseNet-121, ResNet-50)
2. TorchXRayVision pretrained models (trained on chest X-ray datasets)
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List

# TorchXRayVision models
try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False
    print("Warning: torchxrayvision not installed. Install with: pip install torchxrayvision")


# XRV pathologies (18 classes from the "all" model)
XRV_PATHOLOGIES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
    'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture',
    'Lung Opacity', 'Enlarged Cardiomediastinum'
]

# Available XRV pretrained weights
XRV_WEIGHTS = {
    'xrv-all': 'densenet121-res224-all',      # All datasets combined (best general)
    'xrv-chex': 'densenet121-res224-chex',    # CheXpert only
    'xrv-nih': 'densenet121-res224-nih',      # NIH ChestX-ray14
    'xrv-mimic-nb': 'densenet121-res224-mimic_nb',  # MIMIC-CXR (NegBio labels)
    'xrv-mimic-ch': 'densenet121-res224-mimic_ch',  # MIMIC-CXR (CheXpert labels)
    'xrv-rsna': 'densenet121-res224-rsna',    # RSNA Pneumonia
    'xrv-pc': 'densenet121-res224-pc',        # PadChest
}


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 for multi-label chest X-ray classification.
    Standard architecture used in CheXNet and many CheXpert baselines.
    """
    
    def __init__(
        self, 
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Load pretrained DenseNet-121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier with our own
        if dropout > 0:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, num_classes)
            )
        else:
            self.backbone.classifier = nn.Linear(num_features, num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (N, C, H, W)
            
        Returns:
            Logits (N, num_classes) - NOT sigmoid activated
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 for multi-label chest X-ray classification.
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        num_features = self.backbone.fc.in_features
        
        if dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_model(
    model_name: str = 'densenet121',
    num_classes: int = 14,
    pretrained: bool = True,
    dropout: float = 0.0,
    target_labels: List[str] = None
) -> nn.Module:
    """
    Factory function to get a model by name.
    
    Args:
        model_name: One of 'densenet121', 'resnet50', 'xrv-all', 'xrv-chex', etc.
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability before classifier
        target_labels: List of target label names (for XRV models to map outputs)
        
    Returns:
        nn.Module
    """
    model_name = model_name.lower()
    
    # Standard ImageNet pretrained models
    if model_name == 'densenet121':
        return DenseNet121Classifier(num_classes, pretrained, dropout)
    elif model_name == 'resnet50':
        return ResNet50Classifier(num_classes, pretrained, dropout)
    
    # TorchXRayVision pretrained models
    elif model_name.startswith('xrv'):
        if not XRV_AVAILABLE:
            raise ImportError("torchxrayvision not installed. Install with: pip install torchxrayvision")
        return XRVDenseNet(model_name, num_classes, target_labels, dropout)
    
    else:
        available = ['densenet121', 'resnet50'] + list(XRV_WEIGHTS.keys())
        raise ValueError(f"Unknown model: {model_name}. Choose from: {available}")


class XRVDenseNet(nn.Module):
    """
    TorchXRayVision DenseNet-121 pretrained on chest X-ray datasets.
    
    These models are pretrained on multiple chest X-ray datasets and predict
    18 pathology classes. We can either:
    1. Use as a backbone (freeze) and add new classifier
    2. Fine-tune the whole model
    3. Use output mapping if target classes overlap with XRV pathologies
    
    Available pretrained weights:
    - xrv-all: All datasets combined (recommended for general use)
    - xrv-chex: CheXpert only
    - xrv-nih: NIH ChestX-ray14
    - xrv-mimic-nb: MIMIC-CXR with NegBio labels
    - xrv-mimic-ch: MIMIC-CXR with CheXpert labels  
    - xrv-rsna: RSNA Pneumonia Detection
    - xrv-pc: PadChest
    """
    
    def __init__(
        self,
        weights: str = 'xrv-all',
        num_classes: int = 14,
        target_labels: List[str] = None,
        dropout: float = 0.0,
        use_pretrained_head: bool = False
    ):
        """
        Args:
            weights: Which XRV weights to use (e.g., 'xrv-all', 'xrv-chex')
            num_classes: Number of output classes for new head
            target_labels: List of target label names for output mapping
            dropout: Dropout before classifier
            use_pretrained_head: If True, use XRV's pretrained classifier (18 outputs)
                                 and map to target labels. If False, train new head.
        """
        super().__init__()
        
        # Get the actual weight name
        if weights in XRV_WEIGHTS:
            weight_name = XRV_WEIGHTS[weights]
        else:
            weight_name = weights  # Allow direct weight names too
        
        # Load pretrained XRV model
        print(f"Loading TorchXRayVision model: {weight_name}")
        self.xrv_model = xrv.models.DenseNet(weights=weight_name)
        self.xrv_pathologies = list(self.xrv_model.pathologies)
        
        self.num_classes = num_classes
        self.target_labels = target_labels
        self.use_pretrained_head = use_pretrained_head
        
        # Feature dimension (before XRV's classifier)
        self.feature_dim = 1024  # DenseNet-121 features
        
        if use_pretrained_head and target_labels is not None:
            # Map XRV outputs to target labels
            self.output_indices = self._build_output_mapping(target_labels)
            print(f"Using pretrained head with output mapping: {len(self.output_indices)} classes")
            self.classifier = None
        else:
            # Train new classifier head
            print(f"Training new classifier head: {num_classes} classes")
            if dropout > 0:
                self.classifier = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(self.feature_dim, num_classes)
                )
            else:
                self.classifier = nn.Linear(self.feature_dim, num_classes)
            self.output_indices = None
            
    def _build_output_mapping(self, target_labels: List[str]) -> List[int]:
        """
        Build mapping from XRV pathologies to target labels.
        
        Returns list of indices into XRV's 18-class output for each target label.
        Returns -1 for labels not in XRV.
        """
        # Normalize label names for matching
        xrv_normalized = {p.lower().replace('_', ' '): i 
                         for i, p in enumerate(self.xrv_pathologies)}
        
        indices = []
        for label in target_labels:
            label_norm = label.lower().replace('_', ' ')
            
            # Handle special cases
            if label_norm == 'pleural effusion':
                label_norm = 'effusion'
            elif label_norm == 'pleural other':
                label_norm = 'pleural_thickening'  # Best match
                
            if label_norm in xrv_normalized:
                indices.append(xrv_normalized[label_norm])
                print(f"  {label} -> XRV[{xrv_normalized[label_norm]}]: {self.xrv_pathologies[xrv_normalized[label_norm]]}")
            else:
                indices.append(-1)
                print(f"  {label} -> NOT FOUND in XRV (will use zero)")
                
        return indices
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier."""
        # XRV model expects single-channel (grayscale) input
        # Convert RGB to grayscale if needed
        if x.shape[1] == 3:
            # Convert RGB to grayscale using standard weights
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # XRV expects input normalized to [-1024, 1024] range (HU units)
        # Our data is [0, 1], so we scale appropriately
        # XRV internal normalization: (x - mean) / std with mean=0.4949, std=0.2485
        # But actually XRV handles this internally, let's just scale to match expected range
        x = (x - 0.5) * 2048  # Scale from [0,1] to [-1024, 1024]
        
        features = self.xrv_model.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (N, C, H, W), normalized to [0, 1]. C can be 1 or 3.
            
        Returns:
            Logits (N, num_classes)
        """
        # Convert RGB to grayscale if needed
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Scale to XRV expected range [-1024, 1024]
        x = (x - 0.5) * 2048
        
        if self.use_pretrained_head and self.output_indices is not None:
            # Use XRV's pretrained classifier and select relevant outputs
            logits = self.xrv_model(x)  # (N, 18)
            
            # Select outputs for our target labels
            selected = []
            for idx in self.output_indices:
                if idx >= 0:
                    selected.append(logits[:, idx:idx+1])
                else:
                    # Label not in XRV - return zeros
                    selected.append(torch.zeros(logits.size(0), 1, device=logits.device))
            return torch.cat(selected, dim=1)
        else:
            # Extract features and use new classifier
            features = self.xrv_model.features(x)
            out = nn.functional.relu(features, inplace=True)
            out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            return self.classifier(out)
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for param in self.xrv_model.parameters():
            param.requires_grad = False
        if self.classifier is not None:
            for param in self.classifier.parameters():
                param.requires_grad = True
                
    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.xrv_model.parameters():
            param.requires_grad = True
        if self.classifier is not None:
            for param in self.classifier.parameters():
                param.requires_grad = True


def get_xrv_model_info():
    """Print information about available XRV models."""
    print("Available TorchXRayVision pretrained models:")
    print("=" * 60)
    for name, weight in XRV_WEIGHTS.items():
        print(f"  {name:15s} -> {weight}")
    print("\nXRV Pathologies (18 classes):")
    print("-" * 60)
    for i, p in enumerate(XRV_PATHOLOGIES):
        print(f"  {i:2d}: {p}")
    print("\nUsage:")
    print("  model = get_model('xrv-all', num_classes=12, target_labels=LABELS)")
    print("  model = get_model('xrv-chex', num_classes=12, target_labels=LABELS)")
