"""
Saliency and attribution maps for Concept Bottleneck Models.

Provides multiple techniques for understanding which parts of an image 
activate specific concepts or contribute to predictions:

1. GradCAM: Gradient-weighted Class Activation Mapping
2. Integrated Gradients: Path-based attribution
3. Concept Attribution Maps: CBM-specific visualization
4. Smooth gradients with noise
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Callable
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for concept visualization.
    
    Shows which spatial regions in the feature map contribute most to a 
    specific concept activation.
    
    Usage:
        gradcam = GradCAM(model.backbone)
        heatmap = gradcam.generate_cam(image, concept_idx=5)
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: The backbone feature extractor
            target_layer: Specific layer to visualize. If None, uses last conv layer.
        """
        self.model = model
        self.target_layer = target_layer or self._find_target_layer()
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _find_target_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        target = None
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                target = module
        if target is None:
            raise ValueError("No convolutional layer found in model")
        return target
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        concept_layer: nn.Module,
        concept_idx: int,
        use_relu: bool = True
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap for a specific concept.
        
        Args:
            input_tensor: Input image (1, C, H, W)
            concept_layer: The concept bottleneck layer
            concept_idx: Index of the concept to visualize
            use_relu: Whether to apply ReLU to heatmap (show only positive contributions)
        
        Returns:
            Heatmap as numpy array (H, W) normalized to [0, 1]
        """
        self.model.eval()
        concept_layer.eval()
        
        # Forward pass
        input_tensor.requires_grad_(True)
        features = self.model(input_tensor)
        
        # Pool features if needed
        if features.dim() > 2:
            pooled_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        else:
            pooled_features = features
        
        # Get concept activations
        concept_logits = concept_layer(pooled_features)
        
        # Backward pass for specific concept
        self.model.zero_grad()
        concept_layer.zero_grad()
        concept_logits[0, concept_idx].backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        if use_relu:
            cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def __call__(self, *args, **kwargs):
        return self.generate_cam(*args, **kwargs)


class IntegratedGradients:
    """
    Integrated Gradients attribution method.
    
    Computes pixel-level attributions by integrating gradients along a path
    from a baseline (black image) to the actual image.
    
    More stable than vanilla gradients and satisfies theoretical properties.
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: The full model (backbone + concept layer)
        """
        self.model = model
    
    def generate_attribution(
        self,
        input_tensor: torch.Tensor,
        target_concept_idx: int,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50
    ) -> np.ndarray:
        """
        Generate integrated gradients attribution map.
        
        Args:
            input_tensor: Input image (1, C, H, W)
            target_concept_idx: Index of concept to attribute
            baseline: Baseline image (if None, uses zeros)
            n_steps: Number of interpolation steps
        
        Returns:
            Attribution map (C, H, W) as numpy array
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Generate interpolated images
        alphas = torch.linspace(0, 1, n_steps, device=input_tensor.device)
        
        integrated_grads = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated)
            
            # Backward pass
            self.model.zero_grad()
            output[0, target_concept_idx].backward()
            
            # Accumulate gradients
            integrated_grads += interpolated.grad
        
        # Average and scale
        integrated_grads /= n_steps
        integrated_grads *= (input_tensor - baseline)
        
        # Sum across color channels for visualization
        attribution = integrated_grads[0].cpu().numpy()
        attribution = np.abs(attribution).sum(axis=0)  # (H, W)
        
        # Normalize
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        return attribution


class SmoothGrad:
    """
    SmoothGrad: Reduces noise in gradient-based attributions by averaging
    gradients over multiple noisy versions of the input.
    """
    
    def __init__(self, model: nn.Module, noise_level: float = 0.15, n_samples: int = 25):
        """
        Args:
            model: The full model (backbone + concept layer)
            noise_level: Std of Gaussian noise as fraction of input range
            n_samples: Number of noisy samples to average
        """
        self.model = model
        self.noise_level = noise_level
        self.n_samples = n_samples
    
    def generate_attribution(
        self,
        input_tensor: torch.Tensor,
        target_concept_idx: int
    ) -> np.ndarray:
        """
        Generate SmoothGrad attribution map.
        
        Args:
            input_tensor: Input image (1, C, H, W)
            target_concept_idx: Index of concept to attribute
        
        Returns:
            Attribution map (C, H, W) as numpy array
        """
        smooth_grads = torch.zeros_like(input_tensor)
        stdev = self.noise_level * (input_tensor.max() - input_tensor.min())
        
        for _ in range(self.n_samples):
            # Add Gaussian noise
            noise = torch.randn_like(input_tensor) * stdev
            noisy_input = input_tensor + noise
            noisy_input.requires_grad_(True)
            
            # Forward pass
            output = self.model(noisy_input)
            
            # Backward pass
            self.model.zero_grad()
            output[0, target_concept_idx].backward()
            
            # Accumulate gradients
            smooth_grads += noisy_input.grad
        
        # Average
        smooth_grads /= self.n_samples
        
        # Process for visualization
        attribution = smooth_grads[0].cpu().numpy()
        attribution = np.abs(attribution).sum(axis=0)  # (H, W)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        return attribution


class ConceptAttributionMap:
    """
    CBM-specific attribution that shows:
    1. Which pixels activate a concept
    2. Which concepts contribute to a disease prediction
    
    Combines spatial and concept-level attributions.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        concept_layer: nn.Module,
        final_layer_weights: torch.Tensor
    ):
        """
        Args:
            backbone: Feature extractor
            concept_layer: Concept bottleneck layer
            final_layer_weights: Weight matrix (n_classes, n_concepts)
        """
        self.backbone = backbone
        self.concept_layer = concept_layer
        self.final_layer_weights = final_layer_weights
        self.gradcam = GradCAM(backbone)
    
    def generate_disease_attribution(
        self,
        input_tensor: torch.Tensor,
        disease_idx: int,
        top_k: int = 5
    ) -> Tuple[List[int], List[float], np.ndarray]:
        """
        Generate attribution for a disease prediction via concepts.
        
        Args:
            input_tensor: Input image (1, C, H, W)
            disease_idx: Index of disease/class
            top_k: Number of top concepts to visualize
        
        Returns:
            - List of top concept indices
            - List of concept weights for this disease
            - Combined spatial heatmap
        """
        # Get concept weights for this disease
        concept_weights = self.final_layer_weights[disease_idx].cpu().numpy()
        
        # Get top contributing concepts
        top_concept_indices = np.argsort(np.abs(concept_weights))[-top_k:][::-1]
        top_weights = concept_weights[top_concept_indices]
        
        # Generate spatial heatmaps for top concepts
        combined_heatmap = None
        
        for concept_idx, weight in zip(top_concept_indices, top_weights):
            heatmap = self.gradcam.generate_cam(
                input_tensor,
                self.concept_layer,
                int(concept_idx)
            )
            
            # Resize if needed
            target_size = (input_tensor.shape[2], input_tensor.shape[3])
            heatmap_resized = cv2.resize(heatmap, target_size[::-1])
            
            # Weight by concept importance
            weighted_heatmap = heatmap_resized * np.abs(weight)
            
            if combined_heatmap is None:
                combined_heatmap = weighted_heatmap
            else:
                combined_heatmap += weighted_heatmap
        
        # Normalize combined heatmap
        if combined_heatmap is not None:
            combined_heatmap = (combined_heatmap - combined_heatmap.min()) / \
                              (combined_heatmap.max() - combined_heatmap.min() + 1e-8)
        
        return top_concept_indices.tolist(), top_weights.tolist(), combined_heatmap


def visualize_heatmap(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image (can be tensor, numpy, or PIL)
        heatmap: Heatmap array (H, W) in [0, 1]
        alpha: Transparency of overlay
        colormap: Matplotlib colormap name
    
    Returns:
        RGB image with heatmap overlay (H, W, 3)
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        img_np = image.squeeze().cpu().numpy()
        if img_np.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            img_np = np.transpose(img_np, (1, 2, 0))
    elif isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image
    
    # Normalize image to [0, 1]
    if img_np.max() > 1:
        img_np = img_np / 255.0
    
    # Convert grayscale to RGB if needed
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)
    
    # Resize heatmap to match image
    if heatmap.shape != img_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Blend
    overlayed = (1 - alpha) * img_np + alpha * heatmap_colored
    overlayed = np.clip(overlayed, 0, 1)
    
    return overlayed


def save_saliency_visualization(
    image: torch.Tensor,
    heatmap: np.ndarray,
    output_path: str,
    title: str = "",
    concept_name: str = "",
    score: Optional[float] = None
):
    """
    Save a comprehensive saliency visualization with multiple views.
    
    Args:
        image: Original image tensor (1, C, H, W)
        heatmap: Saliency heatmap (H, W)
        output_path: Path to save figure
        title: Main title
        concept_name: Name of the concept
        score: Concept activation score
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img_np = image.squeeze().cpu().numpy()
    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # Normalize for display
    img_display = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    if img_display.ndim == 2:
        axes[0].imshow(img_display, cmap='gray')
    else:
        axes[0].imshow(img_display)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Heatmap only
    im = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Saliency Map")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    overlay = visualize_heatmap(image, heatmap, alpha=0.5)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    # Overall title
    if title:
        title_text = title
        if concept_name:
            title_text += f"\nConcept: {concept_name}"
        if score is not None:
            title_text += f" (score: {score:.3f})"
        fig.suptitle(title_text, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def batch_concept_visualization(
    images: torch.Tensor,
    concept_names: List[str],
    concept_indices: List[int],
    model: nn.Module,
    concept_layer: nn.Module,
    output_dir: str,
    method: str = 'gradcam'
):
    """
    Generate saliency maps for multiple concepts across a batch of images.
    
    Args:
        images: Batch of images (B, C, H, W)
        concept_names: List of concept names
        concept_indices: Indices of concepts to visualize
        model: Backbone model
        concept_layer: Concept bottleneck layer
        output_dir: Directory to save visualizations
        method: 'gradcam', 'integrated_gradients', or 'smoothgrad'
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if method == 'gradcam':
        visualizer = GradCAM(model)
    elif method == 'integrated_gradients':
        full_model = nn.Sequential(model, concept_layer)
        visualizer = IntegratedGradients(full_model)
    elif method == 'smoothgrad':
        full_model = nn.Sequential(model, concept_layer)
        visualizer = SmoothGrad(full_model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    for img_idx in range(min(len(images), 10)):  # Limit to 10 images
        for concept_idx in concept_indices:
            image = images[img_idx:img_idx+1]
            
            if method == 'gradcam':
                heatmap = visualizer.generate_cam(image, concept_layer, concept_idx)
                heatmap = cv2.resize(heatmap, (image.shape[3], image.shape[2]))
            else:
                heatmap = visualizer.generate_attribution(image, concept_idx)
            
            concept_name = concept_names[concept_idx] if concept_idx < len(concept_names) else f"Concept_{concept_idx}"
            
            output_path = os.path.join(
                output_dir,
                f"img{img_idx}_concept{concept_idx}_{method}.png"
            )
            
            save_saliency_visualization(
                image,
                heatmap,
                output_path,
                title=f"Image {img_idx}",
                concept_name=concept_name
            )


__all__ = [
    'GradCAM',
    'IntegratedGradients',
    'SmoothGrad',
    'ConceptAttributionMap',
    'visualize_heatmap',
    'save_saliency_visualization',
    'batch_concept_visualization'
]
