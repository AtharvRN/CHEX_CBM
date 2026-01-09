#!/usr/bin/env python3
"""
Simple test script to verify saliency visualization setup.

This script tests the saliency module with dummy data to ensure
everything is properly installed and working.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.saliency import (
    GradCAM,
    IntegratedGradients,
    SmoothGrad,
    visualize_heatmap,
    save_saliency_visualization
)


def create_dummy_model():
    """Create a simple dummy model for testing."""
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.relu(self.conv3(x))
            return x
    
    class DummyConceptLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, 10)  # 10 concepts
        
        def forward(self, x):
            if x.dim() > 2:
                x = self.pool(x).flatten(1)
            return self.fc(x)
    
    class DummyCBM(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DummyBackbone()
            self.concept_layer = DummyConceptLayer()
        
        def forward(self, x):
            features = self.backbone(x)
            return self.concept_layer(features)
    
    return DummyCBM()


def test_gradcam():
    """Test GradCAM visualization."""
    print("Testing GradCAM...")
    
    model = create_dummy_model()
    model.eval()
    
    # Create dummy image
    image = torch.randn(1, 3, 224, 224)
    
    # Generate GradCAM
    gradcam = GradCAM(model.backbone)
    heatmap = gradcam.generate_cam(image, model.concept_layer, concept_idx=0)
    
    assert heatmap.shape[0] > 0 and heatmap.shape[1] > 0, "Invalid heatmap shape"
    assert heatmap.min() >= 0 and heatmap.max() <= 1, "Heatmap not normalized"
    
    print("✓ GradCAM test passed")
    return image, heatmap


def test_integrated_gradients():
    """Test Integrated Gradients."""
    print("Testing Integrated Gradients...")
    
    model = create_dummy_model()
    model.eval()
    
    image = torch.randn(1, 3, 224, 224)
    
    ig = IntegratedGradients(model)
    heatmap = ig.generate_attribution(image, target_concept_idx=0, n_steps=10)
    
    assert heatmap.shape == (224, 224), f"Invalid heatmap shape: {heatmap.shape}"
    assert heatmap.min() >= 0 and heatmap.max() <= 1, "Heatmap not normalized"
    
    print("✓ Integrated Gradients test passed")
    return heatmap


def test_smoothgrad():
    """Test SmoothGrad."""
    print("Testing SmoothGrad...")
    
    model = create_dummy_model()
    model.eval()
    
    image = torch.randn(1, 3, 224, 224)
    
    sg = SmoothGrad(model, noise_level=0.1, n_samples=5)
    heatmap = sg.generate_attribution(image, target_concept_idx=0)
    
    assert heatmap.shape == (224, 224), f"Invalid heatmap shape: {heatmap.shape}"
    assert heatmap.min() >= 0 and heatmap.max() <= 1, "Heatmap not normalized"
    
    print("✓ SmoothGrad test passed")
    return heatmap


def test_visualization():
    """Test visualization utilities."""
    print("Testing visualization utilities...")
    
    image = torch.randn(1, 3, 224, 224)
    heatmap = np.random.rand(224, 224)
    
    # Test overlay
    overlay = visualize_heatmap(image, heatmap, alpha=0.5)
    assert overlay.shape == (224, 224, 3), f"Invalid overlay shape: {overlay.shape}"
    assert overlay.min() >= 0 and overlay.max() <= 1, "Overlay not normalized"
    
    # Test saving (without actually saving)
    print("✓ Visualization utilities test passed")


def test_all():
    """Run all tests."""
    print("=" * 60)
    print("Saliency Module Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Test individual components
        image, heatmap_gc = test_gradcam()
        heatmap_ig = test_integrated_gradients()
        heatmap_sg = test_smoothgrad()
        test_visualization()
        
        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("The saliency module is ready to use.")
        print("Next steps:")
        print("  1. Train or load a CBM model")
        print("  2. Run: python visualize_concepts.py --help")
        print("  3. Or use the Jupyter notebook: notebooks/concept_saliency.ipynb")
        
        # Create a simple visualization
        print()
        print("Generating sample comparison plot...")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(heatmap_gc, cmap='jet')
        axes[0].set_title('GradCAM')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_ig, cmap='jet')
        axes[1].set_title('Integrated Gradients')
        axes[1].axis('off')
        
        axes[2].imshow(heatmap_sg, cmap='jet')
        axes[2].set_title('SmoothGrad')
        axes[2].axis('off')
        
        fig.suptitle('Saliency Methods Comparison (Dummy Data)', fontweight='bold')
        plt.tight_layout()
        plt.savefig('test_saliency_output.png', dpi=100, bbox_inches='tight')
        print("Saved test visualization to: test_saliency_output.png")
        
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_all()
    exit(0 if success else 1)
