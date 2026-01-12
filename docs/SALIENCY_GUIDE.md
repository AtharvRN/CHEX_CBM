# Concept Saliency Visualization Guide

This guide explains how to use the saliency visualization tools to understand which parts of chest X-ray images activate specific concepts in your CBM models.

## Overview

The saliency module provides multiple attribution methods:

1. **GradCAM** - Gradient-weighted Class Activation Mapping
2. **Integrated Gradients** - Path-based attribution with theoretical guarantees
3. **SmoothGrad** - Noise-averaged gradients for cleaner maps
4. **Concept Attribution Maps** - Disease-level attribution through concepts

## Quick Start

### 1. Generate Saliency Maps for Trained Models

```bash
# For VLG-CBM - visualize specific concepts
python visualize_concepts.py \
    --model_type vlg_cbm \
    --model_dir checkpoints/vlg_cbm_10k \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concepts concepts/chexpert_concepts.txt \
    --output visualizations/vlg_concepts \
    --num_samples 20 \
    --method gradcam

# For Label-Free CBM
python visualize_concepts.py \
    --model_type lf_cbm \
    --model_dir saved_models/lf_cbm_10k \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concepts concepts/chexpert_concepts.txt \
    --output visualizations/lf_concepts \
    --num_samples 20
```

### 2. Disease-Level Attribution

See which concepts contribute to disease predictions:

```bash
python visualize_concepts.py \
    --model_type vlg_cbm \
    --model_dir checkpoints/vlg_cbm_10k \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concepts concepts/chexpert_concepts.txt \
    --disease_attribution \
    --disease_idx 0 \
    --top_k_concepts 10 \
    --output visualizations/disease_attribution \
    --num_samples 20
```

### 3. COVID-QU Dataset

```bash
python visualize_concepts.py \
    --model_type vlg_cbm \
    --model_dir checkpoints/vlg_covidqu \
    --data_dir /workspace/datasets/COVIDQU \
    --label_set covidqu \
    --covidqu_variant infection \
    --concepts concepts/covidqu_irrrag_concepts.txt \
    --output visualizations/covidqu \
    --num_samples 20
```

## Interactive Notebook

For exploratory analysis, use the Jupyter notebook:

```bash
jupyter notebook notebooks/concept_saliency.ipynb
```

The notebook provides:
- Interactive concept exploration
- Comparison of different saliency methods
- Disease-level attribution analysis
- Batch concept analysis across multiple images

## Python API Usage

### Example 1: GradCAM for a Single Concept

```python
import torch
from utils.saliency import GradCAM
from models import get_model
from vlg_cbm_lib.datasets import ConceptLayer, BackboneWithConcepts

# Load your model
backbone = ...  # Your backbone network
concept_layer = ...  # Your concept bottleneck layer
model = BackboneWithConcepts(backbone, concept_layer)
model.eval()

# Load an image
image = ...  # Tensor of shape (1, C, H, W)

# Generate GradCAM
gradcam = GradCAM(backbone)
heatmap = gradcam.generate_cam(
    input_tensor=image,
    concept_layer=concept_layer,
    concept_idx=5  # Visualize concept #5
)

# Visualize
from utils.saliency import save_saliency_visualization
save_saliency_visualization(
    image=image,
    heatmap=heatmap,
    output_path="output/concept5.png",
    concept_name="Air bronchograms",
    score=0.85
)
```

### Example 2: Compare Multiple Methods

```python
from utils.saliency import GradCAM, IntegratedGradients, SmoothGrad
import cv2

# Initialize methods
gradcam = GradCAM(model.backbone)
ig = IntegratedGradients(model)
sg = SmoothGrad(model, noise_level=0.15, n_samples=25)

concept_idx = 10

# Generate heatmaps
heatmap_gc = gradcam.generate_cam(image, concept_layer, concept_idx)
heatmap_gc = cv2.resize(heatmap_gc, (224, 224))

heatmap_ig = ig.generate_attribution(image, concept_idx)

heatmap_sg = sg.generate_attribution(image, concept_idx)

# Compare visually
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(heatmap_gc, cmap='jet')
axes[0].set_title('GradCAM')
axes[1].imshow(heatmap_ig, cmap='jet')
axes[1].set_title('Integrated Gradients')
axes[2].imshow(heatmap_sg, cmap='jet')
axes[2].set_title('SmoothGrad')
plt.show()
```

### Example 3: Disease Attribution Through Concepts

```python
from utils.saliency import ConceptAttributionMap

# Initialize with your trained model components
disease_attr = ConceptAttributionMap(
    backbone=model.backbone,
    concept_layer=model.concept_layer,
    final_layer_weights=W  # Shape: (n_diseases, n_concepts)
)

# Generate attribution for a specific disease
disease_idx = 0  # e.g., Atelectasis
top_concepts, weights, heatmap = disease_attr.generate_disease_attribution(
    input_tensor=image,
    disease_idx=disease_idx,
    top_k=10
)

print("Top contributing concepts:")
for concept_idx, weight in zip(top_concepts, weights):
    print(f"  {concept_names[concept_idx]}: {weight:.4f}")

# Visualize combined heatmap
from utils.saliency import visualize_heatmap
overlay = visualize_heatmap(image, heatmap, alpha=0.5)
plt.imshow(overlay)
plt.title(f"Attribution for {disease_names[disease_idx]}")
plt.show()
```

## Method Comparison

| Method | Speed | Noise Sensitivity | Theoretical Properties | Best For |
|--------|-------|-------------------|----------------------|----------|
| **GradCAM** | Fast ⚡⚡⚡ | Medium | Localization | Quick spatial attribution |
| **Integrated Gradients** | Medium ⚡⚡ | Low | Completeness, Sensitivity | Trustworthy attributions |
| **SmoothGrad** | Slow ⚡ | Very Low | Noise reduction | Clean visualizations |

### Recommendations

- **For quick exploration**: Use GradCAM (fastest, good spatial localization)
- **For publication/validation**: Use Integrated Gradients (theoretical guarantees)
- **For noisy models**: Use SmoothGrad (reduces gradient noise)
- **For clinical validation**: Use all three and compare

## Understanding the Outputs

### Heatmap Interpretation

- **Red/Yellow regions**: High contribution to concept activation
- **Blue/Dark regions**: Low contribution
- **Intensity**: Magnitude of attribution

### Disease Attribution

The disease attribution visualization shows:
1. **Spatial heatmap**: Where in the image concepts are activated
2. **Concept weights**: How much each concept contributes to the disease prediction
3. **Green bars**: Concepts that increase disease probability
4. **Red bars**: Concepts that decrease disease probability

## Integration with Training Pipeline

### Option 1: Generate During Evaluation

Add to your evaluation script:

```python
from utils.saliency import batch_concept_visualization

# After model evaluation
batch_concept_visualization(
    images=test_images[:10],
    concept_names=concept_names,
    concept_indices=[0, 5, 10, 15, 20],
    model=backbone,
    concept_layer=concept_layer,
    output_dir="eval_results/saliency",
    method='gradcam'
)
```

### Option 2: Add to VLG-CBM Training

In `vlg_cbm.py`, after training:

```python
# At the end of main()
if args.visualize_concepts:
    print("\nGenerating concept visualizations...")
    from utils.saliency import batch_concept_visualization
    
    val_loader_viz = DataLoader(val_dataset, batch_size=10, shuffle=False)
    images, _, _ = next(iter(val_loader_viz))
    
    top_concepts = np.argsort(np.abs(W_c[0]))[-10:]  # Top concepts for first disease
    
    batch_concept_visualization(
        images=images.to(device),
        concept_names=filtered_concepts,
        concept_indices=top_concepts.tolist(),
        model=backbone,
        concept_layer=model.concept_layer,
        output_dir=os.path.join(args.output, "concept_viz"),
        method='gradcam'
    )
```

## Troubleshooting

### Issue: Blank or uniform heatmaps

**Cause**: Model not properly loaded or frozen layers

**Solution**:
```python
model.eval()  # Ensure eval mode
# If still issues, check gradient flow:
image.requires_grad_(True)
```

### Issue: Heatmaps look random/noisy

**Cause**: Insufficient training or high learning rate

**Solution**: 
- Use SmoothGrad for noise reduction
- Check model convergence
- Try Integrated Gradients with more steps

### Issue: Size mismatch errors

**Cause**: Heatmap size doesn't match image size

**Solution**:
```python
import cv2
heatmap = cv2.resize(heatmap, (image.shape[3], image.shape[2]))
```

### Issue: CUDA out of memory

**Cause**: Integrated Gradients or SmoothGrad use many forward passes

**Solution**:
- Reduce batch size to 1
- Reduce n_steps (for IG) or n_samples (for SmoothGrad)
- Process images sequentially

## Advanced Usage

### Custom Colormap

```python
from matplotlib.colors import LinearSegmentedColormap

# Create custom colormap
colors = ['blue', 'cyan', 'yellow', 'red']
n_bins = 256
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Use in visualization
overlay = visualize_heatmap(image, heatmap, alpha=0.5, colormap=cmap)
```

### Save All Methods for a Concept

```python
def save_all_methods(image, concept_idx, model, concept_layer, output_prefix):
    methods = {
        'gradcam': GradCAM(model.backbone),
        'integrated_gradients': IntegratedGradients(model),
        'smoothgrad': SmoothGrad(model)
    }
    
    for method_name, method in methods.items():
        if method_name == 'gradcam':
            heatmap = method.generate_cam(image, concept_layer, concept_idx)
            heatmap = cv2.resize(heatmap, (224, 224))
        else:
            heatmap = method.generate_attribution(image, concept_idx)
        
        save_saliency_visualization(
            image, heatmap,
            f"{output_prefix}_{method_name}.png",
            concept_name=concept_names[concept_idx]
        )
```

## Citation

If you use these visualization tools in your research, please cite:

- **GradCAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
- **Integrated Gradients**: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
- **SmoothGrad**: Smilkov et al., "SmoothGrad: removing noise by adding noise", arXiv 2017

## Additional Resources

- Original VLG-CBM paper: https://arxiv.org/pdf/2408.01432
- Label-Free CBM paper: Oikarinen et al., 2023
- ChEX grounding model: https://arxiv.org/pdf/2404.15770

## Support

For issues or questions:
1. Check the examples in `notebooks/concept_saliency.ipynb`
2. Review the docstrings in `utils/saliency.py`
3. Run the standalone script with `--help` for all options
