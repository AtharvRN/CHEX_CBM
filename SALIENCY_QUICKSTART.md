# Saliency Visualization Quick Reference

## Installation Check

```bash
# Test that everything is working
python test_saliency.py
```

## Common Use Cases

### 1. Visualize Top Concepts from Trained VLG-CBM

```bash
python visualize_concepts.py \
    --model_type vlg_cbm \
    --model_dir checkpoints/vlg_cbm_10k \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concepts concepts/chexpert_concepts.txt \
    --pathology_labels \
    --output visualizations/top_concepts \
    --num_samples 20
```

### 2. Visualize Specific Concepts

```bash
python visualize_concepts.py \
    --model_type vlg_cbm \
    --model_dir checkpoints/vlg_cbm_10k \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concepts concepts/chexpert_concepts.txt \
    --concept_indices 0 5 10 15 20 25 \
    --output visualizations/specific_concepts \
    --num_samples 10
```

### 3. Disease Attribution (Most Important!)

Shows which concepts contribute to each disease prediction:

```bash
# For Atelectasis (disease_idx=0)
python visualize_concepts.py \
    --model_type vlg_cbm \
    --model_dir checkpoints/vlg_cbm_10k \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concepts concepts/chexpert_concepts.txt \
    --pathology_labels \
    --disease_attribution \
    --disease_idx 0 \
    --top_k_concepts 10 \
    --output visualizations/atelectasis_attribution \
    --num_samples 20
```

Disease indices for pathology labels:
- 0: Enlarged Cardiomediastinum
- 1: Cardiomegaly
- 2: Lung Opacity
- 3: Lung Lesion
- 4: Edema
- 5: Consolidation
- 6: Pneumonia
- 7: Atelectasis
- 8: Pneumothorax
- 9: Pleural Effusion
- 10: Pleural Other
- 11: Fracture

### 4. Compare Saliency Methods

```bash
# GradCAM (fastest)
python visualize_concepts.py \
    --model_type vlg_cbm \
    --model_dir checkpoints/vlg_cbm_10k \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concepts concepts/chexpert_concepts.txt \
    --method gradcam \
    --output visualizations/gradcam

# Integrated Gradients (most trustworthy)
python visualize_concepts.py \
    --model_type vlg_cbm \
    --model_dir checkpoints/vlg_cbm_10k \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concepts concepts/chexpert_concepts.txt \
    --method integrated_gradients \
    --output visualizations/integrated_gradients

# SmoothGrad (cleanest)
python visualize_concepts.py \
    --model_type vlg_cbm \
    --model_dir checkpoints/vlg_cbm_10k \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concepts concepts/chexpert_concepts.txt \
    --method smoothgrad \
    --output visualizations/smoothgrad
```

### 5. Label-Free CBM

```bash
python visualize_concepts.py \
    --model_type lf_cbm \
    --model_dir saved_models/lf_cbm_10k \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concepts concepts/chexpert_concepts.txt \
    --pathology_labels \
    --output visualizations/lf_cbm \
    --num_samples 20
```

### 6. COVID-QU Dataset

```bash
python visualize_concepts.py \
    --model_type vlg_cbm \
    --model_dir checkpoints/vlg_covidqu \
    --data_dir /workspace/datasets/COVIDQU \
    --label_set covidqu \
    --covidqu_variant infection \
    --concepts concepts/covidqu_irrrag_concepts.txt \
    --disease_attribution \
    --disease_idx 0 \
    --output visualizations/covidqu_covid19 \
    --num_samples 20
```

COVID-QU disease indices:
- 0: COVID-19
- 1: Non-COVID
- 2: Normal

## Python API - Minimal Example

```python
import torch
from utils.saliency import GradCAM
from models import get_model
from vlg_cbm_lib.datasets import ConceptLayer, BackboneWithConcepts

# Load model (replace with your actual loading code)
model = ...  # Your trained CBM model
model.eval()

# Load image
image = torch.randn(1, 3, 224, 224)  # Replace with actual image

# Generate saliency
gradcam = GradCAM(model.backbone)
heatmap = gradcam.generate_cam(
    input_tensor=image,
    concept_layer=model.concept_layer,
    concept_idx=5
)

# Save
from utils.saliency import save_saliency_visualization
save_saliency_visualization(
    image, heatmap, "concept5.png",
    concept_name="Air bronchograms"
)
```

## Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook notebooks/concept_saliency.ipynb
```

## Output Files

Each run creates:
- `sampleN_conceptM_METHOD.png` - Individual concept visualizations
- `sampleN_diseaseM_attribution.png` - Disease attribution plots

## Tips

1. **Start with GradCAM** - fastest for initial exploration
2. **Use disease attribution** - most clinically relevant
3. **Validate with Integrated Gradients** - for publication
4. **Process in batches** - avoid memory issues with large datasets
5. **Compare methods** - different methods reveal different aspects

## Common Options

```bash
--num_samples 20              # Number of images to process
--batch_size 1                # Keep at 1 for visualization
--method gradcam              # gradcam | integrated_gradients | smoothgrad
--top_k_concepts 10           # For disease attribution
--device cuda                 # cuda | cpu
--split valid                 # train | valid | test
```

## Troubleshooting

**Issue**: No gradients flowing
```bash
# Check model is in eval mode and requires_grad is True
```

**Issue**: CUDA out of memory
```bash
# Use CPU or reduce batch_size
--device cpu --batch_size 1
```

**Issue**: Can't find model files
```bash
# Verify paths to model_dir, data_dir, and concepts file
ls -la checkpoints/vlg_cbm_10k/
```

## Full Documentation

See `docs/SALIENCY_GUIDE.md` for complete documentation.
