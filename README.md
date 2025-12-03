# CHEX_CBM: CheXpert Multi-Label Classification with Concept Bottleneck Models

End-to-end fine-tuning for chest X-ray classification on the CheXpert dataset, with support for **Concept Bottleneck Models (CBMs)** including:
- **Standard End-to-End Training**: DenseNet-121/ResNet-50 fine-tuning
- **Label-Free CBM**: CLIP-based concept bottleneck without annotations ([Oikarinen et al., 2023](https://arxiv.org/abs/2304.06129))
- **VLG-CBM**: Vision-Language Guided CBM with grounding ([Srivastava et al., 2024](https://arxiv.org/abs/2408.01432))

## Quick Start

```bash
# Activate environment
source /opt/conda/etc/profile.d/conda.sh && conda activate atharv

# Quick test run (10K samples)
python train.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --limit_samples 10000 \
    --epochs 3 \
    --output checkpoints/test_10k

# Full training
python train.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --epochs 10 \
    --use_pos_weight \
    --output checkpoints/full_run
```

See `commands.txt` for more training examples.

## Dataset

This project uses the **CheXpert-v1.0-small** dataset with 14 pathology labels:

| Label | Description |
|-------|-------------|
| No Finding | No pathology detected |
| Enlarged Cardiomediastinum | |
| Cardiomegaly | Enlarged heart |
| Lung Opacity | |
| Lung Lesion | |
| Edema | Pulmonary edema |
| Consolidation | |
| Pneumonia | |
| Atelectasis | Lung collapse |
| Pneumothorax | |
| Pleural Effusion | Fluid around lungs |
| Pleural Other | |
| Fracture | |
| Support Devices | Medical devices visible |

### Label Values

| Value | Meaning | Default Handling |
|-------|---------|------------------|
| 1.0 | Positive | → 1 |
| 0.0 | Negative | → 0 |
| -1.0 | Uncertain | → 1 (U-Ones) |
| NaN | Not mentioned | → 0 |

## Installation

```bash
cd CHEX_CBM
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train on all 14 labels with default settings
python train.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --output checkpoints/exp1

# Train on 5 competition labels only
python train.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --competition_labels \
    --output checkpoints/exp_competition

# Train with U-Zeros strategy for uncertain labels
python train.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --uncertain_strategy zeros \
    --output checkpoints/exp_uzeros

# Train with class-balanced loss
python train.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --use_pos_weight \
    --output checkpoints/exp_balanced

# Full example with all options
python train.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --model densenet121 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --img_size 224 \
    --uncertain_strategy ones \
    --use_pos_weight \
    --output checkpoints/full_exp
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py \
    --checkpoint checkpoints/exp1/best_model.pth \
    --data_dir /workspace/CheXpert-v1.0-small

# With ROC/PR curves and saved predictions
python evaluate.py \
    --checkpoint checkpoints/exp1/best_model.pth \
    --data_dir /workspace/CheXpert-v1.0-small \
    --plot_curves \
    --save_predictions
```

## Project Structure

```
CHEX_CBM/
├── train.py          # Training script
├── evaluate.py       # Evaluation script
├── dataset.py        # CheXpert dataset class
├── models.py         # Model architectures (DenseNet-121, ResNet-50)
├── requirements.txt  # Dependencies
├── data/
│   └── chexpert_classes.txt  # Label names
└── checkpoints/      # Saved models (created during training)
```

## Metrics

- **AUROC**: Area Under ROC Curve (primary metric)
- **AP**: Average Precision (mAP equivalent for multi-label)
- Per-class and macro-averaged metrics

## References

- [CheXpert Paper (Irvin et al., 2019)](https://arxiv.org/abs/1901.07031)
- [CheXNet (Rajpurkar et al., 2017)](https://arxiv.org/abs/1711.05225)

## Uncertainty Handling Strategies

Following the CheXpert paper:

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **U-Ones** | Map uncertain (-1) to positive (1) | Default, works well for Atelectasis, Edema |
| **U-Zeros** | Map uncertain (-1) to negative (0) | Conservative approach |
| **U-Ignore** | Mask uncertain labels in loss | Not implemented yet |

---

## Concept Bottleneck Models (CBMs)

This repository implements two CBM approaches that provide **interpretable** classification by first predicting human-understandable concepts, then using those concepts to make final predictions.

### Pathology Labels (12 classes)

For CBM training, we focus on 12 pathology labels (excluding "No Finding" and "Support Devices"):

| # | Label |
|---|-------|
| 1 | Enlarged Cardiomediastinum |
| 2 | Cardiomegaly |
| 3 | Lung Opacity |
| 4 | Lung Lesion |
| 5 | Edema |
| 6 | Consolidation |
| 7 | Pneumonia |
| 8 | Atelectasis |
| 9 | Pneumothorax |
| 10 | Pleural Effusion |
| 11 | Pleural Other |
| 12 | Fracture |

### Concepts

We use GPT-generated visual concepts (15 per class, 180 total) stored in:
- `concepts/chexpert_concepts.json` - Structured by class
- `concepts/chexpert_concepts.txt` - Flattened list for Label-Free CBM

---

## Label-Free CBM

**Paper**: [Label-Free Concept Bottleneck Models (Oikarinen et al., 2023)](https://arxiv.org/abs/2304.06129)

Label-Free CBM learns a concept bottleneck layer using CLIP embeddings without requiring concept annotations.

### Pipeline

1. **Save Activations**: Extract backbone (DenseNet/ResNet) features for all images
2. **Compute CLIP Features**: Get text embeddings for all concepts using BioMedCLIP
3. **Filter Concepts**: Remove concepts too similar to class names (CLIP cosine sim > cutoff)
4. **Learn Concept Projection**: Train projection matrix from backbone to concept space
5. **Train Sparse Classifier**: Use GLM-SAGA to learn sparse weights from concepts to labels

### Commands

```bash
# Activate environment
source /opt/conda/etc/profile.d/conda.sh && conda activate atharv
cd /workspace/CHEX_CBM

# Step 1: Full Label-Free CBM training (all steps in one command)
python label_free_cbm.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concept_file concepts/chexpert_concepts.txt \
    --backbone densenet121 \
    --clip_name biomedclip \
    --limit_samples 10000 \
    --batch_size 64 \
    --clip_cutoff 0.28 \
    --interpretability_cutoff 0.1 \
    --lam 0.0002 \
    --n_iters 2000 \
    --output saved_models/lf_cbm_10k \
    --seed 42

# Step 2: Evaluate on validation set
python label_free_cbm.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concept_file concepts/chexpert_concepts.txt \
    --backbone densenet121 \
    --clip_name biomedclip \
    --output saved_models/lf_cbm_10k \
    --eval_only

# Alternative: ResNet-50 backbone
python label_free_cbm.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --concept_file concepts/chexpert_concepts.txt \
    --backbone resnet50 \
    --clip_name biomedclip \
    --limit_samples 10000 \
    --output saved_models/lf_cbm_resnet_10k
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | densenet121 | Backbone model (densenet121 or resnet50) |
| `--clip_name` | biomedclip | CLIP model for concept embeddings |
| `--clip_cutoff` | 0.28 | Max cosine sim between concept and class name |
| `--interpretability_cutoff` | 0.1 | Keep concepts with proj weight > this |
| `--lam` | 0.0002 | Sparsity regularization for SAGA |
| `--n_iters` | 2000 | SAGA iterations |

### Output Structure

```
saved_models/lf_cbm_10k/
├── proj_layer.pth           # Concept projection matrix
├── final_layer.pth          # Sparse classifier weights
├── concepts_filtered.txt    # Concepts after CLIP filtering
├── concepts_selected.txt    # Final selected concepts
├── train_activations.pt     # Cached backbone features
├── val_activations.pt
└── results.json             # Metrics and config
```

---

## VLG-CBM (Vision-Language Guided CBM)

**Paper**: [VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance (Srivastava et al., 2024)](https://arxiv.org/abs/2408.01432)

VLG-CBM uses a grounding model (ChEX) to create pseudo-annotations for concepts, then trains a Concept Bottleneck Layer (CBL) to predict concept activations.

### Pipeline

1. **Generate Annotations**: Use ChEX to detect concepts in each image
2. **Filter Concepts**: Remove concepts with too few annotations
3. **Train CBL**: Train concept prediction layer on filtered concepts
4. **Train Final Layer**: Use GLM-SAGA for sparse classification

### Step 1: Generate ChEX Annotations

```bash
# Activate environment
source /opt/conda/etc/profile.d/conda.sh && conda activate atharv
cd /workspace/CHEX_CBM

# Generate annotations for training images
python generate_annotations.py \
    --concepts concepts/chexpert_concepts.json \
    --data_dir /workspace/CheXpert-v1.0-small \
    --chex_ckpt /workspace/chex_models/chex_stage3/checkpoints/last.ckpt \
    --chex_config /workspace/chex/conf/chex_stage3.yaml \
    --output_dir annotations/train \
    --split train \
    --threshold 0.3 \
    --max_images 10000 \
    --batch_size 1

# Generate annotations for validation images
python generate_annotations.py \
    --concepts concepts/chexpert_concepts.json \
    --data_dir /workspace/CheXpert-v1.0-small \
    --chex_ckpt /workspace/chex_models/chex_stage3/checkpoints/last.ckpt \
    --chex_config /workspace/chex/conf/chex_stage3.yaml \
    --output_dir annotations/valid \
    --split valid \
    --threshold 0.3 \
    --batch_size 1
```

### Step 2: Train VLG-CBM

```bash
# Train VLG-CBM with annotations
python vlg_cbm.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --annotation_dir annotations/train \
    --val_annotation_dir annotations/valid \
    --concepts concepts/chexpert_concepts.json \
    --backbone densenet121 \
    --cbl_epochs 20 \
    --cbl_lr 1e-4 \
    --min_concept_freq 0.01 \
    --saga_lam 0.001 \
    --saga_iters 2000 \
    --limit_samples 10000 \
    --output saved_models/vlg_cbm_10k \
    --seed 42

# Train dense (non-sparse) version
python vlg_cbm.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --annotation_dir annotations/train \
    --val_annotation_dir annotations/valid \
    --concepts concepts/chexpert_concepts.json \
    --backbone densenet121 \
    --cbl_epochs 20 \
    --dense \
    --output saved_models/vlg_cbm_dense_10k
```

### Step 3: Evaluate

```bash
# Evaluate trained model
python vlg_cbm.py \
    --data_dir /workspace/CheXpert-v1.0-small \
    --annotation_dir annotations/train \
    --val_annotation_dir annotations/valid \
    --concepts concepts/chexpert_concepts.json \
    --output saved_models/vlg_cbm_10k \
    --eval_only
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | densenet121 | Backbone model (densenet121 or resnet50) |
| `--cbl_epochs` | 20 | Epochs for CBL training |
| `--cbl_lr` | 1e-4 | Learning rate for CBL |
| `--min_concept_freq` | 0.01 | Min frequency to keep concept (1%) |
| `--saga_lam` | 0.001 | Sparsity regularization |
| `--dense` | False | Use dense final layer instead of sparse |

### Output Structure

```
saved_models/vlg_cbm_10k/
├── backbone.pth             # Fine-tuned backbone (if training)
├── cbl.pth                  # Concept Bottleneck Layer
├── final_layer.pth          # Sparse classifier weights
├── concepts_filtered.json   # Concepts after frequency filtering
├── concept_to_idx.json      # Concept index mapping
└── results.json             # Metrics and config

annotations/train/
├── patient12345_study1_view1.json  # Per-image annotations
├── ...
└── summary.json             # Annotation statistics
```

---

## Project Structure

```
CHEX_CBM/
├── train.py              # Standard end-to-end training
├── evaluate.py           # Evaluation script
├── dataset.py            # CheXpert dataset class
├── models.py             # Model architectures
├── label_free_cbm.py     # Label-Free CBM implementation
├── generate_annotations.py  # ChEX annotation generation
├── vlg_cbm.py            # VLG-CBM training
├── requirements.txt      # Dependencies
├── commands.txt          # Example commands
├── concepts/
│   ├── chexpert_concepts.json   # GPT concepts (structured)
│   └── chexpert_concepts.txt    # GPT concepts (flattened)
├── data/
│   └── chexpert_classes.txt
├── checkpoints/          # Saved standard models
├── saved_models/         # Saved CBM models
├── annotations/          # ChEX annotations for VLG-CBM
└── glm_saga/            # Sparse regression solver
```

---

## Comparison of Methods

| Method | Annotations | Interpretability | Key Components |
|--------|-------------|------------------|----------------|
| **Standard** | None | Low | DenseNet/ResNet → Linear |
| **Label-Free CBM** | None | Medium | Backbone → CLIP Projection → Sparse |
| **VLG-CBM** | ChEX pseudo-labels | High | Backbone → CBL → Sparse |

---

## References

### Datasets
- [CheXpert Paper (Irvin et al., 2019)](https://arxiv.org/abs/1901.07031)
- [CheXNet (Rajpurkar et al., 2017)](https://arxiv.org/abs/1711.05225)

### CBM Methods
- [Label-Free Concept Bottleneck Models (Oikarinen et al., 2023)](https://arxiv.org/abs/2304.06129)
- [VLG-CBM (Srivastava et al., 2024)](https://arxiv.org/abs/2408.01432)

### Grounding
- [ChEX: Interactive Localization and Region Description in Radiology (Boecking et al., 2024)](https://arxiv.org/abs/2404.15770)