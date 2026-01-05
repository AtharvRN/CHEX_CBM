# CHEX_CBM: CheXpert & COVID-QU Classification with Concept Bottleneck Models

End-to-end CXR classifiers plus CBM variants:
- Standard backbone fine-tuning (DenseNet-121/ResNet-50, XRV)
- **Label-Free CBM** (CLIP/XrayCLIP/ChexpertZero pseudo-labels)
- **VLG-CBM** (grounded concept annotations)

Supports CheXpert (multilabel) and COVID-QU (3-class, folder-based).

## Environments
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate atharv
pip install -r requirements.txt
```

## Datasets
- **CheXpert**: CSV-driven (`train.csv`/`valid.csv`) with 14 labels (5 competition / 12 pathology subsets).
- **COVID-QU**: Folder structure (Infection or Lung variants). No CSVs needed for training/eval/CBM; loader walks `{Train,Val,Test}/{COVID-19,Non-COVID,Normal}/images`.
- Concepts: CheXpert sets in `concepts/chexpert_concepts.{json,txt}`; COVID-QU IRR list in `concepts/covidqu_irrrag_concepts.txt`.

Helper (if you still want CSVs): `scripts/prepare_covidqu_csv.py --dataset_root /path/to/COVIDQU --variant infection --output_dir /path/to/COVIDQU`.

## Backbone Training (standard classifier)
CheXpert (multilabel, pathology subset):
```bash
python train.py --data_dir /workspace/CheXpert-v1.0-small \
  --label_set chexpert --pathology_labels \
  --model densenet121 --epochs 10 --batch_size 32 \
  --output checkpoints/chex_patho
```
COVID-QU (3-class, folder mode, infection variant):
```bash
python train.py --data_dir /workspace/datasets/COVIDQU \
  --label_set covidqu --covidqu_variant infection \
  --model densenet121 --epochs 10 --batch_size 64 \
  --output checkpoints/covidqu_backbone
```

## Evaluation (unified `eval.py`)
CheXpert validation:
```bash
python eval.py --data_dir /workspace/CheXpert-v1.0-small \
  --label_set chexpert --pathology_labels \
  --model densenet121 \
  --checkpoint checkpoints/chex_patho/best_model.pth \
  --split valid --output eval_results/chex_patho_val
```
COVID-QU test (accuracy only):
```bash
python eval.py --data_dir /workspace/datasets/COVIDQU \
  --label_set covidqu --covidqu_variant infection \
  --model densenet121 \
  --checkpoint checkpoints/covidqu_backbone/best_model.pth \
  --split test --output eval_results/covidqu_test
```

## Label-Free CBM (LF-CBM)
Uses CLIP-style similarity to create pseudo concept labels; trains a projection + sparse final layer.
COVID-QU:
```bash
python label_free_cbm.py \
  --data_dir /workspace/datasets/COVIDQU \
  --label_set covidqu --covidqu_variant infection \
  --concepts concepts/covidqu_irrrag_concepts.txt \
  --backbone densenet121 --clip_name xrayclip \
  --batch_size 64 --num_workers 8 \
  --output checkpoints/lfcbm_covidqu
```
Notes: `--use_data_parallel` for multi-GPU backbone; `--label_set chexpert` keeps CSV-based CheXpert flow.

## VLG-CBM
Requires concept annotations (ChEX or similar). CheXpert stays CSV-based; COVID-QU uses folder loader.
COVID-QU (after annotations exist at `annotations/covidqu_{train,val}`):
```bash
python vlg_cbm.py \
  --data_dir /workspace/datasets/COVIDQU \
  --label_set covidqu --covidqu_variant infection \
  --concepts concepts/covidqu_irrrag_concepts.txt \
  --annotation_dir annotations/covidqu_train \
  --val_annotation_dir annotations/covidqu_val \
  --backbone densenet121 \
  --output checkpoints/vlg_covidqu
```

## Generating Concept Annotations
`generate_annotations.py` can use TXT concept lists and folder-based COVID-QU (infection/lung). Still depends on ChEX.
Example (COVID-QU, infection variant):
```bash
python generate_annotations.py \
  --data_dir /workspace/datasets/COVIDQU \
  --label_set covidqu --covidqu_variant infection \
  --concepts concepts/covidqu_irrrag_concepts.txt \
  --output_dir annotations/covidqu_train \
  --split train --threshold 0.15
```
(Run again with `--split val` for validation.)

## Scripts & Files
- `train.py` — backbone training (CheXpert multilabel, COVID-QU single-label).
- `eval.py` — unified evaluator (AUROC/AP for CheXpert, accuracy for COVID-QU).
- `label_free_cbm.py` — LF-CBM training.
- `vlg_cbm.py` — VLG-CBM training.
- `generate_annotations.py` — ChEX-based concept annotation generation.
- `scripts/prepare_covidqu_csv.py` — optional COVID-QU CSV builder.
- `concepts/covidqu_irrrag_concepts.txt` — IRR concept set for COVID-QU.

## Notes
- COVID-QU uses folder loading by default (no CSV needed).
- VLG-CBM still requires concept annotations; supply `--annotation_dir/--val_annotation_dir`.
- For speed: use `--num_workers`, larger `--batch_size`, and `--use_data_parallel` where available. Mixed precision can be added if needed.

## External weights
- **CheXagent vision encoder (2-3b)**: downloaded to `~/models/chexagent` via `huggingface_hub.snapshot_download(repo_id="StanfordAIMI/CheXagent-2-3b", cache_dir="~/models/chexagent")`. Point any CheXagent-dependent scripts to that snapshot path.
