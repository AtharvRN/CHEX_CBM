import argparse

from models import XRV_WEIGHTS

AVAILABLE_BACKBONES = ["densenet121", "resnet50"] + list(XRV_WEIGHTS.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="VLG-CBM for CheXpert")
    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to CheXpert-v1.0-small directory")
    parser.add_argument("--concepts", type=str,
                        default="concepts/chexpert_concepts.json",
                        help="Path to concepts file (JSON or TXT)")
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default="annotations/train_chex_stage3",
        help="Directory with training annotations (default: annotations/train_chex_stage3)"
    )
    parser.add_argument(
        "--val_annotation_dir",
        type=str,
        default="annotations/val_chex_stage3",
        help="Directory with validation annotations (default: annotations/val_chex_stage3)"
    )
    parser.add_argument("--competition_labels", action="store_true",
                        help="Use only 5 competition labels")
    parser.add_argument("--uncertain_strategy", type=str, default="ones",
                        choices=["ones", "zeros"])
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--limit_strategy", type=str, default="sequential",
                        choices=["random", "sequential"],
                        help="How to pick samples when limiting the training set")
    parser.add_argument("--seed", type=int, default=42)

    # Models
    parser.add_argument("--backbone", type=str, default="densenet121",
                        choices=AVAILABLE_BACKBONES)
    parser.add_argument("--backbone_ckpt", type=str, default=None,
                        help="Path to finetuned backbone checkpoint")

    # Concept filtering
    parser.add_argument("--confidence_threshold", type=float, default=0.15,
                        help="Threshold for considering a concept detected")
    parser.add_argument("--min_concept_freq", type=float, default=0.01,
                        help="Min frequency for keeping a concept")
    parser.add_argument("--max_concept_freq", type=float, default=0.95,
                        help="Max frequency for keeping a concept")

    # CBL training
    parser.add_argument("--cbl_epochs", type=int, default=20,
                        help="Epochs for concept layer training")
    parser.add_argument("--cbl_lr", type=float, default=5e-4,
                        help="Learning rate for CBL")
    parser.add_argument("--cbl_batch_size", type=int, default=32)
    parser.add_argument("--cbl_hidden_layers", type=int, default=1,
                        help="Hidden layers in concept projection")
    parser.add_argument("--crop_to_concept_prob", type=float, default=0.0,
                        help="Probability of cropping to concept box during training")
    parser.add_argument("--cbl_finetune_backbone", action="store_true",
                        help="Finetune backbone during CBL training")
    parser.add_argument(
        "--resume_concept_layer",
        type=str,
        default=None,
        help="Path to a pre-trained concept_layer.pt to skip CBL training"
    )

    # Final layer
    parser.add_argument("--use_saga", action="store_true", default=True,
                        help="Use sparse SAGA solver for final layer")
    parser.add_argument("--saga_lam", type=float, default=0.0007,
                        help="Sparsity regularization")
    parser.add_argument("--saga_iters", type=int, default=2000)
    parser.add_argument("--saga_max_lr", type=float, default=0.1,
                        help="Initial learning rate for the SAGA solver")
    parser.add_argument("--saga_batch_size", type=int, default=256)
    parser.add_argument("--dense_lr", type=float, default=1e-3,
                        help="Learning rate for dense final layer")
    parser.add_argument("--dense_epochs", type=int, default=100)
    parser.add_argument(
        "--pre_eval_backbone",
        action="store_true",
        help="Evaluate the backbone classifier before CBM training"
    )

    # Output
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--concept_cache", type=str, default=None,
                        help="Path to precomputed annotation cache for training")
    parser.add_argument("--val_concept_cache", type=str, default=None,
                        help="Path to precomputed annotation cache for validation")

    return parser.parse_args()
