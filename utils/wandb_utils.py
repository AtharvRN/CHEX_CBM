"""WandB helper utilities used during training."""

from __future__ import annotations

from typing import Dict, List, Optional

from utils.plotting import (
    plot_confusion_heatmap,
    plot_per_class_auroc,
    plot_pr_curves,
    plot_roc_curves,
    plot_training_curves,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None
    WANDB_AVAILABLE = False


def init_wandb(args, config) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Initialize a wandb run with consistent naming."""
    if not args.use_wandb or not WANDB_AVAILABLE:
        if args.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Skipping wandb logging.")
        return None

    run_name = args.wandb_run_name
    if run_name is None:
        run_name = f"{args.model}_{args.uncertain_strategy}"
        if args.competition_labels:
            run_name += "_comp"
        run_name += f"_lr{args.lr}_bs{args.batch_size}"
        if args.limit_samples:
            run_name += f"_n{args.limit_samples}"

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
        tags=[args.model, args.uncertain_strategy, "competition" if args.competition_labels else "all_labels"],
        reinit=True,
    )
    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    return run


def log_to_wandb(
    epoch: int,
    train_loss: float,
    val_aurocs: Dict[str, float],
    val_aps: Dict[str, float],
    lr: float,
    labels: List[str],
    best_auroc: float,
) -> None:
    """Push scalar metrics for an epoch to wandb."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    log_dict = {
        "epoch": epoch,
        "train/loss": train_loss,
        "val/mean_auroc": val_aurocs.get("mean", float("nan")),
        "val/mean_ap": val_aps.get("mean", float("nan")),
        "train/learning_rate": lr,
        "val/best_auroc": best_auroc,
    }
    for label in labels:
        log_dict[f"val/auroc/{label}"] = val_aurocs.get(label, float("nan"))
        log_dict[f"val/ap/{label}"] = val_aps.get(label, float("nan"))
    wandb.log(log_dict)


def log_plots_to_wandb(history, aurocs, labels, targets, predictions, output_dir) -> None:
    """Log final diagnostic plots."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    training_curves_path = plot_training_curves(history, output_dir)
    wandb.log({"charts/training_curves": wandb.Image(training_curves_path)})

    auroc_bar_path = plot_per_class_auroc(aurocs, labels, output_dir)
    wandb.log({"charts/per_class_auroc": wandb.Image(auroc_bar_path)})

    roc_path = plot_roc_curves(targets, predictions, labels, output_dir)
    wandb.log({"charts/roc_curves": wandb.Image(roc_path)})

    pr_path = plot_pr_curves(targets, predictions, labels, output_dir)
    wandb.log({"charts/pr_curves": wandb.Image(pr_path)})

    heatmap_path = plot_confusion_heatmap(targets, predictions, labels, output_dir)
    wandb.log({"charts/metrics_heatmap": wandb.Image(heatmap_path)})


__all__ = ["WANDB_AVAILABLE", "init_wandb", "log_to_wandb", "log_plots_to_wandb", "wandb"]
