from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from v1_simulation.cli import has_group_override, load_cli_config, merge_overrides
from v1_simulation.config import validate_config
from v1_simulation.simulation import run_bcm_training


def train_command(
    overrides: Optional[List[str]] = typer.Argument(
        None,
        help="Hydra overrides. The bcm_train experiment is added by default unless experiment is set.",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config-path",
        help="Directory containing Hydra YAML configs. Defaults to the project configs directory.",
    ),
    config_name: str = typer.Option("config", "--config-name", help="Root config name without .yaml."),
    override: Optional[List[str]] = typer.Option(
        None,
        "--override",
        "-o",
        help="Hydra override. Can be repeated; positional overrides are also accepted.",
    ),
    run_root: Optional[Path] = typer.Option(
        None,
        "--run-root",
        help="Output root for training artifacts. Overrides paths.run_root.",
    ),
    job_name: Optional[str] = typer.Option(
        None,
        "--job-name",
        help="Job folder name under run_root. Overrides job_name.",
    ),
    natural_image_dir: Optional[Path] = typer.Option(
        None,
        "--natural-image-dir",
        help="Directory containing Van Hateren .iml files. Overrides training.natural_image.dir.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Maximum number of natural-image files per epoch.",
    ),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="BCM training epochs."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Natural-image samples per batch."),
    eta: Optional[float] = typer.Option(None, "--eta", help="BCM learning rate."),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show tqdm progress bars."),
) -> None:
    """Run natural-image BCM training and persist checkpoints/logs."""

    cli_overrides = merge_overrides(overrides, override)
    if not has_group_override(cli_overrides, "experiment"):
        cli_overrides = ["+experiment=bcm_train", *cli_overrides]

    cfg = load_cli_config(config_path=config_path, config_name=config_name, overrides=cli_overrides)
    cfg.mode = "train"
    cfg.training.enabled = True
    if run_root is not None:
        cfg.paths.run_root = run_root
    if job_name is not None:
        cfg.job_name = job_name
    if natural_image_dir is not None:
        cfg.training.natural_image.dir = str(natural_image_dir)
    if limit is not None:
        cfg.training.natural_image.limit = int(limit)
    if epochs is not None:
        cfg.training.bcm.epochs = int(epochs)
    if batch_size is not None:
        cfg.training.bcm.batch_size = int(batch_size)
    if eta is not None:
        cfg.training.bcm.eta = float(eta)

    validate_config(cfg)
    result = run_bcm_training(cfg, show_progress=progress)
    typer.echo(f"Saved training run: {result.run_dir}")
    typer.echo(
        "Training summary: "
        f"steps={result.steps}, "
        f"samples_seen={result.samples_seen}, "
        f"images_seen={result.images_seen}"
    )


__all__ = ["train_command"]
