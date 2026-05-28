from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from v1_simulation.cli import load_cli_config, merge_overrides
from v1_simulation.config import validate_config
from v1_simulation.simulation import run_simulation


def run_command(
    overrides: Optional[List[str]] = typer.Argument(
        None,
        help="Hydra overrides, for example: +experiment=smoke simulation.duration_tau_e=2",
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
        help="Output root for run artifacts. Overrides paths.run_root.",
    ),
    job_name: Optional[str] = typer.Option(
        None,
        "--job-name",
        help="Job folder name under run_root. Overrides job_name.",
    ),
    trained_network: Optional[Path] = typer.Option(
        None,
        "--trained-network",
        help="Checkpoint directory or legacy NPZ to use as model.trained_network_path.",
    ),
) -> None:
    """Run one drifting-grating simulation and persist artifacts."""

    cfg = load_cli_config(
        config_path=config_path,
        config_name=config_name,
        overrides=merge_overrides(overrides, override),
    )
    cfg.mode = "simulate"
    if run_root is not None:
        cfg.paths.run_root = run_root
    if job_name is not None:
        cfg.job_name = job_name
    if trained_network is not None:
        cfg.model.trained_network_path = str(trained_network)

    validate_config(cfg)
    saved = run_simulation(cfg)
    result = saved.result
    typer.echo(f"Saved simulation: {saved.run_dir}")
    typer.echo(
        "Shapes: "
        f"exc={tuple(result.ode.exc.shape)}, "
        f"inh={tuple(result.ode.inh.shape)}, "
        f"time_steps={result.time.size}, "
        f"orientations={result.theta_angles.size}"
    )


__all__ = ["run_command"]
