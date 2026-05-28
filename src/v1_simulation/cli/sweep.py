from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from v1_simulation.cli import load_cli_config, merge_overrides
from v1_simulation.sweeps import expand_grid, run_grid_sweep


def sweep_command(
    overrides: Optional[List[str]] = typer.Argument(
        None,
        help="Hydra overrides used for the base config before expanding sweep.grid.",
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
    output_csv: Optional[Path] = typer.Option(
        None,
        "--output-csv",
        help="Sweep CSV path. Defaults to sweep.output_csv or <run_root>/sweeps/grid.csv.",
    ),
    run_root: Optional[Path] = typer.Option(
        None,
        "--run-root",
        help="Output root for simulation artifacts in each sweep trial.",
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        "--max-workers",
        help="Number of parallel worker threads. Defaults to sweep.max_workers.",
    ),
    resume: Optional[bool] = typer.Option(
        None,
        "--resume/--no-resume",
        help="Skip parameter combinations already completed in output_csv.",
    ),
    job_name_prefix: Optional[str] = typer.Option(
        None,
        "--job-name-prefix",
        help="Prefix for per-trial job names. Defaults to job_name from the config.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print expanded overrides without running simulations.",
    ),
) -> None:
    """Run a parameter sweep using the configured sweep.grid."""

    cli_overrides = merge_overrides(overrides, override)
    if dry_run:
        cfg = load_cli_config(config_path=config_path, config_name=config_name, overrides=cli_overrides)
        points = expand_grid(cfg.sweep.grid)
        typer.echo(f"Expanded {len(points)} sweep point(s):")
        for point in points:
            typer.echo(f"[{point.index}] {' '.join(point.overrides)}")
        return

    result = run_grid_sweep(
        config_path=config_path,
        config_name=config_name,
        base_overrides=cli_overrides,
        output_csv=output_csv,
        run_root=run_root,
        max_workers=max_workers,
        resume=resume,
        job_name_prefix=job_name_prefix,
    )
    typer.echo(f"Saved sweep CSV: {result.output_csv}")
    typer.echo(f"Saved sweep summary: {result.summary_json}")
    typer.echo(f"Rows: {len(result.rows)} (skipped={result.skipped})")


__all__ = ["sweep_command"]
