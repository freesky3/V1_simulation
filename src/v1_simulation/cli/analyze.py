from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from v1_simulation.analysis import (
    load_analysis_inputs_from_run,
    run_analysis,
    write_analysis_result_artifacts,
)
from v1_simulation.cli import load_cli_config, merge_overrides


def analyze_command(
    run_dir: Path = typer.Argument(..., help="Simulation run directory containing responses_exc.npy."),
    overrides: Optional[List[str]] = typer.Argument(
        None,
        help="Hydra overrides for analysis settings, for example: analysis.osi_threshold=0.3",
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
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Directory for analysis artifacts. Defaults to <run_dir>/analysis.",
    ),
    network_path: Optional[Path] = typer.Option(
        None,
        "--network-path",
        help="Network checkpoint directory. Defaults to <run_dir>/network.",
    ),
) -> None:
    """Analyze a saved simulation run and write metrics/artifacts."""

    cfg = load_cli_config(
        config_path=config_path,
        config_name=config_name,
        overrides=merge_overrides(overrides, override),
    )
    inputs = load_analysis_inputs_from_run(run_dir, network_path=network_path)
    result = run_analysis(cfg.analysis, inputs)
    target = write_analysis_result_artifacts(result, output_dir or (run_dir / "analysis"))

    typer.echo(f"Saved analysis: {target}")
    typer.echo(
        "Analysis summary: "
        f"status={result.status}, "
        f"selected={result.selected_indices.size}, "
        f"ensembles={0 if result.communities is None else result.communities.n_ensembles}"
    )


__all__ = ["analyze_command", "load_analysis_inputs_from_run", "write_analysis_result_artifacts"]
