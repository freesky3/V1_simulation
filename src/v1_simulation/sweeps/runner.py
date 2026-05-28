from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from v1_simulation.analysis import run_analysis, write_analysis_result_artifacts
from v1_simulation.config import RootConfig, load_config, validate_config
from v1_simulation.simulation import run_simulation
from v1_simulation.sweeps.grid import SweepPoint, expand_grid
from v1_simulation.sweeps.scoring import score_analysis_result
from v1_simulation.sweeps.summary import (
    read_completed_parameter_keys,
    read_sweep_rows,
    write_sweep_csv,
    write_sweep_summary,
)


@dataclass(frozen=True, slots=True)
class SweepRunResult:
    """Filesystem outputs and rows from a sweep run."""

    rows: list[dict[str, object]]
    output_csv: Path
    summary_json: Path
    skipped: int = 0


def run_grid_sweep(
    *,
    config_path: str | Path | None = None,
    config_name: str = "config",
    base_overrides: Iterable[str] | None = None,
    output_csv: str | Path | None = None,
    run_root: str | Path | None = None,
    max_workers: int | None = None,
    resume: bool | None = None,
    job_name_prefix: str | None = None,
) -> SweepRunResult:
    """Run a configured parameter grid through simulation, analysis, and scoring."""

    base_override_list = list(base_overrides or [])
    base_cfg = load_config(
        config_path=None if config_path is None else str(config_path),
        config_name=config_name,
        overrides=base_override_list,
    )
    target_csv = Path(output_csv or base_cfg.sweep.output_csv or (Path(base_cfg.paths.run_root) / "sweeps" / "grid.csv"))
    resume_enabled = bool(base_cfg.sweep.resume if resume is None else resume)
    workers = int(max_workers if max_workers is not None else base_cfg.sweep.max_workers)
    workers = max(1, workers)

    points = expand_grid(base_cfg.sweep.grid)
    completed_keys = read_completed_parameter_keys(target_csv) if resume_enabled else set()
    existing_rows = read_sweep_rows(target_csv) if resume_enabled else []
    pending = [point for point in points if point.key not in completed_keys]

    configs = [
        _load_trial_config(
            point,
            config_path=config_path,
            config_name=config_name,
            base_overrides=base_override_list,
            run_root=run_root,
            job_name_prefix=job_name_prefix,
        )
        for point in pending
    ]

    new_rows = _run_trials(pending, configs, max_workers=workers)
    rows: list[dict[str, object]] = [dict(row) for row in existing_rows]
    rows.extend(new_rows)
    write_sweep_csv(rows, target_csv)
    summary_json = write_sweep_summary(rows, target_csv.with_suffix(".summary.json"))
    return SweepRunResult(rows=rows, output_csv=target_csv, summary_json=summary_json, skipped=len(points) - len(pending))


def run_sweep_trial(point: SweepPoint, cfg: RootConfig) -> dict[str, object]:
    """Run one sweep point and return a flat CSV-ready result row."""

    row = _base_row(point)
    try:
        validate_config(cfg)
        saved = run_simulation(cfg)
        analysis = run_analysis(cfg.analysis, saved.result.analysis_inputs())
        analysis_dir = write_analysis_result_artifacts(analysis, saved.run_dir / "analysis")
        row.update(score_analysis_result(analysis))
        row.update(
            {
                "status": "completed",
                "run_dir": str(saved.run_dir),
                "analysis_dir": str(analysis_dir),
                "error": None,
            }
        )
    except Exception as exc:  # pragma: no cover - exercised by integration failures.
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return row


def _run_trials(points: list[SweepPoint], configs: list[RootConfig], *, max_workers: int) -> list[dict[str, object]]:
    if max_workers <= 1 or len(points) <= 1:
        return [run_sweep_trial(point, cfg) for point, cfg in zip(points, configs)]

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(run_sweep_trial, point, cfg): point.index
            for point, cfg in zip(points, configs)
        }
        for future in as_completed(future_to_index):
            rows.append(future.result())
    rows.sort(key=lambda row: int(row["trial"]))
    return rows


def _load_trial_config(
    point: SweepPoint,
    *,
    config_path: str | Path | None,
    config_name: str,
    base_overrides: list[str],
    run_root: str | Path | None,
    job_name_prefix: str | None,
) -> RootConfig:
    cfg = load_config(
        config_path=None if config_path is None else str(config_path),
        config_name=config_name,
        overrides=[*base_overrides, *point.overrides],
    )
    cfg.mode = "simulate"
    if run_root is not None:
        cfg.paths.run_root = Path(run_root)
    prefix = job_name_prefix or cfg.job_name
    cfg.job_name = f"{prefix}_sweep_{point.index:04d}"
    return cfg


def _base_row(point: SweepPoint) -> dict[str, object]:
    row: dict[str, object] = {
        "trial": point.index,
        "status": "pending",
        "parameter_key": point.key,
        "overrides": " ".join(point.overrides),
    }
    for key, value in point.parameters.items():
        row[f"param.{key}"] = value
    return row


__all__ = ["SweepRunResult", "run_grid_sweep", "run_sweep_trial"]
