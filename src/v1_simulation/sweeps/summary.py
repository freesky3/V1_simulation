from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from v1_simulation.io.artifacts import json_ready


def write_sweep_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write sweep rows to CSV with a stable, unioned header."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fieldnames = ordered_fieldnames(materialized)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return target


def write_sweep_summary(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write a compact JSON summary for a completed or partial sweep."""

    materialized = [dict(row) for row in rows]
    completed = [row for row in materialized if row.get("status") == "completed"]
    failed = [row for row in materialized if row.get("status") == "failed"]
    best = best_row(completed)
    payload = {
        "total_rows": len(materialized),
        "completed": len(completed),
        "failed": len(failed),
        "best": best,
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)
    return target


def read_sweep_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read existing sweep rows if the CSV exists."""

    source = Path(path)
    if not source.exists():
        return []
    with source.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_completed_parameter_keys(path: str | Path) -> set[str]:
    """Return parameter keys already completed in an existing CSV."""

    return {
        row["parameter_key"]
        for row in read_sweep_rows(path)
        if row.get("status") == "completed" and row.get("parameter_key")
    }


def best_row(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return the completed row with the highest finite score."""

    best: dict[str, Any] | None = None
    best_score = -math.inf
    for row in rows:
        score = _as_float(row.get("score"))
        if score is None or score <= best_score:
            continue
        best_score = score
        best = dict(row)
    return best


def ordered_fieldnames(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "trial",
        "status",
        "analysis_status",
        "score",
        "parameter_key",
        "overrides",
        "run_dir",
        "analysis_dir",
        "error",
        "selected_neurons",
        "n_ensembles",
        "classified_fraction",
        "active_fraction",
        "silent_fraction",
        "rate_mean",
        "rate_max",
        "osi_mean",
        "osi_median",
    ]
    present = {str(key) for row in rows for key in row}
    param_keys = sorted(key for key in present if key.startswith("param."))
    ordered = [key for key in preferred if key in present]
    tail = sorted(present - set(ordered) - set(param_keys))
    return ordered + param_keys + tail


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _csv_value(value: Any) -> Any:
    value = json_ready(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value


__all__ = [
    "best_row",
    "ordered_fieldnames",
    "read_completed_parameter_keys",
    "read_sweep_rows",
    "write_sweep_csv",
    "write_sweep_summary",
]
