from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


_FLAT_STIMULUS_ALIASES = {
    "sigma": "gabor.sigma",
    "gamma": "gabor.gamma",
    "spatial_frequency": "gabor.spatial_frequency",
    "k": "gabor.spatial_frequency",
    "phase": "gabor.phase",
    "psi": "gabor.phase",
}


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One concrete parameter combination in a grid sweep."""

    index: int
    parameters: dict[str, Any]

    @property
    def overrides(self) -> list[str]:
        return parameters_to_overrides(self.parameters)

    @property
    def key(self) -> str:
        return parameter_key(self.parameters)


def expand_grid(grid: Mapping[str, Iterable[Any] | Any]) -> list[SweepPoint]:
    """Expand a mapping of parameter paths to candidate values into sweep points."""

    if not grid:
        return [SweepPoint(index=0, parameters={})]

    keys = [normalize_sweep_key(key) for key in grid]
    value_lists = [_as_value_list(value) for value in grid.values()]
    points: list[SweepPoint] = []
    for index, combo in enumerate(itertools.product(*value_lists)):
        points.append(SweepPoint(index=index, parameters=dict(zip(keys, combo))))
    return points


def grid_size(grid: Mapping[str, Iterable[Any] | Any]) -> int:
    """Return the number of combinations represented by a grid mapping."""

    size = 1
    for value in grid.values():
        size *= len(_as_value_list(value))
    return size


def parameters_to_overrides(parameters: Mapping[str, Any]) -> list[str]:
    """Convert a parameter mapping to Hydra dot-list override strings."""

    return [
        f"{normalize_sweep_key(key)}={format_override_value(value)}"
        for key, value in parameters.items()
    ]


def parameter_key(parameters: Mapping[str, Any]) -> str:
    """Build a stable JSON key for resume/deduplication."""

    normalized = {
        normalize_sweep_key(key): _json_scalar(value)
        for key, value in sorted(parameters.items(), key=lambda item: normalize_sweep_key(item[0]))
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def normalize_sweep_key(key: str) -> str:
    """Normalize legacy sweep paths to the current structured config schema."""

    normalized = str(key).strip()
    if normalized.startswith("model.stimulus."):
        normalized = "stimulus." + normalized[len("model.stimulus.") :]

    if normalized.startswith("stimulus."):
        suffix = normalized[len("stimulus.") :]
        alias = _FLAT_STIMULUS_ALIASES.get(suffix)
        if alias is not None:
            return "stimulus." + alias
    return normalized


def format_override_value(value: Any) -> str:
    """Format Python values for Hydra dot-list overrides."""

    value = _json_scalar(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _as_value_list(value: Iterable[Any] | Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return [_json_scalar(item) for item in value.tolist()]
    if isinstance(value, (str, bytes, Path)) or value is None:
        return [_json_scalar(value)]
    if isinstance(value, Iterable):
        return [_json_scalar(item) for item in value]
    return [_json_scalar(value)]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "SweepPoint",
    "expand_grid",
    "format_override_value",
    "grid_size",
    "normalize_sweep_key",
    "parameter_key",
    "parameters_to_overrides",
]
