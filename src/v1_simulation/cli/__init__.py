from __future__ import annotations

from pathlib import Path
from typing import Iterable

from v1_simulation.config import RootConfig, load_config


def merge_overrides(
    argument_overrides: Iterable[str] | None = None,
    option_overrides: Iterable[str] | None = None,
) -> list[str]:
    """Merge repeated ``--override`` values and positional Hydra overrides."""

    merged: list[str] = []
    if option_overrides:
        merged.extend(str(item) for item in option_overrides)
    if argument_overrides:
        merged.extend(str(item) for item in argument_overrides)
    return merged


def has_group_override(overrides: Iterable[str], group: str) -> bool:
    """Return true when a Hydra override selects or appends a config group."""

    prefix = f"{group}="
    dotted_prefix = f"{group}."
    for raw in overrides:
        item = str(raw).strip()
        while item.startswith("+"):
            item = item[1:]
        if item.startswith(prefix) or item.startswith(dotted_prefix):
            return True
    return False


def load_cli_config(
    *,
    config_path: Path | None = None,
    config_name: str = "config",
    overrides: Iterable[str] | None = None,
) -> RootConfig:
    """Load a typed project config from CLI path/options."""

    return load_config(
        config_path=None if config_path is None else str(config_path),
        config_name=config_name,
        overrides=list(overrides or []),
    )


__all__ = ["has_group_override", "load_cli_config", "merge_overrides"]
