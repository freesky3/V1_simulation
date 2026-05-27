import os
from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from v1_simulation.config.schema import RootConfig
from v1_simulation.config.validation import validate_config

__all__ = ["RootConfig", "load_config", "register_configs", "validate_config"]

def register_configs() -> None:
    """Register RootConfig schema with Hydra's ConfigStore.
    
    This ensures that when configurations are loaded through Hydra, they are
    validated against the structured dataclasses.
    """
    cs = ConfigStore.instance()
    # Register the RootConfig under the name schema or root config
    cs.store(name="config_schema", node=RootConfig)

# Automatically register configurations upon importing the module
register_configs()

def load_config(
    config_path: Optional[str] = None,
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
) -> RootConfig:
    """Safely loads and resolves configuration files using Hydra's Compose API.
    
    Validates the configuration structure against the dataclass schema and resolves
    any string interpolations (e.g. ${seed}).
    
    Args:
        config_path: Absolute or relative path to the configs directory. If None,
                     automatically searches up from this file's parent folder.
        config_name: Name of the entrypoint config YAML file (without extension, defaults to "config").
        overrides: List of CLI-like overrides to apply (e.g. ["model.connectivity.j=2.8"]).
        
    Returns:
        A fully resolved and typed RootConfig instance.
    """
    # 1. Automatically resolve configuration path if not provided
    if config_path is None:
        current_file_dir = Path(__file__).resolve().parent
        for parent in [current_file_dir] + list(current_file_dir.parents)[:4]:
            possible_path = parent / "configs"
            if possible_path.is_dir():
                config_path = str(possible_path)
                break
        if config_path is None:
            config_path = "configs"

    abs_config_path = str(Path(config_path).resolve())

    # 2. Reset Hydra global state if already initialized to allow multiple loads
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # 3. Initialize and compose using Hydra
    with initialize_config_dir(config_dir=abs_config_path, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    # 4. Merge composed config with our structured schema for strict validation and resolution
    schema = OmegaConf.structured(RootConfig)
    merged = OmegaConf.merge(schema, cfg)

    # 5. Convert to typed RootConfig object (resolving references/interpolations)
    typed_config = OmegaConf.to_object(merged)
    
    assert isinstance(typed_config, RootConfig), "Config object must be an instance of RootConfig."
    
    # 6. Validate configuration parameters
    validate_config(typed_config)
    
    return typed_config

