import numpy as np
import types

def _patch_generator_compatibility():
    for name in ('sum', 'prod', 'all', 'any'):
        if hasattr(np, name):
            orig = getattr(np, name)
            def patched(a, *args, _orig=orig, **kwargs):
                if isinstance(a, types.GeneratorType):
                    a = list(a)
                return _orig(a, *args, **kwargs)
            setattr(np, name, patched)

_patch_generator_compatibility()

from v1_simulation.simulation import run_bcm_training, run_simulation


def main() -> None:
    from v1_simulation.cli.main import main as cli_main

    cli_main()


__all__ = ["main", "run_bcm_training", "run_simulation"]
