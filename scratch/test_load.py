import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from v1_simulation.cli import load_cli_config

overrides = [
    "+experiment=bcm_train",
    "solver=diffrax_tsit5",
    "background=none",
    "model.connectivity.j=1.2",
    "training.bcm.epochs=1",
    "training.bcm.batch_size=2",
]

cfg = load_cli_config(config_path=None, config_name="config", overrides=overrides)
print("=== Loaded Config training.natural_image ===")
print("dir:", cfg.training.natural_image.dir)
print("limit:", cfg.training.natural_image.limit)
print("paths.natural_image_dir:", cfg.paths.natural_image_dir)
