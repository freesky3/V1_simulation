import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from v1_simulation.cli import load_cli_config
from v1_simulation.network.builder import build_network_state
from v1_simulation.training.natural_inputs import build_natural_image_l4_drive
from v1_simulation.solvers.wilson_cowan import solve_wilson_cowan_batch
from v1_simulation.simulation.pipeline import _make_simulation_background_trace, default_simulation_time_grid

def diagnose():
    jax.config.update("jax_enable_x64", False)
    
    overrides = [
        "+experiment=bcm_train",
        "solver=diffrax_tsit5",
        "background=none",
        "model.connectivity.j=1.2",
        "training.bcm.epochs=1",
        "training.bcm.batch_size=2", # small batch for test
    ]
    
    cfg = load_cli_config(config_path=None, config_name="config", overrides=overrides)
    cfg.mode = "train"
    cfg.training.enabled = True
    if cfg.training.natural_image.dir is None:
        cfg.training.natural_image.dir = str(cfg.paths.natural_image_dir)
    
    print("Building network...")
    network = build_network_state(cfg)
    time_grid = default_simulation_time_grid(cfg)
    
    print("Building stimulus...")
    natural_drive, natural_sampler = build_natural_image_l4_drive(
        cfg=cfg.training.natural_image,
        stimulus_cfg=cfg.stimulus,
        model_cfg=cfg.model,
        layers_cfg=cfg.model.layers,
        l4_layer=network.layout.l4,
        l4_tunings=network.layout.l4_tunings,
        l4_pref_dirs=network.layout.l4_pref_dirs,
    )
    epoch_samples = list(natural_sampler.make_epoch(
        limit=1,
        shuffle_paths=True,
        shuffle_samples=True,
    ))
    batch = tuple(epoch_samples[:2])
    ext_drive = natural_drive.make_static_batch_func(batch)
    
    bg_trace = _make_simulation_background_trace(
        cfg,
        network=network,
        n_batch=2,
        time=time_grid,
    )
    
    cfg.solver.store_trajectory = True
    print("Running solver...")
    result = solve_wilson_cowan_batch(
        network=network,
        external_drive=ext_drive,
        time=time_grid,
        n_batch=2,
        solver_config=cfg.solver,
        transfer_config=cfg.solver.transfer,
        background_trace=bg_trace,
        store_trajectory=True,
    )
    
    y_final_exc = result.exc_trajectory[-1] # shape (batch, n_exc)
    y_final_inh = result.inh_trajectory[-1] # shape (batch, n_inh)
    y_final = np.concatenate([y_final_exc, y_final_inh], axis=1) # shape (batch, N)
    
    print(f"Final state shape: {y_final.shape}")
    print("exc_trajectory shape:", result.exc_trajectory.shape)
    print("Last 5 time points in time_grid:", time_grid[-5:])
    print("Change in exc_trajectory at the end of simulation:")
    for i in range(1, min(6, len(time_grid))):
        diff = np.max(np.abs(result.exc_trajectory[-i] - result.exc_trajectory[-i-1]))
        print(f"  Step -{i} to -{i+1} max diff: {diff}")
    
    # Actually we can just manually compute numerical derivative approximation for diagnostics
    # since getting the true vector field from the outside is messy.
    dy = (result.exc_trajectory[-1] - result.exc_trajectory[-2]) / (time_grid[-1] - time_grid[-2])
    dy_inh = (result.inh_trajectory[-1] - result.inh_trajectory[-2]) / (time_grid[-1] - time_grid[-2])
    dy = np.concatenate([dy, dy_inh], axis=1)
    
    f_norm_max = np.max(np.abs(dy))
    f_norm_rms = np.sqrt(np.mean(np.square(dy)))
    
    print("=======================================")
    print(f"At end of simulation (t={time_grid[-1]:.4f}s):")
    print(f"Max derivative norm (L_inf): {f_norm_max}")
    print(f"RMS derivative norm (L_2): {f_norm_rms}")
    print(f"Max firing rate: {np.max(y_final)}")
    print("=======================================")
    
    if f_norm_max > 1e-4:
        print("f_norm_max > 1e-4! The absolute tolerance 1e-4 is TOO STRICT for max norm.")
    else:
        print("f_norm_max < 1e-4! There must be another bug.")

if __name__ == "__main__":
    diagnose()
