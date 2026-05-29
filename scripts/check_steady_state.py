#!/usr/bin/env python
from __future__ import annotations

import sys
import numpy as np

def main():
    from v1_simulation.config import load_config
    from v1_simulation.config.validation import validate_config
    from v1_simulation.network.builder import build_network_state
    from v1_simulation.simulation.pipeline import default_training_time_grid
    from v1_simulation.solvers.wilson_cowan import solve_wilson_cowan_batch
    from v1_simulation.training.natural_inputs import build_natural_image_l4_drive

    overrides = sys.argv[1:] if len(sys.argv) > 1 else []
    if not any("experiment=" in o for o in overrides):
        overrides = ["+experiment=bcm_train"] + overrides
    if not any("solver=" in o for o in overrides):
        overrides = ["solver=diffrax_tsit5"] + overrides

    cfg = load_config(overrides=overrides)
    cfg.mode = "train"
    cfg.training.enabled = True
    validate_config(cfg)

    network = build_network_state(cfg)
    natural_drive, natural_sampler = build_natural_image_l4_drive(
        cfg=cfg.training.natural_image,
        stimulus_cfg=cfg.stimulus,
        model_cfg=cfg.model,
        layers_cfg=cfg.model.layers,
        l4_layer=network.layout.l4,
        l4_tunings=network.layout.l4_tunings,
        l4_pref_dirs=network.layout.l4_pref_dirs,
    )

    time_grid = default_training_time_grid(cfg)
    epoch_samples = list(natural_sampler.make_epoch(limit=8, shuffle_paths=True, shuffle_samples=True))
    drive_func = natural_drive.make_static_batch_func(tuple(epoch_samples))
    
    # Calculate the actual batch size produced by the drive (images * patches)
    actual_batch_size = drive_func(0.0).shape[1]

    result = solve_wilson_cowan_batch(
        network=network,
        external_drive=drive_func,
        time=time_grid,
        n_batch=actual_batch_size,
        solver_config=cfg.solver,
        transfer_config=cfg.solver.transfer,
        training_bcm=cfg.training.bcm,
        background_trace=None,
        store_trajectory=True,
        stop_at_steady_state=False,
    )

    exc_traj = np.asarray(result.exc_trajectory) # shape: (time, batch, neurons)
    dt = time_grid[1] - time_grid[0]
    
    # Calculate velocity (abs diff)
    diff = np.abs(np.diff(exc_traj, axis=0)) / dt
    max_vel_over_time = np.max(diff, axis=(1, 2))
    
    print("Max dy/dt over time:")
    for i in range(0, len(max_vel_over_time), len(max_vel_over_time) // 10):
        print(f"  t = {time_grid[i+1]:.4f}: max dy/dt = {max_vel_over_time[i]:.6f}")
    print(f"  t = {time_grid[-1]:.4f}: max dy/dt = {max_vel_over_time[-1]:.6f}")
    
    threshold = 1e-3
    steady_state_idx = np.where(max_vel_over_time < threshold)[0]
    
    if len(steady_state_idx) > 0:
        idx = steady_state_idx[0]
        t_steady = time_grid[idx + 1]
        print(f"\nNetwork reached steady state (max dy/dt < {threshold}) at t = {t_steady:.4f} (step {idx+1}/{len(time_grid)})")
    else:
        print(f"\nNetwork did NOT reach steady state (threshold {threshold}) within the full time grid.")
        print(f"Final max dy/dt: {max_vel_over_time[-1]:.6f}")

if __name__ == "__main__":
    main()
