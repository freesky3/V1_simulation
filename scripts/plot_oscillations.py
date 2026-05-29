#!/usr/bin/env python
from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt

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

    # Note: Using the updated time grid (which is now 30 * tau_e = 0.6s)
    time_grid = default_training_time_grid(cfg)
    epoch_samples = list(natural_sampler.make_epoch(limit=8, shuffle_paths=True, shuffle_samples=True))
    drive_func = natural_drive.make_static_batch_func(tuple(epoch_samples))

    actual_batch_size = drive_func(0.0).shape[1]

    print(f"Integrating for {len(time_grid)} steps (up to t={time_grid[-1]:.3f}s)...")
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
    inh_traj = np.asarray(result.inh_trajectory) # shape: (time, batch, neurons)

    batch_idx = 0
    t = time_grid

    # Calculate population means for the first batch element
    exc_mean = np.mean(exc_traj[:, batch_idx, :], axis=1)
    inh_mean = np.mean(inh_traj[:, batch_idx, :], axis=1)

    # Pick 3 random neurons
    np.random.seed(42)
    sample_exc_idx = np.random.choice(exc_traj.shape[2], 3, replace=False)
    sample_inh_idx = np.random.choice(inh_traj.shape[2], 3, replace=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Population Mean
    axs[0].plot(t, exc_mean, label="Excitatory Population Mean", color="blue", linewidth=2)
    axs[0].plot(t, inh_mean, label="Inhibitory Population Mean", color="red", linewidth=2)
    axs[0].set_title("Population Mean Firing Rates (Batch Element 0)")
    axs[0].set_ylabel("Firing Rate (Hz)")
    axs[0].legend()

    # Plot 2: Single Neurons
    for i, idx in enumerate(sample_exc_idx):
        axs[1].plot(t, exc_traj[:, batch_idx, idx], color="blue", alpha=0.5, 
                    label=f"Exc Neuron {idx}" if i==0 else None)
    for i, idx in enumerate(sample_inh_idx):
        axs[1].plot(t, inh_traj[:, batch_idx, idx], color="red", alpha=0.5, 
                    label=f"Inh Neuron {idx}" if i==0 else None)
    
    axs[1].set_title("Sample Single Neuron Firing Rates")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Firing Rate (Hz)")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("oscillations.png", dpi=300)
    print("Saved trajectory plot to 'oscillations.png'.")

if __name__ == "__main__":
    main()
