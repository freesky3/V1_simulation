#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    from v1_simulation.config import load_config
    from v1_simulation.config.validation import validate_config
    from v1_simulation.network.builder import build_network_state
    from v1_simulation.simulation.pipeline import default_training_time_grid
    from v1_simulation.training.natural_inputs import build_natural_image_l4_drive
    from v1_simulation.training.trainer import BCMTrainer
    from v1_simulation.solvers.wilson_cowan import solve_wilson_cowan_batch

    # Parse and merge user command overrides with requested parameters
    overrides = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Defaults as requested by user:
    # solver=diffrax_tsit5 background=none model.connectivity.j=1.2 solver.early_stop.enabled=true solver.early_stop.min_time=0.2 solver.early_stop.f_atol=1e-4 training.bcm.rate_explosion_threshold=none
    default_overrides = [
        "solver=diffrax_tsit5",
        "background=none",
        "model.connectivity.j=1.2",
        "solver.early_stop.enabled=true",
        "solver.early_stop.min_time=0.2",
        "solver.early_stop.f_atol=1e-4",
        "training.bcm.rate_explosion_threshold=none"
    ]
    
    overrides_keys = [o.split("=")[0].lstrip("+") for o in overrides if "=" in o]
    merged_overrides = ["+experiment=bcm_train"] if not any("experiment" in o for o in overrides_keys) else []
    for do in default_overrides:
        dkey = do.split("=")[0].lstrip("+")
        if dkey not in overrides_keys:
            merged_overrides.append(do)
    merged_overrides.extend(overrides)

    print("Loading config with overrides:")
    for o in merged_overrides:
        print(f"  {o}")
    
    cfg = load_config(overrides=merged_overrides)
    cfg.mode = "train"
    cfg.training.enabled = True
    validate_config(cfg)

    # 1. Build network
    print("\nBuilding network state...")
    network = build_network_state(cfg)
    print(f"  Network built. n_E={network.layout.n_E}, n_I={network.layout.n_I}, n_X={network.layout.n_X}")

    # 2. Build natural image drive and sampler
    print("Building natural image drive...")
    natural_drive, natural_sampler = build_natural_image_l4_drive(
        cfg=cfg.training.natural_image,
        stimulus_cfg=cfg.stimulus,
        model_cfg=cfg.model,
        layers_cfg=cfg.model.layers,
        l4_layer=network.layout.l4,
        l4_tunings=network.layout.l4_tunings,
        l4_pref_dirs=network.layout.l4_pref_dirs,
    )

    # 3. Time grid
    time_grid = default_training_time_grid(cfg)
    print(f"Time grid size: {time_grid.size} steps, t_final={time_grid[-1]:.3f}s")

    # 4. Sampler epoch setup
    batch_size = int(cfg.training.bcm.batch_size)
    limit = 3 * batch_size
    print(f"Sampling epoch with limit={limit}...")
    epoch_samples = list(natural_sampler.make_epoch(
        limit=limit,
        shuffle_paths=True,
        shuffle_samples=True,
    ))
    
    if hasattr(natural_drive, "preload_cache"):
        print("Preloading Gabor cache...")
        natural_drive.preload_cache(epoch_samples)

    # 5. Initialize BCM Trainer
    trainer = BCMTrainer(cfg.training.bcm, network)

    # 6. Run the training for 3 updates
    updates_trajectories = []
    steady_states = []

    print("\nStarting BCM training for 3 updates...")
    for u_idx in range(3):
        start_sample = u_idx * batch_size
        end_sample = min(start_sample + batch_size, len(epoch_samples))
        batch = tuple(epoch_samples[start_sample:end_sample])
        
        # Pad batch if needed
        while len(batch) < batch_size:
            batch = batch + (epoch_samples[len(batch) % len(epoch_samples)],)
            
        print(f"Running update {u_idx + 1}/3 (samples {start_sample} to {end_sample})...")
        
        # Build external drive for the batch
        drive_func = natural_drive.make_static_batch_func(batch)
        
        # Solve WC batch with store_trajectory=True
        dynamics = solve_wilson_cowan_batch(
            network=trainer.state.network,
            external_drive=drive_func,
            time=time_grid,
            n_batch=len(batch),
            solver_config=cfg.solver,
            transfer_config=cfg.solver.transfer,
            training_bcm=cfg.training.bcm,
            background_trace=None,
            store_trajectory=True,
            stop_at_steady_state=None,  # let it read early_stop.enabled=True from config
        )
        
        # Force block until ready if using JAX backend
        try:
            import jax
            jax.block_until_ready(dynamics.exc_trajectory)
        except (ImportError, AttributeError):
            pass

        # Record trajectory and steady state rates for plotting
        updates_trajectories.append({
            "time": dynamics.time,
            "exc_trajectory": np.asarray(dynamics.exc_trajectory),
            "inh_trajectory": np.asarray(dynamics.inh_trajectory),
            "steady_state_reached": dynamics.steady_state_reached,
            "steady_state_index": dynamics.steady_state_index,
        })
        
        steady_states.append({
            "exc": np.asarray(dynamics.exc),  # (n_batch, n_exc)
            "inh": np.asarray(dynamics.inh),  # (n_batch, n_inh)
        })

        # Apply the BCM weight update
        log_row = trainer.train_batch(
            dynamics,
            epoch=1,
            batch_size=len(batch),
            images="plot_script",
        )
        print(f"  Update {u_idx + 1} finished: aE_mean={log_row.aE_mean:.3g}, aI_mean={log_row.aI_mean:.3g}, updated={log_row.updated}")

    # 7. Plotting the results
    print("\nGenerating plots...")
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), dpi=150)
    
    # Modern color palette
    c_exc_mean = "#1f77b4"      # Indigo / Deep Blue
    c_inh_mean = "#d62728"      # Crimson / Coral Red
    c_exc_single = "#a6c8e0"    # Muted light blue
    c_inh_single = "#f3a6a6"    # Muted pink/red
    
    np.random.seed(42)
    
    for u_idx in range(3):
        traj_data = updates_trajectories[u_idx]
        ss_data = steady_states[u_idx]
        
        t = traj_data["time"]
        exc_traj = traj_data["exc_trajectory"]
        inh_traj = traj_data["inh_trajectory"]
        
        # Batch item index to plot trajectories (use batch element 0)
        batch_idx = 0
        
        # Calculate population mean over time for batch element 0
        exc_pop_mean = np.mean(exc_traj[:, batch_idx, :], axis=1)
        inh_pop_mean = np.mean(inh_traj[:, batch_idx, :], axis=1)
        
        # Pick 3 random neurons to show single neuron trajectories
        n_exc = exc_traj.shape[2]
        n_inh = inh_traj.shape[2]
        sample_exc_idx = np.random.choice(n_exc, min(3, n_exc), replace=False)
        sample_inh_idx = np.random.choice(n_inh, min(3, n_inh), replace=False)
        
        # --- LEFT COLUMN: Trajectories ---
        ax_l = axs[u_idx, 0]
        
        # Plot single neurons
        for i, idx in enumerate(sample_exc_idx):
            ax_l.plot(t, exc_traj[:, batch_idx, idx], color=c_exc_single, alpha=0.5, 
                      linewidth=1, label="Sample Exc Neurons" if i == 0 else None)
        for i, idx in enumerate(sample_inh_idx):
            ax_l.plot(t, inh_traj[:, batch_idx, idx], color=c_inh_single, alpha=0.5, 
                      linewidth=1, label="Sample Inh Neurons" if i == 0 else None)
            
        # Plot population mean
        ax_l.plot(t, exc_pop_mean, color=c_exc_mean, linewidth=2.5, label="Excitatory Pop Mean")
        ax_l.plot(t, inh_pop_mean, color=c_inh_mean, linewidth=2.5, label="Inhibitory Pop Mean")
        
        # Draw early stop vertical line if it reached steady state
        if traj_data["steady_state_reached"] and traj_data["steady_state_index"] is not None:
            t_stop = t[traj_data["steady_state_index"]]
            ax_l.axvline(x=t_stop, color="#7f8c8d", linestyle="--", linewidth=1.5,
                         label=f"Early Stop ({t_stop:.3f}s)")
            
        ax_l.set_title(f"Update {u_idx + 1} - Trajectories (Batch Item {batch_idx})", fontsize=12, fontweight="bold")
        ax_l.set_xlabel("Time (s)", fontsize=10)
        ax_l.set_ylabel("Firing Rate (Hz)", fontsize=10)
        ax_l.legend(loc="upper right", frameon=True, fontsize=9)
        ax_l.set_xlim(t[0], t[-1])
        ax_l.grid(True, linestyle=":", alpha=0.6)
        
        # --- RIGHT COLUMN: Steady State Distribution ---
        ax_r = axs[u_idx, 1]
        
        # Excitatory and Inhibitory steady-state rates for all batch samples
        exc_rates = ss_data["exc"].ravel()
        inh_rates = ss_data["inh"].ravel()
        
        # Plot distributions
        ax_r.hist(exc_rates, bins=35, alpha=0.7, color=c_exc_mean, label=f"Exc (Mean={np.mean(exc_rates):.2f} Hz)", density=True)
        ax_r.hist(inh_rates, bins=35, alpha=0.6, color=c_inh_mean, label=f"Inh (Mean={np.mean(inh_rates):.2f} Hz)", density=True)
        
        ax_r.set_title(f"Update {u_idx + 1} - Steady State Distribution (All Batch Items)", fontsize=12, fontweight="bold")
        ax_r.set_xlabel("Steady-State Firing Rate (Hz)", fontsize=10)
        ax_r.set_ylabel("Density", fontsize=10)
        ax_r.legend(loc="upper right", frameon=True, fontsize=9)
        ax_r.grid(True, linestyle=":", alpha=0.6)
        
    plt.suptitle("Neuron Firing Rates During BCM Training (First 3 Updates)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    # Save the plot
    output_path = Path("training_firing_rates.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
