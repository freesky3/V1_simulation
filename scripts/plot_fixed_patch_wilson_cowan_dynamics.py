#!/usr/bin/env python
"""Verify Wilson-Cowan rate stabilization under one fixed natural image patch input.

This script runs a 100-second simulation of the Wilson-Cowan network dynamics
with frozen weights (no plasticity/BCM updates) and no background noise under
a single, fixed natural image patch input. It uses diffrax.Tsit5 with an
adaptive step size controller (PIDController) to verify that the dynamics
eventually converge to a stable fixed point.

Outputs are saved in outputs/YYYY-MM-DD/HH-MM-SS/:
  - fixed_patch.png: The visual image patch presented to the network.
  - patch_stabilization.png: 4-panel plot of firing rate trajectories and convergence.
  - patch_stabilization_summary.json: Diagnostics and stats of the run.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Check if JAX and Diffrax are available
try:
    import diffrax  # noqa: F401
    import jax  # noqa: F401
except ImportError:
    print("Error: This script requires JAX and Diffrax to be installed.", file=sys.stderr)
    print("Please run it in an environment with GPU/JAX/Diffrax support.", file=sys.stderr)
    sys.exit(1)


def main():
    from v1_simulation.config import load_config
    from v1_simulation.config.validation import validate_config
    from v1_simulation.data.natural_images import apply_crop
    from v1_simulation.network.builder import build_network_state
    from v1_simulation.solvers.base import NetworkLayout
    from v1_simulation.solvers.fixed_patch import (
        build_fixed_patch_time_grid,
        evaluate_dy_dt_trajectory,
        evaluate_fixed_patch_convergence,
        solve_static_fixed_patch_diffrax,
    )
    from v1_simulation.solvers.wilson_cowan import _resolve_transfer_functions
    from v1_simulation.training.natural_inputs import build_natural_image_l4_drive

    # ---- 1. Load Config & Apply Overrides ----
    overrides = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Intercept solver.diffrax.solver override to bypass Hydra composition issues
    custom_solver = None
    overrides_for_hydra = []
    for o in overrides:
        key_val = o.split("=")
        if len(key_val) == 2:
            key = key_val[0].strip()
            if key in ("solver.diffrax.solver", "+solver.diffrax.solver"):
                custom_solver = key_val[1].strip()
                continue
        overrides_for_hydra.append(o)

    if not any(o.split("=")[0] in ("experiment", "+experiment") for o in overrides_for_hydra):
        overrides_for_hydra = ["+experiment=bcm_train"] + overrides_for_hydra
    # Force diffrax tsit5 configuration by default if no top-level solver choice is provided
    if not any(o.split("=")[0] == "solver" for o in overrides_for_hydra):
        overrides_for_hydra = ["solver=diffrax_tsit5"] + overrides_for_hydra

    cfg = load_config(overrides=overrides_for_hydra)
    
    # Apply intercepted solver override
    if custom_solver is not None:
        if cfg.solver.diffrax is None:
            from v1_simulation.config.schema import DiffraxSolverConfig
            cfg.solver.diffrax = DiffraxSolverConfig()
        cfg.solver.diffrax.solver = custom_solver

    cfg.mode = "train"
    cfg.training.enabled = True
    
    # Force single patch dataset sampler configuration
    cfg.training.natural_image.patches_per_image = 1
    cfg.training.natural_image.limit = 1
    
    # Ensure background noise / stochasticity is disabled
    cfg.background.enabled = False
    
    # Enforce solver parameters in Hydra config object
    cfg.solver.backend = "diffrax"
    cfg.solver.method = "adaptive"
    if custom_solver is not None:
        cfg.solver.diffrax.solver = custom_solver
    elif cfg.solver.diffrax is None:
        from v1_simulation.config.schema import DiffraxSolverConfig
        cfg.solver.diffrax = DiffraxSolverConfig()
        cfg.solver.diffrax.solver = "tsit5"
    elif not getattr(cfg.solver.diffrax, "solver", None):
        cfg.solver.diffrax.solver = "tsit5"

    # Set seed for reproducibility
    if cfg.seed is None:
        cfg.seed = 42
    np.random.seed(cfg.seed)

    validate_config(cfg)

    # ---- 2. Setup Run Directory ----
    now = datetime.now()
    run_dir = Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory initialized at: {run_dir.resolve()}")

    # ---- 3. Build Network & Visual Drive ----
    print("Building network...")
    network = build_network_state(cfg)
    layout = NetworkLayout.from_network_state(network)
    print(f"  Network built: n_E={layout.n_exc}, n_I={layout.n_inh}, n_X={layout.n_ext}")

    print("Building natural image visual drive...")
    natural_drive, natural_sampler = build_natural_image_l4_drive(
        cfg=cfg.training.natural_image,
        stimulus_cfg=cfg.stimulus,
        model_cfg=cfg.model,
        layers_cfg=cfg.model.layers,
        l4_layer=network.layout.l4,
        l4_tunings=network.layout.l4_tunings,
        l4_pref_dirs=network.layout.l4_pref_dirs,
    )

    # Fetch exactly one patch sample
    epoch_samples = list(natural_sampler.make_epoch(limit=1, shuffle_paths=False, shuffle_samples=False))
    if not epoch_samples:
        raise RuntimeError("No natural image samples were generated by the sampler.")
    sample = epoch_samples[0]
    print(f"Selected image path: {sample.path}")
    print(f"Selected crop box: {sample.crop}")

    # Read and crop the image, then save the visual patch
    image_raw = natural_sampler.dataset.read(sample.path)
    image_cropped = apply_crop(image_raw, sample.crop)
    
    # Plot and save the fixed image patch
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.figure(figsize=(4, 4))
    plt.imshow(image_cropped, cmap="gray")
    plt.title(f"Fixed Patch Input\n{sample.path.name}", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(run_dir / "fixed_patch.png", dpi=300)
    plt.close()
    print("Saved input patch visualization to 'fixed_patch.png'.")

    # Get the visual preprocessed frame and project it to calculate input rates
    frame = natural_drive.preprocessor.transform(image_raw, sample)
    input_rates = natural_drive.projector.project(frame)
    
    # Build time-invariant drive function
    drive_func = natural_drive.make_static_batch_func((sample,))
    
    # Confirm the drive is time-independent
    rate_at_0 = drive_func(0.0)
    rate_at_later = drive_func(float(cfg.simulation.duration_tau_e) * float(cfg.solver.transfer.tau_e))
    assert np.allclose(rate_at_0, rate_at_later), "Error: Drive is not time-independent!"
    print("Confirmed L4 drive is time-independent.")

    # ---- 4. Set Up Time Grid & Transfer Functions ----
    time_grid = build_fixed_patch_time_grid(cfg)
    t0 = time_grid.t0
    t1 = time_grid.t1
    save_ts = time_grid.save_ts
    n_save = save_ts.size
    tau_e = float(cfg.solver.transfer.tau_e)
    tau_i = float(cfg.solver.transfer.tau_i)
    
    # Retrieve transfer functions
    phi_exc, phi_inh = _resolve_transfer_functions(
        transfer_config=cfg.solver.transfer,
        transfer_tables=None,
        phi_exc=None,
        phi_inh=None,
    )

    # ---- 5. JAX-JIT Diffrax Integration ----
    print(f"Compiling and running Diffrax {cfg.solver.diffrax.solver} solver (adaptive step size)...")
    trajectory = solve_static_fixed_patch_diffrax(
        cfg=cfg,
        network=network,
        layout=layout,
        input_rates=input_rates,
        phi_exc=phi_exc,
        phi_inh=phi_inh,
        time_grid=time_grid,
    )
    status = trajectory.status
    num_steps = status.num_steps
    
    print(f"Simulation completed. Solver Result: {status.label} (code: {status.code}, success=0)")
    print(f"Total adaptive integration steps: {num_steps}")

    # ---- 6. Diagnostics & Convergence Check ----
    y_traj = trajectory.y_traj
    y_final = y_traj[-1, :] # shape: (n_rates,)

    convergence = evaluate_fixed_patch_convergence(
        cfg=cfg,
        network=network,
        layout=layout,
        phi_exc=phi_exc,
        phi_inh=phi_inh,
        drive_func=drive_func,
        time_grid=time_grid,
        y_traj=y_traj,
    )
    final_max_abs_drE_dt = convergence.final_max_abs_drE_dt
    final_max_abs_drI_dt = convergence.final_max_abs_drI_dt
    final_max_abs_dy_dt = convergence.final_max_abs_dy_dt
    final_rms_dy_dt = convergence.final_rms_dy_dt
    convergence_window_s = convergence.convergence_window_s
    max_abs_delta_last_1s = convergence.max_abs_delta_last_1s
    rhs_threshold = convergence.rhs_threshold
    peak_to_peak_threshold = convergence.peak_to_peak_threshold
    rhs_converged = convergence.rhs_converged
    window_converged = convergence.window_converged
    is_converged = convergence.converged

    # Population statistics
    exc_traj = y_traj[:, layout.idx_exc] # shape: (n_save, n_exc)
    inh_traj = y_traj[:, layout.idx_inh] # shape: (n_save, n_inh)
    
    mean_E_t = np.mean(exc_traj, axis=1)
    mean_I_t = np.mean(inh_traj, axis=1)
    std_E_t = np.std(exc_traj, axis=1)
    std_I_t = np.std(inh_traj, axis=1)

    final_mean_E = float(mean_E_t[-1])
    final_mean_I = float(mean_I_t[-1])
    final_std_E = float(std_E_t[-1])
    final_std_I = float(std_I_t[-1])

    print("=" * 60)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 60)
    print(f"Final mean E firing rate: {final_mean_E:.4f} Hz")
    print(f"Final mean I firing rate: {final_mean_I:.4f} Hz")
    print(f"Final std E firing rate:  {final_std_E:.4f} Hz")
    print(f"Final std I firing rate:  {final_std_I:.4f} Hz")
    print(f"Final max |dy_E/dt|:      {final_max_abs_drE_dt:.2e} Hz/s")
    print(f"Final max |dy_I/dt|:      {final_max_abs_drI_dt:.2e} Hz/s")
    print(f"Final RMS |dy/dt|:        {final_rms_dy_dt:.2e} Hz/s")
    print(f"Peak-to-Peak change over final {convergence_window_s:g}s: {max_abs_delta_last_1s:.2e} Hz")
    print("-" * 60)
    print(f"RHS converges (max |dy/dt| < {rhs_threshold:g}):       {rhs_converged}")
    print(
        f"Window converges (max P2P last {convergence_window_s:g}s < {peak_to_peak_threshold:g}): "
        f"{window_converged}"
    )
    print(f"--> SYSTEM STABILIZED: {is_converged}")
    print("=" * 60)

    # ---- 7. Select 3 Excitatory and 3 Inhibitory Neurons with the Highest Firing Rates ----
    y_final_exc = y_final[layout.idx_exc]
    y_final_inh = y_final[layout.idx_inh]
    
    # Get local indices of top 3 firing rate neurons
    top_exc_local_idx = np.argsort(y_final_exc)[-3:]
    top_inh_local_idx = np.argsort(y_final_inh)[-3:]
    
    # Map back to global network indices
    sample_exc_indices = layout.idx_exc[top_exc_local_idx].tolist()
    sample_inh_indices = layout.idx_inh[top_inh_local_idx].tolist()
    
    sample_exc_indices = sorted(sample_exc_indices)
    sample_inh_indices = sorted(sample_inh_indices)

    # ---- 8. Compute dy/dt trajectory over time for plotting ----
    print("Evaluating dy/dt trajectory over time...")
    dy_dt_over_time = evaluate_dy_dt_trajectory(
        network=network,
        layout=layout,
        phi_exc=phi_exc,
        phi_inh=phi_inh,
        tau_e=tau_e,
        tau_i=tau_i,
        drive_func=drive_func,
        save_ts=save_ts,
        y_traj=y_traj,
    )
    
    max_dy_dt_E = np.max(np.abs(dy_dt_over_time[:, layout.idx_exc]), axis=1)
    max_dy_dt_I = np.max(np.abs(dy_dt_over_time[:, layout.idx_inh]), axis=1)

    # ---- 9. Plot Results (4 Panels) ----
    print("Generating 4-panel stabilization plot...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 0,0: Population Mean and Std Firing Rates
    axs[0, 0].plot(save_ts, mean_E_t, label="Excitatory Mean", color="#1f77b4", linewidth=2)
    axs[0, 0].fill_between(save_ts, mean_E_t - std_E_t, mean_E_t + std_E_t, color="#1f77b4", alpha=0.15)
    axs[0, 0].plot(save_ts, mean_I_t, label="Inhibitory Mean", color="#d62728", linewidth=2)
    axs[0, 0].fill_between(save_ts, mean_I_t - std_I_t, mean_I_t + std_I_t, color="#d62728", alpha=0.15)
    axs[0, 0].set_title("Population Mean & Std Firing Rates", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Firing Rate (Hz)")
    axs[0, 0].legend()

    # Panel 0,1: Sample Excitatory Neuron Traces
    for idx in sample_exc_indices:
        axs[0, 1].plot(save_ts, y_traj[:, idx], label=f"Neuron #{idx}", alpha=0.8)
    axs[0, 1].set_title("Sample Excitatory Neuron Rates", fontsize=11, fontweight="bold")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Firing Rate (Hz)")
    axs[0, 1].legend()

    # Panel 1,0: Sample Inhibitory Neuron Traces
    for idx in sample_inh_indices:
        axs[1, 0].plot(save_ts, y_traj[:, idx], label=f"Neuron #{idx}", alpha=0.8)
    axs[1, 0].set_title("Sample Inhibitory Neuron Rates", fontsize=11, fontweight="bold")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Firing Rate (Hz)")
    axs[1, 0].legend()

    # Panel 1,1: Convergence speed (Max |dy/dt| over time)
    axs[1, 1].plot(save_ts, max_dy_dt_E, label="Max |dy_E/dt|", color="#1f77b4", alpha=0.8)
    axs[1, 1].plot(save_ts, max_dy_dt_I, label="Max |dy_I/dt|", color="#d62728", alpha=0.8)
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_title("Convergence Speed (Log-Scale Derivative)", fontsize=11, fontweight="bold")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Max |dy/dt| (Hz/s)")
    axs[1, 1].axhline(
        rhs_threshold,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=f"Threshold ({rhs_threshold:g})",
    )
    axs[1, 1].legend()

    plt.suptitle(
        f"Wilson-Cowan Dynamics Under Fixed Patch Input ({sample.path.name})\n"
        f"System Converged: {is_converged} (RHS={rhs_converged}, P2P={window_converged})",
        fontsize=13, fontweight="bold", y=0.97
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(run_dir / "patch_stabilization.png", dpi=300)
    plt.close()
    print("Saved 4-panel visualization to 'patch_stabilization.png'.")

    # ---- 10. Save JSON Summary ----
    summary_data = {
        "t_start": t0,
        "t_stop": t1,
        "solver": f"diffrax_{cfg.solver.diffrax.solver}",
        "num_steps": num_steps,
        "diffrax_result_code": status.code,
        "diffrax_result_str": status.label,
        "rtol": float(cfg.training.bcm.steady_state_rel_tol),
        "atol": float(cfg.training.bcm.steady_state_abs_tol),
        "diffrax_max_steps": int(cfg.solver.diffrax.max_steps),
        "initial_dt_tau_min_fraction": float(cfg.solver.diffrax.initial_dt_tau_min_fraction),
        "trajectory_sample_points": n_save,
        "convergence_window_s": convergence_window_s,
        "dy_dt_threshold": rhs_threshold,
        "peak_to_peak_threshold": peak_to_peak_threshold,
        "final_mean_E": final_mean_E,
        "final_mean_I": final_mean_I,
        "final_std_E": final_std_E,
        "final_std_I": final_std_I,
        "final_max_abs_drE_dt": final_max_abs_drE_dt,
        "final_max_abs_drI_dt": final_max_abs_drI_dt,
        "final_max_abs_dy_dt": final_max_abs_dy_dt,
        "final_rms_dy_dt": final_rms_dy_dt,
        "max_abs_delta_last_1s": max_abs_delta_last_1s,
        "rhs_converged": bool(rhs_converged),
        "window_converged": bool(window_converged),
        "system_stabilized": bool(is_converged),
        "image_path": str(sample.path),
        "crop_box": {
            "top": sample.crop.top if sample.crop else None,
            "left": sample.crop.left if sample.crop else None,
            "height": sample.crop.height if sample.crop else None,
            "width": sample.crop.width if sample.crop else None,
        },
        "sample_exc_indices": sample_exc_indices,
        "sample_inh_indices": sample_inh_indices,
    }

    summary_file = run_dir / "patch_stabilization_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved run summary to {summary_file}.")


if __name__ == "__main__":
    main()
