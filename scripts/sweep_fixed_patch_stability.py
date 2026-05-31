#!/usr/bin/env python
"""Sweep parameters (j and g) to find stable regimes for Wilson-Cowan rate dynamics.

This script runs a grid sweep over the recurrent excitation scaling parameter `j`
and the feedback inhibition scaling parameter `g` under a single, fixed natural
image patch input. For each parameter combination, it integrates the dynamics for
100 seconds using diffrax.Tsit5 with adaptive stepping (PIDController) and records
whether the system successfully stabilizes (converges) without solver crashes.

Outputs are saved in outputs/YYYY-MM-DD/HH-MM-SS/:
  - stability_heatmap.png: Heatmap of the parameter space (Stability & Final Derivatives).
  - sweep_stability_results.json: Full sweep data and stable recommendations.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    cfg.training.natural_image.patches_per_image = 1
    cfg.training.natural_image.limit = 1
    cfg.background.enabled = False
    cfg.solver.backend = "diffrax"
    cfg.solver.method = "adaptive"
    
    # Robustly ensure solver is set
    if cfg.solver.diffrax is None:
        from v1_simulation.config.schema import DiffraxSolverConfig
        cfg.solver.diffrax = DiffraxSolverConfig()
        cfg.solver.diffrax.solver = "tsit5"
    elif not getattr(cfg.solver.diffrax, "solver", None):
        cfg.solver.diffrax.solver = "tsit5"

    if cfg.seed is None:
        cfg.seed = 42
    np.random.seed(cfg.seed)

    validate_config(cfg)

    # ---- 2. Setup Run Directory ----
    now = datetime.now()
    run_dir = Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep output directory initialized at: {run_dir.resolve()}")

    # ---- 3. Build Natural Image Drive (Single Patch) ----
    print("Building natural image visual drive...")
    # Temporarily build network to initialize visual drive/sampler
    temp_network = build_network_state(cfg)
    natural_drive, natural_sampler = build_natural_image_l4_drive(
        cfg=cfg.training.natural_image,
        stimulus_cfg=cfg.stimulus,
        model_cfg=cfg.model,
        layers_cfg=cfg.model.layers,
        l4_layer=temp_network.layout.l4,
        l4_tunings=temp_network.layout.l4_tunings,
        l4_pref_dirs=temp_network.layout.l4_pref_dirs,
    )
    
    epoch_samples = list(natural_sampler.make_epoch(limit=1, shuffle_paths=False, shuffle_samples=False))
    if not epoch_samples:
        raise RuntimeError("No natural image samples found.")
    sample = epoch_samples[0]
    print(f"Fixed visual input patch: {sample.path} (crop: {sample.crop})")

    # Read visual patch input rates
    image_raw = natural_sampler.dataset.read(sample.path)
    frame = natural_drive.preprocessor.transform(image_raw, sample)
    input_rates = natural_drive.projector.project(frame)
    drive_func = natural_drive.make_static_batch_func((sample,))

    # Save visual patch image
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.figure(figsize=(4, 4))
    plt.imshow(apply_crop(image_raw, sample.crop), cmap="gray")
    plt.title(f"Sweep Image Input\n{sample.path.name}", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(run_dir / "fixed_patch.png", dpi=300)
    plt.close()

    # ---- 4. Define Sweep Grids ----
    # Exc scaling (j) and Inh scaling (g)
    j_vals = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    g_vals = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    print(f"Sweep grids: j={j_vals}, g={g_vals} (total combinations: {len(j_vals) * len(g_vals)})")

    # Time grid and transfer functions
    time_grid = build_fixed_patch_time_grid(cfg)
    t1 = time_grid.t1
    save_ts = time_grid.save_ts
    n_save = save_ts.size
    phi_exc, phi_inh = _resolve_transfer_functions(
        transfer_config=cfg.solver.transfer,
        transfer_tables=None,
        phi_exc=None,
        phi_inh=None,
    )
    rtol = float(cfg.training.bcm.steady_state_rel_tol)
    atol = float(cfg.training.bcm.steady_state_abs_tol)

    # ---- 5. Run Grid Sweep ----
    results_grid = []
    
    # Pre-allocate storage for heatmaps
    stability_matrix = np.zeros((len(j_vals), len(g_vals)))
    max_dy_dt_matrix = np.zeros((len(j_vals), len(g_vals)))
    p2p_matrix = np.zeros((len(j_vals), len(g_vals)))

    print("Starting parameter grid sweep...")
    for j_idx, j_val in enumerate(j_vals):
        for g_idx, g_val in enumerate(g_vals):
            print(f"Testing combination: j={j_val:.2f}, g={g_val:.2f}...")
            
            # Rebuild network state with updated parameter overrides
            cfg.model.connectivity.j = j_val
            cfg.model.connectivity.g = g_val
            
            network = build_network_state(cfg)
            layout = NetworkLayout.from_network_state(network)

            # Solve Wilson-Cowan dynamics
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
            y_traj = trajectory.y_traj
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
            is_stable = bool(status.successful and convergence.converged)

            # Store in matrices
            stability_matrix[j_idx, g_idx] = 1.0 if is_stable else 0.0
            max_dy_dt_matrix[j_idx, g_idx] = convergence.final_max_abs_dy_dt
            p2p_matrix[j_idx, g_idx] = convergence.max_abs_delta_last_1s

            # Log this combination
            record = {
                "j": j_val,
                "g": g_val,
                "num_steps": status.num_steps,
                "diffrax_result_code": status.code,
                "diffrax_result_str": status.label,
                "final_max_abs_dy_dt": convergence.final_max_abs_dy_dt,
                "final_rms_dy_dt": convergence.final_rms_dy_dt,
                "max_abs_delta_last_1s": convergence.max_abs_delta_last_1s,
                "stable": is_stable,
            }
            results_grid.append(record)
            print(
                f"  Result: stable={is_stable}, steps={status.num_steps}, "
                f"max|dy/dt|={convergence.final_max_abs_dy_dt:.2e}, "
                f"P2P={convergence.max_abs_delta_last_1s:.2e}"
            )

    # ---- 6. Generate Stability Heatmap & Visual Reports ----
    print("Generating stability parameter space heatmap...")
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Binary Stability Matrix
    im1 = axs[0].imshow(stability_matrix, origin="lower", cmap="RdYlGn", aspect="auto")
    axs[0].set_xticks(np.arange(len(g_vals)))
    axs[0].set_xticklabels([f"{g:.1f}" for g in g_vals])
    axs[0].set_yticks(np.arange(len(j_vals)))
    axs[0].set_yticklabels([f"{j:.2f}" for j in j_vals])
    axs[0].set_xlabel("Inhibition Scaling (g)")
    axs[0].set_ylabel("Recurrent Excitation Scaling (j)")
    axs[0].set_title("Stability Regimes (Green=Stable, Red=Unstable/Crashed)")
    # Draw text annotations in cells
    for (i, k), val in np.ndenumerate(stability_matrix):
        label = "Stable" if val == 1.0 else "Unstable"
        color = "black"
        axs[0].text(k, i, label, ha="center", va="center", color=color, fontweight="bold")

    # Panel 2: Log-scale Final Derivatives
    log_dy_dt = np.log10(np.clip(max_dy_dt_matrix, 1e-6, None))
    im2 = axs[1].imshow(log_dy_dt, origin="lower", cmap="viridis", aspect="auto")
    axs[1].set_xticks(np.arange(len(g_vals)))
    axs[1].set_xticklabels([f"{g:.1f}" for g in g_vals])
    axs[1].set_yticks(np.arange(len(j_vals)))
    axs[1].set_yticklabels([f"{j:.2f}" for j in j_vals])
    axs[1].set_xlabel("Inhibition Scaling (g)")
    axs[1].set_ylabel("Recurrent Excitation Scaling (j)")
    axs[1].set_title("Final State Derivative: log10(max |dy/dt|)")
    fig.colorbar(im2, ax=axs[1], label="log10(Hz/s)")

    plt.suptitle(
        f"Wilson-Cowan Stability Landscape Under Fixed Patch Input ({sample.path.name})\n"
        f"Tolerance: rtol={rtol:.1e}, atol={atol:.1e}",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(run_dir / "stability_heatmap.png", dpi=300)
    plt.close()
    print("Saved 2-panel stability heatmap to 'stability_heatmap.png'.")

    # ---- 7. Compile Stable Parameter Recommendations ----
    stable_combos = [r for r in results_grid if r["stable"]]
    stable_recommendations = []
    for r in stable_combos:
        stable_recommendations.append({"j": r["j"], "g": r["g"]})

    print("=" * 60)
    print("STABILITY SWEEP SUMMARY RECOMMENDATIONS")
    print("=" * 60)
    if stable_combos:
        print(f"Found {len(stable_combos)} stable parameter combinations:")
        for r in stable_combos:
            print(f"  ➜ Stable at: j={r['j']:.2f}, g={r['g']:.2f} (max|dy/dt|={r['final_max_abs_dy_dt']:.2e} Hz/s)")
    else:
        print("  ✗ WARNING: No stable parameter combinations found in the swept grid.")
        print("  ➜ Try further decreasing 'j' (excitation) or increasing 'g' (inhibition).")
    print("=" * 60)

    # ---- 8. Save Sweep JSON Summary ----
    summary_data = {
        "sweep_timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "t_stop": t1,
        "solver": f"diffrax_{cfg.solver.diffrax.solver}",
        "rtol": rtol,
        "atol": atol,
        "diffrax_max_steps": int(cfg.solver.diffrax.max_steps),
        "initial_dt_tau_min_fraction": float(cfg.solver.diffrax.initial_dt_tau_min_fraction),
        "trajectory_sample_points": n_save,
        "convergence_window_s": float(cfg.solver.diagnostics.convergence_window_s),
        "dy_dt_threshold": float(cfg.solver.diagnostics.dy_dt_threshold),
        "peak_to_peak_threshold": float(cfg.solver.diagnostics.peak_to_peak_threshold),
        "j_grid": j_vals,
        "g_grid": g_vals,
        "image_path": str(sample.path),
        "stable_combinations": stable_recommendations,
        "results": results_grid,
    }

    summary_file = run_dir / "sweep_stability_results.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved sweep results summary to {summary_file}.")


if __name__ == "__main__":
    main()
