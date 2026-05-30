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
    import diffrax
    import jax
    import jax.numpy as jnp
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
    from v1_simulation.solvers.jax_backend import _slice_weight_blocks, _transfer_table_arrays
    from v1_simulation.solvers.wilson_cowan import _resolve_transfer_functions, WilsonCowanRHS
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

    # Time grid setup
    t0 = 0.0
    t1 = float(cfg.simulation.t_stop) if (hasattr(cfg, "simulation") and cfg.simulation.t_stop is not None) else 100.0
    tau_e = float(cfg.solver.transfer.tau_e)
    tau_i = float(cfg.solver.transfer.tau_i)
    dt0 = min(tau_e, tau_i) / 10.0
    save_ts = np.linspace(t0, t1, 2000, dtype=np.float64)

    # JAX compilation variables
    j_save_ts = jnp.asarray(save_ts)
    phi_exc, phi_inh = _resolve_transfer_functions(
        transfer_config=cfg.solver.transfer,
        transfer_tables=None,
        phi_exc=None,
        phi_inh=None,
    )
    phi_exc_x, phi_exc_y = _transfer_table_arrays(phi_exc, "phi_exc")
    phi_inh_x, phi_inh_y = _transfer_table_arrays(phi_inh, "phi_inh")
    jax_dtype = jnp.float32 if cfg.solver.jax.dtype == "float32" else jnp.float64
    rtol = float(cfg.training.bcm.steady_state_rel_tol) if hasattr(cfg.training.bcm, "steady_state_rel_tol") else 1e-5
    atol = float(cfg.training.bcm.steady_state_abs_tol) if hasattr(cfg.training.bcm, "steady_state_abs_tol") else 1e-3

    # Define Diffrax vector field and JAX solver function
    def interp_phi(x, xp, fp):
        return jnp.interp(x, xp, fp, left=fp[0], right=fp[-1])

    def vector_field(t, y, args):
        W_exc_a, W_inh_a, mu_ext_a, phi_exc_x_a, phi_exc_y_a, phi_inh_x_a, phi_inh_y_a, tau_e_a, tau_i_a, idx_exc_a, idx_inh_a = args
        mu = W_exc_a @ y[idx_exc_a, :] + W_inh_a @ y[idx_inh_a, :] + mu_ext_a
        dy = jnp.zeros_like(y)
        dy = dy.at[idx_exc_a, :].set((-y[idx_exc_a, :] + interp_phi(tau_e_a * mu[idx_exc_a, :], phi_exc_x_a, phi_exc_y_a)) / tau_e_a)
        dy = dy.at[idx_inh_a, :].set((-y[idx_inh_a, :] + interp_phi(tau_i_a * mu[idx_inh_a, :], phi_inh_x_a, phi_inh_y_a)) / tau_i_a)
        return dy

    @jax.jit
    def run_integration(
        y0_val, W_exc_val, W_inh_val, mu_ext_val, 
        phi_exc_x_val, phi_exc_y_val, phi_inh_x_val, phi_inh_y_val,
        tau_e_val, tau_i_val, idx_exc_val, idx_inh_val, save_ts_val
    ):
        term = diffrax.ODETerm(vector_field)
        solver_name = str(cfg.solver.diffrax.solver).lower()
        if solver_name == "tsit5":
            solver = diffrax.Tsit5()
        elif solver_name == "heun":
            solver = diffrax.Heun()
        elif solver_name == "kvaerno5":
            solver = diffrax.Kvaerno5()
        elif solver_name == "kvaerno3":
            solver = diffrax.Kvaerno3()
        else:
            raise ValueError(f"Unsupported diffrax solver: {solver_name}")
            
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        saveat = diffrax.SaveAt(ts=save_ts_val)

        args = (
            W_exc_val,
            W_inh_val,
            mu_ext_val,
            phi_exc_x_val,
            phi_exc_y_val,
            phi_inh_x_val,
            phi_inh_y_val,
            tau_e_val,
            tau_i_val,
            idx_exc_val,
            idx_inh_val,
        )

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0_val,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=1000000,
        )
        return sol.ys, sol.result, sol.stats["num_steps"]

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
            
            # Prepare JAX arrays and sliced weights
            j_idx_exc = jnp.asarray(layout.idx_exc, dtype=jnp.int32)
            j_idx_inh = jnp.asarray(layout.idx_inh, dtype=jnp.int32)
            W_exc, W_inh, W_ext = _slice_weight_blocks(
                network.weights,
                layout.idx_exc,
                layout.idx_inh,
                layout.idx_ext,
                jnp,
                prefer_sparse=bool(cfg.solver.jax.prefer_sparse),
                dense_max_mb=float(cfg.solver.jax.dense_max_mb),
                dtype=jax_dtype,
            )
            
            y0 = jnp.zeros((layout.n_rates, 1), dtype=jax_dtype)
            ax_0 = jnp.asarray(input_rates[:, np.newaxis], dtype=jax_dtype)
            mu_ext = W_ext @ ax_0

            # Solve Wilson-Cowan dynamics
            ys, sol_result, num_steps_jax = run_integration(
                y0, W_exc, W_inh, mu_ext,
                jnp.asarray(phi_exc_x, dtype=jax_dtype),
                jnp.asarray(phi_exc_y, dtype=jax_dtype),
                jnp.asarray(phi_inh_x, dtype=jax_dtype),
                jnp.asarray(phi_inh_y, dtype=jax_dtype),
                jnp.asarray(tau_e, dtype=jax_dtype),
                jnp.asarray(tau_i, dtype=jax_dtype),
                j_idx_exc,
                j_idx_inh,
                j_save_ts
            )
            
            # Extract variables
            ys = np.asarray(ys)
            num_steps = int(np.asarray(num_steps_jax))
            
            # Check success using diffrax enum comparison
            is_successful = False
            try:
                # Compare directly with diffrax.RESULTS enum
                is_successful = bool(np.asarray(sol_result == diffrax.RESULTS.successful))
                sol_result_val = 0 if is_successful else -1
            except Exception:
                # Fallback to value or attribute checks if not standard
                if hasattr(sol_result, "value"):
                    sol_result_val = int(sol_result.value)
                else:
                    try:
                        sol_result_val = int(sol_result)
                    except (TypeError, ValueError):
                        sol_result_val = -1
                is_successful = (sol_result_val == 0)
            sol_result_str = "successful" if is_successful else str(sol_result)
            
            y_traj = ys[:, :, 0]
            y_final = y_traj[-1, :]

            # Compute final derivatives (dy/dt) using WilsonCowanRHS
            rhs_evaluator = WilsonCowanRHS(
                weights=network.weights,
                layout=layout,
                phi_exc=phi_exc,
                phi_inh=phi_inh,
                tau_exc=tau_e,
                tau_inh=tau_i,
                n_batch=1,
            )
            dy_dt_flat = rhs_evaluator(t1, y_final, drive_func)
            dy_dt = dy_dt_flat.reshape(layout.n_rates, 1)
            
            final_max_abs_drE_dt = float(np.max(np.abs(dy_dt[layout.idx_exc, 0])))
            final_max_abs_drI_dt = float(np.max(np.abs(dy_dt[layout.idx_inh, 0])))
            final_max_abs_dy_dt = max(final_max_abs_drE_dt, final_max_abs_drI_dt)
            final_rms_dy_dt = float(np.sqrt(np.mean(dy_dt ** 2)))

            # Compute peak-to-peak amplitude in the last 1s
            t_start_last_1s = max(t0, t1 - 1.0)
            idx_last_1s = np.argmin(np.abs(save_ts - t_start_last_1s))
            y_traj_last_1s = y_traj[idx_last_1s:, :]
            peak_to_peak_last_1s = np.max(y_traj_last_1s, axis=0) - np.min(y_traj_last_1s, axis=0)
            max_abs_delta_last_1s = float(np.max(peak_to_peak_last_1s))

            # Determine convergence criteria
            rhs_converged = final_max_abs_dy_dt < 1.0
            window_converged = max_abs_delta_last_1s < 0.05
            is_stable = bool(is_successful and rhs_converged and window_converged)

            # Store in matrices
            stability_matrix[j_idx, g_idx] = 1.0 if is_stable else 0.0
            max_dy_dt_matrix[j_idx, g_idx] = final_max_abs_dy_dt
            p2p_matrix[j_idx, g_idx] = max_abs_delta_last_1s

            # Log this combination
            record = {
                "j": j_val,
                "g": g_val,
                "num_steps": num_steps,
                "diffrax_result_code": sol_result_val,
                "diffrax_result_str": sol_result_str,
                "final_max_abs_dy_dt": final_max_abs_dy_dt,
                "final_rms_dy_dt": final_rms_dy_dt,
                "max_abs_delta_last_1s": max_abs_delta_last_1s,
                "stable": is_stable,
            }
            results_grid.append(record)
            print(f"  Result: stable={is_stable}, steps={num_steps}, max|dy/dt|={final_max_abs_dy_dt:.2e}, P2P={max_abs_delta_last_1s:.2e}")

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
