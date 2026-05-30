#!/usr/bin/env python
"""Verify Diffrax Tsit5 reliability and performance at scale (N ≈ 6000).

This script performs four distinct test suites to validate the JAX-based diffrax.Tsit5 solver
under linear exact, non-normal transient, nonlinear random, and project-like Wilson-Cowan dynamics.
It compares different step sizes (dt) and data types (float32 vs float64) and saves diagnostic plots
and tables to runs/verify_diffrax_tsit5_reliability/.
"""
from __future__ import annotations

import os
import json
import sys
import time as time_mod
import numpy as np

# Configure JAX for 64-bit floating point precision
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

# Set up non-interactive matplotlib backend for cloud/headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class StaticDrive:
    """Mock external drive that returns a constant stimulus vector."""
    def __init__(self, val: np.ndarray):
        self.val = val
        self.is_time_dependent = False

    def __call__(self, t: float) -> np.ndarray:
        return self.val


def get_linear_exact_solution(x0: np.ndarray, alpha: np.ndarray, beta: np.ndarray, t: float) -> np.ndarray:
    """Compute exact solution for the block-diagonal linear system at time t."""
    N = x0.shape[0]
    x0_reshaped = x0.reshape(N // 2, 2, -1)
    
    cos_val = np.cos(beta * t)
    sin_val = np.sin(beta * t)
    exp_val = np.exp(-alpha * t)
    
    exp_val = exp_val[:, np.newaxis]
    cos_val = cos_val[:, np.newaxis]
    sin_val = sin_val[:, np.newaxis]
    
    x0_0 = x0_reshaped[:, 0, :]
    x0_1 = x0_reshaped[:, 1, :]
    
    x_t_0 = exp_val * (x0_0 * cos_val + x0_1 * sin_val)
    x_t_1 = exp_val * (-x0_0 * sin_val + x0_1 * cos_val)
    
    x_t = np.stack([x_t_0, x_t_1], axis=1).reshape(N, -1)
    return x_t


def get_transient_exact_solution(x0: np.ndarray, gamma: np.ndarray, eta: np.ndarray, t: float) -> np.ndarray:
    """Compute exact solution for the non-normal transient system at time t."""
    N = x0.shape[0]
    x0_reshaped = x0.reshape(N // 2, 2, -1)
    
    x0_0 = x0_reshaped[:, 0, :]
    x0_1 = x0_reshaped[:, 1, :]
    
    exp_val = np.exp(-gamma * t)[:, np.newaxis]
    
    x_t_0 = x0_0 * exp_val
    x_t_1 = (x0_1 + eta[:, np.newaxis] * t * x0_0) * exp_val
    
    x_t = np.stack([x_t_0, x_t_1], axis=1).reshape(N, -1)
    return x_t


def analytical_solution_linear(x0: np.ndarray, alpha: np.ndarray, beta: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Compute analytical solution trajectory for the linear block system over t_grid."""
    traj = []
    for t in t_grid:
        traj.append(get_linear_exact_solution(x0, alpha, beta, float(t)))
    return np.stack(traj, axis=0)


def analytical_solution_transient(x0: np.ndarray, gamma: np.ndarray, eta: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Compute analytical solution trajectory for the non-normal transient system over t_grid."""
    traj = []
    for t in t_grid:
        traj.append(get_transient_exact_solution(x0, gamma, eta, float(t)))
    return np.stack(traj, axis=0)


def run_linear_exact(y0: np.ndarray, alpha: np.ndarray, beta: np.ndarray, t_grid: np.ndarray, dtype) -> tuple[np.ndarray, float]:
    """Simulate block-diagonal linear system using Diffrax Tsit5."""
    y0_jnp = jnp.array(y0, dtype=dtype)
    alpha_jnp = jnp.array(alpha, dtype=dtype)
    beta_jnp = jnp.array(beta, dtype=dtype)
    t_grid_jnp = jnp.array(t_grid, dtype=dtype)
    
    def vector_field(t, y, args):
        alpha_val, beta_val = args
        N = y.shape[0]
        y_reshaped = y.reshape(N // 2, 2, -1)
        y0_ch = y_reshaped[:, 0, :]
        y1_ch = y_reshaped[:, 1, :]
        
        dy0 = -alpha_val[:, jnp.newaxis] * y0_ch + beta_val[:, jnp.newaxis] * y1_ch
        dy1 = -beta_val[:, jnp.newaxis] * y0_ch - alpha_val[:, jnp.newaxis] * y1_ch
        
        dy = jnp.stack([dy0, dy1], axis=1).reshape(N, -1)
        return dy
    
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.ConstantStepSize()
    dt0 = t_grid_jnp[1] - t_grid_jnp[0]
    
    @jax.jit
    def solve():
        return diffrax.diffeqsolve(
            term,
            solver,
            t0=t_grid_jnp[0],
            t1=t_grid_jnp[-1],
            dt0=dt0,
            y0=y0_jnp,
            args=(alpha_jnp, beta_jnp),
            saveat=diffrax.SaveAt(ts=t_grid_jnp),
            stepsize_controller=stepsize_controller,
        )
    
    t0 = time_mod.perf_counter()
    sol = solve()
    sol.ys.block_until_ready()
    t_elapsed = time_mod.perf_counter() - t0
    
    return np.array(sol.ys), t_elapsed


def run_transient_exact(y0: np.ndarray, gamma: np.ndarray, eta: np.ndarray, t_grid: np.ndarray, dtype) -> tuple[np.ndarray, float]:
    """Simulate block-diagonal non-normal transient system using Diffrax Tsit5."""
    y0_jnp = jnp.array(y0, dtype=dtype)
    gamma_jnp = jnp.array(gamma, dtype=dtype)
    eta_jnp = jnp.array(eta, dtype=dtype)
    t_grid_jnp = jnp.array(t_grid, dtype=dtype)
    
    def vector_field(t, y, args):
        gamma_val, eta_val = args
        N = y.shape[0]
        y_reshaped = y.reshape(N // 2, 2, -1)
        y0_ch = y_reshaped[:, 0, :]
        y1_ch = y_reshaped[:, 1, :]
        
        dy0 = -gamma_val[:, jnp.newaxis] * y0_ch
        dy1 = eta_val[:, jnp.newaxis] * y0_ch - gamma_val[:, jnp.newaxis] * y1_ch
        
        dy = jnp.stack([dy0, dy1], axis=1).reshape(N, -1)
        return dy
    
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.ConstantStepSize()
    dt0 = t_grid_jnp[1] - t_grid_jnp[0]
    
    @jax.jit
    def solve():
        return diffrax.diffeqsolve(
            term,
            solver,
            t0=t_grid_jnp[0],
            t1=t_grid_jnp[-1],
            dt0=dt0,
            y0=y0_jnp,
            args=(gamma_jnp, eta_jnp),
            saveat=diffrax.SaveAt(ts=t_grid_jnp),
            stepsize_controller=stepsize_controller,
        )
    
    t0 = time_mod.perf_counter()
    sol = solve()
    sol.ys.block_until_ready()
    t_elapsed = time_mod.perf_counter() - t0
    
    return np.array(sol.ys), t_elapsed


def run_nonlinear_system(y0: np.ndarray, W: np.ndarray, g: float, t_grid: np.ndarray, dtype) -> tuple[np.ndarray, float, int, int, int]:
    """Simulate dense nonlinear random rate network using Diffrax Tsit5."""
    y0_jnp = jnp.array(y0, dtype=dtype)
    W_jnp = jnp.array(W, dtype=dtype)
    t_grid_jnp = jnp.array(t_grid, dtype=dtype)
    
    def vector_field(t, y, args):
        g_val, W_val = args
        return -y + jnp.tanh(g_val * (W_val @ y))
    
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.ConstantStepSize()
    dt0 = t_grid_jnp[1] - t_grid_jnp[0]
    
    @jax.jit
    def solve():
        return diffrax.diffeqsolve(
            term,
            solver,
            t0=t_grid_jnp[0],
            t1=t_grid_jnp[-1],
            dt0=dt0,
            y0=y0_jnp,
            args=(g, W_jnp),
            saveat=diffrax.SaveAt(ts=t_grid_jnp),
            stepsize_controller=stepsize_controller,
        )
    
    t0 = time_mod.perf_counter()
    sol = solve()
    sol.ys.block_until_ready()
    t_elapsed = time_mod.perf_counter() - t0
    
    stats = getattr(sol, "stats", {})
    num_steps = int(stats.get("num_steps", -1))
    num_accepted = int(stats.get("num_accepted_steps", -1))
    num_rejected = int(stats.get("num_rejected_steps", -1))
    
    return np.array(sol.ys), t_elapsed, num_steps, num_accepted, num_rejected


def run_project_rhs(network, drive_func, t_grid: np.ndarray, dtype, solver_cfg) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate project-like RHS model using solve_wilson_cowan_batch."""
    from v1_simulation.solvers.wilson_cowan import solve_wilson_cowan_batch
    from v1_simulation.solvers.base import SolverOptions
    
    options = SolverOptions(
        backend="diffrax",
        method="adaptive",
        diffrax_solver="tsit5",
        jax_dtype="float32" if dtype == jnp.float32 else "float64",
        store_trajectory=True,
    )
    
    t0 = time_mod.perf_counter()
    result = solve_wilson_cowan_batch(
        network=network,
        external_drive=drive_func,
        time=t_grid,
        n_batch=drive_func.val.shape[1],
        solver_config=solver_cfg,
        options=options,
        store_trajectory=True,
    )
    # Force GPU sync if JAX is backend
    try:
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            result,
        )
    except Exception:
        pass
    t_elapsed = time_mod.perf_counter() - t0
    
    # exc_trajectory has shape (time, batch, neurons)
    # convert to numpy array
    exc_traj = np.asarray(result.exc_trajectory)
    inh_traj = np.asarray(result.inh_trajectory)
    
    return exc_traj, inh_traj, t_elapsed


def main():
    # Setup directories
    out_dir = "runs/verify_diffrax_tsit5_reliability"
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize global metadata for table
    table_rows = []
    summary_data = {}
    
    # Dimension of systems
    N = 6000
    print("=" * 80)
    print(f"VERIFYING DIFFRAX TSIT5 RELIABILITY AT SCALE (N = {N})")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # TEST A: Linear exact block system
    # -------------------------------------------------------------------------
    print("\n[Test A] Linear Exact Block System...")
    np.random.seed(42)
    alpha = np.random.uniform(0.1, 0.5, N // 2)
    beta = np.random.uniform(1.0, 5.0, N // 2)
    x0_linear = np.random.normal(0, 1.0, (N, 1))
    
    dts = [0.02, 0.01, 0.005]
    dtypes = [jnp.float64, jnp.float32]
    
    linear_results = {}
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for dtype in dtypes:
        dtype_str = "float64" if dtype == jnp.float64 else "float32"
        linear_results[dtype_str] = {}
        
        for dt in dts:
            t_grid = np.arange(0.0, 10.0 + dt/2, dt)
            ys_num, t_elapsed = run_linear_exact(x0_linear, alpha, beta, t_grid, dtype)
            ys_exact = analytical_solution_linear(x0_linear, alpha, beta, t_grid)
            
            # Compute errors
            # Rel L2 error at t_final
            final_num = ys_num[-1, :, 0]
            final_exact = ys_exact[-1, :, 0]
            rel_l2_error = np.linalg.norm(final_num - final_exact) / np.linalg.norm(final_exact)
            
            # Relative error trace across time
            num_norms = np.linalg.norm(ys_num[:, :, 0], axis=1)
            exact_norms = np.linalg.norm(ys_exact[:, :, 0], axis=1)
            norm_err_trace = np.abs(num_norms - exact_norms) / np.maximum(exact_norms, 1e-10)
            
            linear_results[dtype_str][dt] = {
                "rel_l2": float(rel_l2_error),
                "runtime": t_elapsed,
                "norm_err_trace": norm_err_trace.tolist()
            }
            
            # Print to stdout
            print(f"  {dtype_str} / dt={dt:<5}: final rel L2 error = {rel_l2_error:.2e} | runtime = {t_elapsed:.3f}s")
            
            # Add to table rows
            status = "yes" if rel_l2_error < (1e-6 if dtype_str == "float64" else 1e-3) else "no"
            table_rows.append({
                "test": "linear_exact",
                "N": N,
                "dtype": dtype_str,
                "dt_rtol": f"dt={dt}",
                "metric": "final_rel_l2",
                "value": f"{rel_l2_error:.2e}",
                "pass": status
            })
            
            # Plot
            lbl = f"dt={dt} ({dtype_str})"
            ax_idx = 0 if dtype_str == "float64" else 1
            axes[ax_idx].plot(t_grid, norm_err_trace, label=lbl)
            
    # Compute convergence ratio for float64
    err_01 = linear_results["float64"][0.01]["rel_l2"]
    err_005 = linear_results["float64"][0.005]["rel_l2"]
    ratio = err_01 / np.maximum(err_005, 1e-15)
    print(f"  float64 Tsit5 convergence ratio (dt=0.01 vs dt=0.005) = {ratio:.2f} (theoretical: ~32.0)")
    
    # Save Test A plot
    axes[0].set_title("Relative Norm Error Trace over Time (Float64)")
    axes[0].set_ylabel("Error")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_title("Relative Norm Error Trace over Time (Float32)")
    axes[1].set_ylabel("Error")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "linear_exact_error.png"), dpi=200)
    plt.close()
    
    summary_data["linear_exact"] = linear_results
    summary_data["linear_convergence_ratio"] = float(ratio)
    
    # -------------------------------------------------------------------------
    # TEST B: Non-normal transient growth
    # -------------------------------------------------------------------------
    print("\n[Test B] Non-normal Transient Growth System...")
    np.random.seed(42)
    gamma = np.random.uniform(0.5, 1.5, N // 2)
    eta = np.random.uniform(5.0, 10.0, N // 2)
    
    x0_transient = np.zeros((N, 1))
    x0_transient[0::2, 0] = 1.0 # First component of each 2-cell loop starts at 1
    x0_transient[1::2, 0] = 0.0 # Second component starts at 0
    
    dt = 0.01
    t_grid = np.arange(0.0, 8.0 + dt/2, dt)
    ys_num_64, t_elapsed_64 = run_transient_exact(x0_transient, gamma, eta, t_grid, jnp.float64)
    ys_exact = analytical_solution_transient(x0_transient, gamma, eta, t_grid)
    
    # Compute errors
    final_num = ys_num_64[-1, :, 0]
    final_exact = ys_exact[-1, :, 0]
    rel_l2_error = np.linalg.norm(final_num - final_exact) / np.linalg.norm(final_exact)
    
    # Check peak amplification of component 1 (index 1 of first loop)
    # The exact peak of block 0 component 1 should be at t_peak = 1 / gamma[0]
    t_peak_exact = 1.0 / gamma[0]
    val_peak_exact = eta[0] * t_peak_exact * np.exp(-gamma[0] * t_peak_exact)
    
    # Find numerical peak for block 0 component 1
    comp1_traj = ys_num_64[:, 1, 0]
    peak_idx = np.argmax(comp1_traj)
    t_peak_num = t_grid[peak_idx]
    val_peak_num = comp1_traj[peak_idx]
    
    print(f"  float64 / dt={dt}: final rel L2 error = {rel_l2_error:.2e} | runtime = {t_elapsed_64:.3f}s")
    print(f"  Block 0 component 1 peak: exact={val_peak_exact:.4f} at t={t_peak_exact:.4f}s | num={val_peak_num:.4f} at t={t_peak_num:.4f}s")
    
    # Save plot for Test B
    plt.figure(figsize=(8, 5))
    plt.plot(t_grid, ys_exact[:, 0, 0], 'k-', label="Component 0 (Exact)")
    plt.plot(t_grid, ys_exact[:, 1, 0], 'r-', label="Component 1 (Exact)")
    plt.plot(t_grid, ys_num_64[:, 0, 0], 'b--', label="Component 0 (Numerical)")
    plt.plot(t_grid, ys_num_64[:, 1, 0], 'g--', label="Component 1 (Numerical)")
    plt.axvline(t_peak_exact, color='gray', linestyle=':', label=f"Theoretical Peak (t={t_peak_exact:.2f}s)")
    plt.title("Transient Growth Amplification in Block 0 (N=6000 scale)")
    plt.xlabel("Time (s)")
    plt.ylabel("Activity")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "transient_amplification_compare.png"), dpi=200)
    plt.close()
    
    status = "yes" if rel_l2_error < 1e-6 else "no"
    table_rows.append({
        "test": "transient_growth",
        "N": N,
        "dtype": "float64",
        "dt_rtol": f"dt={dt}",
        "metric": "final_rel_l2",
        "value": f"{rel_l2_error:.2e}",
        "pass": status
    })
    
    summary_data["transient_growth"] = {
        "rel_l2": float(rel_l2_error),
        "runtime": t_elapsed_64,
        "peak_exact": float(val_peak_exact),
        "peak_num": float(val_peak_num)
    }
    
    # -------------------------------------------------------------------------
    # TEST C: Nonlinear stability boundary sweep (g-sweep)
    # -------------------------------------------------------------------------
    print("\n[Test C] Nonlinear Stability Sweep...")
    np.random.seed(42)
    W_np = np.random.normal(0, 1.0, (N, N)) / np.sqrt(N)
    x0_nonlinear = np.random.normal(0, 0.1, (N, 1))
    
    gs = [0.5, 0.9, 1.1, 1.5]
    dt_c = 0.01
    t_grid_c = np.arange(0.0, 20.0 + dt_c/2, dt_c)
    
    nonlinear_results = {}
    
    plt.figure(figsize=(10, 6))
    
    for g in gs:
        ys_num, t_elapsed, num_steps, num_accepted, num_rejected = run_nonlinear_system(
            x0_nonlinear, W_np, g, t_grid_c, jnp.float32
        )
        
        # Calculate RMS activity over time
        rms_trace = np.sqrt(np.mean(ys_num[:, :, 0]**2, axis=1))
        
        # Stats over the last 20% of simulation time (t in [16.0, 20.0])
        last_20_idx = int(len(t_grid_c) * 0.8)
        mean_rms_last20 = np.mean(rms_trace[last_20_idx:])
        std_rms_last20 = np.std(rms_trace[last_20_idx:])
        final_rms = rms_trace[-1]
        
        nonlinear_results[g] = {
            "mean_rms_last20": float(mean_rms_last20),
            "std_rms_last20": float(std_rms_last20),
            "final_rms": float(final_rms),
            "runtime": t_elapsed,
            "steps": num_steps,
            "accepted": num_accepted,
            "rejected": num_rejected
        }
        
        print(f"  g={g:<5}: final RMS = {final_rms:.4f} | mean RMS (last 20%) = {mean_rms_last20:.4f} +/- {std_rms_last20:.4f} | runtime = {t_elapsed:.3f}s")
        
        # Check stability predictions
        # g < 1 should decay to zero (< 1e-2 in RMS)
        # g > 1 should be active (> 1e-2 in RMS)
        passed_chk = "yes"
        if g < 1.0 and final_rms > 1e-2:
            passed_chk = "no"
        elif g > 1.0 and mean_rms_last20 < 1e-2:
            passed_chk = "no"
            
        table_rows.append({
            "test": f"nonlinear_g{g}",
            "N": N,
            "dtype": "float32",
            "dt_rtol": f"dt={dt_c}",
            "metric": "mean_rms_last20",
            "value": f"{mean_rms_last20:.4f}",
            "pass": passed_chk
        })
        
        plt.plot(t_grid_c, rms_trace, label=f"g={g}")
        
    plt.title(f"Nonlinear Network RMS Activity Sweep (N={N}, float32)")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Activity")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "nonlinear_rms_g_sweep.png"), dpi=200)
    plt.close()
    
    summary_data["nonlinear_g_sweep"] = nonlinear_results
    
    # -------------------------------------------------------------------------
    # TEST D: Project-like RHS model
    # -------------------------------------------------------------------------
    print("\n[Test D] Project-like Wilson-Cowan RHS Test...")
    
    # Try importing configuration files and building the network
    try:
        from v1_simulation.config import load_config
        from v1_simulation.config.validation import validate_config
        from v1_simulation.network.builder import build_network_state
        
        # Load default training config
        cfg = load_config(overrides=["+experiment=bcm_train"])
        cfg.mode = "train"
        cfg.training.enabled = True
        validate_config(cfg)
        
        from v1_simulation.solvers.base import NetworkLayout
        network = build_network_state(cfg)
        layout = NetworkLayout.from_network_state(network)
        print(f"  Network built: n_exc={layout.n_exc}, n_inh={layout.n_inh}, n_ext={layout.n_ext} (n_rates={layout.n_rates})")
        
        # Verify dimension matches target dimension
        print(f"  Actual simulation rate dimension: {layout.n_rates}")
        
        n_batch = 1
        val_ext = np.ones((layout.n_ext, n_batch)) * 0.5
        drive_func = StaticDrive(val_ext)
        
        dt_d = 0.002
        t_grid_d = np.arange(0.0, 1.0 + dt_d/2, dt_d)
        
        dt_ref = 0.001
        t_grid_ref = np.arange(0.0, 1.0 + dt_ref/2, dt_ref)
        
        project_results = {}
        
        plt.figure(figsize=(10, 6))
        
        for dtype_str, dtype in [("float64", jnp.float64), ("float32", jnp.float32)]:
            print(f"  Running solver integration for {dtype_str}...")
            # Candidate solver (dt=0.01)
            exc_01, inh_01, t_01 = run_project_rhs(network, drive_func, t_grid_d, dtype, cfg.solver)
            # Reference solver (dt=0.001)
            exc_ref, inh_ref, t_ref = run_project_rhs(network, drive_func, t_grid_ref, dtype, cfg.solver)
            
            # Compare at t_final
            final_exc_01 = exc_01[-1, 0, :]
            final_inh_01 = inh_01[-1, 0, :]
            final_exc_ref = exc_ref[-1, 0, :]
            final_inh_ref = inh_ref[-1, 0, :]
            
            err_exc = final_exc_01 - final_exc_ref
            err_inh = final_inh_01 - final_inh_ref
            
            num_norm = np.sqrt(np.linalg.norm(final_exc_01)**2 + np.linalg.norm(final_inh_01)**2)
            ref_norm = np.sqrt(np.linalg.norm(final_exc_ref)**2 + np.linalg.norm(final_inh_ref)**2)
            err_norm = np.sqrt(np.linalg.norm(err_exc)**2 + np.linalg.norm(err_inh)**2)
            
            rel_l2 = err_norm / np.maximum(ref_norm, 1e-10)
            
            project_results[dtype_str] = {
                "rel_l2": float(rel_l2),
                "runtime_dt001": t_01,
                "runtime_ref": t_ref
            }
            
            print(f"    {dtype_str} Rel L2 error (dt={dt_d} vs dt={dt_ref}) = {rel_l2:.2e} | runtime = {t_01:.3f}s (ref: {t_ref:.3f}s)")
            
            status = "yes" if rel_l2 < (1e-5 if dtype_str == "float64" else 1e-3) else "no"
            table_rows.append({
                "test": "project_rhs",
                "N": layout.n_rates,
                "dtype": dtype_str,
                "dt_rtol": f"dt={dt_d} vs ref",
                "metric": "final_rel_l2",
                "value": f"{rel_l2:.2e}",
                "pass": status
            })
            
            # Plot comparison of firing rates across neurons
            plt.plot(final_exc_ref, label=f"Ref Exc ({dtype_str})", alpha=0.5)
            plt.plot(final_exc_01, ':', label=f"dt={dt_d} Exc ({dtype_str})")
            
        plt.title(f"Wilson-Cowan Final Rate Profile (dt=0.01 vs dt=0.001, N={layout.n_rates})")
        plt.ylabel("Firing Rate (Hz)")
        plt.xlabel("Neuron Index (Excitatory)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "project_rhs_compare.png"), dpi=200)
        plt.close()
        
        summary_data["project_rhs"] = project_results
        
    except Exception as e:
        print(f"  [Warning] Skipping Test D due to missing config or building errors: {e}")
        table_rows.append({
            "test": "project_rhs",
            "N": "N/A",
            "dtype": "N/A",
            "dt_rtol": "N/A",
            "metric": "final_rel_l2",
            "value": "SKIPPED",
            "pass": "no"
        })
        summary_data["project_rhs"] = {"error": str(e)}

    # -------------------------------------------------------------------------
    # PRINT RESULTS TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY VERIFICATION TABLE")
    print("=" * 80)
    print("| test | N | dtype | dt/rtol | metric | value | pass |")
    print("|---|---:|---|---|---|---:|---|")
    for r in table_rows:
        print(f"| {r['test']} | {r['N']} | {r['dtype']} | {r['dt_rtol']} | {r['metric']} | {r['value']} | {r['pass']} |")
    print("=" * 80)
    
    # Save summary markdown and JSON
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
        
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("# Diffrax Tsit5 Verification Summary\n\n")
        f.write("| test | N | dtype | dt/rtol | metric | value | pass |\n")
        f.write("|---|---:|---|---|---|---:|---|\n")
        for r in table_rows:
            f.write(f"| {r['test']} | {r['N']} | {r['dtype']} | {r['dt_rtol']} | {r['metric']} | {r['value']} | {r['pass']} |\n")
            
    print(f"\nAll verification summaries and plots saved in directory: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
