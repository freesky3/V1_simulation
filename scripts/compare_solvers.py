#!/usr/bin/env python
"""Compare ODE solver accuracy between Tsit5 and Heun on the same batch.

Run from the project root:
    python scripts/compare_solvers.py background=none

Runs 1 batch through both diffrax.Tsit5 (7-stage, 5th order) and
diffrax.Heun (2-stage, 2nd order) with the same fixed time grid, then
reports element-wise differences in firing rates and BCM weight updates.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from time import perf_counter

import numpy as np


def main():
    from v1_simulation.config import load_config
    from v1_simulation.config.validation import validate_config
    from v1_simulation.network.builder import build_network_state
    from v1_simulation.simulation.pipeline import default_training_time_grid
    from v1_simulation.solvers.wilson_cowan import solve_wilson_cowan_batch
    from v1_simulation.training.natural_inputs import build_natural_image_l4_drive
    from v1_simulation.training.trainer import BCMTrainer

    # ---- Load config (use tsit5 as baseline) ----
    overrides = sys.argv[1:] if len(sys.argv) > 1 else []
    if not any("experiment=" in o for o in overrides):
        overrides = ["+experiment=bcm_train"] + overrides
    # Force tsit5 as default
    if not any("solver=" in o for o in overrides):
        overrides = ["solver=diffrax_tsit5"] + overrides

    cfg = load_config(overrides=overrides)
    cfg.mode = "train"
    cfg.training.enabled = True
    validate_config(cfg)

    print("=" * 70)
    print("Solver Accuracy Comparison: Tsit5 vs Heun")
    print("=" * 70)

    # ---- Build network ----
    print("Building network...")
    t0 = perf_counter()
    network = build_network_state(cfg)
    print(f"  Built in {perf_counter() - t0:.2f}s")
    print(f"  n_E={network.layout.n_E}, n_I={network.layout.n_I}, "
          f"n_X={network.layout.n_X}")
    print()

    # ---- Build natural image drive ----
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

    # ---- Build time grid ----
    time_grid = default_training_time_grid(cfg)
    print(f"Time grid: {time_grid.size} steps, "
          f"t=[{time_grid[0]:.4f}, {time_grid[-1]:.4f}], "
          f"dt={time_grid[1]-time_grid[0]:.6f}")
    print()

    # ---- Get 1 batch of samples ----
    epoch_samples = list(natural_sampler.make_epoch(
        limit=cfg.training.natural_image.limit,
        shuffle_paths=True,
        shuffle_samples=True,
    ))
    batch_size = int(cfg.training.bcm.batch_size)
    batch = tuple(epoch_samples[:batch_size])
    print(f"Batch size: {len(batch)}")

    # ---- Build drive function ----
    drive_func = natural_drive.make_static_batch_func(batch)
    print()

    # ---- Solver configs ----
    cfg_tsit5 = deepcopy(cfg)
    cfg_tsit5.solver.diffrax.solver = "tsit5"

    cfg_heun = deepcopy(cfg)
    cfg_heun.solver.diffrax.solver = "heun"

    # ---- Run Tsit5 (reference) ----
    print("--- Running Tsit5 (reference, 5th order, 7 stages/step) ---")
    t0 = perf_counter()
    result_tsit5 = solve_wilson_cowan_batch(
        network=network,
        external_drive=drive_func,
        time=time_grid,
        n_batch=len(batch),
        solver_config=cfg_tsit5.solver,
        transfer_config=cfg_tsit5.solver.transfer,
        training_bcm=cfg_tsit5.training.bcm,
        background_trace=None,
        store_trajectory=False,
        stop_at_steady_state=False,
    )
    # Force GPU sync
    try:
        import jax
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            result_tsit5,
        )
    except ImportError:
        pass
    t_tsit5 = perf_counter() - t0
    print(f"  Time: {t_tsit5:.2f}s")
    print(f"  exc: shape={result_tsit5.exc.shape}, "
          f"mean={np.mean(result_tsit5.exc):.6f}, "
          f"max={np.max(result_tsit5.exc):.6f}")
    print(f"  inh: shape={result_tsit5.inh.shape}, "
          f"mean={np.mean(result_tsit5.inh):.6f}, "
          f"max={np.max(result_tsit5.inh):.6f}")
    print()

    # ---- Run Heun ----
    print("--- Running Heun (candidate, 2nd order, 2 stages/step) ---")
    t0 = perf_counter()
    result_heun = solve_wilson_cowan_batch(
        network=network,
        external_drive=drive_func,
        time=time_grid,
        n_batch=len(batch),
        solver_config=cfg_heun.solver,
        transfer_config=cfg_heun.solver.transfer,
        training_bcm=cfg_heun.training.bcm,
        background_trace=None,
        store_trajectory=False,
        stop_at_steady_state=False,
    )
    try:
        import jax
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            result_heun,
        )
    except ImportError:
        pass
    t_heun = perf_counter() - t0
    print(f"  Time: {t_heun:.2f}s")
    print(f"  exc: shape={result_heun.exc.shape}, "
          f"mean={np.mean(result_heun.exc):.6f}, "
          f"max={np.max(result_heun.exc):.6f}")
    print(f"  inh: shape={result_heun.inh.shape}, "
          f"mean={np.mean(result_heun.inh):.6f}, "
          f"max={np.max(result_heun.inh):.6f}")
    print()

    # ---- Compare firing rates ----
    exc_tsit5 = np.asarray(result_tsit5.exc, dtype=np.float64)
    exc_heun = np.asarray(result_heun.exc, dtype=np.float64)
    inh_tsit5 = np.asarray(result_tsit5.inh, dtype=np.float64)
    inh_heun = np.asarray(result_heun.inh, dtype=np.float64)

    exc_diff = np.abs(exc_tsit5 - exc_heun)
    inh_diff = np.abs(inh_tsit5 - inh_heun)

    # Relative error (avoid div by zero)
    exc_scale = np.maximum(np.abs(exc_tsit5), 1e-10)
    inh_scale = np.maximum(np.abs(inh_tsit5), 1e-10)
    exc_rel = exc_diff / exc_scale
    inh_rel = inh_diff / inh_scale

    print("=" * 70)
    print("FIRING RATE COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30s} {'Excitatory':>15s} {'Inhibitory':>15s}")
    print("-" * 60)
    print(f"{'Mean absolute error':<30s} {np.mean(exc_diff):>15.2e} {np.mean(inh_diff):>15.2e}")
    print(f"{'Max absolute error':<30s} {np.max(exc_diff):>15.2e} {np.max(inh_diff):>15.2e}")
    print(f"{'Mean relative error':<30s} {np.mean(exc_rel):>15.2e} {np.mean(inh_rel):>15.2e}")
    print(f"{'Max relative error':<30s} {np.max(exc_rel):>15.2e} {np.max(inh_rel):>15.2e}")
    print(f"{'Correlation (r)':<30s} {np.corrcoef(exc_tsit5.ravel(), exc_heun.ravel())[0,1]:>15.10f} "
          f"{np.corrcoef(inh_tsit5.ravel(), inh_heun.ravel())[0,1]:>15.10f}")
    print()

    # ---- Compare BCM weight updates ----
    print("=" * 70)
    print("BCM WEIGHT UPDATE COMPARISON")
    print("=" * 70)

    # Apply 1 BCM step with each result
    trainer_tsit5 = BCMTrainer(cfg.training.bcm, network)
    trainer_heun = BCMTrainer(cfg.training.bcm, network)

    # Initialize theta (step 0, no update)
    trainer_tsit5.train_batch(result_tsit5, epoch=1, batch_size=len(batch), images="compare")
    trainer_heun.train_batch(result_heun, epoch=1, batch_size=len(batch), images="compare")

    # Apply actual weight update (step 1)
    log_tsit5 = trainer_tsit5.train_batch(result_tsit5, epoch=1, batch_size=len(batch), images="compare")
    log_heun = trainer_heun.train_batch(result_heun, epoch=1, batch_size=len(batch), images="compare")

    from scipy import sparse

    w_tsit5 = trainer_tsit5.state.network.weights
    w_heun = trainer_heun.state.network.weights
    if sparse.issparse(w_tsit5):
        w_tsit5 = w_tsit5.toarray()
    if sparse.issparse(w_heun):
        w_heun = w_heun.toarray()
    w_tsit5 = np.asarray(w_tsit5, dtype=np.float64)
    w_heun = np.asarray(w_heun, dtype=np.float64)

    w_diff = np.abs(w_tsit5 - w_heun)
    nz_mask = w_tsit5 != 0.0
    w_rel = np.zeros_like(w_diff)
    w_rel[nz_mask] = w_diff[nz_mask] / np.abs(w_tsit5[nz_mask])

    print(f"{'Metric':<35s} {'Value':>15s}")
    print("-" * 50)
    print(f"{'Weight matrix shape':<35s} {str(w_tsit5.shape):>15s}")
    print(f"{'Nonzero elements':<35s} {np.count_nonzero(w_tsit5):>15d}")
    print(f"{'Mean abs weight diff':<35s} {np.mean(w_diff):>15.2e}")
    print(f"{'Max abs weight diff':<35s} {np.max(w_diff):>15.2e}")
    print(f"{'Mean rel weight diff (nonzero)':<35s} {np.mean(w_rel[nz_mask]):>15.2e}")
    print(f"{'Max rel weight diff (nonzero)':<35s} {np.max(w_rel[nz_mask]):>15.2e}")
    print(f"{'Tsit5 aE_mean':<35s} {log_tsit5.aE_mean:>15.6f}")
    print(f"{'Heun  aE_mean':<35s} {log_heun.aE_mean:>15.6f}")
    print(f"{'Tsit5 aI_mean':<35s} {log_tsit5.aI_mean:>15.6f}")
    print(f"{'Heun  aI_mean':<35s} {log_heun.aI_mean:>15.6f}")
    print()

    # ---- Verdict ----
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    max_exc_rel = np.max(exc_rel)
    max_inh_rel = np.max(inh_rel)
    max_w_rel = np.max(w_rel[nz_mask]) if np.any(nz_mask) else 0.0
    corr_exc = np.corrcoef(exc_tsit5.ravel(), exc_heun.ravel())[0, 1]
    corr_inh = np.corrcoef(inh_tsit5.ravel(), inh_heun.ravel())[0, 1]

    all_ok = True
    checks = [
        ("Exc max relative error < 1e-2", max_exc_rel < 1e-2),
        ("Inh max relative error < 1e-2", max_inh_rel < 1e-2),
        ("Weight max relative error < 1e-2", max_w_rel < 1e-2),
        ("Exc correlation > 0.9999", corr_exc > 0.9999),
        ("Inh correlation > 0.9999", corr_inh > 0.9999),
    ]
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {status}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print("  ➜ Heun solver produces results within acceptable tolerance.")
        print(f"  ➜ Speedup: {t_tsit5:.2f}s → {t_heun:.2f}s ({t_tsit5/t_heun:.1f}x)")
    else:
        print("  ➜ WARNING: Heun solver shows significant deviation from Tsit5.")
        print("  ➜ Consider using Tsit5 or reducing the time step size.")

    print()


if __name__ == "__main__":
    main()
