#!/usr/bin/env python
"""Profile each stage of a single BCM training batch.

Run from the project root (requires Van Hateren dataset in data/vanhateren_iml/):

    python scripts/profile_training_batch.py solver=diffrax_tsit5 background=none

    python scripts/profile_training_batch.py solver=scipy_rk4 background=none

The script runs 3 batches:
  batch 0 = warm-up / JIT compile (not representative of steady-state)
  batch 1 = cached JIT (real training speed)
  batch 2 = repeat to confirm stability
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def block_until_ready_tree(x):
    """Recursively call block_until_ready on all JAX arrays in a pytree."""
    try:
        import jax
        return jax.tree_util.tree_map(
            lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
            x,
        )
    except ImportError:
        return x


@dataclass
class TimingRecord:
    """Accumulates per-stage timings across batches."""
    stages: list[str] = field(default_factory=list)
    batch_times: dict[int, dict[str, float]] = field(default_factory=dict)
    _t0: float = field(default=0.0, repr=False)

    def start(self, batch_idx: int):
        self.batch_times[batch_idx] = {}
        self._t0 = perf_counter()

    def mark(self, batch_idx: int, stage: str):
        t = perf_counter()
        elapsed = t - self._t0
        self.batch_times[batch_idx][stage] = elapsed
        if stage not in self.stages:
            self.stages.append(stage)
        self._t0 = t

    def print_table(self):
        batch_ids = sorted(self.batch_times.keys())
        labels = {0: "batch0(compile)", 1: "batch1(cached)", 2: "batch2(cached)"}
        headers = ["Stage"] + [labels.get(i, f"batch{i}") for i in batch_ids]

        # Calculate column widths
        stage_width = max(len(s) for s in self.stages + ["TOTAL"]) + 2
        col_widths = [stage_width] + [max(len(h), 14) + 2 for h in headers[1:]]

        # Header
        row = headers[0].ljust(col_widths[0])
        for j, h in enumerate(headers[1:]):
            row += h.rjust(col_widths[j + 1])
        print(row)
        print("-" * len(row))

        # Stage rows
        totals = {b: 0.0 for b in batch_ids}
        for stage in self.stages:
            row = stage.ljust(col_widths[0])
            for j, b in enumerate(batch_ids):
                t = self.batch_times[b].get(stage, 0.0)
                totals[b] += t
                row += f"{t:>10.2f} s".rjust(col_widths[j + 1])
            print(row)

        # Total row
        print("-" * len(row))
        row = "TOTAL".ljust(col_widths[0])
        for j, b in enumerate(batch_ids):
            row += f"{totals[b]:>10.2f} s".rjust(col_widths[j + 1])
        print(row)


# ---------------------------------------------------------------------------
# Main profiling logic
# ---------------------------------------------------------------------------

def main():
    from v1_simulation.config import load_config
    from v1_simulation.config.validation import validate_config
    from v1_simulation.network.builder import build_network_state
    from v1_simulation.simulation.pipeline import default_training_time_grid
    from v1_simulation.training.natural_inputs import build_natural_image_l4_drive
    from v1_simulation.training.trainer import BCMTrainer
    from v1_simulation.solvers.wilson_cowan import solve_wilson_cowan_batch
    from v1_simulation.stimuli.background import validate_time_grid

    # ---- Load config with CLI overrides ----
    overrides = sys.argv[1:] if len(sys.argv) > 1 else []
    # Auto-add experiment override if not provided
    if not any("experiment=" in o for o in overrides):
        overrides = ["+experiment=bcm_train"] + overrides

    cfg = load_config(overrides=overrides)
    cfg.mode = "train"
    cfg.training.enabled = True
    validate_config(cfg)

    steady_state_enabled = (
        bool(cfg.training.bcm.dynamic_steady_state)
        and not bool(cfg.background.enabled)
    )

    print("=" * 70)
    print("BCM Training Batch Profiler")
    print("=" * 70)
    print(f"Solver backend       : {cfg.solver.backend}")
    print(f"Solver method        : {cfg.solver.method}")
    if cfg.solver.jax:
        print(f"JAX sparse           : {cfg.solver.jax.prefer_sparse}")
        print(f"JAX dense max MB     : {cfg.solver.jax.dense_max_mb}")
    print(f"Background           : {cfg.background.enabled}")
    print(f"Steady state         : {steady_state_enabled}")
    print(f"Batch size           : {cfg.training.bcm.batch_size}")
    print(f"Patches/image        : {cfg.training.natural_image.patches_per_image}")
    print()

    # ---- Build network ----
    print("Building network...")
    t0 = perf_counter()
    network = build_network_state(cfg)
    print(f"  Built in {perf_counter() - t0:.2f}s")
    print(f"  n_E={network.layout.n_E}, n_I={network.layout.n_I}, "
          f"n_X={network.layout.n_X}")
    print(f"  Weights: shape={network.weights.shape}, nnz={network.weights.nnz}")
    print()

    # ---- Build natural image drive + sampler ----
    print("Building natural image drive...")
    t0 = perf_counter()
    natural_drive, natural_sampler = build_natural_image_l4_drive(
        cfg=cfg.training.natural_image,
        stimulus_cfg=cfg.stimulus,
        model_cfg=cfg.model,
        layers_cfg=cfg.model.layers,
        l4_layer=network.layout.l4,
        l4_tunings=network.layout.l4_tunings,
        l4_pref_dirs=network.layout.l4_pref_dirs,
    )
    print(f"  Built in {perf_counter() - t0:.2f}s")
    print()

    # ---- Build time grid ----
    time_grid = default_training_time_grid(cfg)
    print(f"Time grid: {time_grid.size} steps, "
          f"t=[{time_grid[0]:.4f}, {time_grid[-1]:.4f}], "
          f"dt={time_grid[1]-time_grid[0]:.6f}")
    print()

    # ---- Sample batches ----
    epoch_samples = list(natural_sampler.make_epoch(
        limit=cfg.training.natural_image.limit,
        shuffle_paths=True,
        shuffle_samples=True,
    ))
    batch_size = int(cfg.training.bcm.batch_size)
    n_batches_epoch = (len(epoch_samples) + batch_size - 1) // batch_size
    print(f"Total samples: {len(epoch_samples)}, "
          f"batches/epoch: {n_batches_epoch}")

    batches = []
    for i in range(3):
        start = (i * batch_size) % len(epoch_samples)
        end = min(start + batch_size, len(epoch_samples))
        batch = list(epoch_samples[start:end])
        # Pad if needed (shouldn't happen for first 3 batches)
        while len(batch) < batch_size:
            batch.append(epoch_samples[len(batch) % len(epoch_samples)])
        batches.append(tuple(batch))

    print(f"Prepared {len(batches)} batches of size {batch_size}")
    print()

    # ---- JAX info ----
    try:
        import jax
        import jax.numpy as jnp
        print(f"JAX version      : {jax.__version__}")
        print(f"jax_enable_x64   : {jax.config.jax_enable_x64}")
        devices = jax.devices()
        print(f"JAX devices      : {[str(d) for d in devices]}")
        print(f"Default backend  : {jax.default_backend()}")
    except ImportError:
        print("JAX: not available (scipy backend only)")
    print()

    # ---- Create trainer ----
    trainer = BCMTrainer(cfg.training.bcm, network)

    # ---- Profile 3 batches ----
    timer = TimingRecord()

    for bi in range(3):
        batch = batches[bi]
        label = "WARMUP/COMPILE" if bi == 0 else "CACHED"
        print(f"--- Batch {bi} ({label}) ---")
        timer.start(bi)

        # Stage 1: Gabor projection
        drive_func = natural_drive.make_static_batch_func(batch)
        timer.mark(bi, "Gabor projection")

        # Stage 2: Full ODE solve (precompute + transfer + JIT + solve)
        dynamics = solve_wilson_cowan_batch(
            network=trainer.state.network,
            external_drive=drive_func,
            time=time_grid,
            n_batch=len(batch),
            solver_config=cfg.solver,
            transfer_config=cfg.solver.transfer,
            training_bcm=cfg.training.bcm,
            background_trace=None,
            store_trajectory=False,
            stop_at_steady_state=steady_state_enabled,
        )
        # Force GPU synchronization for accurate timing
        block_until_ready_tree(dynamics)
        timer.mark(bi, "ODE solve (total)")

        # Stage 3: BCM plasticity update
        log_row = trainer.train_batch(
            dynamics,
            epoch=1,
            batch_size=len(batch),
            images="profiling",
        )
        timer.mark(bi, "BCM update")

        # Print batch details
        if bi == 0:
            print(f"  exc: dtype={dynamics.exc.dtype}, shape={dynamics.exc.shape}")
            print(f"  inh: dtype={dynamics.inh.dtype}, shape={dynamics.inh.shape}")
            print(f"  time_steps={dynamics.time.size}")
            print(f"  steady_state_reached={dynamics.steady_state_reached}")
            if dynamics.steady_state_index is not None:
                print(f"  steady_state_index={dynamics.steady_state_index}"
                      f"/{time_grid.size}")
        print(f"  aE_mean={log_row.aE_mean:.4g}, aI_mean={log_row.aI_mean:.4g}, "
              f"updated={log_row.updated}")
        print()

    # ---- Results table ----
    print()
    print("=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)
    timer.print_table()
    print()

    # ---- Summary and estimate ----
    cached_total = sum(timer.batch_times[1].values())
    compile_total = sum(timer.batch_times[0].values())
    print(f"JIT compile overhead (batch0 - batch1): "
          f"{compile_total - cached_total:.2f}s")
    print(f"Steady-state batch time (batch1): {cached_total:.2f}s")
    print(f"Batches per epoch: {n_batches_epoch}")
    est_seconds = cached_total * n_batches_epoch
    print(f"Estimated epoch time: {est_seconds:.0f}s "
          f"({est_seconds/60:.1f} min, {est_seconds/3600:.2f} hr)")
    print()

    # Per-stage breakdown for cached batch
    print("Cached batch breakdown:")
    for stage in timer.stages:
        t = timer.batch_times[1].get(stage, 0.0)
        pct = 100.0 * t / cached_total if cached_total > 0 else 0.0
        print(f"  {stage:30s} {t:8.2f}s  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
