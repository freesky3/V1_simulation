import sys
import numpy as np
from pathlib import Path

def main():
    from v1_simulation.config import load_config
    from v1_simulation.config.validation import validate_config
    from v1_simulation.network.builder import build_network_state
    from v1_simulation.simulation.pipeline import default_training_time_grid
    from v1_simulation.training.natural_inputs import build_natural_image_l4_drive
    from v1_simulation.solvers.wilson_cowan import solve_wilson_cowan_batch

    # Exact configuration overrides from your request
    overrides = [
        "+experiment=bcm_train",
        "solver=diffrax_tsit5",
        "background=none",
        "model.connectivity.j=1.2",
        "solver.early_stop.enabled=true",
        "solver.early_stop.min_time=0.2",
        "solver.early_stop.f_atol=1e-4",
        "training.bcm.rate_explosion_threshold=none"
    ]
    
    cfg = load_config(overrides=overrides)
    cfg.mode = "train"
    cfg.training.enabled = True
    validate_config(cfg)

    # 1. Build network
    print("Building network...")
    network = build_network_state(cfg)
    
    # 2. Build drive
    print("Building drive...")
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
    batch_size = int(cfg.training.bcm.batch_size)
    
    epoch_samples = list(natural_sampler.make_epoch(
        limit=batch_size,
        shuffle_paths=True,
        shuffle_samples=True,
    ))
    
    if hasattr(natural_drive, "preload_cache"):
        natural_drive.preload_cache(epoch_samples)
        
    batch = tuple(epoch_samples[:batch_size])
    drive_func = natural_drive.make_static_batch_func(batch)

    # Modify the jax solver code or intercept the values.
    # To see what is going on, let's run the solver.
    print("Running solver...")
    result = solve_wilson_cowan_batch(
        network=network,
        external_drive=drive_func,
        time=time_grid,
        n_batch=batch_size,
        solver_config=cfg.solver,
        transfer_config=cfg.solver.transfer,
        training_bcm=cfg.training.bcm,
        background_trace=None,
        store_trajectory=True,
        stop_at_steady_state=True,
    )
    
    print("\n--- Solver Result ---")
    print(f"steady_state_reached: {result.steady_state_reached}")
    print(f"steady_state_index: {result.steady_state_index}")
    
    # Let's inspect the actual derivatives dy/dt at the end of the trajectory
    # dynamics.exc_trajectory shape: (n_time, n_batch, n_exc)
    exc_traj = result.exc_trajectory
    inh_traj = result.inh_trajectory
    
    dt = time_grid[1] - time_grid[0]
    
    # We calculate the actual numerical derivatives over time for all batch items and all units
    dy_exc = np.diff(exc_traj, axis=0) / dt
    dy_inh = np.diff(inh_traj, axis=0) / dt
    
    print("\nNumerical derivative statistics over time:")
    for step in range(len(time_grid) - 1):
        t = time_grid[step + 1]
        max_dy_exc = np.max(np.abs(dy_exc[step]))
        max_dy_inh = np.max(np.abs(dy_inh[step]))
        max_dy = max(max_dy_exc, max_dy_inh)
        
        y_max_exc = np.max(np.abs(exc_traj[step + 1]))
        y_max_inh = np.max(np.abs(inh_traj[step + 1]))
        y_max = max(y_max_exc, y_max_inh)
        
        atol = 1e-4
        rtol = 1e-4
        threshold = atol + rtol * y_max
        is_steady = max_dy < threshold
        
        if step % 20 == 0 or step == len(time_grid) - 2:
            print(f"t = {t:.3f}s: max |dy/dt| = {max_dy:.6f} Hz/s, threshold = {threshold:.6f} Hz/s, is_steady = {is_steady}")

if __name__ == "__main__":
    main()
