from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from tqdm.auto import tqdm

from v1_simulation.config.schema import RootConfig
from v1_simulation.config.validation import validate_config
from v1_simulation.io.artifacts import TrainingArtifacts
from v1_simulation.network.builder import build_network_state
from v1_simulation.network.empirical import EmpiricalData
from v1_simulation.network.state import NetworkState
from v1_simulation.solvers.base import BatchODEResult
from v1_simulation.solvers.wilson_cowan import solve_wilson_cowan_batch
from v1_simulation.stimuli.background import generate_background_trace, validate_time_grid
from v1_simulation.stimuli.grating import DriftingGratingInput
from v1_simulation.simulation.result import SimulationResult
from v1_simulation.training.checkpoints import save_checkpoint, save_theta
from v1_simulation.training.natural_inputs import build_natural_image_l4_drive
from v1_simulation.training.trainer import BCMTrainer, TrainingResult


SolverCallable = Callable[..., BatchODEResult]


def run_drifting_grating_pipeline(
    cfg: RootConfig,
    *,
    network: NetworkState | None = None,
    empirical: EmpiricalData | None = None,
    time: Sequence[float] | np.ndarray | None = None,
    solver: SolverCallable | None = None,
) -> SimulationResult:
    """Runs a drifting-grating simulation batch using the specified configuration.

    This function sets up the network, generates the stimulus drive and background noise trace,
    and runs the ODE solver to generate the trajectories for all specified stimulus orientations.

    Args:
        cfg: The root configuration object.
        network: Optional pre-constructed NetworkState.
        empirical: Optional EmpiricalData for building network.
        time: Optional time grid.
        solver: Optional custom ODE solver function.

    Returns:
        A SimulationResult containing the solver trajectories and metadata.

    Raises:
        ValueError: If the stimulus kind is not 'drifting_grating'.
    """

    validate_config(cfg)
    if cfg.stimulus.kind != "drifting_grating":
        raise ValueError("cfg.stimulus.kind must be 'drifting_grating' for this simulation pipeline.")

    run_network = network if network is not None else build_network_state(cfg, empirical=empirical)
    theta_angles = build_theta_angles(cfg)
    time_grid = (
        default_simulation_time_grid(cfg)
        if time is None
        else validate_time_grid(np.asarray(time, dtype=float), copy=True)
    )
    stimulus = DriftingGratingInput(
        cfg.stimulus,
        run_network.layout.l4,
        l4_tunings=run_network.layout.l4_tunings,
        l4_pref_dirs=run_network.layout.l4_pref_dirs,
    )
    background_trace = _make_simulation_background_trace(
        cfg,
        network=run_network,
        n_batch=theta_angles.size,
        time=time_grid,
    )
    solver_fn = solve_wilson_cowan_batch if solver is None else solver
    ode = solver_fn(
        network=run_network,
        external_drive=stimulus.make_batched_drive_func(theta_angles),
        time=time_grid,
        n_batch=theta_angles.size,
        solver_config=cfg.solver,
        transfer_config=cfg.solver.transfer,
        background_trace=background_trace,
        store_trajectory=cfg.simulation.store_trajectory,
        stop_at_steady_state=False,
    )

    return SimulationResult(
        ode=ode,
        theta_angles=theta_angles,
        time=time_grid,
        network=run_network,
        metadata=build_simulation_metadata(cfg, run_network, theta_angles, time_grid),
        center_side_fraction=float(cfg.analysis.center_side_fraction),
    )


def build_theta_angles(cfg: RootConfig) -> np.ndarray:
    """Builds the array of stimulus orientation angles in radians.

    Args:
        cfg: The root configuration containing stimulus settings.

    Returns:
        A 1D numpy array of shape (n_theta,) containing orientation angles.
    """
    return np.linspace(0.0, np.pi, int(cfg.stimulus.n_theta), endpoint=False, dtype=float)


def default_simulation_time_grid(cfg: RootConfig) -> np.ndarray:
    """Builds the default time grid for the simulation based on configuration settings.

    Args:
        cfg: The root configuration.

    Returns:
        A 1D numpy array of the validated time grid.

    Raises:
        ValueError: If the resulting time grid has fewer than two points.
    """
    sim = cfg.simulation
    transfer = cfg.solver.transfer
    start = float(sim.t_start)
    stop = (
        float(sim.t_stop)
        if sim.t_stop is not None
        else start + float(sim.duration_tau_e) * float(transfer.tau_e)
    )
    step = float(sim.dt) if sim.dt is not None else float(sim.dt_tau_i_fraction) * float(transfer.tau_i)
    time = validate_time_grid(np.arange(start, stop, step, dtype=float), copy=False)
    if time.size < 2:
        raise ValueError("simulation time grid must contain at least two points.")
    return time


def build_simulation_metadata(
    cfg: RootConfig,
    network: NetworkState,
    theta_angles: np.ndarray,
    time: np.ndarray,
) -> dict[str, object]:
    """Prepares and structures run metadata for the simulation.

    Args:
        cfg: The root configuration.
        network: The NetworkState used in the simulation.
        theta_angles: The stimulus orientations used.
        time: The simulation time grid.

    Returns:
        A dictionary containing structured run metadata.
    """
    return {
        "config": asdict(cfg),
        "network_source": dict(network.source),
        "idx_E": network.idx_E.tolist(),
        "idx_I": network.idx_I.tolist(),
        "idx_X": network.idx_X.tolist(),
        "l23_n_side": int(network.layout.l23.n_side),
        "theta_angles": np.asarray(theta_angles, dtype=float).tolist(),
        "time_steps": int(time.size),
        "t_start": float(time[0]),
        "t_final": float(time[-1]),
    }


def run_bcm_training(
    cfg: RootConfig,
    *,
    network: NetworkState | None = None,
    empirical: EmpiricalData | None = None,
    time: Sequence[float] | np.ndarray | None = None,
    run_root: str | Path | None = None,
    job_name: str | None = None,
    artifacts: TrainingArtifacts | None = None,
    drive=None,
    sampler=None,
    solver: SolverCallable | None = None,
    show_progress: bool = True,
) -> TrainingResult:
    """Runs schema-driven natural-image BCM training.

    The pipeline assembles network, natural-image drive, solver, trainer, and
    artifacts from ``RootConfig``. The trainer owns BCM state transitions; this
    function only orchestrates dependencies and persistence.

    Args:
        cfg: The root configuration for the simulation.
        network: Optional initial network state. If not provided, it is built from the config.
        empirical: Optional empirical data for network initialization.
        time: Optional time grid for solver simulation. If not provided, a default grid is used.
        run_root: Optional root directory path to store run artifacts.
        job_name: Optional custom job name for artifact directories.
        artifacts: Optional pre-configured training artifacts helper.
        drive: Optional custom external drive for natural image inputs.
        sampler: Optional custom image sampler.
        solver: Optional custom ODE solver function.
        show_progress: Whether to display a progress bar for the training loop.

    Returns:
        The training result containing final states and metadata.

    Raises:
        ValueError: If config is invalid or BCM training is disabled.
        RuntimeError: If no natural-image samples were processed.
    """

    validate_config(cfg)
    if not cfg.training.enabled:
        raise ValueError("cfg.training.enabled must be true for BCM training.")

    run_network = network
    if run_network is None:
        run_network = build_network_state(cfg, empirical=empirical)

    natural_drive = drive
    natural_sampler = sampler
    if natural_drive is None or natural_sampler is None:
        natural_drive, natural_sampler = build_natural_image_l4_drive(
            cfg=cfg.training.natural_image,
            stimulus_cfg=cfg.stimulus,
            model_cfg=cfg.model,
            layers_cfg=cfg.model.layers,
            l4_layer=run_network.layout.l4,
            l4_tunings=run_network.layout.l4_tunings,
            l4_pref_dirs=run_network.layout.l4_pref_dirs,
        )

    time_grid = (
        default_training_time_grid(cfg)
        if time is None
        else validate_time_grid(np.asarray(time, dtype=float), copy=True)
    )
    run_artifacts = artifacts or TrainingArtifacts.create(
        Path(cfg.paths.run_root) if run_root is None else run_root,
        job_name=cfg.job_name if job_name is None else job_name,
    )
    solver_fn = solve_wilson_cowan_batch if solver is None else solver
    trainer = BCMTrainer(cfg.training.bcm, run_network)

    save_checkpoint(run_artifacts.run_dir, "network_initial", trainer.state.network, metadata={"step": 0})

    image_count = 0
    for epoch in range(1, int(cfg.training.bcm.epochs) + 1):
        epoch_samples = tuple(
            natural_sampler.make_epoch(
                limit=cfg.training.natural_image.limit,
                shuffle_paths=True,
                shuffle_samples=True,
            )
        )
        if hasattr(natural_drive, "preload_cache"):
            natural_drive.preload_cache(epoch_samples)
        image_count += _count_unique_paths(epoch_samples)
        batch_size = int(cfg.training.bcm.batch_size)
        n_batches = (len(epoch_samples) + batch_size - 1) // batch_size

        progress = tqdm(
            _iter_batches(epoch_samples, batch_size),
            total=n_batches,
            desc=f"BCM epoch {epoch}/{cfg.training.bcm.epochs}",
            unit="batch",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for batch in progress:
            background_trace = _make_background_trace(
                cfg,
                network=trainer.state.network,
                n_batch=len(batch),
                time=time_grid,
                step=trainer.state.step,
            )
            dynamics = solver_fn(
                network=trainer.state.network,
                external_drive=natural_drive.make_static_batch_func(batch),
                time=time_grid,
                n_batch=len(batch),
                solver_config=cfg.solver,
                transfer_config=cfg.solver.transfer,
                training_bcm=cfg.training.bcm,
                background_trace=background_trace,
                store_trajectory=False,
                stop_at_steady_state=_steady_state_enabled(cfg),
            )
            log_row = trainer.train_batch(
                dynamics,
                epoch=epoch,
                batch_size=len(batch),
                images=_sample_paths_for_log(batch),
            )
            run_artifacts.append_log(log_row)
            postfix = dict(
                samples=trainer.state.samples_seen,
                aE=f"{log_row.aE_mean:.3g}",
                aI=f"{log_row.aI_mean:.3g}",
                updated=log_row.updated,
            )
            if log_row.skipped_bad_batch:
                postfix["BAD"] = trainer.state.consecutive_bad_batches
            progress.set_postfix(**postfix)

            if trainer.state.step % int(cfg.training.bcm.save_every) == 0:
                save_checkpoint(
                    run_artifacts.run_dir,
                    "network_latest",
                    trainer.state.network,
                    metadata=_checkpoint_metadata(trainer, image_count),
                )

    if trainer.state.theta is None:
        raise RuntimeError("BCM training did not process any natural-image samples.")

    save_checkpoint(
        run_artifacts.run_dir,
        "network_final",
        trainer.state.network,
        metadata=_checkpoint_metadata(trainer, image_count),
    )
    save_theta(run_artifacts.run_dir, trainer.state.theta)
    run_artifacts.save_json(
        "run_config.json",
        {
            "config": asdict(cfg),
            "training": {
                "batches": trainer.state.step,
                "images": image_count,
                "samples": trainer.state.samples_seen,
                "time_steps": int(time_grid.size),
                "t_final": float(time_grid[-1]),
                "steady_state_enabled": _steady_state_enabled(cfg),
            },
        },
    )

    return TrainingResult(
        run_dir=run_artifacts.run_dir,
        network=trainer.state.network,
        theta=trainer.state.theta,
        steps=trainer.state.step,
        samples_seen=trainer.state.samples_seen,
        images_seen=image_count,
    )


def default_training_time_grid(cfg: RootConfig) -> np.ndarray:
    """Generates the default time grid for training solver execution.

    Args:
        cfg: The root configuration schema containing solver and transfer parameters.

    Returns:
        A validated numpy array representing the time grid sequence.
    """
    transfer = cfg.solver.transfer
    # Use configurable duration_tau_e (defaults to 30 * tau_e) to reach stable limit cycles.
    stop = float(cfg.training.bcm.duration_tau_e) * float(transfer.tau_e)

    step = float(cfg.training.bcm.dt_tau_i_fraction) * float(transfer.tau_i)
    return validate_time_grid(np.arange(0.0, stop, step, dtype=float), copy=False)


def _make_simulation_background_trace(
    cfg: RootConfig,
    *,
    network: NetworkState,
    n_batch: int,
    time: np.ndarray,
):
    if not cfg.background.enabled:
        return None
    base_seed = cfg.background.seed if cfg.background.seed is not None else cfg.seed + 200000
    return generate_background_trace(
        cfg.background,
        n_exc=network.layout.n_E,
        n_inh=network.layout.n_I,
        n_batch=n_batch,
        time=time,
        seed=np.random.SeedSequence([int(base_seed), 0]),
    )


def _make_background_trace(
    cfg: RootConfig,
    *,
    network: NetworkState,
    n_batch: int,
    time: np.ndarray,
    step: int,
):
    if not cfg.background.enabled:
        return None
    base_seed = cfg.background.seed if cfg.background.seed is not None else cfg.seed + 100000
    seed = np.random.SeedSequence([int(base_seed), int(step)])
    return generate_background_trace(
        cfg.background,
        n_exc=network.layout.n_E,
        n_inh=network.layout.n_I,
        n_batch=n_batch,
        time=time,
        seed=seed,
    )


def _steady_state_enabled(cfg: RootConfig) -> bool:
    return bool(cfg.training.bcm.dynamic_steady_state) and not bool(cfg.background.enabled)


def _iter_batches(items: Sequence, batch_size: int) -> Iterable[tuple]:
    for start in range(0, len(items), int(batch_size)):
        yield tuple(items[start : start + int(batch_size)])


def _sample_paths_for_log(samples: Sequence) -> str:
    return ";".join(str(_sample_path(sample)) for sample in samples)


def _count_unique_paths(samples: Sequence) -> int:
    return len({_sample_path(sample) for sample in samples})


def _sample_path(sample) -> Path | str:
    return getattr(sample, "path", sample)


def _checkpoint_metadata(trainer: BCMTrainer, images_seen: int) -> dict[str, int]:
    return {
        "step": trainer.state.step,
        "samples_seen": trainer.state.samples_seen,
        "images_seen": images_seen,
    }
