from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from v1_simulation.config.schema import RootConfig
from v1_simulation.io.artifacts import SimulationArtifacts
from v1_simulation.network.empirical import EmpiricalData
from v1_simulation.network.state import NetworkState
from v1_simulation.simulation.pipeline import SolverCallable, run_drifting_grating_pipeline
from v1_simulation.simulation.result import SimulationResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SavedSimulationRun:
    """Holds the result and directory path of a persisted simulation run.

    Attributes:
        result: The SimulationResult object containing trajectories and network state.
        run_dir: The directory where the simulation run was saved.
    """
    result: SimulationResult
    run_dir: Path


def run_simulation(
    cfg: RootConfig,
    *,
    network: NetworkState | None = None,
    empirical: EmpiricalData | None = None,
    time: Sequence[float] | np.ndarray | None = None,
    run_root: str | Path | None = None,
    job_name: str | None = None,
    artifacts: SimulationArtifacts | None = None,
    solver: SolverCallable | None = None,
) -> SavedSimulationRun:
    """Runs and persists one drifting-grating simulation based on the config.

    Args:
        cfg: The root configuration for the simulation.
        network: Optional pre-constructed NetworkState. If not provided, it is built.
        empirical: Optional empirical data for network construction.
        time: Optional time grid for the simulation.
        run_root: Optional root directory path for run artifacts.
        job_name: Optional job name used to identify the run directory.
        artifacts: Optional SimulationArtifacts helper.
        solver: Optional ODE solver callable.

    Returns:
        A SavedSimulationRun container enclosing the result and run directory.
    """
    result = run_drifting_grating_pipeline(
        cfg,
        network=network,
        empirical=empirical,
        time=time,
        solver=solver,
    )
    run_artifacts = artifacts or SimulationArtifacts.create(
        Path(cfg.paths.run_root) if run_root is None else run_root,
        job_name=cfg.job_name if job_name is None else job_name,
    )
    run_artifacts.save_result(result, save_network=cfg.simulation.save_network)
    logger.info("Saved simulation run to %s", run_artifacts.run_dir)
    return SavedSimulationRun(result=result, run_dir=run_artifacts.run_dir)
