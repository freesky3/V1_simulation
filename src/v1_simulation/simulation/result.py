from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from v1_simulation.analysis.types import AnalysisInputs
from v1_simulation.network.state import NetworkState
from v1_simulation.solvers.base import BatchODEResult


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """In-memory output of one drifting-grating simulation pipeline run."""

    ode: BatchODEResult
    theta_angles: NDArray[np.float64]
    time: NDArray[np.float64]
    network: NetworkState
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exc_responses(self) -> NDArray[np.float64]:
        """Excitatory trajectories shaped for analysis: ``(n_exc, n_theta, n_time)``."""
        if self.ode.exc_trajectory is None:
            raise ValueError("SimulationResult does not contain excitatory trajectories.")
        return np.transpose(self.ode.exc_trajectory, (2, 1, 0))

    @property
    def inh_responses(self) -> NDArray[np.float64]:
        """Inhibitory trajectories shaped for analysis: ``(n_inh, n_theta, n_time)``."""
        if self.ode.inh_trajectory is None:
            raise ValueError("SimulationResult does not contain inhibitory trajectories.")
        return np.transpose(self.ode.inh_trajectory, (2, 1, 0))

    @property
    def aE_all(self) -> NDArray[np.float64]:
        """Legacy batch-major E trajectories: ``(n_theta, n_exc, n_time)``."""
        if self.ode.exc_trajectory is None:
            raise ValueError("SimulationResult does not contain excitatory trajectories.")
        return np.transpose(self.ode.exc_trajectory, (1, 2, 0))

    @property
    def aI_all(self) -> NDArray[np.float64]:
        """Legacy batch-major I trajectories: ``(n_theta, n_inh, n_time)``."""
        if self.ode.inh_trajectory is None:
            raise ValueError("SimulationResult does not contain inhibitory trajectories.")
        return np.transpose(self.ode.inh_trajectory, (1, 2, 0))

    def analysis_inputs(self) -> AnalysisInputs:
        """Build analysis-ready inputs for the excitatory L2/3 population."""
        exc_idx = self.network.idx_E
        l23 = self.network.layout.l23
        distance = l23.distance_matrix()[np.ix_(exc_idx, exc_idx)]
        return AnalysisInputs(
            responses=self.exc_responses,
            coords=l23.coords[exc_idx],
            distance=distance,
            theta_angles=self.theta_angles,
        )
