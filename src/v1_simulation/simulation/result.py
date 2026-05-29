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
    center_side_fraction: float = 1.0

    @property
    def exc_responses(self) -> NDArray[np.float64]:
        """Excitatory trajectories shaped for analysis: ``(n_exc, n_theta, n_time)``."""
        if self.ode.exc_trajectory is None:
            raise ValueError("SimulationResult does not contain excitatory trajectories.")
        traj = np.transpose(self.ode.exc_trajectory, (2, 1, 0))
        if self.center_side_fraction < 1.0:
            l23 = self.network.layout.l23
            coords_E = l23.coords[self.network.idx_E]
            half_side = (l23.region_size * self.center_side_fraction) / 2.0
            in_center_E = (np.abs(coords_E[:, 0]) <= half_side) & (np.abs(coords_E[:, 1]) <= half_side)
            traj = traj[in_center_E]
        return traj

    @property
    def inh_responses(self) -> NDArray[np.float64]:
        """Inhibitory trajectories shaped for analysis: ``(n_inh, n_theta, n_time)``."""
        if self.ode.inh_trajectory is None:
            raise ValueError("SimulationResult does not contain inhibitory trajectories.")
        traj = np.transpose(self.ode.inh_trajectory, (2, 1, 0))
        if self.center_side_fraction < 1.0:
            l23 = self.network.layout.l23
            coords_I = l23.coords[self.network.idx_I]
            half_side = (l23.region_size * self.center_side_fraction) / 2.0
            in_center_I = (np.abs(coords_I[:, 0]) <= half_side) & (np.abs(coords_I[:, 1]) <= half_side)
            traj = traj[in_center_I]
        return traj

    @property
    def aE_all(self) -> NDArray[np.float64]:
        """Legacy batch-major E trajectories: ``(n_theta, n_exc, n_time)``."""
        if self.ode.exc_trajectory is None:
            raise ValueError("SimulationResult does not contain excitatory trajectories.")
        traj = np.transpose(self.ode.exc_trajectory, (1, 2, 0))
        if self.center_side_fraction < 1.0:
            l23 = self.network.layout.l23
            coords_E = l23.coords[self.network.idx_E]
            half_side = (l23.region_size * self.center_side_fraction) / 2.0
            in_center_E = (np.abs(coords_E[:, 0]) <= half_side) & (np.abs(coords_E[:, 1]) <= half_side)
            traj = traj[:, in_center_E, :]
        return traj

    @property
    def aI_all(self) -> NDArray[np.float64]:
        """Legacy batch-major I trajectories: ``(n_theta, n_inh, n_time)``."""
        if self.ode.inh_trajectory is None:
            raise ValueError("SimulationResult does not contain inhibitory trajectories.")
        traj = np.transpose(self.ode.inh_trajectory, (1, 2, 0))
        if self.center_side_fraction < 1.0:
            l23 = self.network.layout.l23
            coords_I = l23.coords[self.network.idx_I]
            half_side = (l23.region_size * self.center_side_fraction) / 2.0
            in_center_I = (np.abs(coords_I[:, 0]) <= half_side) & (np.abs(coords_I[:, 1]) <= half_side)
            traj = traj[:, in_center_I, :]
        return traj

    @property
    def exc_mean(self) -> NDArray[np.float64]:
        """Excitatory mean rates: ``(n_batch, n_exc)``."""
        val = self.ode.exc
        if self.center_side_fraction < 1.0:
            l23 = self.network.layout.l23
            coords_E = l23.coords[self.network.idx_E]
            half_side = (l23.region_size * self.center_side_fraction) / 2.0
            in_center_E = (np.abs(coords_E[:, 0]) <= half_side) & (np.abs(coords_E[:, 1]) <= half_side)
            val = val[:, in_center_E]
        return val

    @property
    def inh_mean(self) -> NDArray[np.float64]:
        """Inhibitory mean rates: ``(n_batch, n_inh)``."""
        val = self.ode.inh
        if self.center_side_fraction < 1.0:
            l23 = self.network.layout.l23
            coords_I = l23.coords[self.network.idx_I]
            half_side = (l23.region_size * self.center_side_fraction) / 2.0
            in_center_I = (np.abs(coords_I[:, 0]) <= half_side) & (np.abs(coords_I[:, 1]) <= half_side)
            val = val[:, in_center_I]
        return val

    def analysis_inputs(self) -> AnalysisInputs:
        """Build analysis-ready inputs for the excitatory L2/3 population."""
        exc_idx = self.network.idx_E
        l23 = self.network.layout.l23
        coords = l23.coords[exc_idx]
        distance = l23.distance_matrix()[np.ix_(exc_idx, exc_idx)]
        if self.center_side_fraction < 1.0:
            half_side = (l23.region_size * self.center_side_fraction) / 2.0
            in_center_E = (np.abs(coords[:, 0]) <= half_side) & (np.abs(coords[:, 1]) <= half_side)
            coords = coords[in_center_E]
            distance = distance[np.ix_(in_center_E, in_center_E)]
        return AnalysisInputs(
            responses=self.exc_responses,
            coords=coords,
            distance=distance,
            theta_angles=self.theta_angles,
        )
