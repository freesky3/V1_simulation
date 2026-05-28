from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from v1_simulation.network.state import NetworkState
from v1_simulation.training.bcm import BCMThetaState, validate_bcm_config
from v1_simulation.training.plasticity import bcm_training_step, make_bcm_row_sum_limits

if TYPE_CHECKING:
    from v1_simulation.config.schema import TrainingBCMConfig
    from v1_simulation.solvers.base import BatchODEResult


@dataclass(slots=True)
class BCMTrainingState:
    network: NetworkState
    theta: BCMThetaState | None = None
    step: int = 0
    samples_seen: int = 0


@dataclass(frozen=True, slots=True)
class BatchTrainingLog:
    step: int
    epoch: int
    batch_size: int
    samples_seen: int
    images: str
    updated: int
    aE_mean: float
    aI_mean: float
    aE_max: float
    aI_max: float
    conv_aE: float
    conv_aI: float
    steady_state_reached: int
    steady_state_index: int
    steady_state_start_index: int
    time_steps: int
    t_final: float
    weight_stats: dict[str, float] = field(default_factory=dict)
    theta_stats: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    network: NetworkState
    theta: BCMThetaState
    steps: int
    samples_seen: int
    images_seen: int


class BCMTrainer:
    """Owns BCM training state transitions, independent from solvers and artifact IO.

    Attributes:
        config: The BCM training configuration.
        row_sum_limits: The row sum limits for EE and IE connection blocks.
        state: The current training state including network state, sliding thresholds,
            and samples seen.
    """

    def __init__(self, config: TrainingBCMConfig, network: NetworkState) -> None:
        validate_bcm_config(config, include_loop_fields=True)
        self.config = config
        self.row_sum_limits = make_bcm_row_sum_limits(network, config)
        self.state = BCMTrainingState(network=network)

    def train_batch(
        self,
        dynamics: BatchODEResult,
        *,
        epoch: int,
        batch_size: int,
        images: str = "",
    ) -> BatchTrainingLog:
        """Applies a plastic update step on the network weights using the ODE solver results.

        This computes synaptic plasticity adjustments via the BCM learning rule, updates the
        sliding threshold state (theta), applies normalization/clipping constraints, and
        gathers step-level statistics for logging.

        Args:
            dynamics: The result of the ODE solver containing firing rates, convergence
                metrics, and steady-state information.
            epoch: The current epoch index.
            batch_size: The number of samples processed in this batch.
            images: Semicolon-separated paths of images used in this batch.

        Returns:
            A BatchTrainingLog object containing statistics and diagnostics for this step.
        """
        self.state.step += 1
        self.state.samples_seen += int(batch_size)

        result = bcm_training_step(
            network=self.state.network,
            x_E=dynamics.exc,
            y_E=dynamics.exc,
            y_I=dynamics.inh,
            theta=self.state.theta,
            config=self.config,
            row_sum_limits=self.row_sum_limits,
        )
        self.state.network = result.network
        self.state.theta = result.theta

        return BatchTrainingLog(
            step=self.state.step,
            epoch=int(epoch),
            batch_size=int(batch_size),
            samples_seen=self.state.samples_seen,
            images=images,
            updated=int(result.updated),
            aE_mean=float(np.mean(dynamics.exc)),
            aI_mean=float(np.mean(dynamics.inh)) if dynamics.inh.size else 0.0,
            aE_max=float(np.max(dynamics.exc)),
            aI_max=float(np.max(dynamics.inh)) if dynamics.inh.size else 0.0,
            conv_aE=float(np.mean(dynamics.exc_convergence)),
            conv_aI=float(np.mean(dynamics.inh_convergence)) if dynamics.inh_convergence.size else 0.0,
            steady_state_reached=int(dynamics.steady_state_reached),
            steady_state_index=_optional_index(dynamics.steady_state_index),
            steady_state_start_index=_optional_index(dynamics.steady_state_start_index),
            time_steps=int(dynamics.time.size),
            t_final=float(dynamics.time[-1]),
            weight_stats=_training_weight_stats(self.state.network),
            theta_stats=_theta_stats(result.theta_for_update, self.config.theta_floor),
        )


def _training_weight_stats(network: NetworkState) -> dict[str, float]:
    weights = network.weights
    idx_E = network.idx_E
    idx_I = network.idx_I

    stats: dict[str, float] = {}
    stats.update(_nonzero_stats(weights[np.ix_(idx_E, idx_E)], "W_EE"))
    stats.update(_row_sum_stats(weights[np.ix_(idx_E, idx_E)], "W_EE_row_sum"))
    stats.update(_nonzero_stats(weights[np.ix_(idx_I, idx_E)], "W_IE"))
    stats.update(_row_sum_stats(weights[np.ix_(idx_I, idx_E)], "W_IE_row_sum"))
    return stats


def _nonzero_stats(values, prefix: str) -> dict[str, float]:
    data = values.data if sparse.issparse(values) else np.asarray(values, dtype=float).ravel()
    nonzero = data[data != 0.0]
    if nonzero.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(nonzero)),
        f"{prefix}_p95": float(np.percentile(nonzero, 95)),
        f"{prefix}_p99": float(np.percentile(nonzero, 99)),
        f"{prefix}_max": float(np.max(nonzero)),
    }


def _row_sum_stats(values, prefix: str) -> dict[str, float]:
    row_sums = _row_sums(values)
    if row_sums.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(row_sums)),
        f"{prefix}_p95": float(np.percentile(row_sums, 95)),
        f"{prefix}_max": float(np.max(row_sums)),
    }


def _row_sums(values) -> np.ndarray:
    if sparse.issparse(values):
        return np.asarray(values.sum(axis=1)).ravel().astype(float)
    return np.sum(np.asarray(values, dtype=float), axis=1)


def _theta_stats(theta: BCMThetaState, theta_floor: float | None) -> dict[str, float]:
    stats: dict[str, float] = {}
    for name, values in {"theta_E": theta.E, "theta_I": theta.I}.items():
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            stats[f"{name}_min"] = 0.0
            stats[f"{name}_p05"] = 0.0
            stats[f"{name}_median"] = 0.0
            stats[f"{name}_floor_fraction"] = 0.0
            continue
        stats[f"{name}_min"] = float(np.min(values))
        stats[f"{name}_p05"] = float(np.percentile(values, 5))
        stats[f"{name}_median"] = float(np.median(values))
        stats[f"{name}_floor_fraction"] = (
            0.0 if theta_floor is None else float(np.mean(values <= float(theta_floor)))
        )
    return stats


def _optional_index(value: int | None) -> int:
    return -1 if value is None else int(value)
