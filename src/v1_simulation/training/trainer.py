from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from v1_simulation.network.state import NetworkState
from v1_simulation.training.bcm import BCMThetaState, validate_bcm_config
from v1_simulation.training.plasticity import (
    BCMEfferentUpdateIndex,
    bcm_training_step,
    make_bcm_efferent_update_index,
    make_bcm_row_sum_limits,
)

if TYPE_CHECKING:
    from v1_simulation.config.schema import SolverConfig, TrainingBCMConfig
    from v1_simulation.solvers.base import BatchODEResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BCMTrainingState:
    network: NetworkState
    theta: BCMThetaState | None = None
    step: int = 0
    samples_seen: int = 0
    consecutive_bad_batches: int = 0


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
    aE_saturation_fraction: float
    aI_saturation_fraction: float
    aE_active_fraction: float
    aE_active_mean: float
    aI_active_fraction: float
    aI_active_mean: float
    conv_aE: float
    conv_aI: float
    steady_state_reached: int
    steady_state_index: int
    steady_state_start_index: int
    summary_start_index: int
    summary_end_index: int
    summary_window_size: int
    time_steps: int
    t_final: float
    y_diff_max: float
    y_diff_rms: float
    dy_max: float
    dy_rms: float
    skipped_bad_batch: bool = False
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

    def __init__(
        self,
        config: TrainingBCMConfig,
        network: NetworkState,
        *,
        solver_config: SolverConfig | None = None,
    ) -> None:
        validate_bcm_config(config, include_loop_fields=True)
        self.config = config
        self._jax_updater = None
        # Convert weights to dense once upfront to avoid CSR↔dense round trips
        # every batch during training. Checkpointing converts back to CSR on save.
        if sparse.issparse(network.weights):
            dense_network = NetworkState(
                layout=network.layout,
                connectivity=network.connectivity,
                weights=network.weights.toarray(),
                source=network.source,
            )
        else:
            dense_network = network
        self._rate_max = _solver_rate_max(solver_config)
        # Cache the dense topology mask once to avoid CSR→dense conversion
        # and _validate_weights_follow_topology on every batch.
        conn = network.connectivity
        self._cached_topology = conn.toarray().astype(bool) if sparse.issparse(conn) else np.asarray(conn, dtype=bool)
        self._update_index = make_bcm_efferent_update_index(
            self._cached_topology,
            dense_network.idx_E,
            dense_network.idx_I,
        )
        clipped = _clip_initial_efferent_excitatory_weights(
            dense_network,
            self._update_index,
            w_max=config.w_max if bool(config.clip_initial_weights) else None,
        )
        if clipped:
            logger.info("Clipped %d initial BCM-plastic weights to w_max=%s.", clipped, config.w_max)
        self.row_sum_limits = make_bcm_row_sum_limits(dense_network, config)
        self.state = BCMTrainingState(network=dense_network)
        self._weight_stats_index = _TrainingWeightStatsIndex(
            ee=_BlockWeightStatsIndex.from_update_index(self._update_index.target_E_source_E),
            ie=_BlockWeightStatsIndex.from_update_index(self._update_index.target_I_source_E),
        )
        if _should_use_jax_bcm(solver_config):
            try:
                from v1_simulation.training.jax_bcm import JAXBCMUpdater

                self._jax_updater = JAXBCMUpdater(
                    config=config,
                    row_sum_limits=self.row_sum_limits,
                    update_index=self._update_index,
                    dtype=getattr(solver_config.jax, "dtype", "float64") if solver_config is not None else "float64",
                )
                self.state.network = self._jax_updater.to_device_network(self.state.network)
            except RuntimeError as exc:
                logger.warning("JAX BCM updater unavailable; falling back to NumPy BCM updates: %s", exc)

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

        If the ODE dynamics show signs of instability (NaN/Inf, excessively high
        firing rates, or many neurons saturating near the transfer function ceiling),
        the BCM weight update is skipped and the batch is flagged.

        Args:
            dynamics: The result of the ODE solver containing firing rates, convergence
                metrics, and steady-state information.
            epoch: The current epoch index.
            batch_size: The number of samples processed in this batch.
            images: Semicolon-separated paths of images used in this batch.

        Returns:
            A BatchTrainingLog object containing statistics and diagnostics for this step.

        Raises:
            RuntimeError: If the number of consecutive bad batches exceeds
                ``config.max_consecutive_bad_batches``.
        """
        self.state.step += 1
        self.state.samples_seen += int(batch_size)

        bad_reason = self._detect_bad_batch(dynamics)
        skipped = bad_reason is not None

        if skipped:
            self.state.consecutive_bad_batches += 1
            logger.warning(
                "Bad batch detected at step %d (epoch %d): %s. "
                "Skipping BCM update. Consecutive bad batches: %d/%d.",
                self.state.step,
                epoch,
                bad_reason,
                self.state.consecutive_bad_batches,
                self.config.max_consecutive_bad_batches,
            )
            max_consecutive = int(self.config.max_consecutive_bad_batches)
            if self.state.consecutive_bad_batches >= max_consecutive:
                raise RuntimeError(
                    f"Training halted: {self.state.consecutive_bad_batches} consecutive "
                    f"bad batches detected (limit: {max_consecutive}). "
                    f"Last reason: {bad_reason}. "
                    f"The network may be in an unstable regime. "
                    f"Consider lowering model.connectivity.j or increasing g."
                )
            result_updated = 0
        else:
            try:
                if self._jax_updater is None:
                    result = bcm_training_step(
                        network=self.state.network,
                        x_E=dynamics.exc,
                        y_E=dynamics.exc,
                        y_I=dynamics.inh,
                        theta=self.state.theta,
                        config=self.config,
                        row_sum_limits=self.row_sum_limits,
                        _cached_topology=self._cached_topology,
                        copy_weights=False,
                        update_index=self._update_index,
                    )
                else:
                    result = self._jax_updater.training_step(
                        network=self.state.network,
                        x_E=dynamics.exc,
                        y_E=dynamics.exc,
                        y_I=dynamics.inh,
                        theta=self.state.theta,
                    )
                self.state.consecutive_bad_batches = 0
                self.state.network = result.network
                self.state.theta = result.theta
                result_updated = int(result.updated)
            except ValueError as e:
                if "NaN" in str(e) or "infinite" in str(e):
                    self.state.consecutive_bad_batches += 1
                    logger.warning(
                        "Late bad batch detected at step %d (epoch %d): %s. "
                        "Skipping BCM update. Consecutive bad batches: %d/%d.",
                        self.state.step,
                        epoch,
                        str(e),
                        self.state.consecutive_bad_batches,
                        self.config.max_consecutive_bad_batches,
                    )
                    max_consecutive = int(self.config.max_consecutive_bad_batches)
                    if self.state.consecutive_bad_batches >= max_consecutive:
                        raise RuntimeError(
                            f"Training halted: {self.state.consecutive_bad_batches} consecutive "
                            f"bad batches detected (limit: {max_consecutive}). "
                            f"Last reason: {str(e)}. "
                            f"The network may be in an unstable regime."
                        )
                    result_updated = 0
                    skipped = True
                else:
                    raise

        aE_max = float(np.nanmax(dynamics.exc)) if dynamics.exc.size else 0.0
        aI_max = float(np.nanmax(dynamics.inh)) if dynamics.inh.size else 0.0
        aE_sat = _saturation_fraction(dynamics.exc, self._rate_max)
        aI_sat = _saturation_fraction(dynamics.inh, self._rate_max) if dynamics.inh.size else 0.0
        active_threshold = float(self.config.active_rate_threshold)
        aE_active_fraction, aE_active_mean = _active_rate_stats(dynamics.exc, active_threshold)
        aI_active_fraction, aI_active_mean = _active_rate_stats(dynamics.inh, active_threshold)

        summary_start = _optional_index(getattr(dynamics, "summary_start_index", None))
        summary_end = _optional_index(getattr(dynamics, "summary_end_index", None))
        summary_w_size = int(summary_end - summary_start) if summary_start >= 0 and summary_end >= 0 else -1

        return BatchTrainingLog(
            step=self.state.step,
            epoch=int(epoch),
            batch_size=int(batch_size),
            samples_seen=self.state.samples_seen,
            images=images,
            updated=result_updated,
            aE_mean=float(np.nanmean(dynamics.exc)),
            aI_mean=float(np.nanmean(dynamics.inh)) if dynamics.inh.size else 0.0,
            aE_max=aE_max,
            aI_max=aI_max,
            aE_saturation_fraction=aE_sat,
            aI_saturation_fraction=aI_sat,
            aE_active_fraction=aE_active_fraction,
            aE_active_mean=aE_active_mean,
            aI_active_fraction=aI_active_fraction,
            aI_active_mean=aI_active_mean,
            conv_aE=float(np.mean(dynamics.exc_convergence)),
            conv_aI=float(np.mean(dynamics.inh_convergence)) if dynamics.inh_convergence.size else 0.0,
            steady_state_reached=int(dynamics.steady_state_reached),
            steady_state_index=_optional_index(dynamics.steady_state_index),
            steady_state_start_index=_optional_index(dynamics.steady_state_start_index),
            summary_start_index=summary_start,
            summary_end_index=summary_end,
            summary_window_size=summary_w_size,
            time_steps=int(dynamics.time.size),
            t_final=float(dynamics.time[-1]),
            y_diff_max=float(getattr(dynamics, "y_diff_max", float("nan"))),
            y_diff_rms=float(getattr(dynamics, "y_diff_rms", float("nan"))),
            dy_max=float(getattr(dynamics, "dy_max", float("nan"))),
            dy_rms=float(getattr(dynamics, "dy_rms", float("nan"))),
            skipped_bad_batch=skipped,
            weight_stats=(
                self._jax_updater.weight_stats(self.state.network.weights)
                if self._jax_updater is not None
                else _training_weight_stats(self.state.network, self._weight_stats_index)
            ),
            theta_stats=_theta_stats(
                result.theta_for_update if not skipped else self.state.theta,
                self.config.theta_floor,
            ) if self.state.theta is not None else {},
        )

    def _detect_bad_batch(self, dynamics: BatchODEResult) -> str | None:
        """Checks ODE dynamics for signs of instability.

        Returns a human-readable reason string if the batch is bad, or None if healthy.
        """
        exc = np.asarray(dynamics.exc, dtype=float)
        inh = np.asarray(dynamics.inh, dtype=float) if dynamics.inh.size else np.array([])

        # Check NaN / Inf
        if np.any(~np.isfinite(exc)):
            return "NaN or Inf in excitatory firing rates"
        if inh.size and np.any(~np.isfinite(inh)):
            return "NaN or Inf in inhibitory firing rates"

        if bool(getattr(self.config, "require_steady_state", False)) and not bool(dynamics.steady_state_reached):
            return "steady state was not reached"

        y_diff_threshold = getattr(self.config, "y_diff_max_threshold", None)
        if y_diff_threshold is not None:
            y_diff_max = float(getattr(dynamics, "y_diff_max", float("nan")))
            if np.isfinite(y_diff_max) and y_diff_max > float(y_diff_threshold):
                return f"y_diff_max {y_diff_max:.3g} exceeds threshold {float(y_diff_threshold):.3g}"

        dy_threshold = getattr(self.config, "dy_max_threshold", None)
        if dy_threshold is not None:
            dy_max = float(getattr(dynamics, "dy_max", float("nan")))
            if np.isfinite(dy_max) and dy_max > float(dy_threshold):
                return f"dy_max {dy_max:.3g} exceeds threshold {float(dy_threshold):.3g}"

        threshold = getattr(self.config, 'rate_explosion_threshold', None)
        if threshold is not None:
            threshold = float(threshold)
            max_E = float(np.max(exc)) if exc.size else 0.0
            if max_E > threshold:
                return (
                    f"max excitatory rate {max_E:.1f} Hz exceeds "
                    f"threshold {threshold:.1f} Hz"
                )
            if inh.size:
                max_I = float(np.max(inh))
                # Inhibitory neurons can fire faster; use 2x threshold
                if max_I > threshold * 2.0:
                    return (
                        f"max inhibitory rate {max_I:.1f} Hz exceeds "
                        f"threshold {threshold * 2.0:.1f} Hz"
                    )

        # Check saturation fraction (fraction of neurons near rate_max).
        sat_thr = getattr(self.config, 'saturation_fraction_threshold', None)
        if sat_thr is not None and self._rate_max is not None:
            threshold_fraction = float(sat_thr)
            e_fraction = _saturation_fraction(exc, self._rate_max)
            if e_fraction > threshold_fraction:
                return (
                    f"{e_fraction:.1%} of E neurons near rate_max "
                    f"{self._rate_max:.1f} Hz (threshold: {threshold_fraction:.1%})"
                )
            if inh.size:
                i_fraction = _saturation_fraction(inh, self._rate_max)
                if i_fraction > threshold_fraction:
                    return (
                        f"{i_fraction:.1%} of I neurons near rate_max "
                        f"{self._rate_max:.1f} Hz (threshold: {threshold_fraction:.1%})"
                    )

        return None


@dataclass(frozen=True, slots=True)
class _BlockWeightStatsIndex:
    rows: np.ndarray
    cols: np.ndarray
    local_rows: np.ndarray
    row_count: int

    @classmethod
    def from_update_index(cls, index) -> "_BlockWeightStatsIndex":
        return cls(
            rows=index.rows,
            cols=index.cols,
            local_rows=index.local_rows,
            row_count=index.row_count,
        )


@dataclass(frozen=True, slots=True)
class _TrainingWeightStatsIndex:
    ee: _BlockWeightStatsIndex
    ie: _BlockWeightStatsIndex


def _clip_initial_efferent_excitatory_weights(
    network: NetworkState,
    index: BCMEfferentUpdateIndex,
    *,
    w_max: float | None,
) -> int:
    if w_max is None:
        return 0
    weights = np.asarray(network.weights, dtype=float)
    clipped = 0
    for block in (index.target_E_source_E, index.target_I_source_E):
        if block.rows.size == 0:
            continue
        values = weights[block.rows, block.cols]
        clipped_values = np.clip(values, 0.0, float(w_max))
        clipped += int(np.count_nonzero(np.abs(clipped_values - values) > 1.0e-12))
        weights[block.rows, block.cols] = clipped_values
    return clipped


def _training_weight_stats(
    network: NetworkState,
    index: _TrainingWeightStatsIndex | None = None,
) -> dict[str, float]:
    weights = network.weights
    if sparse.issparse(weights):
        weights_dense = weights.toarray()
    else:
        weights_dense = np.asarray(weights)

    idx_E = network.idx_E
    idx_I = network.idx_I

    stats: dict[str, float] = {}
    if index is not None:
        stats.update(_indexed_block_stats(weights_dense, index.ee, "W_EE", "W_EE_row_sum"))
        stats.update(_indexed_block_stats(weights_dense, index.ie, "W_IE", "W_IE_row_sum"))
        return stats

    w_ee = weights_dense[np.ix_(idx_E, idx_E)]
    stats.update(_nonzero_stats(w_ee, "W_EE"))
    stats.update(_row_sum_stats(w_ee, "W_EE_row_sum"))

    w_ie = weights_dense[np.ix_(idx_I, idx_E)]
    stats.update(_nonzero_stats(w_ie, "W_IE"))
    stats.update(_row_sum_stats(w_ie, "W_IE_row_sum"))
    return stats


def _indexed_block_stats(
    weights: np.ndarray,
    index: _BlockWeightStatsIndex,
    weight_prefix: str,
    row_sum_prefix: str,
) -> dict[str, float]:
    values = weights[index.rows, index.cols] if index.rows.size else np.array([], dtype=float)
    row_sums = np.bincount(index.local_rows, weights=values, minlength=index.row_count)
    stats = _nonzero_stats(values, weight_prefix)
    stats.update(_row_sum_vector_stats(row_sums, row_sum_prefix))
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
    return _row_sum_vector_stats(row_sums, prefix)


def _row_sum_vector_stats(row_sums, prefix: str) -> dict[str, float]:
    row_sums = np.asarray(row_sums, dtype=float).ravel()
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


def _solver_rate_max(solver_config: SolverConfig | None) -> float | None:
    if solver_config is None:
        return None
    transfer = getattr(solver_config, "transfer", None)
    rate_max = getattr(transfer, "rate_max", None)
    if rate_max is None:
        return None
    value = float(rate_max)
    return value if np.isfinite(value) and value > 0.0 else None


def _saturation_fraction(values, rate_max: float | None) -> float:
    if rate_max is None:
        return 0.0
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr >= 0.99 * float(rate_max)))


def _active_rate_stats(values, threshold: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 0.0
    active = finite > float(threshold)
    if not np.any(active):
        return 0.0, 0.0
    return float(np.mean(active)), float(np.mean(finite[active]))


def _should_use_jax_bcm(solver_config) -> bool:
    if solver_config is None:
        return False
    if getattr(solver_config, "backend", None) not in {"jax-rk4", "diffrax"}:
        return False
    jax_cfg = getattr(solver_config, "jax", None)
    return jax_cfg is not None and not bool(getattr(jax_cfg, "prefer_sparse", True))
