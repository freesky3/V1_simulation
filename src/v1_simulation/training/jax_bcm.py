from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

from v1_simulation.network.state import NetworkState
from v1_simulation.solvers.jax_utils import require_jax
from v1_simulation.training.bcm import BCMThetaState, initialize_theta, validate_bcm_config
from v1_simulation.training.plasticity import (
    BCMBlockUpdateIndex,
    BCMEfferentUpdateIndex,
    BCMRowSumLimits,
    BCMTrainingStepResult,
)

if TYPE_CHECKING:
    from v1_simulation.config.schema import TrainingBCMConfig


@dataclass(frozen=True, slots=True)
class _DeviceBlockIndex:
    rows: object
    cols: object
    local_rows: object
    local_cols: object
    row_sum_max: object


class JAXBCMUpdater:
    """JIT-compiled BCM updater for dense JAX training weights."""

    def __init__(
        self,
        *,
        config: TrainingBCMConfig,
        row_sum_limits: BCMRowSumLimits,
        update_index: BCMEfferentUpdateIndex,
        dtype: str = "float64",
    ) -> None:
        validate_bcm_config(config)
        self.config = config
        self._jax, self._jnp = require_jax("bcm-jax")
        self._dtype = self._jnp.float32 if dtype == "float32" else self._jnp.float64
        self._ee = self._device_block(update_index.target_E_source_E, row_sum_limits.target_E_source_E)
        self._ie = self._device_block(update_index.target_I_source_E, row_sum_limits.target_I_source_E)
        self._update = _make_bcm_update_kernel(self._jax, self._jnp)
        self._stats = _make_weight_stats_kernel(self._jax, self._jnp)

    def to_device_network(self, network: NetworkState) -> NetworkState:
        weights = self.to_device_weights(network.weights)
        if weights is network.weights:
            return network
        return NetworkState(
            layout=network.layout,
            connectivity=network.connectivity,
            weights=weights,
            source=network.source,
        )

    def to_device_weights(self, weights):
        if sparse.issparse(weights):
            weights = weights.toarray()
        return self._jnp.asarray(weights, dtype=self._dtype)

    def training_step(
        self,
        *,
        network: NetworkState,
        x_E: ArrayLike,
        y_E: ArrayLike,
        y_I: ArrayLike,
        theta: BCMThetaState | None,
    ) -> BCMTrainingStepResult:
        if theta is None:
            initialized = initialize_theta(y_E, y_I, self.config)
            return BCMTrainingStepResult(
                network=self.to_device_network(network),
                theta=initialized,
                theta_for_update=initialized,
                updated=False,
            )

        x = self._as_batch("x_E", x_E)
        y_e = self._as_batch("y_E", y_E)
        y_i = self._as_batch("y_I", y_I)
        weights = self.to_device_weights(network.weights)
        theta_e = self._jnp.asarray(theta.E, dtype=self._dtype)
        theta_i = self._jnp.asarray(theta.I, dtype=self._dtype)

        next_weights, next_theta_e, next_theta_i, update_theta_e, update_theta_i = self._update(
            weights,
            x,
            y_e,
            y_i,
            theta_e,
            theta_i,
            self._ee.rows,
            self._ee.cols,
            self._ee.local_rows,
            self._ee.local_cols,
            self._ee.row_sum_max,
            self._ie.rows,
            self._ie.cols,
            self._ie.local_rows,
            self._ie.local_cols,
            self._ie.row_sum_max,
            self._scalar(self.config.eta),
            self._scalar(self.config.theta_eps),
            self._scalar(self.config.theta_beta),
            self._scalar(_theta_floor(self.config)),
            self._scalar(float("inf") if self.config.w_max is None else self.config.w_max),
            self._jnp.asarray(self.config.theta_update_order == "pre"),
        )

        next_network = NetworkState(
            layout=network.layout,
            connectivity=network.connectivity,
            weights=next_weights,
            source=network.source,
        )
        return BCMTrainingStepResult(
            network=next_network,
            theta=BCMThetaState(E=np.asarray(next_theta_e), I=np.asarray(next_theta_i)),
            theta_for_update=BCMThetaState(E=np.asarray(update_theta_e), I=np.asarray(update_theta_i)),
            updated=True,
        )

    def weight_stats(self, weights) -> dict[str, float]:
        weights = self.to_device_weights(weights)
        values = self._stats(
            weights,
            self._ee.rows,
            self._ee.cols,
            self._ee.local_rows,
            self._ee.row_sum_max,
            self._ie.rows,
            self._ie.cols,
            self._ie.local_rows,
            self._ie.row_sum_max,
        )
        stats = [float(np.asarray(value)) for value in values]
        keys = (
            "W_EE_mean",
            "W_EE_p95",
            "W_EE_p99",
            "W_EE_max",
            "W_EE_row_sum_mean",
            "W_EE_row_sum_p95",
            "W_EE_row_sum_max",
            "W_IE_mean",
            "W_IE_p95",
            "W_IE_p99",
            "W_IE_max",
            "W_IE_row_sum_mean",
            "W_IE_row_sum_p95",
            "W_IE_row_sum_max",
        )
        return dict(zip(keys, stats, strict=True))

    def _device_block(
        self,
        index: BCMBlockUpdateIndex,
        row_sum_max: ArrayLike | None,
    ) -> _DeviceBlockIndex:
        return _DeviceBlockIndex(
            rows=self._jnp.asarray(index.rows, dtype=self._jnp.int32),
            cols=self._jnp.asarray(index.cols, dtype=self._jnp.int32),
            local_rows=self._jnp.asarray(index.local_rows, dtype=self._jnp.int32),
            local_cols=self._jnp.asarray(index.local_cols, dtype=self._jnp.int32),
            row_sum_max=self._jnp.asarray(
                _limits_or_inf(row_sum_max, index.row_count),
                dtype=self._dtype,
            ),
        )

    def _as_batch(self, name: str, value: ArrayLike):
        arr = np.asarray(value, dtype=np.float32 if self._dtype == self._jnp.float32 else np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 1D vector or a 2D batch matrix.")
        if arr.shape[0] == 0:
            raise ValueError(f"{name} batch must not be empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return self._jnp.asarray(arr, dtype=self._dtype)

    def _scalar(self, value: float | None):
        return self._jnp.asarray(float(value), dtype=self._dtype)


def _make_bcm_update_kernel(jax, jnp):
    def update_theta_vector(theta, response, beta, floor):
        response_ms = jnp.mean(jnp.square(response), axis=0)
        return jnp.maximum((1.0 - beta) * theta + beta * response_ms, floor)

    def update_block(weights, x, y, theta, rows, cols, local_rows, local_cols, row_sum_max, eta, theta_eps, w_max):
        gain = eta * y * (y - jnp.maximum(theta, theta_eps)[jnp.newaxis, :])
        delta = gain.T @ x / x.shape[0]
        edge_values = weights[rows, cols] + delta[local_rows, local_cols]
        edge_values = jnp.maximum(edge_values, 0.0)
        edge_values = jnp.minimum(edge_values, w_max)
        row_totals = jnp.bincount(local_rows, weights=edge_values, length=row_sum_max.shape[0])
        row_scale = jnp.where(row_totals > 0.0, row_sum_max / row_totals, 1.0)
        row_scale = jnp.minimum(row_scale, 1.0)
        edge_values = edge_values * row_scale[local_rows]
        return weights.at[rows, cols].set(edge_values)

    def run(
        weights,
        x_e,
        y_e,
        y_i,
        theta_e,
        theta_i,
        ee_rows,
        ee_cols,
        ee_local_rows,
        ee_local_cols,
        ee_row_sum_max,
        ie_rows,
        ie_cols,
        ie_local_rows,
        ie_local_cols,
        ie_row_sum_max,
        eta,
        theta_eps,
        theta_beta,
        theta_floor,
        w_max,
        theta_update_pre,
    ):
        next_theta_e = update_theta_vector(theta_e, y_e, theta_beta, theta_floor)
        next_theta_i = update_theta_vector(theta_i, y_i, theta_beta, theta_floor)
        update_theta_e = jnp.where(theta_update_pre, next_theta_e, theta_e)
        update_theta_i = jnp.where(theta_update_pre, next_theta_i, theta_i)
        weights = update_block(
            weights,
            x_e,
            y_e,
            update_theta_e,
            ee_rows,
            ee_cols,
            ee_local_rows,
            ee_local_cols,
            ee_row_sum_max,
            eta,
            theta_eps,
            w_max,
        )
        weights = update_block(
            weights,
            x_e,
            y_i,
            update_theta_i,
            ie_rows,
            ie_cols,
            ie_local_rows,
            ie_local_cols,
            ie_row_sum_max,
            eta,
            theta_eps,
            w_max,
        )
        return weights, next_theta_e, next_theta_i, update_theta_e, update_theta_i

    return jax.jit(run)


def _make_weight_stats_kernel(jax, jnp):
    def nonzero_stats(values):
        if values.size == 0:
            zero = jnp.asarray(0.0, dtype=values.dtype)
            return zero, zero, zero, zero
        mask = values != 0.0
        count = jnp.sum(mask)
        nz_sum = jnp.sum(jnp.where(mask, values, 0.0))
        mean = jnp.where(count > 0, nz_sum / count, 0.0)
        masked = jnp.where(mask, values, -jnp.inf)
        max_value = jnp.where(count > 0, jnp.max(masked), 0.0)
        sorted_values = jnp.sort(values)
        zero_count = values.size - count

        def percentile(q):
            rank = jnp.maximum(0, jnp.ceil(q * count).astype(jnp.int32) - 1)
            idx = jnp.minimum(values.size - 1, zero_count.astype(jnp.int32) + rank)
            return jnp.where(count > 0, sorted_values[idx], 0.0)

        return mean, percentile(0.95), percentile(0.99), max_value

    def row_sum_stats(values, local_rows, row_sum_max):
        if row_sum_max.size == 0:
            zero = jnp.asarray(0.0, dtype=values.dtype)
            return zero, zero, zero
        row_sums = jnp.bincount(local_rows, weights=values, length=row_sum_max.shape[0])
        return jnp.mean(row_sums), jnp.percentile(row_sums, 95.0), jnp.max(row_sums)

    def block_stats(weights, rows, cols, local_rows, row_sum_max):
        values = weights[rows, cols] if rows.size else jnp.asarray([], dtype=weights.dtype)
        return (*nonzero_stats(values), *row_sum_stats(values, local_rows, row_sum_max))

    def run(weights, ee_rows, ee_cols, ee_local_rows, ee_row_sum_max, ie_rows, ie_cols, ie_local_rows, ie_row_sum_max):
        return (
            *block_stats(weights, ee_rows, ee_cols, ee_local_rows, ee_row_sum_max),
            *block_stats(weights, ie_rows, ie_cols, ie_local_rows, ie_row_sum_max),
        )

    return jax.jit(run)


def _limits_or_inf(row_sum_max: ArrayLike | None, row_count: int) -> np.ndarray:
    if row_sum_max is None:
        return np.full(int(row_count), np.inf, dtype=float)
    limits = np.asarray(row_sum_max, dtype=float)
    if limits.ndim == 0:
        return np.full(int(row_count), float(limits), dtype=float)
    return limits


def _theta_floor(config: TrainingBCMConfig) -> float:
    floor = float(config.theta_eps)
    if config.theta_floor is not None:
        floor = max(float(config.theta_floor), floor)
    return floor
