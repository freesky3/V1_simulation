from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from v1_simulation.network.state import NetworkState
from v1_simulation.network.weights import (
    as_connection_mask,
    as_dense_weights,
    limit_row_sums,
    row_sums,
    validate_indices,
)
from v1_simulation.training.bcm import BCMThetaState, bcm_delta, initialize_theta, update_theta, validate_bcm_config
from v1_simulation.training.bcm import bcm_gain

if TYPE_CHECKING:
    from v1_simulation.config.schema import TrainingBCMConfig


@dataclass(frozen=True, slots=True)
class BCMRowSumLimits:
    """Initial row-sum caps for target/source E blocks."""

    target_E_source_E: NDArray[np.float64] | None = None
    target_I_source_E: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class BCMTrainingStepResult:
    """Result of one BCM training batch."""

    network: NetworkState
    theta: BCMThetaState
    theta_for_update: BCMThetaState
    updated: bool


@dataclass(frozen=True, slots=True)
class BCMBlockUpdateIndex:
    """Sparse edge index for one target/source E update block."""

    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    local_rows: NDArray[np.int64]
    local_cols: NDArray[np.int64]
    row_count: int


@dataclass(frozen=True, slots=True)
class BCMEfferentUpdateIndex:
    """Sparse edge indices for target E/source E and target I/source E blocks."""

    target_E_source_E: BCMBlockUpdateIndex
    target_I_source_E: BCMBlockUpdateIndex


def make_bcm_efferent_update_index(
    topology: NDArray[np.bool_],
    idx_E: ArrayLike,
    idx_I: ArrayLike,
) -> BCMEfferentUpdateIndex:
    """Builds cached sparse edge indices for in-place BCM training updates."""

    idx_E_arr = validate_indices("idx_E", idx_E, topology.shape[0])
    idx_I_arr = validate_indices("idx_I", idx_I, topology.shape[0], allow_empty=True)
    return BCMEfferentUpdateIndex(
        target_E_source_E=_make_bcm_block_update_index(topology, idx_E_arr, idx_E_arr),
        target_I_source_E=_make_bcm_block_update_index(topology, idx_I_arr, idx_E_arr),
    )


def _make_bcm_block_update_index(
    topology: NDArray[np.bool_],
    target_idx: NDArray[np.int64],
    source_idx: NDArray[np.int64],
) -> BCMBlockUpdateIndex:
    local_rows, local_cols = np.nonzero(topology[np.ix_(target_idx, source_idx)])
    return BCMBlockUpdateIndex(
        rows=target_idx[local_rows].astype(np.int64, copy=False),
        cols=source_idx[local_cols].astype(np.int64, copy=False),
        local_rows=local_rows.astype(np.int64, copy=False),
        local_cols=local_cols.astype(np.int64, copy=False),
        row_count=int(target_idx.size),
    )


def update_efferent_excitatory_weights(
    weights: ArrayLike | sparse.spmatrix,
    connection_mask: ArrayLike | sparse.spmatrix,
    idx_E: ArrayLike,
    idx_I: ArrayLike,
    x_E: ArrayLike,
    y_E: ArrayLike,
    y_I: ArrayLike,
    theta: BCMThetaState,
    config: TrainingBCMConfig,
    *,
    row_sum_max_E: float | ArrayLike | None = None,
    row_sum_max_I: float | ArrayLike | None = None,
    _cached_topology: NDArray[np.bool_] | None = None,
    copy_weights: bool = True,
    update_index: BCMEfferentUpdateIndex | None = None,
) -> NDArray[np.float64] | sparse.csr_matrix:
    """Updates the efferent weights originating from excitatory source neurons.

    Matrix convention is ``weights[target, source]``. Updated blocks are:
    target E/source E and target I/source E. Does not mutate the input weights.

    Args:
        weights: The current weights matrix.
        connection_mask: The boolean connection topology mask.
        idx_E: Index array of excitatory neurons in the network.
        idx_I: Index array of inhibitory neurons in the network.
        x_E: Pre-synaptic excitatory responses.
        y_E: Post-synaptic excitatory responses.
        y_I: Post-synaptic inhibitory responses.
        theta: The BCM sliding-threshold state.
        config: The BCM configuration settings.
        row_sum_max_E: Optional row-sum limit for target E neurons.
        row_sum_max_I: Optional row-sum limit for target I neurons.
        _cached_topology: Optional pre-computed dense boolean topology mask.
            When provided, skips redundant dense conversions and validation
            for hot-loop performance.
        copy_weights: If True, preserve pure-function behavior by copying the
            full dense weight matrix before updating. If False, update the
            provided dense matrix in place. Sparse inputs are always copied.
        update_index: Optional cached sparse edge indices. Used only with
            dense in-place updates to avoid materializing full EE/IE blocks.

    Returns:
        The updated weights matrix (dense array or sparse matrix matching input).

    Raises:
        ValueError: If indexing or dimensions are mismatched.
    """

    validate_bcm_config(config)
    input_is_sparse = sparse.issparse(weights)

    if _cached_topology is not None:
        # Fast path: caller guarantees weights are dense and topology is valid.
        W = np.asarray(weights, dtype=float)
        topology = _cached_topology
    else:
        W = as_dense_weights(weights)
        topology = as_connection_mask(connection_mask, W.shape)
        _validate_weights_follow_topology(W, topology)

    idx_E_arr = validate_indices("idx_E", idx_E, W.shape[0])
    idx_I_arr = validate_indices("idx_I", idx_I, W.shape[0], allow_empty=True)
    if np.intersect1d(idx_E_arr, idx_I_arr).size:
        raise ValueError("idx_E and idx_I must be disjoint.")
    if idx_E_arr.size and int(idx_E_arr.max()) >= W.shape[1]:
        raise ValueError("idx_E contains source indices outside the weight columns.")

    updated = W.copy() if copy_weights or input_is_sparse else W

    if not copy_weights and not input_is_sparse and update_index is not None:
        _update_excitatory_block_edges_inplace(
            updated,
            update_index.target_E_source_E,
            x=x_E,
            y=y_E,
            theta=theta.E,
            config=config,
            row_sum_max=row_sum_max_E,
        )
        _update_excitatory_block_edges_inplace(
            updated,
            update_index.target_I_source_E,
            x=x_E,
            y=y_I,
            theta=theta.I,
            config=config,
            row_sum_max=row_sum_max_I,
        )
        return updated

    target_E_source_E = np.ix_(idx_E_arr, idx_E_arr)
    target_I_source_E = np.ix_(idx_I_arr, idx_E_arr)

    # Extract dense sub-blocks once and pass directly to avoid re-densification
    updated[target_E_source_E] = update_excitatory_block(
        weights=W[target_E_source_E],
        connection_mask=topology[target_E_source_E],
        x=x_E,
        y=y_E,
        theta=theta.E,
        config=config,
        row_sum_max=row_sum_max_E,
        _skip_densify=True,
    )
    updated[target_I_source_E] = update_excitatory_block(
        weights=W[target_I_source_E],
        connection_mask=topology[target_I_source_E],
        x=x_E,
        y=y_I,
        theta=theta.I,
        config=config,
        row_sum_max=row_sum_max_I,
        _skip_densify=True,
    )

    if input_is_sparse:
        return sparse.csr_matrix(updated)
    return updated


def _update_excitatory_block_edges_inplace(
    weights: NDArray[np.float64],
    index: BCMBlockUpdateIndex,
    *,
    x: ArrayLike,
    y: ArrayLike,
    theta: ArrayLike,
    config: TrainingBCMConfig,
    row_sum_max: float | ArrayLike | None,
) -> None:
    if index.rows.size == 0:
        return

    gain = bcm_gain(
        y,
        theta,
        eta=float(config.eta),
        theta_eps=float(config.theta_eps),
    )
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        delta_edges = gain[index.local_rows] * x_arr[index.local_cols]
    elif x_arr.ndim == 2:
        if gain.ndim != 2:
            raise ValueError("BCM x and y must both be 1D or both be 2D batch arrays.")
        if x_arr.shape[0] != gain.shape[0]:
            raise ValueError("Batch x and y must have the same leading dimension.")
        if x_arr.shape[0] == 0:
            raise ValueError("BCM batch arrays must not be empty.")
        delta = gain.T @ x_arr / x_arr.shape[0]
        delta_edges = delta[index.local_rows, index.local_cols]
    else:
        raise ValueError("BCM x must be a 1D vector or a 2D batch matrix.")

    weights[index.rows, index.cols] += delta_edges
    edge_values = weights[index.rows, index.cols]
    np.maximum(edge_values, 0.0, out=edge_values)
    if config.w_max is not None:
        np.minimum(edge_values, float(config.w_max), out=edge_values)
    weights[index.rows, index.cols] = edge_values

    _limit_edge_row_sums_inplace(weights, index, row_sum_max)


def _limit_edge_row_sums_inplace(
    weights: NDArray[np.float64],
    index: BCMBlockUpdateIndex,
    row_sum_max: float | ArrayLike | None,
) -> None:
    if row_sum_max is None or index.rows.size == 0:
        return

    limits = np.asarray(row_sum_max, dtype=float)
    if limits.ndim == 0:
        limits = np.full(index.row_count, float(limits), dtype=float)
    if limits.shape != (index.row_count,):
        raise ValueError(f"row_sum_max shape {limits.shape} does not match {(index.row_count,)}.")
    if not np.all(np.isfinite(limits)) or np.any(limits < 0.0):
        raise ValueError("row_sum_max must contain finite non-negative values.")

    edge_values = weights[index.rows, index.cols]
    row_totals = np.bincount(index.local_rows, weights=edge_values, minlength=index.row_count)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_scale = np.divide(limits, row_totals, out=np.ones_like(row_totals), where=row_totals > 0.0)
    row_scale = np.minimum(row_scale, 1.0)
    edge_values *= row_scale[index.local_rows]
    edge_values[(limits[index.local_rows] == 0.0) & (row_totals[index.local_rows] > 0.0)] = 0.0
    weights[index.rows, index.cols] = edge_values


def update_excitatory_block(
    *,
    weights: ArrayLike,
    connection_mask: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    theta: ArrayLike,
    config: TrainingBCMConfig,
    row_sum_max: float | ArrayLike | None = None,
    _skip_densify: bool = False,
) -> NDArray[np.float64]:
    """Applies BCM update rule and row-sum limits to a single target/source sub-block.

    Args:
        weights: The weights sub-block.
        connection_mask: The connection mask sub-block.
        x: Pre-synaptic responses.
        y: Post-synaptic responses.
        theta: Post-synaptic threshold vector.
        config: The BCM configuration settings.
        row_sum_max: Optional row-sum limit for the sub-block.
        _skip_densify: If True, skip as_dense_weights/as_connection_mask copies
            (caller guarantees inputs are already dense numpy arrays).

    Returns:
        The updated sub-block weights as a dense 2D float array.
    """

    validate_bcm_config(config)
    if _skip_densify:
        W = np.asarray(weights, dtype=float)
        topology = np.asarray(connection_mask, dtype=bool)
    else:
        W = as_dense_weights(weights, name="weights")
        topology = as_connection_mask(connection_mask, W.shape)

    delta = bcm_delta(
        x=x,
        y=y,
        theta=theta,
        eta=float(config.eta),
        theta_eps=float(config.theta_eps),
    )
    if delta.shape != W.shape:
        raise ValueError(f"BCM delta shape {delta.shape} does not match weight block shape {W.shape}.")

    updated = W + topology.astype(float) * delta
    updated = np.maximum(updated, 0.0)
    if config.w_max is not None:
        updated = np.minimum(updated, float(config.w_max))
    updated = updated * topology
    return limit_row_sums(updated, row_sum_max)


def make_bcm_row_sum_limits(
    network: NetworkState,
    config: TrainingBCMConfig,
) -> BCMRowSumLimits:
    """Creates the initial row-sum caps for BCM target/source E blocks from config.

    Args:
        network: The current NetworkState.
        config: The BCM configuration settings containing row_sum_max_scale.

    Returns:
        The calculated BCMRowSumLimits.
    """

    return initial_row_sum_limits(
        initial_weights=network.weights,
        idx_E=network.idx_E,
        idx_I=network.idx_I,
        scale=config.row_sum_max_scale,
    )


def initial_row_sum_limits(
    *,
    initial_weights: ArrayLike | sparse.spmatrix,
    idx_E: ArrayLike,
    idx_I: ArrayLike,
    scale: float | None,
) -> BCMRowSumLimits:
    """Computes initial target/source E row-sum caps based on a scaling factor.

    Args:
        initial_weights: Initial weights matrix.
        idx_E: Index array of excitatory neurons.
        idx_I: Index array of inhibitory neurons.
        scale: Scaling factor to multiply initial row sums by.

    Returns:
        The calculated BCMRowSumLimits.

    Raises:
        ValueError: If scale is invalid or indices are out of range.
    """

    if scale is None:
        return BCMRowSumLimits()
    scale = float(scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("training.bcm.row_sum_max_scale must be finite and non-negative.")

    W = as_dense_weights(initial_weights)
    idx_E_arr = validate_indices("idx_E", idx_E, W.shape[0])
    idx_I_arr = validate_indices("idx_I", idx_I, W.shape[0], allow_empty=True)

    return BCMRowSumLimits(
        target_E_source_E=row_sums(W[np.ix_(idx_E_arr, idx_E_arr)]) * scale,
        target_I_source_E=row_sums(W[np.ix_(idx_I_arr, idx_E_arr)]) * scale,
    )


def bcm_training_step(
    *,
    network: NetworkState,
    x_E: ArrayLike,
    y_E: ArrayLike,
    y_I: ArrayLike,
    theta: BCMThetaState | None,
    config: TrainingBCMConfig,
    row_sum_limits: BCMRowSumLimits | None = None,
    _cached_topology: NDArray[np.bool_] | None = None,
    copy_weights: bool = True,
    update_index: BCMEfferentUpdateIndex | None = None,
) -> BCMTrainingStepResult:
    """Runs a single schema-configured BCM training step.

    This function handles threshold initialization on the first step, handles
    pre- vs post- order threshold updates, applies BCM plasticity to excitatory
    efferents, and enforces row-sum limits.

    Args:
        network: The current NetworkState.
        x_E: Pre-synaptic excitatory responses.
        y_E: Post-synaptic excitatory responses.
        y_I: Post-synaptic inhibitory responses.
        theta: The current BCMThetaState. If None, initializes the thresholds.
        config: The BCM configuration settings.
        row_sum_limits: Optional BCMRowSumLimits.
        _cached_topology: Optional pre-computed dense boolean topology mask.
        copy_weights: If True, return an updated network with copied weights.
            If False, mutate dense network weights in place for training hot loops.
        update_index: Optional cached sparse edge indices for dense in-place updates.

    Returns:
        A BCMTrainingStepResult containing the updated network state, thresholds, and flag.
    """

    validate_bcm_config(config)
    limits = BCMRowSumLimits() if row_sum_limits is None else row_sum_limits

    if theta is None:
        initialized = initialize_theta(y_E, y_I, config)
        return BCMTrainingStepResult(
            network=network,
            theta=initialized,
            theta_for_update=initialized,
            updated=False,
        )

    theta_for_update = theta
    next_theta = theta
    if config.theta_update_order == "pre":
        next_theta = update_theta(theta, y_E, y_I, config)
        theta_for_update = next_theta

    next_weights = update_efferent_excitatory_weights(
        network.weights,
        network.connectivity,
        network.idx_E,
        network.idx_I,
        x_E=x_E,
        y_E=y_E,
        y_I=y_I,
        theta=theta_for_update,
        config=config,
        row_sum_max_E=limits.target_E_source_E,
        row_sum_max_I=limits.target_I_source_E,
        _cached_topology=_cached_topology,
        copy_weights=copy_weights,
        update_index=update_index,
    )

    if config.theta_update_order == "post":
        next_theta = update_theta(theta, y_E, y_I, config)

    next_network = network if next_weights is network.weights else NetworkState(
        layout=network.layout,
        connectivity=network.connectivity,
        weights=next_weights,
    )

    return BCMTrainingStepResult(
        network=next_network,
        theta=next_theta,
        theta_for_update=theta_for_update,
        updated=True,
    )


def _validate_weights_follow_topology(W: NDArray[np.float64], topology: NDArray[np.bool_]) -> None:
    if np.any((~topology) & (W != 0.0)):
        raise ValueError("weights contain nonzero values outside connection_mask topology.")
