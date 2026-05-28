from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from v1_simulation.network.empirical import EmpiricalWeightSamples
from v1_simulation.network.state import NetworkState, PopulationLayout


@dataclass(frozen=True, slots=True)
class WeightSpec:
    j: float
    g: float
    ee_scale: float = 1.0
    ei_scale: float = 1.0
    ex_scale: float = 1.0
    ie_scale: float = 1.0
    ii_scale: float = 1.0
    ix_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.j <= 0.0:
            raise ValueError(f"j must be positive, got {self.j}.")
        if self.g <= 0.0:
            raise ValueError(f"g must be positive, got {self.g}.")
        for name in ("ee_scale", "ei_scale", "ex_scale", "ie_scale", "ii_scale", "ix_scale"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")


def as_dense_weights(
    weights: ArrayLike | sparse.spmatrix,
    *,
    name: str = "weights",
) -> NDArray[np.float64]:
    """Converts a dense or sparse weight representation to a dense 2D float array.

    Args:
        weights: The input weights matrix (dense array-like or sparse matrix).
        name: Name of the variable for error reporting.

    Returns:
        A copy of the weights as a dense 2D float array.

    Raises:
        ValueError: If weights are not 2D or contain non-finite values.
    """

    if sparse.issparse(weights):
        arr = weights.toarray().astype(float, copy=False)
    else:
        arr = np.asarray(weights, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr.astype(float, copy=True)


def as_connection_mask(
    mask: ArrayLike | sparse.spmatrix,
    shape: tuple[int, int],
    *,
    name: str = "connection_mask",
) -> NDArray[np.bool_]:
    """Validates and converts a connection topology mask to a boolean 2D array.

    Masks must be boolean or numeric 0/1. Fractional masks are rejected because
    topology should select connections, not scale existing weights or updates.

    Args:
        mask: The input mask matrix (dense array-like or sparse matrix).
        shape: The expected shape of the matrix.
        name: Name of the variable for error reporting.

    Returns:
        A copy of the connection mask as a boolean 2D array.

    Raises:
        ValueError: If the shape does not match or elements are not boolean/binary.
    """

    if sparse.issparse(mask):
        arr = mask.toarray()
    else:
        arr = np.asarray(mask)

    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}.")

    if arr.dtype == bool:
        return arr.astype(bool, copy=True)

    numeric = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} must be boolean or contain only 0/1 values.")
    return numeric.astype(bool)


def validate_indices(
    name: str,
    indices: ArrayLike,
    upper_bound: int,
    *,
    allow_empty: bool = False,
) -> NDArray[np.int64]:
    """Validates a 1D array of unique integer indices.

    Args:
        name: Name of the variable for error reporting.
        indices: 1D index array-like.
        upper_bound: Exclusive upper bound for the index values.
        allow_empty: If True, permits empty index arrays.

    Returns:
        A validated copy of the index array.

    Raises:
        ValueError: If array is not 1D, has wrong values, duplicates, or is out of range.
    """

    raw = np.asarray(indices)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1D index array.")
    if raw.size == 0:
        if allow_empty:
            return np.array([], dtype=np.int64)
        raise ValueError(f"{name} must not be empty.")

    idx = raw.astype(np.int64, copy=False)
    if not np.all(np.asarray(raw, dtype=float) == idx):
        raise ValueError(f"{name} must contain integer indices.")
    if np.any(idx < 0) or np.any(idx >= int(upper_bound)):
        raise ValueError(f"{name} contains out-of-range indices.")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{name} contains duplicate indices.")
    return idx.astype(np.int64, copy=True)


def limit_row_sums(
    weights: ArrayLike,
    row_sum_max: float | ArrayLike | None,
) -> NDArray[np.float64]:
    """Scales rows of a weights matrix down to not exceed a configured limit.

    Args:
        weights: The input weights matrix.
        row_sum_max: The row-sum cap(s). Can be a scalar float or a 1D array of limits.

    Returns:
        The scaled weights matrix.

    Raises:
        ValueError: If limits dimensions or values are invalid.
    """

    arr = as_dense_weights(weights, name="weights")
    if row_sum_max is None:
        return arr

    limits = np.asarray(row_sum_max, dtype=float)
    if limits.ndim == 0:
        limits = np.full(arr.shape[0], float(limits), dtype=float)

    if limits.shape != (arr.shape[0],):
        raise ValueError(f"row_sum_max shape {limits.shape} does not match {(arr.shape[0],)}.")
    if not np.all(np.isfinite(limits)) or np.any(limits < 0.0):
        raise ValueError("row_sum_max must contain finite non-negative values.")

    row_totals = np.sum(arr, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.divide(limits, row_totals, out=np.ones_like(row_totals), where=row_totals > 0.0)
    scale = np.minimum(scale, 1.0)
    limited = arr * scale[:, np.newaxis]
    limited[(limits == 0.0) & (row_totals > 0.0), :] = 0.0
    return limited


def row_sums(weights: ArrayLike | sparse.spmatrix) -> NDArray[np.float64]:
    """Calculates dense float row sums for dense or sparse matrices.

    Args:
        weights: The input weights matrix.

    Returns:
        A 1D array of row sums.
    """

    if sparse.issparse(weights):
        return np.asarray(weights.sum(axis=1)).ravel().astype(float)
    return np.sum(as_dense_weights(weights, name="weights"), axis=1)


def sample_weights(
    layout: PopulationLayout,
    connectivity: sparse.csr_matrix,
    spec: WeightSpec,
    samples: EmpiricalWeightSamples,
    rng: np.random.Generator,
) -> NetworkState:
    """Samples signed sparse synaptic weights based on the connectivity mask and empirical samples.

    Args:
        layout: The network population spatial layout.
        connectivity: Boolean sparse connectivity mask indicating existing connections.
        spec: The weight scaling configuration containing coupling strength parameters.
        samples: Empirical connection weight samples for each connection block.
        rng: Random generator stream.

    Returns:
        The NetworkState containing layout, connectivity, and sampled signed weights.
    """

    q = sparse.csr_matrix(connectivity, dtype=bool)
    if q.shape != layout.shape:
        raise ValueError(f"connectivity shape {q.shape} does not match layout shape {layout.shape}.")

    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    data: list[NDArray[np.float64]] = []

    blocks = (
        (layout.idx_E, layout.idx_E, +spec.j * spec.ee_scale, samples.ee),
        (layout.idx_E, layout.idx_I, -spec.j * spec.g * spec.ei_scale, samples.ei),
        (layout.idx_E, layout.idx_X, +spec.j * spec.ex_scale, samples.ex),
        (layout.idx_I, layout.idx_E, +spec.j * spec.ie_scale, samples.ie),
        (layout.idx_I, layout.idx_I, -spec.j * spec.g * spec.ii_scale, samples.ii),
        (layout.idx_I, layout.idx_X, +spec.j * spec.ix_scale, samples.ix),
    )

    for target_idx, source_idx, gain, block_samples in blocks:
        _append_weight_block(rows, cols, data, q, target_idx, source_idx, gain, block_samples, rng)

    if not rows:
        weights = sparse.csr_matrix(layout.shape, dtype=float)
    else:
        weights = sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=layout.shape,
            dtype=float,
        ).tocsr()

    return NetworkState(layout=layout, connectivity=q, weights=weights)


def _append_weight_block(
    rows: list[NDArray[np.int64]],
    cols: list[NDArray[np.int64]],
    data: list[NDArray[np.float64]],
    q: sparse.csr_matrix,
    target_idx: NDArray[np.int64],
    source_idx: NDArray[np.int64],
    gain: float,
    samples: NDArray[np.float64],
    rng: np.random.Generator,
) -> None:
    block = q[np.ix_(target_idx, source_idx)]
    local_rows, local_cols = block.nonzero()
    if local_rows.size == 0:
        return
    rows.append(target_idx[local_rows])
    cols.append(source_idx[local_cols])
    data.append(float(gain) * rng.choice(samples, size=local_rows.size))
