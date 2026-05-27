from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
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
