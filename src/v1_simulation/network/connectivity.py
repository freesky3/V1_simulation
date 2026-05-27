from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from v1_simulation.network.empirical import ConnectionProbabilities
from v1_simulation.network.state import PopulationLayout

PopulationName = Literal["E", "I", "X"]


@dataclass(frozen=True, slots=True)
class SpatialKernel:
    sigma_narrow: float
    sigma_broad: float
    kappa: float

    def __post_init__(self) -> None:
        if self.sigma_narrow <= 0.0 or self.sigma_broad <= 0.0:
            raise ValueError("Kernel sigmas must be positive.")
        if self.sigma_narrow >= self.sigma_broad:
            raise ValueError("sigma_narrow must be smaller than sigma_broad.")
        if not 0.0 <= self.kappa <= 1.0:
            raise ValueError("kappa must be in [0, 1].")

    def evaluate(self, distance: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluates the spatial kernel for a given array of distances.

        Computes the weighted sum of a narrow and a broad Gaussian profile based on
        sigma_narrow, sigma_broad, and kappa.

        Args:
            distance: Array of pairwise distances.

        Returns:
            Array of kernel score values of the same shape as distance.
        """
        distance = np.asarray(distance, dtype=float)
        d2 = distance * distance
        narrow = np.exp(-d2 / (2.0 * self.sigma_narrow**2))
        broad = np.exp(-d2 / (2.0 * self.sigma_broad**2))
        return self.kappa * narrow + (1.0 - self.kappa) * broad


@dataclass(frozen=True, slots=True)
class ConnectionBlock:
    """One directed block, named by target population then source population."""

    target: PopulationName
    source: PopulationName
    probability: float
    sign: int

    @property
    def key(self) -> str:
        return f"{self.target.lower()}{self.source.lower()}"


@dataclass(frozen=True, slots=True)
class ConnectivitySpec:
    """Connectivity probabilities named as target/source blocks."""

    p_ee: float
    p_ei: float
    p_ex: float
    p_ie: float
    p_ii: float
    p_ix: float
    periodic: bool = True
    equalize_indegree: bool = True

    @classmethod
    def from_probabilities(
        cls,
        probabilities: ConnectionProbabilities,
        *,
        periodic: bool,
        equalize_indegree: bool = True,
    ) -> "ConnectivitySpec":
        return cls(
            p_ee=probabilities.ee,
            p_ei=probabilities.ei,
            p_ex=probabilities.ex,
            p_ie=probabilities.ie,
            p_ii=probabilities.ii,
            p_ix=probabilities.ix,
            periodic=periodic,
            equalize_indegree=equalize_indegree,
        )

    @property
    def blocks(self) -> tuple[ConnectionBlock, ...]:
        return (
            ConnectionBlock("E", "E", self.p_ee, +1),
            ConnectionBlock("E", "I", self.p_ei, -1),
            ConnectionBlock("E", "X", self.p_ex, +1),
            ConnectionBlock("I", "E", self.p_ie, +1),
            ConnectionBlock("I", "I", self.p_ii, -1),
            ConnectionBlock("I", "X", self.p_ix, +1),
        )

    def probability_for(self, target: PopulationName, source: PopulationName) -> float:
        key = f"p_{target.lower()}{source.lower()}"
        return float(getattr(self, key))


def probability_matrix(
    layout: PopulationLayout,
    kernel: SpatialKernel,
    spec: ConnectivitySpec,
) -> NDArray[np.float64]:
    """Builds the dense Bernoulli connection probability matrix without sampling.

    Args:
        layout: The network population spatial layout.
        kernel: The spatial kernel describing decay over distance.
        spec: The connectivity specifications containing block probabilities.

    Returns:
        A 2D dense float array representing connection probabilities between all neuron pairs.
    """

    probabilities = np.zeros(layout.shape, dtype=float)
    for target_idx, source_idx, distance, block in _iter_distance_blocks(layout, spec):
        valid = np.ones(distance.shape, dtype=bool)
        if block.target == block.source and block.target in {"E", "I"}:
            valid &= target_idx[:, None] != source_idx[None, :]
        block_prob = probability_block(
            kernel.evaluate(distance),
            block.probability,
            valid_mask=valid,
            equalize_rows=spec.equalize_indegree,
        )
        probabilities[np.ix_(target_idx, source_idx)] = block_prob
    return probabilities


def sample_connectivity(
    layout: PopulationLayout,
    kernel: SpatialKernel,
    spec: ConnectivitySpec,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    """Samples a boolean sparse connectivity matrix from the connection probabilities.

    Args:
        layout: The network population spatial layout.
        kernel: The spatial kernel describing decay over distance.
        spec: The connectivity specifications.
        rng: Random generator stream for sampling.

    Returns:
        A boolean CSR sparse matrix representing the connection mask.
    """

    q = np.zeros(layout.shape, dtype=bool)
    for target_idx, source_idx, distance, block in _iter_distance_blocks(layout, spec):
        valid = np.ones(distance.shape, dtype=bool)
        if block.target == block.source and block.target in {"E", "I"}:
            valid &= target_idx[:, None] != source_idx[None, :]
        block_prob = probability_block(
            kernel.evaluate(distance),
            block.probability,
            valid_mask=valid,
            equalize_rows=spec.equalize_indegree,
        )
        q[np.ix_(target_idx, source_idx)] = rng.random(block_prob.shape) < block_prob
    return sparse.csr_matrix(q)


def probability_block(
    score: NDArray[np.float64],
    target_probability: float,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
    equalize_rows: bool = True,
) -> NDArray[np.float64]:
    """Scales a spatial score matrix to match a target connection probability.

    Args:
        score: Array of pairwise spatial scores.
        target_probability: The desired mean connection probability for valid pairs.
        valid_mask: Boolean mask indicating which connections are allowed.
        equalize_rows: If True, scales each row independently so that the average indegree
            per target neuron is equalized. Otherwise, scales the block globally.

    Returns:
        Array of connection probabilities scaled to the target probability.
    """

    if not 0.0 <= target_probability <= 1.0:
        raise ValueError(f"Connection probabilities must be in [0, 1], got {target_probability}.")

    score = np.asarray(score, dtype=float)
    if valid_mask is None:
        valid = np.ones(score.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != score.shape:
            raise ValueError(f"valid_mask shape {valid.shape} does not match score shape {score.shape}.")

    out = np.zeros(score.shape, dtype=float)
    if target_probability == 0.0 or not np.any(valid):
        return out

    base = np.where(valid, score, 0.0)
    if np.any(base[valid] < 0.0):
        raise ValueError("Connectivity score must be non-negative.")
    if equalize_rows:
        return _row_equalized_probability(base, valid, target_probability)
    return _globally_scaled_probability(base, valid, target_probability)


def _iter_distance_blocks(
    layout: PopulationLayout,
    spec: ConnectivitySpec,
):
    dist_l23 = layout.l23.distance_matrix(periodic=spec.periodic)
    dist_x_to_l23 = layout.l23.distance_to(layout.l4, periodic=spec.periodic)
    local_x = np.arange(layout.l4.n_cells, dtype=np.int64)
    blocks = spec.blocks

    yield layout.idx_E, layout.idx_E, dist_l23[np.ix_(layout.idx_E, layout.idx_E)], blocks[0]
    yield layout.idx_E, layout.idx_I, dist_l23[np.ix_(layout.idx_E, layout.idx_I)], blocks[1]
    yield layout.idx_E, layout.idx_X, dist_x_to_l23[np.ix_(layout.idx_E, local_x)], blocks[2]
    yield layout.idx_I, layout.idx_E, dist_l23[np.ix_(layout.idx_I, layout.idx_E)], blocks[3]
    yield layout.idx_I, layout.idx_I, dist_l23[np.ix_(layout.idx_I, layout.idx_I)], blocks[4]
    yield layout.idx_I, layout.idx_X, dist_x_to_l23[np.ix_(layout.idx_I, local_x)], blocks[5]


def _row_equalized_probability(
    base: NDArray[np.float64],
    valid: NDArray[np.bool_],
    target_probability: float,
) -> NDArray[np.float64]:
    valid_counts = valid.sum(axis=1, keepdims=True)
    positive_rows = (valid_counts[:, 0] > 0) & (base.sum(axis=1) > 0.0)
    if np.any((valid_counts[:, 0] > 0) & ~positive_rows):
        raise ValueError("Cannot assign positive probability to a row with zero connectivity score.")

    lo = np.zeros((base.shape[0], 1), dtype=float)
    hi = np.ones((base.shape[0], 1), dtype=float)

    active = positive_rows[:, None]
    for _ in range(80):
        means = _row_valid_mean(np.clip(hi * base, 0.0, 1.0), valid, valid_counts)
        needs_more = active & (means < target_probability)
        if not np.any(needs_more):
            break
        hi = np.where(needs_more, hi * 2.0, hi)
    else:
        raise ValueError(f"Could not scale rows to target probability {target_probability}.")

    for _ in range(48):
        mid = (lo + hi) / 2.0
        means = _row_valid_mean(np.clip(mid * base, 0.0, 1.0), valid, valid_counts)
        hi = np.where(active & (means > target_probability), mid, hi)
        lo = np.where(active & (means <= target_probability), mid, lo)

    out = np.clip(((lo + hi) / 2.0) * base, 0.0, 1.0)
    out[~valid] = 0.0
    return out


def _globally_scaled_probability(
    base: NDArray[np.float64],
    valid: NDArray[np.bool_],
    target_probability: float,
) -> NDArray[np.float64]:
    if np.sum(base[valid]) == 0.0:
        raise ValueError("Cannot assign positive probability to a zero connectivity score matrix.")

    lo = 0.0
    hi = 1.0
    for _ in range(80):
        if np.mean(np.clip(hi * base[valid], 0.0, 1.0)) >= target_probability:
            break
        hi *= 2.0
    else:
        raise ValueError(f"Could not scale block to target probability {target_probability}.")

    for _ in range(48):
        mid = (lo + hi) / 2.0
        if np.mean(np.clip(mid * base[valid], 0.0, 1.0)) > target_probability:
            hi = mid
        else:
            lo = mid

    out = np.clip(((lo + hi) / 2.0) * base, 0.0, 1.0)
    out[~valid] = 0.0
    return out


def _row_valid_mean(
    probability: NDArray[np.float64],
    valid: NDArray[np.bool_],
    valid_counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    row_sums = np.sum(np.where(valid, probability, 0.0), axis=1, keepdims=True)
    return np.divide(row_sums, valid_counts, out=np.zeros_like(row_sums), where=valid_counts > 0)
