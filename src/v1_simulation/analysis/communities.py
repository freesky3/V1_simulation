from __future__ import annotations

from typing import TYPE_CHECKING

import bct
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist, squareform

from v1_simulation.analysis.clusters import relabel_consecutive
from v1_simulation.analysis.types import CommunityResult

if TYPE_CHECKING:
    from v1_simulation.config.schema import LouvainConfig


def identify_communities(
    activity_trace: ArrayLike,
    config: "LouvainConfig",
    *,
    rng: np.random.Generator | None = None,
) -> CommunityResult:
    """Performs consensus Louvain community detection on a neural activity trace.

    Calculates cosine similarity, builds a thresholded proportional graph, runs Louvain
    multiple times, forms an agreement matrix, and resolves a consensus partition. Small
    or weakly connected clusters are pruned based on configured thresholds.

    Args:
        activity_trace: Firing rates of shape (n_neurons, n_time).
        config: Louvain community detection configuration schema.
        rng: Optional random number generator for reproducible seed generation.

    Returns:
        A CommunityResult containing labels, similarity matrix, and agreement matrix.

    Raises:
        ValueError: If activity_trace dimensions are invalid, contains NaN/inf, or config
            parameters are out of bounds.
    """
    trace = np.asarray(activity_trace, dtype=float)
    if trace.ndim != 2:
        raise ValueError("activity_trace must have shape (n_neurons, n_time).")
    if trace.shape[0] < 1:
        raise ValueError("activity_trace must contain at least one neuron.")
    if not np.all(np.isfinite(trace)):
        raise ValueError("activity_trace contains NaN or infinite values.")

    _validate_louvain_config(config)
    local_rng = np.random.default_rng() if rng is None else rng

    similarity = cosine_similarity_matrix(trace)
    graph = bct.threshold_proportional(similarity, float(config.thr_prop))
    graph = bct.weight_conversion(graph, "normalize")

    partitions = np.empty((trace.shape[0], int(config.num_runs)), dtype=np.int64)
    for run_idx in range(int(config.num_runs)):
        seed = _next_seed(local_rng)
        labels, _ = bct.community_louvain(graph, gamma=float(config.gamma), seed=seed)
        partitions[:, run_idx] = np.asarray(labels, dtype=np.int64)

    agreement = agreement_matrix(partitions) / float(config.num_runs)
    consensus_seed = _next_seed(local_rng)
    consensus = np.asarray(
        bct.consensus_und(
            agreement,
            tau=float(config.consensus_tau),
            reps=int(config.consensus_reps),
            seed=consensus_seed,
        ),
        dtype=float,
    )
    labels = _drop_weak_or_small_clusters(
        consensus,
        graph > 0.0,
        min_module_degree=float(config.min_module_degree),
        min_cluster_size=int(config.min_cluster_size),
    )
    final_labels = relabel_consecutive(np.nan_to_num(labels, nan=0.0))

    return CommunityResult(
        labels=final_labels,
        similarity=similarity,
        agreement=agreement,
        diagnostics={
            "thr_prop": float(config.thr_prop),
            "gamma": float(config.gamma),
            "num_runs": int(config.num_runs),
            "consensus_tau": float(config.consensus_tau),
            "consensus_reps": int(config.consensus_reps),
            "min_module_degree": float(config.min_module_degree),
            "min_cluster_size": int(config.min_cluster_size),
            "n_ensembles": int(np.unique(final_labels[final_labels != 0]).size),
            "classified_neurons": int(np.sum(final_labels != 0)),
        },
    )


def cosine_similarity_matrix(activity_trace: ArrayLike) -> NDArray[np.float64]:
    """Computes the pairwise cosine similarity matrix between neural activity traces.

    Self-similarity (diagonal) is set to 0.0.

    Args:
        activity_trace: Firing rates of shape (n_neurons, n_time).

    Returns:
        A square similarity matrix of shape (n_neurons, n_neurons).

    Raises:
        ValueError: If activity_trace shape is not 2D.
    """
    trace = np.asarray(activity_trace, dtype=float)
    if trace.ndim != 2:
        raise ValueError("activity_trace must have shape (n_neurons, n_time).")
    if trace.shape[0] == 1:
        return np.zeros((1, 1), dtype=float)
    distances = pdist(trace, metric="cosine")
    similarity = 1.0 - squareform(distances)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(similarity, 0.0)
    return similarity.astype(float, copy=False)


def agreement_matrix(partitions: ArrayLike) -> NDArray[np.float64]:
    """Constructs a node-by-node agreement matrix from a set of partition runs.

    For each run, increments the entry for any pair of nodes assigned to the same community.

    Args:
        partitions: Partition matrix of shape (n_nodes, n_partitions).

    Returns:
        A square agreement matrix of shape (n_nodes, n_nodes).

    Raises:
        ValueError: If partitions shape is not 2D.
    """
    values = np.asarray(partitions)
    if values.ndim != 2:
        raise ValueError("partitions must have shape (n_nodes, n_partitions).")
    n_nodes, n_partitions = values.shape
    agreement = np.zeros((n_nodes, n_nodes), dtype=float)
    for run_idx in range(n_partitions):
        labels = values[:, run_idx]
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            if members.size:
                agreement[np.ix_(members, members)] += 1.0
    return agreement


def _drop_weak_or_small_clusters(
    labels: NDArray[np.float64],
    graph_binary: NDArray[np.bool_],
    *,
    min_module_degree: float,
    min_cluster_size: int,
) -> NDArray[np.float64]:
    cleaned = np.asarray(labels, dtype=float).copy()
    for c_id in np.unique(cleaned[np.isfinite(cleaned)]):
        members = np.flatnonzero(cleaned == c_id)
        if members.size == 0:
            continue
        degree = graph_binary[np.ix_(members, members)].sum(axis=1)
        cleaned[members[degree < float(min_module_degree)]] = np.nan

    for c_id in np.unique(cleaned[np.isfinite(cleaned)]):
        if np.sum(cleaned == c_id) < int(min_cluster_size):
            cleaned[cleaned == c_id] = np.nan
    return cleaned


def _validate_louvain_config(config: "LouvainConfig") -> None:
    if not 0.0 < float(config.thr_prop) <= 1.0:
        raise ValueError("config.thr_prop must be in (0, 1].")
    if float(config.gamma) <= 0.0:
        raise ValueError("config.gamma must be positive.")
    if int(config.num_runs) <= 0:
        raise ValueError("config.num_runs must be positive.")
    if not 0.0 <= float(config.consensus_tau) <= 1.0:
        raise ValueError("config.consensus_tau must be in [0, 1].")
    if int(config.consensus_reps) <= 0:
        raise ValueError("config.consensus_reps must be positive.")
    if float(config.min_module_degree) < 0.0:
        raise ValueError("config.min_module_degree must be non-negative.")
    if int(config.min_cluster_size) <= 0:
        raise ValueError("config.min_cluster_size must be positive.")


def _next_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
