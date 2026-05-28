from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from v1_simulation.analysis.clusters import cluster_members, labels_array


def generate_grid_positions(n_side: int, region_size: float) -> NDArray[np.float64]:
    """Generates 2D Cartesian grid coordinates for a given grid side size and region size.

    Args:
        n_side: Number of nodes along one side of the square grid.
        region_size: The total width/height of the square area.

    Returns:
        A numpy array of shape (n_side * n_side, 2) containing coordinate points.

    Raises:
        ValueError: If n_side or region_size is not positive.
    """
    if int(n_side) <= 0:
        raise ValueError("n_side must be positive.")
    if float(region_size) <= 0.0:
        raise ValueError("region_size must be positive.")
    half_size = float(region_size) / 2.0
    x = np.linspace(-half_size, half_size, int(n_side))
    y = np.linspace(-half_size, half_size, int(n_side))
    grid_x, grid_y = np.meshgrid(x, y)
    return np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(float, copy=False)


def distance_matrix(
    coords: ArrayLike,
    *,
    region_size: float | None = None,
    periodic: bool = False,
) -> NDArray[np.float64]:
    """Computes the pairwise Euclidean distance matrix between coordinate points.

    Supports periodic boundary conditions (toroidal topology) if requested.

    Args:
        coords: Coordinates of points, shape (n_neurons, 2).
        region_size: The size of the region; required if periodic is True.
        periodic: Whether to use periodic boundary conditions.

    Returns:
        A square numpy array of shape (n_neurons, n_neurons) of pairwise distances.

    Raises:
        ValueError: If coords shape is invalid, coords contains inf/NaN, or region_size is invalid.
    """
    points = np.asarray(coords, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coords must have shape (n_neurons, 2).")
    if not np.all(np.isfinite(points)):
        raise ValueError("coords contains NaN or infinite values.")

    delta = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    if periodic:
        if region_size is None or float(region_size) <= 0.0:
            raise ValueError("region_size must be positive when periodic=True.")
        abs_delta = np.abs(delta)
        delta = np.minimum(abs_delta, float(region_size) - abs_delta)
    return np.sqrt(np.sum(delta**2, axis=2))


def select_center_indices(n_side: int, *, side_fraction: float = 0.5) -> NDArray[np.int64]:
    """Selects the indices of neurons located in the center square fraction of a grid.

    Useful to avoid boundary effects in spatial analyses.

    Args:
        n_side: Number of nodes along one side of the full grid.
        side_fraction: The linear fraction of the center region to select (in (0, 1]).

    Returns:
        A 1D numpy array of integer indices of the selected center neurons.

    Raises:
        ValueError: If n_side is not positive or side_fraction is out of range.
    """
    if int(n_side) <= 0:
        raise ValueError("n_side must be positive.")
    if not 0.0 < float(side_fraction) <= 1.0:
        raise ValueError("side_fraction must be in (0, 1].")
    center_side = max(1, int(round(int(n_side) * float(side_fraction))))
    start = (int(n_side) - center_side) // 2
    end = start + center_side
    grid = np.arange(int(n_side) * int(n_side), dtype=np.int64).reshape(int(n_side), int(n_side))
    return grid[start:end, start:end].ravel().copy()


def cluster_spatial_metrics(
    labels: ArrayLike,
    distances: ArrayLike,
) -> dict[int, dict[str, float]]:
    """Computes spatial clustering metrics for each detected community.

    Metrics include mean pairwise distance and mean nearest neighbor distance.

    Args:
        labels: Community labels for all neurons.
        distances: Pairwise distance matrix of shape (n_neurons, n_neurons).

    Returns:
        A dictionary mapping each non-zero cluster ID to a dict containing:
            - 'mean_pairwise_distance': Average distance between all members.
            - 'nearest_neighbor_distance': Average distance to each member's nearest neighbor
              within the same cluster.

    Raises:
        ValueError: If distances is not a square matrix.
    """
    dist = np.asarray(distances, dtype=float)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("distances must be a square matrix.")

    label_values = labels_array(labels, n_neurons=dist.shape[0])
    metrics: dict[int, dict[str, float]] = {}
    for c_id, members in cluster_members(label_values).items():
        if members.size < 2:
            continue
        mean_dist, nnd_values = _cluster_spatial_values(dist, members)
        metrics[c_id] = {
            "mean_pairwise_distance": float(mean_dist),
            "nearest_neighbor_distance": float(np.mean(nnd_values)),
        }
    return metrics


def _cluster_spatial_values(
    distances: NDArray[np.float64],
    members: NDArray[np.int64],
) -> tuple[float, NDArray[np.float64]]:
    sub_dist = np.array(distances[np.ix_(members, members)], dtype=float, copy=True)
    pairwise = sub_dist[np.triu_indices_from(sub_dist, k=1)]
    pairwise = pairwise[np.isfinite(pairwise)]
    if pairwise.size == 0:
        return np.nan, np.array([], dtype=float)
    np.fill_diagonal(sub_dist, np.inf)
    nnd_values = np.min(sub_dist, axis=1)
    return float(np.mean(pairwise)), nnd_values[np.isfinite(nnd_values)]
