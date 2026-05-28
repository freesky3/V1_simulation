import numpy as np
import pytest

from v1_simulation.analysis.spatial import (
    cluster_spatial_metrics,
    distance_matrix,
    generate_grid_positions,
    select_center_indices,
)


def test_generate_grid_positions() -> None:
    # 2x2 grid in a 2.0x2.0 region
    coords = generate_grid_positions(2, 2.0)
    assert coords.shape == (4, 2)
    # Boundary-placed nodes for 2x2: linspace(-1.0, 1.0, 2)
    expected = np.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
    ])
    assert np.allclose(coords, expected)

    # Validation errors
    with pytest.raises(ValueError, match="n_side must be positive"):
        generate_grid_positions(0, 1.0)
    with pytest.raises(ValueError, match="region_size must be positive"):
        generate_grid_positions(2, -1.0)


def test_distance_matrix() -> None:
    # Points on a 1D-like ring mapping to toroidal distances
    coords = np.array([
        [0.0, 0.0],
        [0.8, 0.0],
        [1.8, 0.0],
    ])
    # Case 1: Non-periodic
    dist_np = distance_matrix(coords, periodic=False)
    assert dist_np.shape == (3, 3)
    assert np.isclose(dist_np[0, 1], 0.8)
    assert np.isclose(dist_np[0, 2], 1.8)

    # Case 2: Periodic with region_size = 2.0
    dist_p = distance_matrix(coords, region_size=2.0, periodic=True)
    # Toroidal wrapping: distance between 0.0 and 1.8 is min(1.8, 2.0 - 1.8) = 0.2
    assert np.isclose(dist_p[0, 2], 0.2)

    # Validation errors
    with pytest.raises(ValueError, match="coords must have shape"):
        distance_matrix(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="coords contains NaN or infinite values"):
        distance_matrix([[1.0, np.nan], [2.0, 2.0]])
    with pytest.raises(ValueError, match="region_size must be positive"):
        distance_matrix(coords, periodic=True)


def test_select_center_indices() -> None:
    # 4x4 grid (16 cells)
    # indices:
    #  0  1  2  3
    #  4  5  6  7
    #  8  9 10 11
    # 12 13 14 15
    # Center square with side_fraction=0.5 -> 2x2 area -> indices [5, 6, 9, 10]
    idx = select_center_indices(4, side_fraction=0.5)
    assert np.array_equal(np.sort(idx), np.array([5, 6, 9, 10], dtype=np.int64))

    # Error handling
    with pytest.raises(ValueError, match="n_side must be positive"):
        select_center_indices(0)
    with pytest.raises(ValueError, match="side_fraction must be in"):
        select_center_indices(4, side_fraction=1.5)


def test_cluster_spatial_metrics() -> None:
    # 5 neurons
    # distances
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [10.0, 10.0],
        [11.0, 10.0],
    ])
    dist = distance_matrix(coords)
    # labels: neurons 0, 1, 2 in cluster 1; neurons 3, 4 in cluster 2
    labels = np.array([1, 1, 1, 2, 2])

    metrics = cluster_spatial_metrics(labels, dist)

    assert set(metrics.keys()) == {1, 2}

    # For cluster 1: members are [0, 1, 2]
    # pairwise distances: d(0,1)=1.0, d(0,2)=1.0, d(1,2)=sqrt(2) ~ 1.414
    # mean_pairwise_distance: (1.0 + 1.0 + 1.414) / 3 = 1.138
    # nearest neighbors: NN(0)=1 (dist 1.0), NN(1)=0 (dist 1.0), NN(2)=0 (dist 1.0).
    # nearest_neighbor_distance = 1.0
    assert metrics[1]["mean_pairwise_distance"] == pytest.approx((2.0 + np.sqrt(2)) / 3)
    assert metrics[1]["nearest_neighbor_distance"] == pytest.approx(1.0)

    # For cluster 2: members are [3, 4]
    # pairwise: d(3,4) = 1.0
    assert metrics[2]["mean_pairwise_distance"] == pytest.approx(1.0)
    assert metrics[2]["nearest_neighbor_distance"] == pytest.approx(1.0)

    # Mismatched distance matrix
    with pytest.raises(ValueError, match="distances must be a square matrix"):
        cluster_spatial_metrics(labels, np.ones((5, 4)))
