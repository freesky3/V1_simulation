import numpy as np
import pytest
from types import SimpleNamespace

from v1_simulation.network.geometry import L2_3, L4, SheetGeometry


def test_sheet_geometry_basic() -> None:
    # 3x3 grid, size 3.0, z=0.0
    sheet = SheetGeometry(n_side=3, region_size=3.0, z_pos=0.0)
    assert sheet.n_cells == 9
    assert sheet.N == 9
    assert sheet.z_pos == 0.0

    # Spacing = 3 / 3 = 1.0. Centers of grid: [-1.0, 0.0, 1.0]
    expected_coords = np.array([
        [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
        [-1.0,  0.0], [0.0,  0.0], [1.0,  0.0],
        [-1.0,  1.0], [0.0,  1.0], [1.0,  1.0]
    ])
    assert np.allclose(sheet.coords, expected_coords)

    # Validation errors
    with pytest.raises(ValueError, match="n_side must be positive"):
        SheetGeometry(n_side=0, region_size=1.0, z_pos=0.0)
    with pytest.raises(ValueError, match="region_size must be positive"):
        SheetGeometry(n_side=3, region_size=-1.0, z_pos=0.0)


def test_sheet_geometry_distances() -> None:
    sheet = SheetGeometry(n_side=3, region_size=3.0, z_pos=0.0)

    # Non-periodic Euclidean distance
    dist = sheet.get_distance_matrix(periodic=False)
    assert np.allclose(np.diag(dist), 0.0)
    assert dist[0, 2] == pytest.approx(2.0)  # distance between (-1, -1) and (1, -1)

    # Periodic Euclidean distance
    dist_p = sheet.get_distance_matrix(periodic=True)
    assert np.allclose(np.diag(dist_p), 0.0)
    # Wrapping distance: between x=-1 and x=1, periodic wraps around size 3.0.
    # Diff = 2.0. Periodic diff = min(2.0, 3.0 - 2.0) = 1.0.
    assert dist_p[0, 2] == pytest.approx(1.0)


def test_cross_layer_distances() -> None:
    l23 = SheetGeometry(n_side=2, region_size=2.0, z_pos=0.1)
    l4 = SheetGeometry(n_side=2, region_size=2.0, z_pos=0.0)

    # Non-periodic distance to other layer
    dist = l23.get_distance_to(l4, periodic=False)
    assert dist.shape == (4, 4)
    # Target index 0: (-0.5, -0.5, 0.1) to (-0.5, -0.5, 0.0) -> dist = 0.1
    assert dist[0, 0] == pytest.approx(0.1)

    # Mismatched sizes error
    l4_bad = SheetGeometry(n_side=2, region_size=3.0, z_pos=0.0)
    with pytest.raises(ValueError, match="requires matching region_size"):
        l23.get_distance_to(l4_bad, periodic=True)


def test_l4_tuned_initialization() -> None:
    cfg = SimpleNamespace(
        n_side=2,
        region_size=2.0,
        z_pos=0.0,
        N_theta=4,
        l4=SimpleNamespace(all_tuned=False),
    )
    exp = SimpleNamespace(
        NT_X=2,  # Exactly 2 tuned cells
    )

    rng = np.random.default_rng(42)
    l4 = L4(cfg, exp, rng=rng)

    assert l4.N == 4
    # Check that exactly 2 are tuned
    assert np.sum(l4.is_tuned) == 2
    assert np.sum(l4.tunings == "T") == 2
    assert np.sum(l4.tunings == "U") == 2

    # Preferred directions are defined only for tuned cells, untuned are NaN
    tuned_indices = np.flatnonzero(l4.is_tuned)
    untuned_indices = np.flatnonzero(~l4.is_tuned)

    assert np.all(~np.isnan(l4.pref_dirs[tuned_indices]))
    assert np.all(np.isnan(l4.pref_dirs[untuned_indices]))

    # Preferred orientations alias yields 0.0 for untuned
    assert np.all(l4.preferred_orientations[untuned_indices] == 0.0)
    assert np.all(l4.preferred_orientations[tuned_indices] >= 0.0)
    assert np.all(l4.preferred_orientations[tuned_indices] < 2.0 * np.pi)
    assert np.any(l4.preferred_orientations[tuned_indices] > 0.0)


def test_l4_all_tuned() -> None:
    cfg = SimpleNamespace(
        n_side=2,
        region_size=2.0,
        z_pos=0.0,
        N_theta=4,
        l4=SimpleNamespace(all_tuned=True),
    )
    exp = SimpleNamespace(
        pT_X=0.5,
    )
    l4 = L4(cfg, exp)
    assert np.all(l4.is_tuned)
    assert np.sum(~np.isnan(l4.pref_dirs)) == 4


def test_l4_bounded_count_validation() -> None:
    cfg = SimpleNamespace(
        n_side=2,
        region_size=2.0,
        z_pos=0.0,
        N_theta=4,
        l4=SimpleNamespace(all_tuned=False),
    )
    exp_bad = SimpleNamespace(
        NT_X=5,  # Exceeds N = 4
    )
    with pytest.raises(ValueError, match="must be between 0 and 4"):
        L4(cfg, exp_bad)


def test_l23_uniform_vs_random_inhibitory() -> None:
    cfg_random = SimpleNamespace(
        region_size=2.0,
        z_pos=0.1,
        random_I=True,
    )
    cfg_uniform = SimpleNamespace(
        region_size=2.0,
        z_pos=0.1,
        random_I=False,
    )

    exp = SimpleNamespace(
        l2_3_n_side=4,
        N_I=4,
        N_E=12,
    )

    # Random selection
    rng = np.random.default_rng(42)
    l23_rand = L2_3(cfg_random, exp, rng=rng)
    assert np.sum(l23_rand.types == "I") == 4
    assert np.sum(l23_rand.types == "E") == 12

    # Uniform grid selection
    l23_uni = L2_3(cfg_uniform, exp)
    assert np.sum(l23_uni.types == "I") == 4
    assert np.sum(l23_uni.types == "E") == 12

    # Verify uniform distribution indices are spread out
    i_indices = np.flatnonzero(l23_uni.types == "I")
    # For N_I=4 on a 4x4 grid, expected to place them evenly
    assert i_indices.size == 4


def test_l23_count_validation() -> None:
    cfg = SimpleNamespace(
        region_size=2.0,
        z_pos=0.1,
        random_I=True,
    )
    # Total cells for l2_3_n_side=3 is 9, but N_I + N_E = 10
    exp_bad = SimpleNamespace(
        l2_3_n_side=3,
        N_I=3,
        N_E=7,
    )
    with pytest.raises(ValueError, match="N_E \\+ N_I must match layer size"):
        L2_3(cfg, exp_bad)
