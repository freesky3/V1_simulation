import numpy as np
import pytest
from types import SimpleNamespace

from v1_simulation.stimuli.receptive_fields import (
    GaborConfig,
    GaborRFConfig,
    L4GaborBank,
    VisualGrid,
    gabor_bank,
    gabor_kernel,
)


def test_gabor_config_validation() -> None:
    # Valid
    cfg = GaborConfig(sigma=0.5, gamma=1.0, spatial_frequency=2.0, phase=0.0)
    assert cfg.sigma == 0.5

    # Invalid sigma
    with pytest.raises(ValueError, match="sigma must be positive"):
        GaborConfig(sigma=0.0, gamma=1.0, spatial_frequency=2.0, phase=0.0)

    # Invalid gamma
    with pytest.raises(ValueError, match="gamma must be positive"):
        GaborConfig(sigma=0.5, gamma=-0.1, spatial_frequency=2.0, phase=0.0)


def test_gabor_rf_config_validation() -> None:
    gcfg = GaborConfig(sigma=0.5, gamma=1.0, spatial_frequency=2.0, phase=0.0)
    # Valid
    rf_cfg = GaborRFConfig(stimulus_size=2.0, resolution=10, gabor=gcfg)
    assert rf_cfg.stimulus_size == 2.0

    # Invalid size
    with pytest.raises(ValueError, match="stimulus_size must be positive"):
        GaborRFConfig(stimulus_size=-1.0, resolution=10, gabor=gcfg)

    # Invalid resolution
    with pytest.raises(ValueError, match="resolution must be greater than 1"):
        GaborRFConfig(stimulus_size=2.0, resolution=1, gabor=gcfg)


def test_visual_grid() -> None:
    grid = VisualGrid.centered_midpoint(size=2.0, resolution=2)
    # resolution=2, size=2.0 -> dx = 1.0. Midpoints: [-0.5, 0.5]
    assert np.array_equal(grid.x_axis, np.array([-0.5, 0.5]))
    assert np.array_equal(grid.y_axis, np.array([-0.5, 0.5]))
    assert grid.dx == 1.0
    assert grid.dy == 1.0
    assert grid.area_element == 1.0

    # Aliases
    assert np.array_equal(grid.X, grid.x)
    assert np.array_equal(grid.Y, grid.y)

    with pytest.raises(ValueError, match="size must be positive"):
        VisualGrid.centered_midpoint(size=0.0, resolution=2)
    with pytest.raises(ValueError, match="resolution must be greater than 1"):
        VisualGrid.centered_midpoint(size=2.0, resolution=1)


def test_gabor_kernel_tuned_vs_untuned() -> None:
    grid = VisualGrid.centered_midpoint(size=2.0, resolution=20)
    cfg = GaborConfig(sigma=0.5, gamma=0.5, spatial_frequency=2.0, phase=0.0)

    # Tuned cell with orientation = 0
    k_tuned = gabor_kernel(grid, cfg, theta_pref=0.0, is_tuned=True)
    # Untuned cell
    k_untuned = gabor_kernel(grid, cfg, theta_pref=np.pi / 4, is_tuned=False)

    assert k_tuned.shape == (20, 20)
    assert k_untuned.shape == (20, 20)

    # Untuned Gabor ignores orientation theta_pref (forced to 0.0) and sets gamma to 1.0 (isotropic)
    # So it should be circular symmetric. Let's check rotation symmetry.
    assert np.allclose(k_untuned, k_untuned.T)  # symmetric across diagonal


def test_gabor_bank() -> None:
    grid = VisualGrid.centered_midpoint(size=2.0, resolution=5)
    cfg = GaborConfig(sigma=0.5, gamma=1.0, spatial_frequency=2.0, phase=0.0)

    thetas = np.array([0.0, np.pi / 2])
    is_tuned = np.array([True, False])

    bank = gabor_bank(grid, cfg, theta_pref=thetas, is_tuned=is_tuned)
    assert bank.shape == (2, 5, 5)


def test_l4_gabor_bank_lazy_loading() -> None:
    # Mock L4 layer
    l4_layer = SimpleNamespace(
        coords=np.array([[0.0, 0.0], [1.0, 1.0]]),
        tunings=["T", "U"],
        pref_dirs=[np.pi / 4, np.nan],
        N=2,
    )
    gcfg = GaborConfig(sigma=0.5, gamma=1.0, spatial_frequency=2.0, phase=0.0)
    rf_cfg = GaborRFConfig(stimulus_size=2.0, resolution=5, gabor=gcfg)

    bank = L4GaborBank(rf_cfg, l4_layer)
    assert bank._filters is None  # initially not loaded

    # Access property triggers evaluation
    filters = bank.filters
    assert filters.shape == (2, 5, 5)
    assert bank._filters is not None
    # Read-only flag check
    assert not filters.flags.writeable

    # Length mismatch validation
    l4_bad = SimpleNamespace(
        coords=np.array([[0.0, 0.0]]),  # len 1
        tunings=["T", "U"],            # len 2
        pref_dirs=[np.pi / 4, np.nan],  # len 2
        N=2,
    )
    with pytest.raises(ValueError, match="must have the same length"):
        L4GaborBank(rf_cfg, l4_bad)
