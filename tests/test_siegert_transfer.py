import numpy as np
import pytest
from types import SimpleNamespace

from v1_simulation.transfer.siegert import (
    TransferGrid,
    TransferTable,
    build_transfer_table,
    build_transfer_tables,
    integrate_siegert_kernel,
    siegert_kernel,
    siegert_rate,
)


def test_transfer_grid() -> None:
    grid = TransferGrid.symmetric(2.0, points_per_unit=100)
    assert grid.mu_min == -2.0
    assert grid.mu_max == 2.0
    assert grid.n_points == 200
    assert np.allclose(grid.values(), np.linspace(-2.0, 2.0, 200))

    with pytest.raises(ValueError, match="mu_max must be positive"):
        TransferGrid.symmetric(-1.0)


def test_transfer_table_validation() -> None:
    mu = np.array([0.0, 1.0, 2.0])
    rate = np.array([0.0, 10.0, 20.0])

    # Valid
    table = TransferTable(mu, rate)
    assert np.array_equal(table.as_arrays()[0], mu)

    # 1D check
    with pytest.raises(ValueError, match="must be one-dimensional"):
        TransferTable(mu.reshape(-1, 1), rate)

    # Shape mismatch
    with pytest.raises(ValueError, match="must have the same shape"):
        TransferTable(mu, rate[:2])

    # Minimum size
    with pytest.raises(ValueError, match="needs at least two points"):
        TransferTable(mu[:1], rate[:1])

    # Not strictly increasing
    with pytest.raises(ValueError, match="must be strictly increasing"):
        TransferTable(np.array([1.0, 0.0]), np.array([10.0, 20.0]))


def test_transfer_table_interpolation() -> None:
    mu = np.array([-1.0, 0.0, 1.0])
    rate = np.array([10.0, 20.0, 30.0])
    table = TransferTable(mu, rate)

    # In-bounds interpolation
    assert table(0.5) == pytest.approx(25.0)
    # Vectorized interpolation
    assert np.allclose(table([-0.5, 0.5]), np.array([15.0, 25.0]))

    # Out-of-bounds clipping (left/right defaults)
    assert table(-2.0) == pytest.approx(10.0)
    assert table(2.0) == pytest.approx(30.0)


def test_siegert_kernel() -> None:
    # Scalar evaluation
    assert np.isfinite(siegert_kernel(0.0))

    # Test middle range (normal path)
    x_mid = np.array([-1.0, 0.0, 1.0])
    y_mid = np.exp(x_mid * x_mid) * (1.0 + scipy_erf(x_mid))
    assert np.allclose(siegert_kernel(x_mid), y_mid)

    # Test large positive (exp overflow avoidance)
    large_pos = siegert_kernel(6.0)
    assert large_pos == pytest.approx(2.0 * np.exp(36.0))

    # Test large negative (underflow avoidance)
    large_neg = siegert_kernel(-6.0)
    val = -6.0
    expected_neg = -1.0 / (np.sqrt(np.pi) * val) * (1.0 - 0.5 / 36.0 + 0.75 / (36.0 * 36.0))
    assert large_neg == pytest.approx(expected_neg)


def test_integrate_siegert_kernel() -> None:
    # Valid integration
    lower = np.array([-1.0, 0.0])
    upper = np.array([1.0, 2.0])
    integral = integrate_siegert_kernel(lower, upper, grid_step=1e-3)
    assert integral.shape == (2,)
    assert np.all(integral >= 0.0)

    # Max upper threshold -> np.inf
    inf_integral = integrate_siegert_kernel([0.0], [27.0], max_upper=26.0)
    assert np.isinf(inf_integral[0])

    # NaN preservation
    nan_integral = integrate_siegert_kernel([np.nan], [1.0])
    assert np.isnan(nan_integral[0])

    # Mismatch/ValueError check
    with pytest.raises(ValueError, match="grid_step must be positive"):
        integrate_siegert_kernel(0.0, 1.0, grid_step=0.0)


def test_siegert_rate() -> None:
    cfg = SimpleNamespace(
        v_r=-5.0,
        theta=0.0,
        tau_rp=0.002,
        sigma_t=1.0,
        tau_e=0.02,
        tau_i=0.01,
    )

    # Scalar computation
    rate = siegert_rate(0.0, tau_m=0.02, cfg_transfer=cfg)
    assert rate > 0.0

    # Vectorized computation
    rates = siegert_rate(np.array([-1.0, 1.0]), tau_m=0.02, cfg_transfer=cfg)
    assert rates.shape == (2,)
    assert rates[1] > rates[0]  # larger mean input -> higher rate

    # Parameter boundaries
    with pytest.raises(ValueError, match="tau_m must be positive"):
        siegert_rate(0.0, tau_m=-0.01, cfg_transfer=cfg)

    # Extreme input range limits
    with pytest.raises(ValueError, match="supported range"):
        siegert_rate(101.0, tau_m=0.02, cfg_transfer=cfg)


def test_build_transfer_tables() -> None:
    cfg = SimpleNamespace(
        v_r=-5.0,
        theta=0.0,
        tau_rp=0.002,
        sigma_t=1.0,
        tau_e=0.02,
        tau_i=0.01,
    )
    grid = TransferGrid(-2.0, 2.0, 20)

    # Build single table
    table = build_transfer_table(tau_m=0.02, cfg_transfer=cfg, grid=grid)
    assert isinstance(table, TransferTable)
    assert table.rate.shape == (20,)

    # Build dual tables
    tables = build_transfer_tables(cfg_transfer=cfg, grid=grid)
    assert isinstance(tables.excitatory, TransferTable)
    assert isinstance(tables.inhibitory, TransferTable)


# Helper matching scipy's erf to compare in test
def scipy_erf(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf
    return erf(x)
