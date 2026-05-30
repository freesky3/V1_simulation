from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.special import erf

if TYPE_CHECKING:
    from v1_simulation.config.schema import TransferConfig

DEFAULT_ASYMPTOTIC_THRESHOLD = 5.5
DEFAULT_INTEGRAL_GRID_STEP = 1e-3
DEFAULT_MAX_INTEGRAL_UPPER = 26.0


@dataclass(frozen=True, slots=True)
class TransferGrid:
    """Defines a grid of membrane potential (mu) values for transfer table calculations."""
    mu_min: float
    mu_max: float
    n_points: int

    @classmethod
    def symmetric(cls, mu_max: float, points_per_unit: int = 1000) -> "TransferGrid":
        """Creates a symmetric TransferGrid ranging from -mu_max to mu_max.

        Args:
            mu_max: The absolute maximum value of mu. Must be positive.
            points_per_unit: Number of grid points per unit of mu.

        Returns:
            A TransferGrid instance.

        Raises:
            ValueError: If mu_max is not positive.
        """
        if mu_max <= 0:
            raise ValueError("mu_max must be positive.")
        n_points = max(2, int(points_per_unit * mu_max))
        return cls(mu_min=-float(mu_max), mu_max=float(mu_max), n_points=n_points)

    def values(self) -> NDArray[np.float64]:
        """Generates a linear grid of mu values."""
        return np.linspace(self.mu_min, self.mu_max, self.n_points, dtype=float)


@dataclass(frozen=True, slots=True)
class TransferTable:
    """Look-up table for neuron firing rates as a function of mean input (mu)."""
    mu: NDArray[np.float64]
    rate: NDArray[np.float64]
    rate_max: float | None = None

    def __post_init__(self) -> None:
        if self.mu.ndim != 1 or self.rate.ndim != 1:
            raise ValueError("mu and rate must be one-dimensional arrays.")
        if self.mu.shape != self.rate.shape:
            raise ValueError("mu and rate must have the same shape.")
        if self.mu.size < 2:
            raise ValueError("transfer table needs at least two points.")
        if not np.all(np.diff(self.mu) > 0):
            raise ValueError("mu values must be strictly increasing.")
        if self.rate_max is not None and self.rate_max <= 0:
            raise ValueError("rate_max must be positive when set.")

    def __call__(self, mu: ArrayLike) -> float | NDArray[np.float64]:
        """Interpolates firing rates for the given mu inputs.

        Args:
            mu: Input mean potential values.

        Returns:
            The interpolated firing rate(s), clipped to [0, rate_max] if rate_max is set.
        """
        arr = np.asarray(mu, dtype=float)
        was_scalar = arr.ndim == 0
        out = np.interp(arr, self.mu, self.rate, left=self.rate[0], right=self.rate[-1])
        if self.rate_max is not None:
            out = np.clip(out, 0.0, self.rate_max)
        return float(out) if was_scalar else out

    def as_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns the raw mu and rate arrays."""
        return self.mu, self.rate


@dataclass(frozen=True, slots=True)
class TransferTables:
    """Excitatory and inhibitory transfer tables wrapper."""
    excitatory: TransferTable
    inhibitory: TransferTable


def siegert_kernel(
    x: ArrayLike,
    *,
    asymptotic_threshold: float = DEFAULT_ASYMPTOTIC_THRESHOLD,
) -> float | NDArray[np.float64]:
    """Evaluates the Siegert integral integrand: exp(x^2) * (1 + erf(x)).

    Uses asymptotic expansions for extreme negative values to avoid numerical
    underflow/overflow, and handles NaN inputs gracefully.

    Args:
        x: Input array-like values.
        asymptotic_threshold: Threshold above which asymptotic expansion is used.

    Returns:
        The evaluated integrand values.
    """
    arr = np.asarray(x, dtype=float)
    was_scalar = arr.ndim == 0
    arr = np.atleast_1d(arr)

    # Initialize with NaN to ensure unhandled elements (e.g. NaNs) propagate correctly
    out = np.full_like(arr, np.nan, dtype=float)

    large_pos = arr > asymptotic_threshold
    mid = (~large_pos) & (arr >= -asymptotic_threshold)
    large_neg = arr < -asymptotic_threshold

    if np.any(large_pos):
        val = arr[large_pos]
        out[large_pos] = 2.0 * np.exp(val * val)

    if np.any(mid):
        val = arr[mid]
        out[mid] = np.exp(val * val) * (1.0 + erf(val))

    if np.any(large_neg):
        val = arr[large_neg]
        val2 = val * val
        out[large_neg] = (
            -1.0
            / (np.sqrt(np.pi) * val)
            * (1.0 - 0.5 / val2 + 0.75 / (val2 * val2))
        )

    return float(out[0]) if was_scalar else out


def integrate_siegert_kernel(
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    grid_step: float = DEFAULT_INTEGRAL_GRID_STEP,
    max_upper: float = DEFAULT_MAX_INTEGRAL_UPPER,
) -> NDArray[np.float64]:
    """Integrates the Siegert kernel between lower and upper limits using trapezoidal integration.

    Args:
        lower: Lower integration limits.
        upper: Upper integration limits.
        grid_step: Grid spacing step size for the numerical integration.
        max_upper: Maximum allowed upper limit. Values above this return infinity.

    Returns:
        A numpy array containing the integrated values.

    Raises:
        ValueError: If grid_step is not positive.
    """
    if grid_step <= 0:
        raise ValueError("grid_step must be positive.")

    lower_arr, upper_arr = np.broadcast_arrays(
        np.atleast_1d(np.asarray(lower, dtype=float)),
        np.atleast_1d(np.asarray(upper, dtype=float)),
    )

    out = np.empty_like(upper_arr, dtype=float)
    finite = upper_arr <= max_upper
    out[~finite] = np.inf

    # Ensure NaN values are preserved correctly instead of becoming np.inf
    nans = np.isnan(lower_arr) | np.isnan(upper_arr)
    out[nans] = np.nan

    if not np.any(finite):
        return out

    bounds = np.concatenate([lower_arr[finite], upper_arr[finite]])
    valid_bounds = bounds[~np.isnan(bounds)]
    if valid_bounds.size == 0:
        out[finite] = np.nan
        return out

    grid_min = float(np.min(valid_bounds))
    grid_max = float(np.max(valid_bounds))

    if grid_min == grid_max:
        out[finite] = 0.0
        out[nans] = np.nan
        return out

    n_grid = max(2, int(np.ceil((grid_max - grid_min) / grid_step)) + 1)
    grid = np.linspace(grid_min, grid_max, n_grid, dtype=float)

    antiderivative = cumulative_trapezoid(
        siegert_kernel(grid),
        grid,
        initial=0.0,
    )

    lower_vals = np.interp(lower_arr[finite], grid, antiderivative)
    upper_vals = np.interp(upper_arr[finite], grid, antiderivative)
    out[finite] = upper_vals - lower_vals
    out[nans] = np.nan

    return out


def siegert_rate(
    mu: ArrayLike,
    *,
    tau_m: float,
    cfg_transfer: TransferConfig,
    grid_step: float = DEFAULT_INTEGRAL_GRID_STEP,
) -> float | NDArray[np.float64]:
    """Calculates the steady-state firing rate using the Siegert formula.

    Args:
        mu: Mean synaptic input potential.
        tau_m: Membrane time constant.
        cfg_transfer: Transfer configuration containing v_r, theta, tau_rp, sigma_t.
        grid_step: Step size for numerical integration of the Siegert kernel.

    Returns:
        The calculated firing rate(s).

    Raises:
        ValueError: If tau_m is not positive, or if mu contains values outside [-100, 100].
    """
    if tau_m <= 0:
        raise ValueError("tau_m must be positive.")

    arr = np.asarray(mu, dtype=float)
    was_scalar = arr.ndim == 0
    arr = np.atleast_1d(arr)

    if np.any(np.abs(arr) > 100):
        raise ValueError("mu contains values outside the supported range [-100, 100].")

    lower = (cfg_transfer.v_r - arr) / cfg_transfer.sigma_t
    upper = (cfg_transfer.theta - arr) / cfg_transfer.sigma_t

    integral = integrate_siegert_kernel(lower, upper, grid_step=grid_step)
    rate = 1.0 / (cfg_transfer.tau_rp + tau_m * np.sqrt(np.pi) * integral)

    return float(rate[0]) if was_scalar else rate


def build_transfer_table(
    *,
    tau_m: float,
    cfg_transfer: TransferConfig,
    grid: TransferGrid,
    grid_step: float = DEFAULT_INTEGRAL_GRID_STEP,
    rate_max: float | None = None,
) -> TransferTable:
    """Builds a lookup table of firing rates for a given grid of mu values.

    Args:
        tau_m: Membrane time constant.
        cfg_transfer: Transfer configuration.
        grid: The grid of mu values.
        grid_step: Step size for Siegert kernel integration.
        rate_max: Optional maximum firing rate clamp.

    Returns:
        A TransferTable instance.
    """
    mu = grid.values()
    rate = siegert_rate(mu, tau_m=tau_m, cfg_transfer=cfg_transfer, grid_step=grid_step)
    return TransferTable(mu=mu, rate=np.asarray(rate, dtype=float), rate_max=rate_max)


def build_transfer_tables(
    *,
    cfg_transfer: TransferConfig,
    grid: TransferGrid,
    grid_step: float = DEFAULT_INTEGRAL_GRID_STEP,
) -> TransferTables:
    """Builds lookup tables for both excitatory and inhibitory neuron populations.

    Args:
        cfg_transfer: Transfer configuration.
        grid: The grid of mu values.
        grid_step: Step size for Siegert kernel integration.

    Returns:
        A TransferTables instance containing excitatory and inhibitory lookup tables.
    """
    rate_max = getattr(cfg_transfer, 'rate_max', None)
    return TransferTables(
        excitatory=build_transfer_table(
            tau_m=cfg_transfer.tau_e,
            cfg_transfer=cfg_transfer,
            grid=grid,
            grid_step=grid_step,
            rate_max=rate_max,
        ),
        inhibitory=build_transfer_table(
            tau_m=cfg_transfer.tau_i,
            cfg_transfer=cfg_transfer,
            grid=grid,
            grid_step=grid_step,
            rate_max=rate_max,
        ),
    )