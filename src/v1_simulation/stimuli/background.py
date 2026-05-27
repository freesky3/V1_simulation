from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from v1_simulation.config.schema import BackgroundConfig

FloatArray = NDArray[np.float64]
InterpolationMode = Literal["linear", "sample_hold"]


@dataclass(frozen=True, slots=True)
class OUParams:
    """Parameters for a stationary Ornstein-Uhlenbeck process.

    Attributes:
        mean: Stationary mean of the process.
        stationary_std: Stationary standard deviation of the process.
        tau: Relaxation time constant of the process (in seconds).
    """

    mean: float = 0.0
    stationary_std: float = 0.0
    tau: float = 0.05

    def __post_init__(self) -> None:
        mean = _finite_float(self.mean, "mean")
        stationary_std = _finite_float(self.stationary_std, "stationary_std")
        tau = _finite_float(self.tau, "tau")

        if stationary_std < 0.0:
            raise ValueError("stationary_std must be non-negative.")
        if tau <= 0.0:
            raise ValueError("tau must be positive.")

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "stationary_std", stationary_std)
        object.__setattr__(self, "tau", tau)


@dataclass(frozen=True, slots=True)
class BackgroundTrace:
    """Time-major background drive consumed by solver steps.

    Attributes:
        time: 1D time grid array of shape (n_time,).
        exc: 3D array of excitatory background inputs of shape (n_time, n_batch, n_exc).
        inh: 3D array of inhibitory background inputs of shape (n_time, n_batch, n_inh).
        interpolation: Interpolation strategy ("linear" or "sample_hold").
    """

    time: FloatArray
    exc: FloatArray
    inh: FloatArray
    interpolation: InterpolationMode = "linear"

    def __post_init__(self) -> None:
        time = validate_time_grid(self.time, copy=True)
        exc = _trace_array(self.exc, "exc")
        inh = _trace_array(self.inh, "inh")
        interpolation = _interpolation_mode(self.interpolation)

        if exc.shape[0] != time.size:
            raise ValueError(
                f"exc time dimension {exc.shape[0]} does not match time size {time.size}."
            )
        if inh.shape[0] != time.size:
            raise ValueError(
                f"inh time dimension {inh.shape[0]} does not match time size {time.size}."
            )
        if exc.shape[1] != inh.shape[1]:
            raise ValueError(
                f"exc batch size {exc.shape[1]} does not match inh batch size {inh.shape[1]}."
            )

        time.setflags(write=False)
        exc.setflags(write=False)
        inh.setflags(write=False)

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "exc", exc)
        object.__setattr__(self, "inh", inh)
        object.__setattr__(self, "interpolation", interpolation)

    @property
    def n_time(self) -> int:
        """Returns the number of time points in the trace."""
        return self.time.size

    @property
    def n_batch(self) -> int:
        """Returns the batch size (number of parallel simulations)."""
        return self.exc.shape[1]

    @property
    def n_exc(self) -> int:
        """Returns the number of excitatory neurons."""
        return self.exc.shape[2]

    @property
    def n_inh(self) -> int:
        """Returns the number of inhibitory neurons."""
        return self.inh.shape[2]

    def validate_shape(self, *, n_exc: int, n_inh: int, n_batch: int) -> None:
        """Validates that trace dimensions match solver expectations.

        Args:
            n_exc: Expected number of excitatory neurons.
            n_inh: Expected number of inhibitory neurons.
            n_batch: Expected batch size.

        Raises:
            ValueError: If shape parameters do not match.
        """
        expected = (
            _positive_int(n_batch, "n_batch"),
            _non_negative_int(n_exc, "n_exc"),
            _non_negative_int(n_inh, "n_inh"),
        )
        actual = (self.n_batch, self.n_exc, self.n_inh)
        if actual != expected:
            raise ValueError(
                "Background trace shape mismatch: "
                f"got batch/exc/inh={actual}, expected {expected}."
            )

    def value_at(self, t: float) -> tuple[FloatArray, FloatArray]:
        """Interpolates background values at a specific time point `t`.

        Args:
            t: Target time point (in seconds).

        Returns:
            A tuple (exc_t, inh_t) of arrays for the batch at time `t`.
        """
        if self.interpolation == "sample_hold":
            index = int(np.searchsorted(self.time, _finite_float(t, "t"), side="right") - 1)
            index = min(max(index, 0), self.n_time - 1)
            return self.exc[index], self.inh[index]

        return (
            _linear_value_at(self.time, self.exc, t),
            _linear_value_at(self.time, self.inh, t),
        )

    def rk4_samples(self) -> RK4BackgroundSamples:
        """Vectorized precomputation of background values at RK4 step stages.

        Precomputes values for left interval endpoints, midpoints, and right
        endpoints corresponding to the trace's time grid segments.

        Returns:
            An RK4BackgroundSamples instance containing the pre-sampled arrays.
        """
        if self.n_time < 2:
            raise ValueError("RK4 background samples require at least two time points.")

        if self.interpolation == "linear":
            exc_mid = 0.5 * (self.exc[:-1] + self.exc[1:])
            inh_mid = 0.5 * (self.inh[:-1] + self.inh[1:])
        elif self.interpolation == "sample_hold":
            exc_mid = self.exc[:-1]
            inh_mid = self.inh[:-1]
        else:
            raise ValueError(f"Unsupported interpolation: {self.interpolation}")

        return RK4BackgroundSamples(
            exc_left=self.exc[:-1],
            inh_left=self.inh[:-1],
            exc_mid=exc_mid,
            inh_mid=inh_mid,
            exc_right=self.exc[1:],
            inh_right=self.inh[1:],
        )


@dataclass(frozen=True, slots=True)
class RK4BackgroundSamples:
    """Pre-sampled background values at RK4 stage points for each solver interval.

    Each array has shape (n_intervals, n_batch, n_units).
    """

    exc_left: FloatArray
    inh_left: FloatArray
    exc_mid: FloatArray
    inh_mid: FloatArray
    exc_right: FloatArray
    inh_right: FloatArray


def generate_background_trace(
    config: BackgroundConfig,
    *,
    n_exc: int,
    n_inh: int,
    n_batch: int,
    time: FloatArray,
    seed: int | np.random.SeedSequence | None = None,
) -> BackgroundTrace | None:
    """Generates a background noise trace from a BackgroundConfig schema.

    Args:
        config: The background configuration dataclass.
        n_exc: Number of excitatory units.
        n_inh: Number of inhibitory units.
        n_batch: Batch size.
        time: 1D time grid array.
        seed: Random number generator seed.

    Returns:
        A generated BackgroundTrace, or None if background is disabled.
    """
    if not bool(config.enabled):
        return None

    return generate_ou_background(
        n_exc=n_exc,
        n_inh=n_inh,
        n_batch=n_batch,
        time=time,
        exc=OUParams(
            mean=config.mu_e,
            stationary_std=config.sigma_e,
            tau=config.tau_e,
        ),
        inh=OUParams(
            mean=config.mu_i,
            stationary_std=config.sigma_i,
            tau=config.tau_i,
        ),
        seed=config.seed if seed is None else seed,
        interpolation=_interpolation_mode(config.interpolation),
    )


def generate_ou_background(
    *,
    n_exc: int,
    n_inh: int,
    n_batch: int,
    time: FloatArray,
    exc: OUParams,
    inh: OUParams,
    seed: int | np.random.SeedSequence | None = None,
    interpolation: InterpolationMode = "linear",
) -> BackgroundTrace:
    """Generates Ornstein-Uhlenbeck background traces for population layers.

    Args:
        n_exc: Number of excitatory units.
        n_inh: Number of inhibitory units.
        n_batch: Batch size.
        time: 1D time grid array.
        exc: Excitatory population OU parameters.
        inh: Inhibitory population OU parameters.
        seed: Random seed or seed sequence.
        interpolation: Interpolation strategy ("linear" or "sample_hold").

    Returns:
        A generated BackgroundTrace.
    """
    time = validate_time_grid(time, copy=True)
    n_exc = _non_negative_int(n_exc, "n_exc")
    n_inh = _non_negative_int(n_inh, "n_inh")
    n_batch = _positive_int(n_batch, "n_batch")

    exc_rng, inh_rng = _split_population_rngs(seed)
    exc_trace = _generate_ou_population(
        n_units=n_exc,
        n_batch=n_batch,
        time=time,
        params=exc,
        rng=exc_rng,
    )
    inh_trace = _generate_ou_population(
        n_units=n_inh,
        n_batch=n_batch,
        time=time,
        params=inh,
        rng=inh_rng,
    )

    return BackgroundTrace(
        time=time,
        exc=exc_trace,
        inh=inh_trace,
        interpolation=interpolation,
    )


def validate_time_grid(value: FloatArray, *, copy: bool = True) -> FloatArray:
    """Validates that a time grid array is strictly increasing and finite.

    Args:
        value: Input time array-like.
        copy: If True, copies the input array.

    Returns:
        A validated 1D float64 NumPy array.
    """
    time = np.array(value, dtype=np.float64, copy=copy)

    if time.ndim != 1 or time.size == 0:
        raise ValueError("time must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing.")

    return time


def _generate_ou_population(
    *,
    n_units: int,
    n_batch: int,
    time: FloatArray,
    params: OUParams,
    rng: np.random.Generator,
) -> FloatArray:
    values = np.empty((time.size, n_batch, n_units), dtype=np.float64)

    if n_units == 0:
        return values

    if params.stationary_std == 0.0:
        values.fill(params.mean)
        return values

    values[0] = params.mean + params.stationary_std * rng.standard_normal((n_batch, n_units))
    for step, dt in enumerate(np.diff(time), start=1):
        alpha = float(np.exp(-dt / params.tau))
        innovation_std = params.stationary_std * np.sqrt(max(0.0, 1.0 - alpha * alpha))
        values[step] = (
            params.mean
            + alpha * (values[step - 1] - params.mean)
            + innovation_std * rng.standard_normal((n_batch, n_units))
        )

    return values


def _linear_value_at(time: FloatArray, values: FloatArray, t: float) -> FloatArray:
    t = _finite_float(t, "t")

    if t <= time[0]:
        return values[0]
    if t >= time[-1]:
        return values[-1]

    right = int(np.searchsorted(time, t, side="right"))
    left = right - 1
    weight = (t - time[left]) / (time[right] - time[left])
    return (1.0 - weight) * values[left] + weight * values[right]


def _split_population_rngs(
    seed: int | np.random.SeedSequence | None,
) -> tuple[np.random.Generator, np.random.Generator]:
    seed_sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    exc_seed, inh_seed = seed_sequence.spawn(2)
    return np.random.default_rng(exc_seed), np.random.default_rng(inh_seed)


def _trace_array(value: FloatArray, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")

    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (n_time, n_batch, n_units).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _interpolation_mode(value: str) -> InterpolationMode:
    if value not in {"linear", "sample_hold"}:
        raise ValueError("interpolation must be 'linear' or 'sample_hold'.")
    return value


def _finite_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _positive_int(value: int, name: str) -> int:
    value = _non_negative_int(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value

