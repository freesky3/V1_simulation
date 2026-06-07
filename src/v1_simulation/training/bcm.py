from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from v1_simulation.config.schema import TrainingBCMConfig


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BCMThetaState:
    """Immutable BCM sliding-threshold state for E and I target populations."""

    E: FloatArray
    I: FloatArray

    def __post_init__(self) -> None:
        theta_E = as_1d_float("theta.E", self.E, copy=True)
        theta_I = as_1d_float("theta.I", self.I, copy=True)
        theta_E.setflags(write=False)
        theta_I.setflags(write=False)
        object.__setattr__(self, "E", theta_E)
        object.__setattr__(self, "I", theta_I)


def validate_bcm_config(config: TrainingBCMConfig, *, include_loop_fields: bool = False) -> None:
    """Validates the BCM parameters from a TrainingBCMConfig schema.

    Args:
        config: The BCM configuration object to validate.
        include_loop_fields: If True, validates training loop parameters too.

    Raises:
        ValueError: If any parameter value is out of range or invalid.
    """

    eta = _finite_float(config.eta, "training.bcm.eta")
    if eta < 0.0:
        raise ValueError("training.bcm.eta must be non-negative.")

    beta = _finite_float(config.theta_beta, "training.bcm.theta_beta")
    if not 0.0 < beta <= 1.0:
        raise ValueError("training.bcm.theta_beta must be in (0.0, 1.0].")

    theta_eps = _finite_float(config.theta_eps, "training.bcm.theta_eps")
    if theta_eps <= 0.0:
        raise ValueError("training.bcm.theta_eps must be positive.")

    if config.theta_update_order not in {"pre", "post"}:
        raise ValueError("training.bcm.theta_update_order must be 'pre' or 'post'.")

    _validate_optional_positive(config.theta_init, "training.bcm.theta_init")
    _validate_optional_positive(config.theta_floor, "training.bcm.theta_floor")
    _validate_optional_positive(config.w_max, "training.bcm.w_max")
    if not isinstance(config.clip_initial_weights, bool):
        raise ValueError("training.bcm.clip_initial_weights must be a boolean.")

    row_sum_max_scale = _optional_finite_float(config.row_sum_max_scale, "training.bcm.row_sum_max_scale")
    if row_sum_max_scale is not None and row_sum_max_scale < 0.0:
        raise ValueError("training.bcm.row_sum_max_scale must be non-negative when provided.")

    if not include_loop_fields:
        return

    if int(config.epochs) <= 0:
        raise ValueError("training.bcm.epochs must be positive.")
    if int(config.batch_size) <= 0:
        raise ValueError("training.bcm.batch_size must be positive.")
    if int(config.save_every) <= 0:
        raise ValueError("training.bcm.save_every must be positive.")

    if _finite_float(config.steady_state_abs_tol, "training.bcm.steady_state_abs_tol") <= 0.0:
        raise ValueError("training.bcm.steady_state_abs_tol must be positive.")
    if _finite_float(config.steady_state_rel_tol, "training.bcm.steady_state_rel_tol") <= 0.0:
        raise ValueError("training.bcm.steady_state_rel_tol must be positive.")
    if int(config.steady_state_window) <= 0:
        raise ValueError("training.bcm.steady_state_window must be positive.")
    if _finite_float(config.steady_state_min_tau, "training.bcm.steady_state_min_tau") < 0.0:
        raise ValueError("training.bcm.steady_state_min_tau must be non-negative.")
    _validate_optional_positive(config.y_diff_max_threshold, "training.bcm.y_diff_max_threshold")
    _validate_optional_positive(config.dy_max_threshold, "training.bcm.dy_max_threshold")
    _validate_optional_positive(config.rate_explosion_threshold, "training.bcm.rate_explosion_threshold")
    sat_fraction = _finite_float(config.saturation_fraction_threshold, "training.bcm.saturation_fraction_threshold")
    if sat_fraction < 0.0 or sat_fraction > 1.0:
        raise ValueError("training.bcm.saturation_fraction_threshold must be in [0.0, 1.0].")
    if _finite_float(config.active_rate_threshold, "training.bcm.active_rate_threshold") < 0.0:
        raise ValueError("training.bcm.active_rate_threshold must be non-negative.")
    if int(config.max_consecutive_bad_batches) <= 0:
        raise ValueError("training.bcm.max_consecutive_bad_batches must be positive.")


def initialize_theta(
    y_E: ArrayLike,
    y_I: ArrayLike,
    config: TrainingBCMConfig,
) -> BCMThetaState:
    """Initializes the BCM thresholds from config or the first batch of neuron responses.

    Args:
        y_E: Initial responses of excitatory target neurons.
        y_I: Initial responses of inhibitory target neurons.
        config: The BCM configuration settings.

    Returns:
        The initialized BCMThetaState.
    """

    validate_bcm_config(config)
    theta_init = config.theta_init
    if theta_init is None:
        theta_E = mean_squared_response(y_E)
        theta_I = mean_squared_response(y_I)
    else:
        theta_E = np.full(response_width(y_E), float(theta_init), dtype=float)
        theta_I = np.full(response_width(y_I), float(theta_init), dtype=float)

    return BCMThetaState(
        E=np.maximum(theta_E, float(config.theta_eps)),
        I=np.maximum(theta_I, float(config.theta_eps)),
    )


def update_theta(
    theta: BCMThetaState,
    y_E: ArrayLike,
    y_I: ArrayLike,
    config: TrainingBCMConfig,
) -> BCMThetaState:
    """Updates the BCM sliding threshold state based on new response batch.

    Args:
        theta: The current BCMThetaState.
        y_E: Batch responses of excitatory target neurons.
        y_I: Batch responses of inhibitory target neurons.
        config: The BCM configuration settings.

    Returns:
        A new BCMThetaState containing the updated thresholds.
    """

    validate_bcm_config(config)
    floor = float(config.theta_eps)
    if config.theta_floor is not None:
        floor = max(float(config.theta_floor), floor)

    return BCMThetaState(
        E=update_theta_vector(theta.E, y_E, beta=float(config.theta_beta), floor=floor),
        I=update_theta_vector(theta.I, y_I, beta=float(config.theta_beta), floor=floor),
    )


def update_theta_vector(
    theta: ArrayLike,
    response: ArrayLike,
    *,
    beta: float,
    floor: float,
) -> FloatArray:
    """Updates a single threshold vector using sliding threshold rule.

    Args:
        theta: The current 1D threshold array.
        response: The neuron responses (1D or 2D batch).
        beta: The sliding threshold update rate (smoothing factor).
        floor: Minimum allowed value for the thresholds.

    Returns:
        The updated threshold vector.

    Raises:
        ValueError: If shapes or parameters are invalid.
    """
    theta_arr = as_1d_float("theta", theta)
    response_ms = mean_squared_response(response)

    if theta_arr.shape != response_ms.shape:
        raise ValueError(
            f"theta shape {theta_arr.shape} does not match response width {response_ms.shape}."
        )

    if not 0.0 < float(beta) <= 1.0:
        raise ValueError("BCM theta beta must be in (0.0, 1.0].")
    if not np.isfinite(float(floor)) or float(floor) <= 0.0:
        raise ValueError("BCM theta floor must be positive and finite.")

    updated = (1.0 - float(beta)) * theta_arr + float(beta) * response_ms
    return np.maximum(updated, float(floor))


def bcm_gain(
    y: ArrayLike,
    theta: ArrayLike,
    *,
    eta: float,
    theta_eps: float,
) -> FloatArray:
    """Calculates the BCM post-synaptic gain: eta * y * (y - theta).

    Args:
        y: Post-synaptic responses (1D or 2D batch).
        theta: Firing rate threshold vector.
        eta: Learning rate.
        theta_eps: Small positive threshold epsilon to ensure stability.

    Returns:
        The calculated BCM gain array.

    Raises:
        ValueError: If shapes or dimensions are invalid.
    """

    eta = _finite_float(eta, "eta")
    theta_eps = _finite_float(theta_eps, "theta_eps")
    if eta < 0.0:
        raise ValueError("BCM eta must be non-negative.")
    if theta_eps <= 0.0:
        raise ValueError("BCM theta_eps must be positive.")

    y_arr = as_float_array("y", y)
    theta_arr = np.maximum(as_1d_float("theta", theta), theta_eps)

    if y_arr.ndim == 1:
        if y_arr.shape != theta_arr.shape:
            raise ValueError(f"y shape {y_arr.shape} does not match theta shape {theta_arr.shape}.")
        return eta * y_arr * (y_arr - theta_arr)

    if y_arr.ndim == 2:
        if y_arr.shape[1] != theta_arr.size:
            raise ValueError(f"y width {y_arr.shape[1]} does not match theta width {theta_arr.size}.")
        return eta * y_arr * (y_arr - theta_arr[np.newaxis, :])

    raise ValueError("BCM response y must be a 1D vector or a 2D batch matrix.")


def bcm_delta(
    x: ArrayLike,
    y: ArrayLike,
    theta: ArrayLike,
    *,
    eta: float,
    theta_eps: float,
) -> FloatArray:
    """Calculates the BCM weight update matrix (outer product of gain and pre-synaptic activity).

    Args:
        x: Pre-synaptic activity (1D or 2D batch).
        y: Post-synaptic activity (1D or 2D batch).
        theta: Firing rate threshold vector.
        eta: Learning rate.
        theta_eps: Threshold epsilon.

    Returns:
        A 2D array of weight updates (target x source).

    Raises:
        ValueError: If shapes, batches, or dimensions are invalid.
    """

    x_arr = as_float_array("x", x)
    gain = bcm_gain(y, theta, eta=eta, theta_eps=theta_eps)

    if x_arr.ndim != gain.ndim:
        raise ValueError("BCM x and y must both be 1D or both be 2D batch arrays.")

    if x_arr.ndim == 1:
        return gain[:, np.newaxis] * x_arr[np.newaxis, :]

    if x_arr.ndim != 2:
        raise ValueError("BCM x must be a 1D vector or a 2D batch matrix.")
    if x_arr.shape[0] != gain.shape[0]:
        raise ValueError("Batch x and y must have the same leading dimension.")
    if x_arr.shape[0] == 0:
        raise ValueError("BCM batch arrays must not be empty.")

    return np.einsum("bt,bs->ts", gain, x_arr) / x_arr.shape[0]


def mean_squared_response(response: ArrayLike) -> FloatArray:
    """Computes the mean squared response (average over batch dimension if 2D).

    Args:
        response: Firing rate responses (1D or 2D batch).

    Returns:
        A 1D array of mean squared responses.
    """
    arr = as_float_array("response", response)
    if arr.ndim == 1:
        return arr**2
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError("BCM response batch must not be empty.")
        return np.mean(arr**2, axis=0)
    raise ValueError("BCM responses must be 1D or 2D arrays.")


def response_width(response: ArrayLike) -> int:
    """Returns the number of neurons/features in the response.

    Args:
        response: Firing rate responses (1D or 2D batch).

    Returns:
        The width (number of columns or elements).
    """
    arr = as_float_array("response", response)
    if arr.ndim == 1:
        return arr.shape[0]
    if arr.ndim == 2:
        return arr.shape[1]
    raise ValueError("BCM responses must be 1D or 2D arrays.")


def as_float_array(name: str, value: ArrayLike, *, copy: bool = False) -> FloatArray:
    arr = np.array(value, dtype=float, copy=True) if copy else np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def as_1d_float(name: str, value: ArrayLike, *, copy: bool = False) -> FloatArray:
    arr = as_float_array(name, value, copy=copy)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return arr


def _validate_optional_positive(value: float | None, path: str) -> None:
    value = _optional_finite_float(value, path)
    if value is not None and value <= 0.0:
        raise ValueError(f"{path} must be positive when provided.")


def _optional_finite_float(value: float | None, path: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, path)


def _finite_float(value: float, path: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{path} must be finite.")
    return value
