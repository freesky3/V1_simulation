from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from v1_simulation.stimuli.background import BackgroundTrace, RK4BackgroundSamples, validate_time_grid

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BatchODEResult:
    """The result of a batch ODE solver run.

    Attributes:
        exc: Firing rates for excitatory neurons. Shape: (n_batch, n_exc).
        inh: Firing rates for inhibitory neurons. Shape: (n_batch, n_inh).
        exc_trajectory: Firing rate trajectory for excitatory neurons.
            Shape: (n_time, n_batch, n_exc) or None.
        inh_trajectory: Firing rate trajectory for inhibitory neurons.
            Shape: (n_time, n_batch, n_inh) or None.
        time: The solver time points. Shape: (n_time,).
        exc_convergence: Firing rate changes for E units at convergence/exit.
        inh_convergence: Firing rate changes for I units at convergence/exit.
        steady_state_reached: Whether steady state was successfully reached.
        steady_state_index: Time index where steady state was reached, or None.
        steady_state_start_index: Time index where steady state detection window started, or None.
    """

    exc: FloatArray
    inh: FloatArray
    exc_trajectory: FloatArray | None
    inh_trajectory: FloatArray | None
    time: FloatArray
    exc_convergence: FloatArray
    inh_convergence: FloatArray
    steady_state_reached: bool = False
    steady_state_index: int | None = None
    steady_state_start_index: int | None = None


def validate_background_trace(
    trace: BackgroundTrace | None,
    *,
    n_exc: int,
    n_inh: int,
    n_batch: int,
    time: FloatArray,
) -> None:
    """Validates that a background trace matches solver dimensions and time grid.

    Args:
        trace: The background trace to validate, or None.
        n_exc: Expected number of excitatory neurons.
        n_inh: Expected number of inhibitory neurons.
        n_batch: Expected batch size (number of stimuli).
        time: The solver time grid array.

    Raises:
        ValueError: If trace shape or time grid does not match the solver parameters.
    """
    if trace is None:
        return

    trace.validate_shape(n_exc=n_exc, n_inh=n_inh, n_batch=n_batch)
    solver_time = validate_time_grid(time, copy=True)
    if trace.time.shape != solver_time.shape or not np.allclose(
        trace.time,
        solver_time,
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise ValueError("Background trace time grid does not match solver time grid.")


def prepare_rk4_background(
    trace: BackgroundTrace | None,
    *,
    n_exc: int,
    n_inh: int,
    n_batch: int,
    time: FloatArray,
) -> RK4BackgroundSamples | None:
    """Validates a background trace and prepares its sampled values for RK4 solver steps.

    Args:
        trace: The background trace, or None.
        n_exc: Expected number of excitatory neurons.
        n_inh: Expected number of inhibitory neurons.
        n_batch: Expected batch size.
        time: Solver time grid array.

    Returns:
        Sampled RK4 background values, or None if the input trace is None.
    """
    validate_background_trace(
        trace,
        n_exc=n_exc,
        n_inh=n_inh,
        n_batch=n_batch,
        time=time,
    )
    if trace is None:
        return None
    return trace.rk4_samples()

