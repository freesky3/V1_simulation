from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from v1_simulation.stimuli.background import BackgroundTrace, RK4BackgroundSamples, validate_time_grid

if TYPE_CHECKING:
    from v1_simulation.config.schema import SolverConfig, TrainingBCMConfig
    from v1_simulation.network.state import NetworkState, PopulationLayout

FloatArray = NDArray[np.float64]
BackendName = Literal["scipy", "jax-rk4", "diffrax"]
ScipyMethod = Literal["RK4", "RK45", "DOP853", "BDF", "Radau", "LSODA"]


class ExternalDrive(Protocol):
    """Continuous-time external drive for L4 source neurons.

    Implementations return either ``(n_x,)`` for a single batch item or
    ``(n_x, n_batch)`` for batched simulation.
    """

    def __call__(self, t: float) -> ArrayLike:
        ...


TransferFunction = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True, slots=True)
class NetworkLayout:
    """Index layout consumed by Wilson-Cowan solvers.

    ``idx_exc`` and ``idx_inh`` index dynamic L2/3 rate rows. ``idx_ext`` indexes
    the external L4 source columns in the connectivity/weight matrix.
    """

    idx_exc: NDArray[np.int64]
    idx_inh: NDArray[np.int64]
    idx_ext: NDArray[np.int64]

    def __post_init__(self) -> None:
        idx_exc = _index_array(self.idx_exc, "idx_exc")
        idx_inh = _index_array(self.idx_inh, "idx_inh")
        idx_ext = _index_array(self.idx_ext, "idx_ext")

        if np.intersect1d(idx_exc, idx_inh).size:
            raise ValueError("idx_exc and idx_inh must be disjoint.")

        n_rates = idx_exc.size + idx_inh.size
        if idx_exc.size and int(idx_exc.max()) >= n_rates:
            raise ValueError("idx_exc contains values outside the dynamic L2/3 rate range.")
        if idx_inh.size and int(idx_inh.max()) >= n_rates:
            raise ValueError("idx_inh contains values outside the dynamic L2/3 rate range.")

        object.__setattr__(self, "idx_exc", idx_exc)
        object.__setattr__(self, "idx_inh", idx_inh)
        object.__setattr__(self, "idx_ext", idx_ext)

    @classmethod
    def from_population_layout(cls, layout: PopulationLayout) -> "NetworkLayout":
        return cls(idx_exc=layout.idx_E, idx_inh=layout.idx_I, idx_ext=layout.idx_X)

    @classmethod
    def from_network_state(cls, network: NetworkState) -> "NetworkLayout":
        return cls.from_population_layout(network.layout)

    @property
    def n_exc(self) -> int:
        return int(self.idx_exc.size)

    @property
    def n_inh(self) -> int:
        return int(self.idx_inh.size)

    @property
    def n_ext(self) -> int:
        return int(self.idx_ext.size)

    @property
    def n_rates(self) -> int:
        return self.n_exc + self.n_inh


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """Runtime options derived from ``solver`` YAML plus call-site choices."""

    backend: BackendName = "scipy"
    method: str = "RK4"
    store_trajectory: bool = True
    early_stop_enabled: bool = False
    early_stop_min_time: float = 0.2
    early_stop_min_steps: int = 20
    early_stop_f_atol: float = 1.0e-4
    early_stop_f_rtol: float = 1.0e-4
    early_stop_norm: str = "max"
    early_stop_rk4_window: int = 5
    early_stop_only_static_input: bool = True
    jax_prefer_sparse: bool = True
    jax_dense_max_mb: float = 128.0
    diffrax_solver: str = "tsit5"
    jax_dtype: str = "float64"
    steady_state_tail_points: int = 1

    @classmethod
    def from_config(
        cls,
        solver: SolverConfig,
        *,
        training_bcm: TrainingBCMConfig | None = None,
        store_trajectory: bool = True,
        stop_at_steady_state: bool | None = None,
    ) -> "SolverOptions":
        """Creates SolverOptions derived from YAML solver config and optional training specs.

        Args:
            solver: The parsed SolverConfig configuration.
            training_bcm: Optional TrainingBCMConfig containing steady state parameters.
            store_trajectory: Whether to store and return the full time trajectory.
            stop_at_steady_state: Optional override for early steady-state stopping.

        Returns:
            A resolved SolverOptions instance.
        """
        # Override from training config if steady_state overrides early_stop config,
        # but prefer early_stop config fields directly from solver configuration if they exist.
        should_stop = solver.early_stop.enabled
        min_time = solver.early_stop.min_time
        f_atol = solver.early_stop.f_atol
        f_rtol = solver.early_stop.f_rtol
        window = solver.early_stop.rk4_window
        min_steps = solver.early_stop.min_steps
        norm = solver.early_stop.norm
        only_static = solver.early_stop.only_static_input

        # Fallback to older BCM params if early_stop is not enabled but BCM dynamic_steady_state is true
        if training_bcm is not None and not should_stop:
            if training_bcm.dynamic_steady_state:
                should_stop = True
                f_atol = float(training_bcm.steady_state_abs_tol)
                f_rtol = float(training_bcm.steady_state_rel_tol)
                window = int(training_bcm.steady_state_window)
                min_time = float(training_bcm.steady_state_min_tau) * float(solver.transfer.tau_e)

        if stop_at_steady_state is not None:
            should_stop = bool(stop_at_steady_state)

        jax_cfg = solver.jax
        diffrax_cfg = solver.diffrax
        return cls(
            backend=solver.backend,  # type: ignore[arg-type]
            method=str(solver.method),
            store_trajectory=bool(store_trajectory),
            early_stop_enabled=should_stop,
            early_stop_f_atol=f_atol,
            early_stop_f_rtol=f_rtol,
            early_stop_rk4_window=window,
            early_stop_min_time=min_time,
            early_stop_min_steps=min_steps,
            early_stop_norm=norm,
            early_stop_only_static_input=only_static,
            jax_prefer_sparse=True if jax_cfg is None else bool(jax_cfg.prefer_sparse),
            jax_dense_max_mb=128.0 if jax_cfg is None else float(jax_cfg.dense_max_mb),
            diffrax_solver="tsit5" if diffrax_cfg is None else str(diffrax_cfg.solver),
            jax_dtype="float64" if jax_cfg is None else getattr(jax_cfg, "dtype", "float64"),
            steady_state_tail_points=1 if diffrax_cfg is None else getattr(diffrax_cfg, "steady_state_tail_points", 1),
        )


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


def validate_solver_options(options: SolverOptions) -> None:
    """Validates that method and backend selections are compatible.

    Args:
        options: The solver options to validate.

    Raises:
        ValueError: If the combination of backend and method is unsupported.
    """
    if options.backend == "scipy":
        if options.method not in {"RK4", "RK45", "DOP853", "BDF", "Radau", "LSODA"}:
            raise ValueError(
                "solver.method must be one of RK4, RK45, DOP853, BDF, Radau, or LSODA "
                "when solver.backend is 'scipy'."
            )
        return

    if options.backend == "jax-rk4":
        if options.method != "RK4":
            raise ValueError("solver.backend 'jax-rk4' requires solver.method 'RK4'.")
        return

    if options.backend == "diffrax":
        if options.method != "adaptive":
            raise ValueError("solver.backend 'diffrax' requires solver.method 'adaptive'.")
        return

    raise ValueError(f"Unknown solver backend: {options.backend!r}")


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


def summary_start_index(n_time: int, steady_state_start_index: int | None = None) -> int:
    """Returns the first time index used for output summary statistics.

    Args:
        n_time: Total number of time steps.
        steady_state_start_index: Optional index where steady state started. If None,
            defaults to the last 1/3 of the trajectory.

    Returns:
        The 0-indexed start time index for summarizing.
    """

    if n_time <= 0:
        raise ValueError("n_time must be positive.")
    if steady_state_start_index is not None:
        return max(0, min(int(steady_state_start_index), n_time - 1))
    return int(n_time * 2 / 3)


def pack_trajectory_result(
    trajectory: FloatArray,
    *,
    layout: NetworkLayout,
    time: FloatArray,
    store_trajectory: bool,
    steady_state_reached: bool = False,
    steady_state_index: int | None = None,
    steady_state_start_index: int | None = None,
) -> BatchODEResult:
    """Packs a full dynamic trajectory into the public result shape.

    Args:
        trajectory: Array with shape ``(n_time, n_rates, n_batch)``.
        layout: The network indexing layout.
        time: Array of time points.
        store_trajectory: Whether to store and return the full trajectories.
        steady_state_reached: Whether early steady-state stopping was triggered.
        steady_state_index: Time index where steady state was reached.
        steady_state_start_index: Time index where steady state detection window started.

    Returns:
        A filled BatchODEResult.
    """

    y_t = np.asarray(trajectory, dtype=np.float64)
    time = validate_time_grid(time, copy=True)
    if y_t.ndim != 3:
        raise ValueError("trajectory must have shape (n_time, n_rates, n_batch).")
    if y_t.shape[0] != time.size:
        raise ValueError("trajectory time dimension must match time.")
    if y_t.shape[1] != layout.n_rates:
        raise ValueError("trajectory rate dimension must match layout.n_rates.")

    exc_t = np.transpose(y_t[:, layout.idx_exc, :], (0, 2, 1))
    inh_t = np.transpose(y_t[:, layout.idx_inh, :], (0, 2, 1))
    start = summary_start_index(time.size, steady_state_start_index)

    exc_tail = exc_t[start:]
    inh_tail = inh_t[start:]
    exc = np.mean(exc_tail, axis=0)
    inh = np.mean(inh_tail, axis=0)
    exc_convergence = np.mean(np.std(exc_tail, axis=0), axis=1)
    inh_convergence = np.mean(np.std(inh_tail, axis=0), axis=1)

    return BatchODEResult(
        exc=exc,
        inh=inh,
        exc_trajectory=exc_t if store_trajectory else None,
        inh_trajectory=inh_t if store_trajectory else None,
        time=time,
        exc_convergence=exc_convergence,
        inh_convergence=inh_convergence,
        steady_state_reached=steady_state_reached,
        steady_state_index=steady_state_index,
        steady_state_start_index=steady_state_start_index,
    )


def pack_summary_result(
    *,
    mean_rates: FloatArray,
    std_rates: FloatArray,
    layout: NetworkLayout,
    time: FloatArray,
    steady_state_reached: bool = False,
    steady_state_index: int | None = None,
    steady_state_start_index: int | None = None,
) -> BatchODEResult:
    """Packs streaming summary statistics into the public result shape.

    Args:
        mean_rates: Mean firing rates with shape (n_rates, n_batch).
        std_rates: Standard deviation of rates with shape (n_rates, n_batch).
        layout: The network indexing layout.
        time: Solver time grid array.
        steady_state_reached: Whether early steady-state stopping was triggered.
        steady_state_index: Time index where steady state was reached.
        steady_state_start_index: Time index where steady state detection window started.

    Returns:
        A filled BatchODEResult with empty trajectory fields.
    """

    mean = np.asarray(mean_rates, dtype=np.float64)
    std = np.asarray(std_rates, dtype=np.float64)
    if mean.shape != std.shape:
        raise ValueError("mean_rates and std_rates must have the same shape.")
    if mean.ndim != 2 or mean.shape[0] != layout.n_rates:
        raise ValueError("summary arrays must have shape (n_rates, n_batch).")

    return BatchODEResult(
        exc=np.transpose(mean[layout.idx_exc, :]),
        inh=np.transpose(mean[layout.idx_inh, :]),
        exc_trajectory=None,
        inh_trajectory=None,
        time=validate_time_grid(time, copy=True),
        exc_convergence=np.mean(std[layout.idx_exc, :], axis=0),
        inh_convergence=np.mean(std[layout.idx_inh, :], axis=0),
        steady_state_reached=steady_state_reached,
        steady_state_index=steady_state_index,
        steady_state_start_index=steady_state_start_index,
    )


def validate_external_drive_value(
    value: ArrayLike,
    *,
    n_ext: int,
    n_batch: int,
) -> FloatArray:
    """Validates the shape and values of the external drive function output.

    Args:
        value: The value returned by the external drive function.
        n_ext: Expected number of external inputs (L4 neurons).
        n_batch: Expected number of batch items (stimulus orientations).

    Returns:
        A float array of shape (n_ext, n_batch).
    """
    drive = np.asarray(value, dtype=np.float64)
    if drive.ndim == 1:
        if n_batch != 1:
            raise ValueError(
                f"external drive returned shape {drive.shape}; expected ({n_ext}, {n_batch})."
            )
        drive = drive[:, np.newaxis]
    if drive.shape != (n_ext, n_batch):
        raise ValueError(f"external drive shape {drive.shape} != ({n_ext}, {n_batch}).")
    return drive


def _index_array(values: ArrayLike, name: str) -> NDArray[np.int64]:
    arr = np.asarray(values, dtype=np.int64).reshape(-1).copy()
    if arr.size and np.any(arr < 0):
        raise ValueError(f"{name} must contain non-negative indices.")
    if np.unique(arr).size != arr.size:
        raise ValueError(f"{name} must not contain duplicate indices.")
    arr.setflags(write=False)
    return arr
