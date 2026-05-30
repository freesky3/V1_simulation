from __future__ import annotations

from collections import deque

import numpy as np
from scipy.integrate import RK45, solve_ivp

from v1_simulation.solvers.base import (
    BatchODEResult,
    ExternalDrive,
    FloatArray,
    NetworkLayout,
    SolverOptions,
    pack_summary_result,
    pack_trajectory_result,
    prepare_rk4_background,
)


def solve_scipy(
    rhs,
    external_drive: ExternalDrive,
    layout: NetworkLayout,
    n_batch: int,
    time: FloatArray,
    options: SolverOptions,
) -> BatchODEResult:
    """Dispatches the ODE system to the appropriate SciPy solver backend method.

    Args:
        rhs: The callable right-hand side evaluator of the network dynamics.
        external_drive: Continuous-time L4 stimulus drive.
        layout: The network indexing layout.
        n_batch: The batch size (number of orientations/stimuli).
        time: The target time grid points.
        options: SolverOptions containing backend/method/tolerance settings.

    Returns:
        The BatchODEResult containing trajectories or summarized statistics.
    """
    _require_two_time_points(time)

    if options.method == "RK4":
        return solve_fixed_rk4(rhs, external_drive, layout, n_batch, time, options)

    if options.early_stop_enabled:
        if options.method != "RK45":
            raise ValueError("Early stopping is only supported for SciPy RK4 and RK45.")
        return solve_rk45_with_early_stop(rhs, external_drive, layout, n_batch, time, options)

    y0 = np.zeros((layout.n_rates, n_batch), dtype=np.float64).ravel()
    sol = solve_ivp(
        lambda t, y: rhs(t, y, external_drive, background=_background_at(rhs.background_trace, t)),
        (float(time[0]), float(time[-1])),
        y0,
        method=options.method,
        t_eval=time,
    )
    if not sol.success:
        raise RuntimeError(f"SciPy solver failed: {sol.message}")

    trajectory = np.asarray(sol.y.T, dtype=np.float64).reshape(sol.t.size, layout.n_rates, n_batch)
    return pack_trajectory_result(
        trajectory,
        layout=layout,
        time=np.asarray(sol.t, dtype=np.float64),
        store_trajectory=options.store_trajectory,
    )


def solve_fixed_rk4(
    rhs,
    external_drive: ExternalDrive,
    layout: NetworkLayout,
    n_batch: int,
    time: FloatArray,
    options: SolverOptions,
) -> BatchODEResult:
    """Solves the system using a custom fixed-step fourth-order Runge-Kutta (RK4) integration loop.

    Args:
        rhs: The callable right-hand side evaluator.
        external_drive: Continuous-time L4 stimulus drive.
        layout: The network indexing layout.
        n_batch: The batch size.
        time: The target time grid points.
        options: SolverOptions settings.

    Returns:
        The BatchODEResult containing trajectories or summarized statistics.
    """
    y = np.zeros((layout.n_rates, n_batch), dtype=np.float64).ravel()
    background = prepare_rk4_background(
        rhs.background_trace,
        n_exc=layout.n_exc,
        n_inh=layout.n_inh,
        n_batch=n_batch,
        time=time,
    )

    trajectory = None
    if options.store_trajectory:
        trajectory = np.empty((time.size, layout.n_rates, n_batch), dtype=np.float64)
        trajectory[0] = y.reshape(layout.n_rates, n_batch)

    summary = _TrajectorySummary(layout.n_rates, n_batch)
    checker = _SteadyStateChecker.from_options(options, rhs) if options.early_stop_enabled else None
    tail = deque([y.copy()], maxlen=(checker.window + 1 if checker is not None else 1))

    steady_reached = False
    steady_index = None
    steady_start = None

    y_probe = None
    t_probe = None
    if options.diagnostics_enabled:
        t_probe = max(float(time[0]), float(time[-1]) - options.diagnostics_probe_dt)

    summary_start = int(time.size * 2 / 3)
    if not options.store_trajectory and summary_start == 0:
        summary.add(y)

    final_index = time.size - 1
    for step in range(1, time.size):
        t0 = float(time[step - 1])
        dt = float(time[step] - time[step - 1])
        y_prev = y.copy()

        k1 = rhs(t0, y, external_drive, background=_rk4_background_stage(background, "left", step - 1))
        k2 = rhs(
            t0 + 0.5 * dt,
            y + 0.5 * dt * k1,
            external_drive,
            background=_rk4_background_stage(background, "mid", step - 1),
        )
        k3 = rhs(
            t0 + 0.5 * dt,
            y + 0.5 * dt * k2,
            external_drive,
            background=_rk4_background_stage(background, "mid", step - 1),
        )
        k4 = rhs(
            t0 + dt,
            y + dt * k3,
            external_drive,
            background=_rk4_background_stage(background, "right", step - 1),
        )
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if trajectory is not None:
            trajectory[step] = y.reshape(layout.n_rates, n_batch)
        else:
            if t_probe is not None and y_probe is None and t0 >= t_probe:
                y_probe = y.copy()
            if checker is not None:
                tail.append(y.copy())
            if step >= summary_start:
                summary.add(y)

        if checker is not None and checker.reached(float(time[step]), y_prev, y):
            steady_reached = True
            steady_index = step
            steady_start = max(0, step - checker.window)
            final_index = step
            if trajectory is None:
                summary = _TrajectorySummary(layout.n_rates, n_batch)
                for item in tail:
                    summary.add(item)
            break

    out_time = time[: final_index + 1]
    if trajectory is not None:
        return pack_trajectory_result(
            trajectory[: final_index + 1],
            layout=layout,
            time=out_time,
            store_trajectory=True,
            steady_state_reached=steady_reached,
            steady_state_index=steady_index,
            steady_state_start_index=steady_start,
        )

    y_diff_max = float("nan")
    y_diff_rms = float("nan")
    dy_max = float("nan")
    dy_rms = float("nan")

    if options.diagnostics_enabled:
        y_final = y if summary.count == 0 else summary.sum_y / summary.count
        if options.diagnostics_eval_dy_at == "final":
            y_eval = y
        else:
            y_eval = y_final

        dy = rhs(float(out_time[-1]), y_eval.ravel(), external_drive, background=None).reshape(layout.n_rates, n_batch)
        if options.diagnostics_variables == "exc":
            dy = dy[layout.idx_exc, :]
            
        dy_max = float(np.max(np.abs(dy)))
        dy_rms = float(np.sqrt(np.mean(dy**2)))

        y_p = y_probe if y_probe is not None else y0
        if trajectory is not None:
            idx = max(0, np.searchsorted(out_time, out_time[-1] - options.diagnostics_probe_dt))
            y_p = trajectory[idx].ravel()

        y_diff = y - y_p
        y_diff = y_diff.reshape(layout.n_rates, n_batch)
        if options.diagnostics_variables == "exc":
            y_diff = y_diff[layout.idx_exc, :]
            
        y_diff_max = float(np.max(np.abs(y_diff)))
        y_diff_rms = float(np.sqrt(np.mean(y_diff**2)))

    if summary.count == 0:
        summary.add(y)
    return summary.pack(
        layout=layout,
        time=out_time,
        steady_state_reached=steady_reached,
        steady_state_index=steady_index,
        steady_state_start_index=steady_start,
        y_diff_max=y_diff_max,
        y_diff_rms=y_diff_rms,
        dy_max=dy_max,
        dy_rms=dy_rms,
    )


def solve_rk45_with_early_stop(
    rhs,
    external_drive: ExternalDrive,
    layout: NetworkLayout,
    n_batch: int,
    time: FloatArray,
    options: SolverOptions,
) -> BatchODEResult:
    """Solves the system using adaptive-step RK45 integration with early stopping on steady state.

    Args:
        rhs: The callable right-hand side evaluator.
        external_drive: Continuous-time L4 stimulus drive.
        layout: The network indexing layout.
        n_batch: The batch size.
        time: The target time grid points.
        options: SolverOptions settings.

    Returns:
        The BatchODEResult containing trajectories or summarized statistics.
    """
    y0 = np.zeros((layout.n_rates, n_batch), dtype=np.float64).ravel()
    solver = RK45(
        lambda t, y: rhs(t, y, external_drive, background=_background_at(rhs.background_trace, t)),
        float(time[0]),
        y0,
        float(time[-1]),
    )
    checker = _SteadyStateChecker.from_options(options, rhs)

    ys = [y0.copy()]
    ts = [float(solver.t)]
    steady_reached = False

    while solver.status == "running":
        y_prev = solver.y.copy()
        solver.step()
        if solver.status == "failed":
            raise RuntimeError("RK45 failed while solving Wilson-Cowan dynamics.")

        ys.append(solver.y.copy())
        ts.append(float(solver.t))
        if checker.reached(float(solver.t), y_prev, solver.y):
            steady_reached = True
            break

    trajectory = np.asarray(ys, dtype=np.float64).reshape(len(ys), layout.n_rates, n_batch)
    out_time = np.asarray(ts, dtype=np.float64)
    steady_index = len(out_time) - 1 if steady_reached else None
    steady_start = max(0, int(steady_index) - checker.window) if steady_reached else None

    y_diff_max = float("nan")
    y_diff_rms = float("nan")
    dy_max = float("nan")
    dy_rms = float("nan")

    if options.diagnostics_enabled:
        y_final = np.mean(trajectory[max(0, steady_start):], axis=0) if steady_start is not None else trajectory[-1]
        if options.diagnostics_eval_dy_at == "final":
            y_eval = trajectory[-1]
        else:
            y_eval = y_final

        dy = rhs(float(out_time[-1]), y_eval.ravel(), external_drive, background=None).reshape(layout.n_rates, n_batch)
        if options.diagnostics_variables == "exc":
            dy = dy[layout.idx_exc, :]
            
        dy_max = float(np.max(np.abs(dy)))
        dy_rms = float(np.sqrt(np.mean(dy**2)))

        idx = max(0, np.searchsorted(out_time, out_time[-1] - options.diagnostics_probe_dt))
        y_p = trajectory[idx]

        y_diff = trajectory[-1] - y_p
        if options.diagnostics_variables == "exc":
            y_diff = y_diff[layout.idx_exc, :]
            
        y_diff_max = float(np.max(np.abs(y_diff)))
        y_diff_rms = float(np.sqrt(np.mean(y_diff**2)))

    return pack_trajectory_result(
        trajectory,
        layout=layout,
        time=out_time,
        store_trajectory=options.store_trajectory,
        steady_state_reached=steady_reached,
        steady_state_index=steady_index,
        steady_state_start_index=steady_start,
        y_diff_max=y_diff_max,
        y_diff_rms=y_diff_rms,
        dy_max=dy_max,
        dy_rms=dy_rms,
    )


class _TrajectorySummary:
    def __init__(self, n_rates: int, n_batch: int) -> None:
        self.n_rates = int(n_rates)
        self.n_batch = int(n_batch)
        self.sum_y = np.zeros((self.n_rates, self.n_batch), dtype=np.float64)
        self.sumsq_y = np.zeros_like(self.sum_y)
        self.count = 0

    def add(self, y_flat: FloatArray) -> None:
        y = np.asarray(y_flat, dtype=np.float64).reshape(self.n_rates, self.n_batch)
        self.sum_y += y
        self.sumsq_y += y * y
        self.count += 1

    def pack(
        self,
        *,
        layout: NetworkLayout,
        time: FloatArray,
        steady_state_reached: bool,
        steady_state_index: int | None,
        steady_state_start_index: int | None,
    ) -> BatchODEResult:
        if self.count <= 0:
            raise ValueError("Cannot pack an empty trajectory summary.")
        mean = self.sum_y / self.count
        variance = np.maximum(self.sumsq_y / self.count - mean * mean, 0.0)
        return pack_summary_result(
            mean_rates=mean,
            std_rates=np.sqrt(variance),
            layout=layout,
            time=time,
            steady_state_reached=steady_state_reached,
            steady_state_index=steady_state_index,
            steady_state_start_index=steady_state_start_index,
        )


class _SteadyStateChecker:
    @classmethod
    def from_options(cls, options: SolverOptions, rhs) -> "_SteadyStateChecker":
        min_time = options.steady_state_min_time
        if min_time is None:
            min_time = 5.0 * max(float(rhs.tau_exc), float(rhs.tau_inh))
        return cls(
            abs_tol=options.steady_state_abs_tol,
            rel_tol=options.steady_state_rel_tol,
            window=options.steady_state_window,
            min_time=min_time,
        )

    def __init__(self, *, abs_tol: float, rel_tol: float, window: int, min_time: float) -> None:
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.window = max(1, int(window))
        self.min_time = float(min_time)
        self._stable_steps = 0

    def reached(self, t: float, y_prev: FloatArray, y: FloatArray) -> bool:
        if float(t) < self.min_time:
            self._stable_steps = 0
            return False

        scale = max(1.0, float(np.max(np.abs(y))))
        delta = float(np.max(np.abs(y - y_prev)))
        if delta <= self.abs_tol + self.rel_tol * scale:
            self._stable_steps += 1
        else:
            self._stable_steps = 0
        return self._stable_steps >= self.window


def _background_at(trace, t: float):
    if trace is None:
        return None
    return trace.value_at(float(t))


def _rk4_background_stage(samples, stage: str, index: int):
    if samples is None:
        return None
    if stage == "left":
        return samples.exc_left[index], samples.inh_left[index]
    if stage == "mid":
        return samples.exc_mid[index], samples.inh_mid[index]
    if stage == "right":
        return samples.exc_right[index], samples.inh_right[index]
    raise ValueError(f"Unsupported RK4 background stage: {stage!r}")


def _require_two_time_points(time: FloatArray) -> None:
    if np.asarray(time).size < 2:
        raise ValueError("solver time grid must contain at least two points.")
