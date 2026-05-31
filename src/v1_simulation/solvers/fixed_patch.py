from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from v1_simulation.solvers.base import ExternalDrive, FloatArray, NetworkLayout
from v1_simulation.solvers.jax_rhs import make_wilson_cowan_jax_rhs
from v1_simulation.solvers.jax_utils import (
    make_diffrax_solver,
    require_diffrax,
    require_jax,
    slice_weight_blocks,
    transfer_table_arrays,
)
from v1_simulation.solvers.wilson_cowan import WilsonCowanRHS

if TYPE_CHECKING:
    from v1_simulation.config.schema import RootConfig
    from v1_simulation.network.state import NetworkState
    from v1_simulation.solvers.base import TransferFunction


_static_diffrax_cache = {}


@dataclass(frozen=True, slots=True)
class FixedPatchTimeGrid:
    t0: float
    t1: float
    dt0: float
    save_ts: FloatArray


@dataclass(frozen=True, slots=True)
class DiffraxStatus:
    result: object
    code: int
    label: str
    successful: bool
    num_steps: int


@dataclass(frozen=True, slots=True)
class FixedPatchTrajectory:
    ys: FloatArray
    status: DiffraxStatus

    @property
    def y_traj(self) -> FloatArray:
        return self.ys[:, :, 0]


@dataclass(frozen=True, slots=True)
class FixedPatchConvergence:
    final_max_abs_drE_dt: float
    final_max_abs_drI_dt: float
    final_max_abs_dy_dt: float
    final_rms_dy_dt: float
    max_abs_delta_window: float
    rhs_threshold: float
    peak_to_peak_threshold: float
    convergence_window_s: float
    rhs_converged: bool
    window_converged: bool
    converged: bool
    dy_dt: FloatArray

    @property
    def max_abs_delta_last_1s(self) -> float:
        return self.max_abs_delta_window


def build_fixed_patch_time_grid(cfg: RootConfig) -> FixedPatchTimeGrid:
    tau_e = float(cfg.solver.transfer.tau_e)
    tau_i = float(cfg.solver.transfer.tau_i)
    t0 = float(cfg.simulation.t_start)
    t1 = (
        float(cfg.simulation.t_stop)
        if cfg.simulation.t_stop is not None
        else t0 + float(cfg.simulation.duration_tau_e) * tau_e
    )
    dt0 = min(tau_e, tau_i) * float(cfg.solver.diffrax.initial_dt_tau_min_fraction)
    n_save = int(cfg.solver.diagnostics.trajectory_sample_points)
    save_ts = np.linspace(t0, t1, n_save, dtype=np.float64)
    return FixedPatchTimeGrid(t0=t0, t1=t1, dt0=dt0, save_ts=save_ts)


def solve_static_fixed_patch_diffrax(
    *,
    cfg: RootConfig,
    network: NetworkState,
    layout: NetworkLayout,
    input_rates: ArrayLike,
    phi_exc: TransferFunction,
    phi_inh: TransferFunction,
    time_grid: FixedPatchTimeGrid,
) -> FixedPatchTrajectory:
    jax, jnp = require_jax("diffrax")
    diffrax = require_diffrax()

    dtype = jnp.float32 if cfg.solver.jax.dtype == "float32" else jnp.float64
    W_exc, W_inh, W_ext = slice_weight_blocks(
        network.weights,
        layout.idx_exc,
        layout.idx_inh,
        layout.idx_ext,
        jnp,
        prefer_sparse=bool(cfg.solver.jax.prefer_sparse),
        dense_max_mb=float(cfg.solver.jax.dense_max_mb),
        dtype=dtype,
    )

    phi_exc_x, phi_exc_y, phi_exc_rate_max = transfer_table_arrays(phi_exc, "phi_exc")
    phi_inh_x, phi_inh_y, phi_inh_rate_max = transfer_table_arrays(phi_inh, "phi_inh")

    y0 = jnp.zeros((layout.n_rates, 1), dtype=dtype)
    ax_0 = jnp.asarray(np.asarray(input_rates, dtype=np.float64)[:, np.newaxis], dtype=dtype)
    mu_ext = W_ext @ ax_0

    solver_name = str(cfg.solver.diffrax.solver).lower()
    cache_key = (solver_name, str(cfg.solver.jax.dtype), int(cfg.solver.diffrax.max_steps))
    if cache_key not in _static_diffrax_cache:
        _static_diffrax_cache[cache_key] = _make_static_fixed_patch_runner(
            jax,
            jnp,
            diffrax,
            solver_name=solver_name,
            max_steps=int(cfg.solver.diffrax.max_steps),
        )

    run = _static_diffrax_cache[cache_key]
    ys, sol_result, num_steps_jax = run(
        y0,
        W_exc,
        W_inh,
        mu_ext,
        jnp.asarray(phi_exc_x, dtype=dtype),
        jnp.asarray(phi_exc_y, dtype=dtype),
        jnp.asarray(phi_exc_rate_max, dtype=dtype),
        jnp.asarray(phi_inh_x, dtype=dtype),
        jnp.asarray(phi_inh_y, dtype=dtype),
        jnp.asarray(phi_inh_rate_max, dtype=dtype),
        jnp.asarray(float(cfg.solver.transfer.tau_e), dtype=dtype),
        jnp.asarray(float(cfg.solver.transfer.tau_i), dtype=dtype),
        jnp.asarray(layout.idx_exc, dtype=jnp.int32),
        jnp.asarray(layout.idx_inh, dtype=jnp.int32),
        jnp.asarray(float(time_grid.t0), dtype=dtype),
        jnp.asarray(float(time_grid.t1), dtype=dtype),
        jnp.asarray(float(time_grid.dt0), dtype=dtype),
        jnp.asarray(time_grid.save_ts, dtype=dtype),
        jnp.asarray(float(cfg.training.bcm.steady_state_rel_tol), dtype=dtype),
        jnp.asarray(float(cfg.training.bcm.steady_state_abs_tol), dtype=dtype),
    )
    jax.block_until_ready(ys)

    status = parse_diffrax_status(
        diffrax,
        sol_result,
        num_steps=int(np.asarray(num_steps_jax)),
    )
    return FixedPatchTrajectory(ys=np.asarray(ys, dtype=np.float64), status=status)


def _make_static_fixed_patch_runner(jax, jnp, diffrax, *, solver_name: str, max_steps: int):
    wc_rhs = make_wilson_cowan_jax_rhs(jnp, is_static=True)

    def vector_field(t, y, args):
        (
            W_exc,
            W_inh,
            mu_ext,
            phi_exc_x,
            phi_exc_y,
            phi_exc_rate_max,
            phi_inh_x,
            phi_inh_y,
            phi_inh_rate_max,
            tau_e,
            tau_i,
            idx_exc,
            idx_inh,
        ) = args
        return wc_rhs(
            y,
            y[:0, :],
            0.0,
            0.0,
            W_exc,
            W_inh,
            y[:0, :],
            mu_ext,
            idx_exc,
            idx_inh,
            phi_exc_x,
            phi_exc_y,
            phi_exc_rate_max,
            phi_inh_x,
            phi_inh_y,
            phi_inh_rate_max,
            tau_e,
            tau_i,
        )

    def run(
        y0,
        W_exc,
        W_inh,
        mu_ext,
        phi_exc_x,
        phi_exc_y,
        phi_exc_rate_max,
        phi_inh_x,
        phi_inh_y,
        phi_inh_rate_max,
        tau_e,
        tau_i,
        idx_exc,
        idx_inh,
        t0,
        t1,
        dt0,
        save_ts,
        rtol,
        atol,
    ):
        term = diffrax.ODETerm(vector_field)
        solver = make_diffrax_solver(diffrax, solver_name)
        args = (
            W_exc,
            W_inh,
            mu_ext,
            phi_exc_x,
            phi_exc_y,
            phi_exc_rate_max,
            phi_inh_x,
            phi_inh_y,
            phi_inh_rate_max,
            tau_e,
            tau_i,
            idx_exc,
            idx_inh,
        )
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=args,
            saveat=diffrax.SaveAt(ts=save_ts),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            throw=False,
        )
        return sol.ys, sol.result, sol.stats["num_steps"]

    return jax.jit(run)


def parse_diffrax_status(diffrax, sol_result, *, num_steps: int) -> DiffraxStatus:
    try:
        successful = bool(np.asarray(sol_result == diffrax.RESULTS.successful))
        code = 0 if successful else -1
    except Exception:
        if hasattr(sol_result, "value"):
            code = int(sol_result.value)
        else:
            try:
                code = int(sol_result)
            except (TypeError, ValueError):
                code = -1
        successful = code == 0

    label = "successful" if successful else str(sol_result)
    return DiffraxStatus(
        result=sol_result,
        code=code,
        label=label,
        successful=successful,
        num_steps=int(num_steps),
    )


def evaluate_fixed_patch_convergence(
    *,
    cfg: RootConfig,
    network: NetworkState,
    layout: NetworkLayout,
    phi_exc: TransferFunction,
    phi_inh: TransferFunction,
    drive_func: ExternalDrive,
    time_grid: FixedPatchTimeGrid,
    y_traj: ArrayLike,
) -> FixedPatchConvergence:
    y_arr = np.asarray(y_traj, dtype=np.float64)
    y_final = y_arr[-1, :]
    tau_e = float(cfg.solver.transfer.tau_e)
    tau_i = float(cfg.solver.transfer.tau_i)

    rhs_evaluator = WilsonCowanRHS(
        weights=network.weights,
        layout=layout,
        phi_exc=phi_exc,
        phi_inh=phi_inh,
        tau_exc=tau_e,
        tau_inh=tau_i,
        n_batch=1,
    )

    dy_dt_flat = rhs_evaluator(time_grid.t1, y_final, drive_func)
    dy_dt = dy_dt_flat.reshape(layout.n_rates, 1)
    dy_dt_exc = dy_dt[layout.idx_exc, 0]
    dy_dt_inh = dy_dt[layout.idx_inh, 0]

    final_max_abs_drE_dt = float(np.max(np.abs(dy_dt_exc)))
    final_max_abs_drI_dt = float(np.max(np.abs(dy_dt_inh)))
    final_max_abs_dy_dt = max(final_max_abs_drE_dt, final_max_abs_drI_dt)
    final_rms_dy_dt = float(np.sqrt(np.mean(dy_dt**2)))

    convergence_window_s = float(cfg.solver.diagnostics.convergence_window_s)
    window_start = max(time_grid.t0, time_grid.t1 - convergence_window_s)
    window_start_idx = int(np.argmin(np.abs(time_grid.save_ts - window_start)))
    y_window = y_arr[window_start_idx:, :]
    peak_to_peak = np.max(y_window, axis=0) - np.min(y_window, axis=0)
    max_abs_delta_window = float(np.max(peak_to_peak))

    rhs_threshold = float(cfg.solver.diagnostics.dy_dt_threshold)
    peak_to_peak_threshold = float(cfg.solver.diagnostics.peak_to_peak_threshold)
    rhs_converged = final_max_abs_dy_dt < rhs_threshold
    window_converged = max_abs_delta_window < peak_to_peak_threshold
    return FixedPatchConvergence(
        final_max_abs_drE_dt=final_max_abs_drE_dt,
        final_max_abs_drI_dt=final_max_abs_drI_dt,
        final_max_abs_dy_dt=final_max_abs_dy_dt,
        final_rms_dy_dt=final_rms_dy_dt,
        max_abs_delta_window=max_abs_delta_window,
        rhs_threshold=rhs_threshold,
        peak_to_peak_threshold=peak_to_peak_threshold,
        convergence_window_s=convergence_window_s,
        rhs_converged=bool(rhs_converged),
        window_converged=bool(window_converged),
        converged=bool(rhs_converged and window_converged),
        dy_dt=dy_dt,
    )


def evaluate_dy_dt_trajectory(
    *,
    network: NetworkState,
    layout: NetworkLayout,
    phi_exc: TransferFunction,
    phi_inh: TransferFunction,
    tau_e: float,
    tau_i: float,
    drive_func: ExternalDrive,
    save_ts: ArrayLike,
    y_traj: ArrayLike,
) -> FloatArray:
    rhs_evaluator = WilsonCowanRHS(
        weights=network.weights,
        layout=layout,
        phi_exc=phi_exc,
        phi_inh=phi_inh,
        tau_exc=float(tau_e),
        tau_inh=float(tau_i),
        n_batch=1,
    )
    values = [
        rhs_evaluator(float(t), np.asarray(y_flat, dtype=np.float64), drive_func)
        for t, y_flat in zip(np.asarray(save_ts, dtype=np.float64), np.asarray(y_traj, dtype=np.float64))
    ]
    return np.stack(values)
