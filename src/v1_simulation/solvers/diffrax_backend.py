from __future__ import annotations

import numpy as np

from v1_simulation.solvers.base import (
    BatchODEResult,
    ExternalDrive,
    FloatArray,
    NetworkLayout,
    SolverOptions,
    pack_summary_result,
    pack_trajectory_result,
    validate_external_drive_value,
)
from v1_simulation.solvers.jax_inputs import precompute_diffrax_background, precompute_diffrax_inputs
from v1_simulation.solvers.jax_rhs import make_wilson_cowan_jax_rhs
from v1_simulation.solvers.jax_utils import (
    make_diffrax_solver,
    require_diffrax,
    require_jax,
    slice_weight_blocks,
    transfer_table_arrays,
)


_diffrax_solve_cache = {}


def solve_diffrax(
    *,
    rhs,
    external_drive: ExternalDrive,
    layout: NetworkLayout,
    n_batch: int,
    time: FloatArray,
    options: SolverOptions,
) -> BatchODEResult:
    """Solves the Wilson-Cowan system using a JAX-based Diffrax solver."""
    if options.method != "adaptive":
        raise ValueError("solver.backend 'diffrax' requires solver.method 'adaptive'.")

    jax, jnp = require_jax("diffrax")
    diffrax = require_diffrax()

    is_static = getattr(external_drive, "is_time_dependent", None) is False
    if is_static:
        ax_0 = validate_external_drive_value(external_drive(0.0), n_ext=layout.n_ext, n_batch=n_batch)
        ax_t = np.zeros((2, 1, 1), dtype=np.float64)
    else:
        ax_t = precompute_diffrax_inputs(
            external_drive,
            time,
            n_ext=layout.n_ext,
            n_batch=n_batch,
        )
        if getattr(external_drive, "is_time_dependent", None) is None:
            is_static = bool(np.all(ax_t == ax_t[0]))
        ax_0 = ax_t[0]

    if options.early_stop_enabled and options.early_stop_only_static_input and not is_static:
        raise ValueError(
            "Early stopping by steady-state is only valid for static deterministic input. "
            "Disable it for OU/time-varying background."
        )

    bg_e, bg_i = precompute_diffrax_background(
        rhs.background_trace,
        layout=layout,
        n_batch=n_batch,
        time=time,
    )
    phi_exc_x, phi_exc_y, phi_exc_rate_max = transfer_table_arrays(rhs.phi_exc, "phi_exc")
    phi_inh_x, phi_inh_y, phi_inh_rate_max = transfer_table_arrays(rhs.phi_inh, "phi_inh")

    dtype = jnp.float32 if options.jax_dtype == "float32" else jnp.float64
    W_exc, W_inh, W_ext = slice_weight_blocks(
        rhs.weights,
        layout.idx_exc,
        layout.idx_inh,
        layout.idx_ext,
        jnp,
        prefer_sparse=options.jax_prefer_sparse,
        dense_max_mb=options.jax_dense_max_mb,
        dtype=dtype,
    )

    y0 = jnp.zeros((layout.n_rates, int(n_batch)), dtype=dtype)
    tail_points = options.steady_state_tail_points
    cache_key = (
        options.diffrax_solver,
        options.store_trajectory,
        is_static,
        tail_points,
        options.early_stop_enabled,
        options.early_stop_min_time,
        options.early_stop_f_atol,
        options.early_stop_f_rtol,
        options.early_stop_norm,
        options.diffrax_max_steps,
        options.diffrax_initial_dt_tau_min_fraction,
        options.diagnostics_enabled,
        options.diagnostics_probe_dt,
        options.diagnostics_eval_dy_at,
        options.diagnostics_variables,
    )
    if cache_key not in _diffrax_solve_cache:
        _diffrax_solve_cache[cache_key] = make_diffrax_diffeqsolve(
            jax,
            jnp,
            diffrax,
            solver_name=options.diffrax_solver,
            store_trajectory=options.store_trajectory,
            is_static=is_static,
            tail_points=tail_points,
            early_stop_enabled=options.early_stop_enabled,
            early_stop_min_time=options.early_stop_min_time,
            early_stop_f_atol=options.early_stop_f_atol,
            early_stop_f_rtol=options.early_stop_f_rtol,
            early_stop_norm=options.early_stop_norm,
            max_steps=options.diffrax_max_steps,
            initial_dt_tau_min_fraction=options.diffrax_initial_dt_tau_min_fraction,
            diagnostics_enabled=options.diagnostics_enabled,
            diagnostics_probe_dt=options.diagnostics_probe_dt,
            diagnostics_eval_dy_at=options.diagnostics_eval_dy_at,
            diagnostics_variables=options.diagnostics_variables,
        )
    run = _diffrax_solve_cache[cache_key]

    if is_static:
        mu_ext = W_ext @ jnp.asarray(ax_0, dtype=dtype)
    else:
        mu_ext = jnp.zeros((layout.n_rates, int(n_batch)), dtype=dtype)

    if options.store_trajectory:
        ts_save_val = time
    elif options.diagnostics_enabled:
        t_probe = max(float(time[0]), float(time[-1]) - options.diagnostics_probe_dt)
        idx = np.searchsorted(time, t_probe)
        idx = min(idx, len(time) - 1)
        t_probe_actual = float(time[idx])

        if options.steady_state_tail_points > 1:
            ts_list = [t_probe_actual] + time[-options.steady_state_tail_points :].tolist()
            ts_save_val = np.unique(ts_list)
        else:
            ts_save_val = np.array([t_probe_actual, float(time[-1])])
    elif options.steady_state_tail_points > 1:
        ts_save_val = time[-options.steady_state_tail_points :]
    else:
        ts_save_val = np.array([float(time[-1])])

    out = run(
        y0,
        W_exc,
        W_inh,
        W_ext,
        mu_ext,
        jnp.asarray(layout.idx_exc, dtype=jnp.int32),
        jnp.asarray(layout.idx_inh, dtype=jnp.int32),
        jnp.asarray(time, dtype=dtype),
        jnp.asarray(ts_save_val, dtype=dtype),
        jnp.asarray(ax_t, dtype=dtype),
        jnp.asarray(bg_e, dtype=dtype),
        jnp.asarray(bg_i, dtype=dtype),
        jnp.asarray(phi_exc_x, dtype=dtype),
        jnp.asarray(phi_exc_y, dtype=dtype),
        jnp.asarray(phi_exc_rate_max, dtype=dtype),
        jnp.asarray(phi_inh_x, dtype=dtype),
        jnp.asarray(phi_inh_y, dtype=dtype),
        jnp.asarray(phi_inh_rate_max, dtype=dtype),
        jnp.asarray(float(rhs.tau_exc), dtype=dtype),
        jnp.asarray(float(rhs.tau_inh), dtype=dtype),
    )

    if options.store_trajectory:
        y_all, ss_reached, ss_index, y_diff_max, y_diff_rms, dy_max, dy_rms = out
        jax.block_until_ready(y_all)
        ss_reached_bool = bool(ss_reached)
        ss_index_int = int(ss_index) if ss_reached_bool else None
        ss_start_index = max(0, ss_index_int - options.steady_state_tail_points) if ss_index_int is not None else None
        return pack_trajectory_result(
            np.asarray(y_all, dtype=np.float64),
            layout=layout,
            time=time,
            store_trajectory=True,
            steady_state_reached=ss_reached_bool,
            steady_state_index=ss_index_int,
            steady_state_start_index=ss_start_index,
            y_diff_max=float(y_diff_max),
            y_diff_rms=float(y_diff_rms),
            dy_max=float(dy_max),
            dy_rms=float(dy_rms),
        )

    mean, std, ss_reached, ss_index, y_diff_max, y_diff_rms, dy_max, dy_rms = out
    jax.block_until_ready(mean)
    ss_reached_bool = bool(ss_reached)
    ss_index_int = int(ss_index) if ss_reached_bool else None
    ss_start_index = max(0, ss_index_int - options.steady_state_tail_points) if ss_index_int is not None else None
    return pack_summary_result(
        mean_rates=np.asarray(mean, dtype=np.float64),
        std_rates=np.asarray(std, dtype=np.float64),
        layout=layout,
        time=time,
        steady_state_reached=ss_reached_bool,
        steady_state_index=ss_index_int,
        steady_state_start_index=ss_start_index,
        y_diff_max=float(y_diff_max),
        y_diff_rms=float(y_diff_rms),
        dy_max=float(dy_max),
        dy_rms=float(dy_rms),
    )


def make_diffrax_diffeqsolve(
    jax,
    jnp,
    diffrax,
    *,
    solver_name: str,
    store_trajectory: bool,
    is_static: bool,
    tail_points: int,
    early_stop_enabled: bool = False,
    early_stop_min_time: float = 0.0,
    early_stop_f_atol: float = 1e-4,
    early_stop_f_rtol: float = 1e-4,
    early_stop_norm: str = "max",
    max_steps: int = 4096,
    initial_dt_tau_min_fraction: float = 0.1,
    diagnostics_enabled: bool = True,
    diagnostics_probe_dt: float = 1.0,
    diagnostics_eval_dy_at: str = "mean",
    diagnostics_variables: str = "exc",
):
    """Creates a JIT-compiled Diffrax solve kernel for Wilson-Cowan dynamics."""
    wc_rhs = make_wilson_cowan_jax_rhs(jnp, is_static=is_static)

    def vector_field(t, y, args):
        (
            W_exc,
            W_inh,
            W_ext,
            mu_ext,
            ax_interp,
            bg_e_interp,
            bg_i_interp,
            phi_exc_x,
            phi_exc_y,
            phi_exc_rate_max,
            phi_inh_x,
            phi_inh_y,
            phi_inh_rate_max,
            tau_exc,
            tau_inh,
            idx_exc,
            idx_inh,
        ) = args
        ax = y[:0, :] if is_static else ax_interp.evaluate(t)
        curr_bg_e = bg_e_interp.evaluate(t)
        curr_bg_i = bg_i_interp.evaluate(t)
        return wc_rhs(
            y,
            ax,
            curr_bg_e,
            curr_bg_i,
            W_exc,
            W_inh,
            W_ext,
            mu_ext,
            idx_exc,
            idx_inh,
            phi_exc_x,
            phi_exc_y,
            phi_exc_rate_max,
            phi_inh_x,
            phi_inh_y,
            phi_inh_rate_max,
            tau_exc,
            tau_inh,
        )

    def run(
        y0,
        W_exc,
        W_inh,
        W_ext,
        mu_ext,
        idx_exc,
        idx_inh,
        time,
        ts_save,
        ax_t,
        bg_e,
        bg_i,
        phi_exc_x,
        phi_exc_y,
        phi_exc_rate_max,
        phi_inh_x,
        phi_inh_y,
        phi_inh_rate_max,
        tau_exc,
        tau_inh,
    ):
        bg_e_interp = diffrax.LinearInterpolation(time, bg_e)
        bg_i_interp = diffrax.LinearInterpolation(time, bg_i)
        ax_interp = None if is_static else diffrax.LinearInterpolation(time, ax_t)
        args = (
            W_exc,
            W_inh,
            W_ext,
            mu_ext,
            ax_interp,
            bg_e_interp,
            bg_i_interp,
            phi_exc_x,
            phi_exc_y,
            phi_exc_rate_max,
            phi_inh_x,
            phi_inh_y,
            phi_inh_rate_max,
            tau_exc,
            tau_inh,
            idx_exc,
            idx_inh,
        )

        term = diffrax.ODETerm(vector_field)
        solver = make_diffrax_solver(diffrax, solver_name)
        stepsize_controller = diffrax.ConstantStepSize()
        dt0 = jnp.minimum(time[1] - time[0], initial_dt_tau_min_fraction * jnp.minimum(tau_exc, tau_inh))

        saveat = diffrax.SaveAt(ts=ts_save) if store_trajectory else diffrax.SaveAt(ts=ts_save, t1=True)

        if early_stop_enabled:

            def cond_fn(state_or_t, y=None, args_in=None, **kwargs):
                if y is None:
                    t = state_or_t.t
                    y_val = state_or_t.y
                    args_val = getattr(state_or_t, "args", args)
                else:
                    t = state_or_t
                    y_val = y
                    args_val = args_in

                dy = vector_field(t, y_val, args_val)
                if early_stop_norm == "max":
                    f_norm = jnp.max(jnp.abs(dy))
                    y_norm = jnp.max(jnp.abs(y_val))
                else:
                    f_norm = jnp.sqrt(jnp.mean(jnp.square(dy)))
                    y_norm = jnp.sqrt(jnp.mean(jnp.square(y_val)))

                is_steady = f_norm < early_stop_f_atol + early_stop_f_rtol * y_norm
                return jnp.logical_and(t >= early_stop_min_time, is_steady)

            if hasattr(diffrax, "Event"):
                event = diffrax.Event(cond_fn)
            else:
                event = diffrax.DiscreteTerminatingEvent(cond_fn)
        else:
            event = None

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=time[0],
            t1=time[-1],
            dt0=dt0,
            y0=y0,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            event=event,
            max_steps=max_steps,
            throw=False,
        )

        y_all = sol.ys

        finite_rows = jnp.all(jnp.isfinite(y_all), axis=tuple(range(1, y_all.ndim)))
        finite_count = jnp.sum(finite_rows.astype(jnp.int32))

        if early_stop_enabled:
            ss_reached = jnp.where(sol.result == diffrax.RESULTS.event_occurred, 1, 0)
            # Diffrax stats["num_steps"] is the internal solver step count, not
            # an index into saved values. Use the exclusive end of finite saved
            # rows so downstream summary windows stay on the returned time grid.
            ss_index = finite_count
        else:
            ss_reached = 0
            ss_index = -1

        if store_trajectory:
            return y_all, ss_reached, ss_index, jnp.nan, jnp.nan, jnp.nan, jnp.nan

        finite_rank = jnp.cumsum(finite_rows.astype(jnp.int32))
        has_finite = finite_count > 0
        summary_start_rank = jnp.maximum(jnp.int32(1), finite_count - jnp.asarray(tail_points, dtype=jnp.int32) + 1)
        summary_mask = jnp.logical_and(finite_rows, finite_rank >= summary_start_rank)
        summary_mask_3d = summary_mask[:, jnp.newaxis, jnp.newaxis]
        summary_count = jnp.maximum(jnp.sum(summary_mask.astype(y_all.dtype)), jnp.asarray(1, dtype=y_all.dtype))
        y_sum = jnp.sum(jnp.where(summary_mask_3d, y_all, jnp.zeros_like(y_all)), axis=0)
        y_mean_masked = y_sum / summary_count
        centered = jnp.where(summary_mask_3d, y_all - y_mean_masked, jnp.zeros_like(y_all))
        y_std_masked = jnp.sqrt(jnp.sum(jnp.square(centered), axis=0) / summary_count)

        final_mask = jnp.logical_and(finite_rows, finite_rank == finite_count)
        final_mask_3d = final_mask[:, jnp.newaxis, jnp.newaxis]
        y_final_masked = jnp.sum(jnp.where(final_mask_3d, y_all, jnp.zeros_like(y_all)), axis=0)

        probe_mask = jnp.logical_and(finite_rows, finite_rank == 1)
        probe_mask_3d = probe_mask[:, jnp.newaxis, jnp.newaxis]
        y_probe_masked = jnp.sum(jnp.where(probe_mask_3d, y_all, jnp.zeros_like(y_all)), axis=0)

        y_mean = jnp.where(has_finite, y_mean_masked, y_all[-1])
        y_std = jnp.where(has_finite, y_std_masked, jnp.full_like(y_mean_masked, jnp.nan))
        y_final = jnp.where(has_finite, y_final_masked, y_all[-1])
        y_probe = jnp.where(has_finite, y_probe_masked, y_all[0])

        y_diff_max = jnp.nan
        y_diff_rms = jnp.nan
        dy_max = jnp.nan
        dy_rms = jnp.nan

        if diagnostics_enabled:
            y_diff = y_final - y_probe
            if diagnostics_variables == "exc":
                y_diff = y_diff[idx_exc]

            y_diff_max = jnp.max(jnp.abs(y_diff))
            y_diff_rms = jnp.sqrt(jnp.mean(jnp.square(y_diff)))

            y_eval = y_mean if diagnostics_eval_dy_at == "mean" else y_final
            dy = vector_field(time[-1], y_eval, args)
            if diagnostics_variables == "exc":
                dy = dy[idx_exc]

            dy_max = jnp.max(jnp.abs(dy))
            dy_rms = jnp.sqrt(jnp.mean(jnp.square(dy)))

        return y_mean, y_std, ss_reached, ss_index, y_diff_max, y_diff_rms, dy_max, dy_rms

    return jax.jit(run)
