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
from v1_simulation.solvers.jax_inputs import precompute_rk4_background, precompute_rk4_inputs
from v1_simulation.solvers.jax_rhs import make_wilson_cowan_jax_rhs
from v1_simulation.solvers.jax_utils import require_jax, slice_weight_blocks, transfer_table_arrays


_jax_rk4_solve_cache = {}


def solve_jax_rk4(
    *,
    rhs,
    external_drive: ExternalDrive,
    layout: NetworkLayout,
    n_batch: int,
    time: FloatArray,
    options: SolverOptions,
) -> BatchODEResult:
    """Solves the Wilson-Cowan system using a JIT-compiled JAX RK4 solver."""
    if options.method != "RK4":
        raise ValueError("solver.backend 'jax-rk4' requires solver.method 'RK4'.")

    jax, jnp = require_jax("jax-rk4")
    is_static = getattr(external_drive, "is_time_dependent", None) is False
    if is_static:
        ax_0 = validate_external_drive_value(external_drive(0.0), n_ext=layout.n_ext, n_batch=n_batch)
        dummy_shape = (time.size - 1, 1, 1)
        ax_left = np.zeros(dummy_shape, dtype=np.float64)
        ax_mid = np.zeros(dummy_shape, dtype=np.float64)
        ax_right = np.zeros(dummy_shape, dtype=np.float64)
    else:
        ax_left, ax_mid, ax_right = precompute_rk4_inputs(
            external_drive,
            time,
            n_ext=layout.n_ext,
            n_batch=n_batch,
        )
        if getattr(external_drive, "is_time_dependent", None) is None:
            is_static = bool(
                np.all(ax_left == ax_left[0]) and np.all(ax_mid == ax_left[0]) and np.all(ax_right == ax_left[0])
            )
        ax_0 = ax_left[0]

    if options.early_stop_enabled and options.early_stop_only_static_input and not is_static:
        raise ValueError(
            "Early stopping by steady-state is only valid for static deterministic input. "
            "Disable it for OU/time-varying background."
        )

    bg_left_e, bg_mid_e, bg_right_e, bg_left_i, bg_mid_i, bg_right_i = precompute_rk4_background(
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
    cache_key = (
        options.store_trajectory,
        is_static,
        options.early_stop_enabled,
        options.early_stop_min_time,
        options.early_stop_f_atol,
        options.early_stop_f_rtol,
        options.early_stop_norm,
        options.early_stop_rk4_window,
        options.early_stop_min_steps,
    )
    if cache_key not in _jax_rk4_solve_cache:
        _jax_rk4_solve_cache[cache_key] = make_jax_rk4(
            jax,
            jnp,
            store_trajectory=options.store_trajectory,
            is_static=is_static,
            early_stop_enabled=options.early_stop_enabled,
            early_stop_min_time=options.early_stop_min_time,
            early_stop_f_atol=options.early_stop_f_atol,
            early_stop_f_rtol=options.early_stop_f_rtol,
            early_stop_norm=options.early_stop_norm,
            early_stop_rk4_window=options.early_stop_rk4_window,
            early_stop_min_steps=options.early_stop_min_steps,
        )
    run = _jax_rk4_solve_cache[cache_key]

    if is_static:
        mu_ext = W_ext @ jnp.asarray(ax_0, dtype=dtype)
    else:
        mu_ext = jnp.zeros((layout.n_rates, int(n_batch)), dtype=dtype)

    out = run(
        y0,
        W_exc,
        W_inh,
        W_ext,
        mu_ext,
        jnp.asarray(layout.idx_exc, dtype=jnp.int32),
        jnp.asarray(layout.idx_inh, dtype=jnp.int32),
        jnp.asarray(time, dtype=dtype),
        jnp.asarray(ax_left, dtype=dtype),
        jnp.asarray(ax_mid, dtype=dtype),
        jnp.asarray(ax_right, dtype=dtype),
        jnp.asarray(bg_left_e, dtype=dtype),
        jnp.asarray(bg_mid_e, dtype=dtype),
        jnp.asarray(bg_right_e, dtype=dtype),
        jnp.asarray(bg_left_i, dtype=dtype),
        jnp.asarray(bg_mid_i, dtype=dtype),
        jnp.asarray(bg_right_i, dtype=dtype),
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
        y_all, ss_reached, ss_index = out
        jax.block_until_ready(y_all)
        ss_reached_bool = bool(ss_reached)
        ss_index_int = int(ss_index) if ss_reached_bool else None
        ss_start_index = max(0, ss_index_int - options.early_stop_rk4_window) if ss_index_int is not None else None
        return pack_trajectory_result(
            np.asarray(y_all, dtype=np.float64),
            layout=layout,
            time=time,
            store_trajectory=True,
            steady_state_reached=ss_reached_bool,
            steady_state_index=ss_index_int,
            steady_state_start_index=ss_start_index,
        )

    mean, std, ss_reached, ss_index = out
    jax.block_until_ready(mean)
    ss_reached_bool = bool(ss_reached)
    ss_index_int = int(ss_index) if ss_reached_bool else None
    ss_start_index = max(0, ss_index_int - options.early_stop_rk4_window) if ss_index_int is not None else None
    return pack_summary_result(
        mean_rates=np.asarray(mean, dtype=np.float64),
        std_rates=np.asarray(std, dtype=np.float64),
        layout=layout,
        time=time,
        steady_state_reached=ss_reached_bool,
        steady_state_index=ss_index_int,
        steady_state_start_index=ss_start_index,
    )


def make_jax_rk4(
    jax,
    jnp,
    *,
    store_trajectory: bool,
    is_static: bool,
    early_stop_enabled: bool = False,
    early_stop_min_time: float = 0.0,
    early_stop_f_atol: float = 1e-4,
    early_stop_f_rtol: float = 1e-4,
    early_stop_norm: str = "l2",
    early_stop_rk4_window: int = 5,
    early_stop_min_steps: int = 20,
):
    """Creates a JIT-compiled JAX RK4 kernel for the Wilson-Cowan system."""
    wc_rhs = make_wilson_cowan_jax_rhs(jnp, is_static=is_static)

    def run(
        y0,
        W_exc,
        W_inh,
        W_ext,
        mu_ext,
        idx_exc,
        idx_inh,
        time,
        ax_left,
        ax_mid,
        ax_right,
        bg_left_e,
        bg_mid_e,
        bg_right_e,
        bg_left_i,
        bg_mid_i,
        bg_right_i,
        phi_exc_x,
        phi_exc_y,
        phi_exc_rate_max,
        phi_inh_x,
        phi_inh_y,
        phi_inh_rate_max,
        tau_exc,
        tau_inh,
    ):
        params = (
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
        dts = time[1:] - time[:-1]

        def scan_step(y, xs):
            dt, ax_l, ax_m, ax_r, bg_l_e, bg_m_e, bg_r_e, bg_l_i, bg_m_i, bg_r_i = xs
            k1 = wc_rhs(y, ax_l, bg_l_e, bg_l_i, *params)
            k2 = wc_rhs(y + 0.5 * dt * k1, ax_m, bg_m_e, bg_m_i, *params)
            k3 = wc_rhs(y + 0.5 * dt * k2, ax_m, bg_m_e, bg_m_i, *params)
            k4 = wc_rhs(y + dt * k3, ax_r, bg_r_e, bg_r_i, *params)
            y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return y_next, y_next

        if not early_stop_enabled:
            _, ys = jax.lax.scan(
                scan_step,
                y0,
                (
                    dts,
                    ax_left,
                    ax_mid,
                    ax_right,
                    bg_left_e,
                    bg_mid_e,
                    bg_right_e,
                    bg_left_i,
                    bg_mid_i,
                    bg_right_i,
                ),
            )
            y_all = jnp.concatenate([y0[jnp.newaxis, :, :], ys], axis=0)
            ss_reached = 0
            ss_index = -1
        else:
            max_steps = time.shape[0] - 1
            ys_init = jnp.zeros((max_steps + 1, y0.shape[0], y0.shape[1]), dtype=y0.dtype)
            ys_init = ys_init.at[0].set(y0)
            init_val = (jnp.int32(0), y0, ys_init, jnp.int32(0))

            def cond_fun(val):
                step_idx, _, _, stable_steps = val
                not_done = step_idx < max_steps
                not_stable = stable_steps < early_stop_rk4_window
                return jnp.logical_and(not_done, not_stable)

            def body_fun(val):
                step_idx, y, ys, stable_steps = val
                xs = (
                    dts[step_idx],
                    ax_left[step_idx],
                    ax_mid[step_idx],
                    ax_right[step_idx],
                    bg_left_e[step_idx],
                    bg_mid_e[step_idx],
                    bg_right_e[step_idx],
                    bg_left_i[step_idx],
                    bg_mid_i[step_idx],
                    bg_right_i[step_idx],
                )
                _, y_next = scan_step(y, xs)
                ys_next = ys.at[step_idx + 1].set(y_next)

                dt = dts[step_idx]
                f_est = (y_next - y) / dt

                if early_stop_norm == "max":
                    f_norm = jnp.max(jnp.abs(f_est))
                    y_norm = jnp.max(jnp.abs(y_next))
                else:
                    f_norm = jnp.sqrt(jnp.mean(jnp.square(f_est)))
                    y_norm = jnp.sqrt(jnp.mean(jnp.square(y_next)))

                is_stable = f_norm < early_stop_f_atol + early_stop_f_rtol * y_norm
                t = time[step_idx + 1]
                can_stop = jnp.logical_and(t >= early_stop_min_time, step_idx + 1 >= early_stop_min_steps)
                is_stable = jnp.logical_and(is_stable, can_stop)

                stable_steps = jnp.where(is_stable, stable_steps + 1, 0)
                return (step_idx + 1, y_next, ys_next, stable_steps)

            final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)
            step_idx, _, y_all, stable_steps = final_val

            ss_reached = jnp.where(stable_steps >= early_stop_rk4_window, 1, 0)
            ss_index = step_idx

        if store_trajectory:
            return y_all, ss_reached, ss_index

        if early_stop_enabled:
            valid_len = ss_index + 1
            start_idx = jnp.maximum(0, valid_len - valid_len // 3)
            mask = jnp.arange(y_all.shape[0]) >= start_idx
            mask = jnp.logical_and(mask, jnp.arange(y_all.shape[0]) < valid_len)
            mask = mask[:, jnp.newaxis, jnp.newaxis]
            sum_mask = jnp.maximum(1, jnp.sum(mask, axis=0))
            tail_mean = jnp.sum(y_all * mask, axis=0) / sum_mask
            tail_std = jnp.sqrt(jnp.sum(jnp.square(y_all - tail_mean) * mask, axis=0) / sum_mask)
            return tail_mean, tail_std, ss_reached, ss_index

        tail = y_all[int(time.shape[0] * 2 / 3) :, :, :]
        return jnp.mean(tail, axis=0), jnp.std(tail, axis=0), ss_reached, ss_index

    return jax.jit(run)
