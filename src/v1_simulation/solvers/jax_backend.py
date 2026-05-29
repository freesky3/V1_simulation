from __future__ import annotations

import importlib.util
import warnings

import numpy as np
from scipy import sparse as scipy_sparse

from v1_simulation.solvers.base import (
    BatchODEResult,
    ExternalDrive,
    FloatArray,
    NetworkLayout,
    SolverOptions,
    pack_summary_result,
    pack_trajectory_result,
    prepare_rk4_background,
    validate_external_drive_value,
)


def is_jax_available() -> bool:
    return importlib.util.find_spec("jax") is not None


def is_diffrax_available() -> bool:
    return importlib.util.find_spec("diffrax") is not None


_jax_rk4_solve_cache = {}
_diffrax_solve_cache = {}


def _slice_weight_blocks(
    weights,
    idx_exc: np.ndarray,
    idx_inh: np.ndarray,
    idx_ext: np.ndarray,
    jnp,
    *,
    prefer_sparse: bool,
    dense_max_mb: float,
    dtype=None,
):
    """Pre-slice weight matrix into excitatory, inhibitory and external blocks.

    Slicing is performed in SciPy (Python layer) before any JAX JIT boundary.
    Each block is then converted to a JAX dense or sparse array.

    Args:
        weights: The full weight matrix (scipy sparse or dense numpy).
        idx_exc: Column indices for excitatory source neurons.
        idx_inh: Column indices for inhibitory source neurons.
        idx_ext: Column indices for external (L4) source neurons.
        jnp: The jax.numpy module.
        prefer_sparse: Whether to use JAX BCOO sparse format for large blocks.
        dense_max_mb: Maximum size (MB) allowed for a dense fallback.
        dtype: Optional target JAX data type.

    Returns:
        Tuple (W_exc, W_inh, W_ext) as JAX arrays.
    """
    w = scipy_sparse.csc_matrix(weights) if scipy_sparse.issparse(weights) else weights

    if scipy_sparse.issparse(w):
        w_exc = scipy_sparse.csr_matrix(w[:, idx_exc])
        w_inh = scipy_sparse.csr_matrix(w[:, idx_inh])
        w_ext = scipy_sparse.csr_matrix(w[:, idx_ext])
    else:
        w_exc = w[:, idx_exc]
        w_inh = w[:, idx_inh]
        w_ext = w[:, idx_ext]

    return (
        _prepare_jax_matrix(w_exc, jnp, prefer_sparse=prefer_sparse, dense_max_mb=dense_max_mb, dtype=dtype),
        _prepare_jax_matrix(w_inh, jnp, prefer_sparse=prefer_sparse, dense_max_mb=dense_max_mb, dtype=dtype),
        _prepare_jax_matrix(w_ext, jnp, prefer_sparse=prefer_sparse, dense_max_mb=dense_max_mb, dtype=dtype),
    )


def solve_jax_rk4(
    *,
    rhs,
    external_drive: ExternalDrive,
    layout: NetworkLayout,
    n_batch: int,
    time: FloatArray,
    options: SolverOptions,
) -> BatchODEResult:
    """Solves the Wilson-Cowan system using a JIT-compiled JAX RK4 solver.

    Weight blocks W_exc, W_inh, W_ext are pre-sliced in Python (outside JIT)
    to avoid dynamic indexing inside the compiled function.

    Args:
        rhs: The callable right-hand side evaluator of the network dynamics.
        external_drive: Continuous-time L4 stimulus drive.
        layout: The network indexing layout.
        n_batch: The batch size (number of stimuli).
        time: The target time grid points.
        options: SolverOptions settings.

    Returns:
        The BatchODEResult containing trajectories or summarized statistics.
    """
    if options.method != "RK4":
        raise ValueError("solver.backend 'jax-rk4' requires solver.method 'RK4'.")
    if options.stop_at_steady_state:
        warnings.warn(
            "jax-rk4 currently evaluates the full time grid; use scipy RK4/RK45 for early stopping.",
            RuntimeWarning,
            stacklevel=2,
        )

    jax, jnp = _require_jax("jax-rk4")
    is_static = getattr(external_drive, "is_time_dependent", None) is False
    if is_static:
        ax_0 = validate_external_drive_value(external_drive(0.0), n_ext=layout.n_ext, n_batch=n_batch)
        dummy_shape = (time.size - 1, 1, 1)
        ax_left = np.zeros(dummy_shape, dtype=np.float64)
        ax_mid = np.zeros(dummy_shape, dtype=np.float64)
        ax_right = np.zeros(dummy_shape, dtype=np.float64)
    else:
        ax_left, ax_mid, ax_right = _precompute_rk4_inputs(
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

    bg_left_e, bg_mid_e, bg_right_e, bg_left_i, bg_mid_i, bg_right_i = _precompute_rk4_background(
        rhs.background_trace,
        layout=layout,
        n_batch=n_batch,
        time=time,
    )
    phi_exc_x, phi_exc_y = _transfer_table_arrays(rhs.phi_exc, "phi_exc")
    phi_inh_x, phi_inh_y = _transfer_table_arrays(rhs.phi_inh, "phi_inh")

    dtype = jnp.float32 if options.jax_dtype == "float32" else jnp.float64

    # Pre-slice weight blocks in Python (scipy layer) before any JIT boundary.
    W_exc, W_inh, W_ext = _slice_weight_blocks(
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
    cache_key = (options.store_trajectory, is_static)
    if cache_key not in _jax_rk4_solve_cache:
        _jax_rk4_solve_cache[cache_key] = _make_jax_rk4(
            jax, jnp, store_trajectory=options.store_trajectory, is_static=is_static
        )
    run = _jax_rk4_solve_cache[cache_key]

    # Precompute static external contribution here (Python layer) when is_static=True.
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
        jnp.asarray(phi_inh_x, dtype=dtype),
        jnp.asarray(phi_inh_y, dtype=dtype),
        jnp.asarray(float(rhs.tau_exc), dtype=dtype),
        jnp.asarray(float(rhs.tau_inh), dtype=dtype),
    )

    if options.store_trajectory:
        jax.block_until_ready(out)
        return pack_trajectory_result(
            np.asarray(out, dtype=np.float64),
            layout=layout,
            time=time,
            store_trajectory=True,
        )

    mean, std = out
    jax.block_until_ready(mean)
    return pack_summary_result(
        mean_rates=np.asarray(mean, dtype=np.float64),
        std_rates=np.asarray(std, dtype=np.float64),
        layout=layout,
        time=time,
    )


def solve_diffrax(
    *,
    rhs,
    external_drive: ExternalDrive,
    layout: NetworkLayout,
    n_batch: int,
    time: FloatArray,
    options: SolverOptions,
) -> BatchODEResult:
    """Solves the system using JAX-based Diffrax solver.

    Weight blocks W_exc, W_inh, W_ext are pre-sliced in Python (outside JIT)
    to avoid dynamic indexing inside the compiled function.

    Args:
        rhs: The callable right-hand side evaluator of the network dynamics.
        external_drive: Continuous-time L4 stimulus drive.
        layout: The network indexing layout.
        n_batch: The batch size.
        time: The target time grid points.
        options: SolverOptions settings.

    Returns:
        The BatchODEResult.
    """
    if options.method != "adaptive":
        raise ValueError("solver.backend 'diffrax' requires solver.method 'adaptive'.")
    if options.stop_at_steady_state:
        warnings.warn(
            "diffrax backend currently evaluates the full time grid; use scipy RK4/RK45 for early stopping.",
            RuntimeWarning,
            stacklevel=2,
        )

    jax, jnp = _require_jax("diffrax")
    diffrax = _require_diffrax()

    is_static = getattr(external_drive, "is_time_dependent", None) is False
    if is_static:
        ax_0 = validate_external_drive_value(external_drive(0.0), n_ext=layout.n_ext, n_batch=n_batch)
        ax_t = np.zeros((2, 1, 1), dtype=np.float64)
    else:
        ax_t = _precompute_diffrax_inputs(
            external_drive,
            time,
            n_ext=layout.n_ext,
            n_batch=n_batch,
        )
        if getattr(external_drive, "is_time_dependent", None) is None:
            is_static = bool(np.all(ax_t == ax_t[0]))
        ax_0 = ax_t[0]

    bg_e, bg_i = _precompute_diffrax_background(
        rhs.background_trace,
        layout=layout,
        n_batch=n_batch,
        time=time,
    )

    phi_exc_x, phi_exc_y = _transfer_table_arrays(rhs.phi_exc, "phi_exc")
    phi_inh_x, phi_inh_y = _transfer_table_arrays(rhs.phi_inh, "phi_inh")

    dtype = jnp.float32 if options.jax_dtype == "float32" else jnp.float64

    # Pre-slice weight blocks in Python (scipy layer) before any JIT boundary.
    W_exc, W_inh, W_ext = _slice_weight_blocks(
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
    cache_key = (options.diffrax_solver, options.store_trajectory, is_static, tail_points)
    if cache_key not in _diffrax_solve_cache:
        _diffrax_solve_cache[cache_key] = _make_diffrax_diffeqsolve(
            jax,
            jnp,
            diffrax,
            solver_name=options.diffrax_solver,
            store_trajectory=options.store_trajectory,
            is_static=is_static,
            tail_points=tail_points,
        )
    run = _diffrax_solve_cache[cache_key]

    # Precompute static external contribution here (Python layer) when is_static=True.
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
        jnp.asarray(ax_t, dtype=dtype),
        jnp.asarray(bg_e, dtype=dtype),
        jnp.asarray(bg_i, dtype=dtype),
        jnp.asarray(phi_exc_x, dtype=dtype),
        jnp.asarray(phi_exc_y, dtype=dtype),
        jnp.asarray(phi_inh_x, dtype=dtype),
        jnp.asarray(phi_inh_y, dtype=dtype),
        jnp.asarray(float(rhs.tau_exc), dtype=dtype),
        jnp.asarray(float(rhs.tau_inh), dtype=dtype),
    )

    if options.store_trajectory:
        jax.block_until_ready(out)
        return pack_trajectory_result(
            np.asarray(out, dtype=np.float64),
            layout=layout,
            time=time,
            store_trajectory=True,
        )

    mean, std = out
    jax.block_until_ready(mean)
    return pack_summary_result(
        mean_rates=np.asarray(mean, dtype=np.float64),
        std_rates=np.asarray(std, dtype=np.float64),
        layout=layout,
        time=time,
    )


def _make_diffrax_diffeqsolve(
    jax,
    jnp,
    diffrax,
    *,
    solver_name: str,
    store_trajectory: bool,
    is_static: bool,
    tail_points: int,
):
    """Creates a JIT-compiled JAX function to solve the Wilson-Cowan ODEs using Diffrax.

    Receives pre-sliced weight sub-matrices W_exc, W_inh, W_ext and (when is_static=True)
    the precomputed static external drive contribution mu_ext. No weight indexing occurs
    inside this JIT-compiled function.

    Args:
        jax: The jax module.
        jnp: The jax.numpy module.
        diffrax: The diffrax module.
        solver_name: The name of the Diffrax solver to use (e.g., 'tsit5').
        store_trajectory: Whether to return the full rate trajectories at all time points.
        is_static: Whether the external L4 stimulus drive is constant over time.

    Returns:
        A JIT-compiled callable `run` that executes the ODE integration.
    """
    def interp_phi(x, xp, fp):
        return jnp.interp(x, xp, fp, left=fp[0], right=fp[-1])

    if is_static:
        def vector_field(t, y, args):
            W_exc, W_inh, mu_ext, bg_e_interp, bg_i_interp, phi_exc_x, phi_exc_y, phi_inh_x, phi_inh_y, tau_exc, tau_inh, idx_exc, idx_inh = args
            curr_bg_e = bg_e_interp.evaluate(t)
            curr_bg_i = bg_i_interp.evaluate(t)

            mu = W_exc @ y[idx_exc, :] + W_inh @ y[idx_inh, :] + mu_ext

            dy = jnp.zeros_like(y)
            dy = dy.at[idx_exc, :].set(
                (-y[idx_exc, :] + interp_phi(tau_exc * mu[idx_exc, :] + curr_bg_e, phi_exc_x, phi_exc_y))
                / tau_exc
            )
            dy = dy.at[idx_inh, :].set(
                (-y[idx_inh, :] + interp_phi(tau_inh * mu[idx_inh, :] + curr_bg_i, phi_inh_x, phi_inh_y))
                / tau_inh
            )
            return dy
    else:
        def vector_field(t, y, args):
            W_exc, W_inh, W_ext, ax_interp, bg_e_interp, bg_i_interp, phi_exc_x, phi_exc_y, phi_inh_x, phi_inh_y, tau_exc, tau_inh, idx_exc, idx_inh = args
            ax = ax_interp.evaluate(t)
            curr_bg_e = bg_e_interp.evaluate(t)
            curr_bg_i = bg_i_interp.evaluate(t)

            mu = W_exc @ y[idx_exc, :] + W_inh @ y[idx_inh, :] + W_ext @ ax

            dy = jnp.zeros_like(y)
            dy = dy.at[idx_exc, :].set(
                (-y[idx_exc, :] + interp_phi(tau_exc * mu[idx_exc, :] + curr_bg_e, phi_exc_x, phi_exc_y))
                / tau_exc
            )
            dy = dy.at[idx_inh, :].set(
                (-y[idx_inh, :] + interp_phi(tau_inh * mu[idx_inh, :] + curr_bg_i, phi_inh_x, phi_inh_y))
                / tau_inh
            )
            return dy

    def run(
        y0,
        W_exc,
        W_inh,
        W_ext,
        mu_ext,
        idx_exc,
        idx_inh,
        time,
        ax_t,
        bg_e,
        bg_i,
        phi_exc_x,
        phi_exc_y,
        phi_inh_x,
        phi_inh_y,
        tau_exc,
        tau_inh,
    ):
        bg_e_interp = diffrax.LinearInterpolation(time, bg_e)
        bg_i_interp = diffrax.LinearInterpolation(time, bg_i)

        if is_static:
            args = (
                W_exc,
                W_inh,
                mu_ext,
                bg_e_interp,
                bg_i_interp,
                phi_exc_x,
                phi_exc_y,
                phi_inh_x,
                phi_inh_y,
                tau_exc,
                tau_inh,
                idx_exc,
                idx_inh,
            )
        else:
            ax_interp = diffrax.LinearInterpolation(time, ax_t)
            args = (
                W_exc,
                W_inh,
                W_ext,
                ax_interp,
                bg_e_interp,
                bg_i_interp,
                phi_exc_x,
                phi_exc_y,
                phi_inh_x,
                phi_inh_y,
                tau_exc,
                tau_inh,
                idx_exc,
                idx_inh,
            )

        term = diffrax.ODETerm(vector_field)
        if solver_name == "tsit5":
            solver = diffrax.Tsit5()
        elif solver_name == "heun":
            solver = diffrax.Heun()
        else:
            raise ValueError(f"Unsupported diffrax solver: {solver_name}")
        stepsize_controller = diffrax.ConstantStepSize()
        dt0 = time[1] - time[0]
        if store_trajectory:
            saveat = diffrax.SaveAt(ts=time)
        elif tail_points > 1:
            saveat = diffrax.SaveAt(ts=time[-tail_points:])
        else:
            saveat = diffrax.SaveAt(t1=True)

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
        )

        y_all = sol.ys
        if store_trajectory:
            return y_all

        if tail_points > 1:
            return jnp.mean(y_all, axis=0), jnp.std(y_all, axis=0)
        else:
            # SaveAt(t1=True) returns ys with shape (1, n_rates, n_batch).
            # Squeeze the leading singleton so downstream gets (n_rates, n_batch).
            y_final = y_all[0]
            return y_final, jnp.zeros_like(y_final)

    return jax.jit(run)


def _precompute_diffrax_inputs(
    external_drive: ExternalDrive,
    time: FloatArray,
    *,
    n_ext: int,
    n_batch: int,
) -> FloatArray:
    """Precompute deterministic forcing traces for Diffrax.

    External drive is treated as a pre-sampled deterministic input,
    not as a Diffrax-native stochastic process.
    """
    ax_t = []
    for t in time:
        ax_t.append(validate_external_drive_value(external_drive(float(t)), n_ext=n_ext, n_batch=n_batch))
    return np.stack(ax_t)


def _precompute_diffrax_background(
    trace,
    *,
    layout: NetworkLayout,
    n_batch: int,
    time: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Precompute deterministic forcing traces for Diffrax.

    Background activity is treated as a pre-sampled deterministic input,
    not as a Diffrax-native stochastic process.
    """
    from v1_simulation.solvers.base import validate_background_trace
    validate_background_trace(
        trace,
        n_exc=layout.n_exc,
        n_inh=layout.n_inh,
        n_batch=n_batch,
        time=time,
    )
    if trace is None:
        n_steps = time.size
        return (
            np.zeros((n_steps, layout.n_exc, n_batch), dtype=np.float64),
            np.zeros((n_steps, layout.n_inh, n_batch), dtype=np.float64),
        )

    return (
        np.transpose(trace.exc, (0, 2, 1)),
        np.transpose(trace.inh, (0, 2, 1)),
    )


def _make_jax_rk4(jax, jnp, *, store_trajectory: bool, is_static: bool):
    """Creates a JIT-compiled JAX RK4 ODE solver for the Wilson-Cowan system.

    Receives pre-sliced weight sub-matrices W_exc, W_inh, W_ext and (when is_static=True)
    the precomputed static external drive contribution mu_ext. No weight indexing occurs
    inside this JIT-compiled function.

    Args:
        jax: The jax module.
        jnp: The jax.numpy module.
        store_trajectory: Whether to return the full rate trajectories.
        is_static: Whether the external L4 stimulus drive is constant over time.

    Returns:
        A JIT-compiled callable `run`.
    """
    def interp_phi(x, xp, fp):
        return jnp.interp(x, xp, fp, left=fp[0], right=fp[-1])

    if is_static:
        def wc_rhs(
            y,
            _ax,
            bg_e,
            bg_i,
            W_exc,
            W_inh,
            _W_ext,
            idx_exc,
            idx_inh,
            phi_exc_x,
            phi_exc_y,
            phi_inh_x,
            phi_inh_y,
            tau_exc,
            tau_inh,
            mu_ext,
        ):
            mu = W_exc @ y[idx_exc, :] + W_inh @ y[idx_inh, :] + mu_ext
            dy = jnp.zeros_like(y)
            dy = dy.at[idx_exc, :].set(
                (-y[idx_exc, :] + interp_phi(tau_exc * mu[idx_exc, :] + bg_e, phi_exc_x, phi_exc_y))
                / tau_exc
            )
            dy = dy.at[idx_inh, :].set(
                (-y[idx_inh, :] + interp_phi(tau_inh * mu[idx_inh, :] + bg_i, phi_inh_x, phi_inh_y))
                / tau_inh
            )
            return dy
    else:
        def wc_rhs(
            y,
            ax,
            bg_e,
            bg_i,
            W_exc,
            W_inh,
            W_ext,
            idx_exc,
            idx_inh,
            phi_exc_x,
            phi_exc_y,
            phi_inh_x,
            phi_inh_y,
            tau_exc,
            tau_inh,
            mu_ext,
        ):
            mu = W_exc @ y[idx_exc, :] + W_inh @ y[idx_inh, :] + W_ext @ ax
            dy = jnp.zeros_like(y)
            dy = dy.at[idx_exc, :].set(
                (-y[idx_exc, :] + interp_phi(tau_exc * mu[idx_exc, :] + bg_e, phi_exc_x, phi_exc_y))
                / tau_exc
            )
            dy = dy.at[idx_inh, :].set(
                (-y[idx_inh, :] + interp_phi(tau_inh * mu[idx_inh, :] + bg_i, phi_inh_x, phi_inh_y))
                / tau_inh
            )
            return dy

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
        phi_inh_x,
        phi_inh_y,
        tau_exc,
        tau_inh,
    ):
        params = (
            W_exc,
            W_inh,
            W_ext,
            idx_exc,
            idx_inh,
            phi_exc_x,
            phi_exc_y,
            phi_inh_x,
            phi_inh_y,
            tau_exc,
            tau_inh,
            mu_ext,
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
        if store_trajectory:
            return y_all
        tail = y_all[int(time.shape[0] * 2 / 3) :, :, :]
        return jnp.mean(tail, axis=0), jnp.std(tail, axis=0)

    return jax.jit(run)


def _precompute_rk4_inputs(
    external_drive: ExternalDrive,
    time: FloatArray,
    *,
    n_ext: int,
    n_batch: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    left = []
    mid = []
    right = []
    for t0, t1 in zip(time[:-1], time[1:]):
        dt = float(t1 - t0)
        left.append(validate_external_drive_value(external_drive(float(t0)), n_ext=n_ext, n_batch=n_batch))
        mid.append(validate_external_drive_value(external_drive(float(t0 + 0.5 * dt)), n_ext=n_ext, n_batch=n_batch))
        right.append(validate_external_drive_value(external_drive(float(t1)), n_ext=n_ext, n_batch=n_batch))
    return np.stack(left), np.stack(mid), np.stack(right)


def _precompute_rk4_background(
    trace,
    *,
    layout: NetworkLayout,
    n_batch: int,
    time: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    samples = prepare_rk4_background(
        trace,
        n_exc=layout.n_exc,
        n_inh=layout.n_inh,
        n_batch=n_batch,
        time=time,
    )
    if samples is None:
        n_steps = time.size - 1
        return (
            np.zeros((n_steps, layout.n_exc, n_batch), dtype=np.float64),
            np.zeros((n_steps, layout.n_exc, n_batch), dtype=np.float64),
            np.zeros((n_steps, layout.n_exc, n_batch), dtype=np.float64),
            np.zeros((n_steps, layout.n_inh, n_batch), dtype=np.float64),
            np.zeros((n_steps, layout.n_inh, n_batch), dtype=np.float64),
            np.zeros((n_steps, layout.n_inh, n_batch), dtype=np.float64),
        )

    return (
        np.transpose(samples.exc_left, (0, 2, 1)),
        np.transpose(samples.exc_mid, (0, 2, 1)),
        np.transpose(samples.exc_right, (0, 2, 1)),
        np.transpose(samples.inh_left, (0, 2, 1)),
        np.transpose(samples.inh_mid, (0, 2, 1)),
        np.transpose(samples.inh_right, (0, 2, 1)),
    )


def _transfer_table_arrays(phi, name: str) -> tuple[FloatArray, FloatArray]:
    if hasattr(phi, "as_arrays"):
        x, y = phi.as_arrays()
    elif hasattr(phi, "mu") and hasattr(phi, "rate"):
        x, y = phi.mu, phi.rate
    else:
        raise ValueError(f"{name} must be a TransferTable-like object for jax-rk4.")
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _prepare_jax_matrix(matrix, jnp, *, prefer_sparse: bool, dense_max_mb: float, dtype=None):
    if dtype is None:
        dtype = jnp.float64
    if prefer_sparse and scipy_sparse.issparse(matrix):
        from jax.experimental import sparse as jax_sparse

        coo = matrix.tocoo()
        indices = np.column_stack([coo.row, coo.col]).astype(np.int32, copy=False)
        return jax_sparse.BCOO((jnp.asarray(coo.data, dtype=dtype), jnp.asarray(indices)), shape=coo.shape)

    np_dtype = np.float32 if dtype == jnp.float32 else np.float64
    dense_mb = np.prod(matrix.shape) * np_dtype().itemsize / 1024.0**2
    if dense_mb > float(dense_max_mb):
        raise RuntimeError(f"Dense JAX weights fallback would require {dense_mb:.1f} MB.")
    return jnp.asarray(matrix.toarray() if scipy_sparse.issparse(matrix) else matrix, dtype=dtype)


def _require_jax(backend_name: str = "jax-rk4"):
    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"solver.backend='{backend_name}' requested, but jax is not installed.\n"
            f"This backend requires installing the optional JAX dependencies.\n"
            "Try: pip install -e \".[jax]\""
        ) from exc
    return jax, jnp


def _require_diffrax():
    try:
        import diffrax
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "solver.backend='diffrax' requested, but diffrax is not installed.\n"
            "Diffrax backend requires installing the optional JAX dependencies.\n"
            "Try: pip install -e \".[jax]\""
        ) from exc
    return diffrax
