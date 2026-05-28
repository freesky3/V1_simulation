from __future__ import annotations

import importlib.util
import warnings

import numpy as np
from scipy import sparse as scipy_sparse

from v1_simulation.solvers.base import (
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
    ax_left, ax_mid, ax_right = _precompute_rk4_inputs(
        external_drive,
        time,
        n_ext=layout.n_ext,
        n_batch=n_batch,
    )
    bg_left_e, bg_mid_e, bg_right_e, bg_left_i, bg_mid_i, bg_right_i = _precompute_rk4_background(
        rhs.background_trace,
        layout=layout,
        n_batch=n_batch,
        time=time,
    )
    phi_exc_x, phi_exc_y = _transfer_table_arrays(rhs.phi_exc, "phi_exc")
    phi_inh_x, phi_inh_y = _transfer_table_arrays(rhs.phi_inh, "phi_inh")
    weights = _prepare_jax_matrix(
        rhs.weights,
        jnp,
        prefer_sparse=options.jax_prefer_sparse,
        dense_max_mb=options.jax_dense_max_mb,
    )

    y0 = jnp.zeros((layout.n_rates, int(n_batch)), dtype=jnp.asarray(time).dtype)
    cache_key = options.store_trajectory
    if cache_key not in _jax_rk4_solve_cache:
        _jax_rk4_solve_cache[cache_key] = _make_jax_rk4(jax, jnp, store_trajectory=options.store_trajectory)
    run = _jax_rk4_solve_cache[cache_key]
    out = run(
        y0,
        weights,
        jnp.asarray(layout.idx_exc, dtype=jnp.int32),
        jnp.asarray(layout.idx_inh, dtype=jnp.int32),
        jnp.asarray(layout.idx_ext, dtype=jnp.int32),
        jnp.asarray(time),
        jnp.asarray(ax_left),
        jnp.asarray(ax_mid),
        jnp.asarray(ax_right),
        jnp.asarray(bg_left_e),
        jnp.asarray(bg_mid_e),
        jnp.asarray(bg_right_e),
        jnp.asarray(bg_left_i),
        jnp.asarray(bg_mid_i),
        jnp.asarray(bg_right_i),
        jnp.asarray(phi_exc_x),
        jnp.asarray(phi_exc_y),
        jnp.asarray(phi_inh_x),
        jnp.asarray(phi_inh_y),
        jnp.asarray(float(rhs.tau_exc)),
        jnp.asarray(float(rhs.tau_inh)),
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

    ax_t = _precompute_diffrax_inputs(
        external_drive,
        time,
        n_ext=layout.n_ext,
        n_batch=n_batch,
    )
    bg_e, bg_i = _precompute_diffrax_background(
        rhs.background_trace,
        layout=layout,
        n_batch=n_batch,
        time=time,
    )

    phi_exc_x, phi_exc_y = _transfer_table_arrays(rhs.phi_exc, "phi_exc")
    phi_inh_x, phi_inh_y = _transfer_table_arrays(rhs.phi_inh, "phi_inh")
    weights = _prepare_jax_matrix(
        rhs.weights,
        jnp,
        prefer_sparse=options.jax_prefer_sparse,
        dense_max_mb=options.jax_dense_max_mb,
    )

    y0 = jnp.zeros((layout.n_rates, int(n_batch)), dtype=jnp.asarray(time).dtype)
    cache_key = (options.diffrax_solver, options.store_trajectory)
    if cache_key not in _diffrax_solve_cache:
        _diffrax_solve_cache[cache_key] = _make_diffrax_diffeqsolve(
            jax,
            jnp,
            diffrax,
            solver_name=options.diffrax_solver,
            store_trajectory=options.store_trajectory,
        )
    run = _diffrax_solve_cache[cache_key]

    out = run(
        y0,
        weights,
        jnp.asarray(layout.idx_exc, dtype=jnp.int32),
        jnp.asarray(layout.idx_inh, dtype=jnp.int32),
        jnp.asarray(layout.idx_ext, dtype=jnp.int32),
        jnp.asarray(time),
        jnp.asarray(ax_t),
        jnp.asarray(bg_e),
        jnp.asarray(bg_i),
        jnp.asarray(phi_exc_x),
        jnp.asarray(phi_exc_y),
        jnp.asarray(phi_inh_x),
        jnp.asarray(phi_inh_y),
        jnp.asarray(float(rhs.tau_exc)),
        jnp.asarray(float(rhs.tau_inh)),
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
):
    """Creates a JIT-compiled JAX function to solve the Wilson-Cowan ODEs using Diffrax.

    Args:
        jax: The jax module.
        jnp: The jax.numpy module.
        diffrax: The diffrax module.
        solver_name: The name of the Diffrax solver to use (e.g., 'tsit5').
        store_trajectory: Whether to return the full rate trajectories at all time points.

    Returns:
        A JIT-compiled callable `run` that executes the ODE integration.
    """
    def interp_phi(x, xp, fp):
        return jnp.interp(x, xp, fp, left=fp[0], right=fp[-1])

    def run(
        y0,
        weights,
        idx_exc,
        idx_inh,
        idx_ext,
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
        # The forcing traces are linearly interpolated between simulation grid points.
        # This differs from RK4 left/mid/right sampling and may not be numerically identical.
        ax_interp = diffrax.LinearInterpolation(time, ax_t)
        bg_e_interp = diffrax.LinearInterpolation(time, bg_e)
        bg_i_interp = diffrax.LinearInterpolation(time, bg_i)

        def vector_field(t, y, args):
            ax = ax_interp.evaluate(t)
            curr_bg_e = bg_e_interp.evaluate(t)
            curr_bg_i = bg_i_interp.evaluate(t)

            sources = jnp.zeros((weights.shape[1], y.shape[1]), dtype=y.dtype)
            sources = sources.at[idx_exc, :].set(y[idx_exc, :])
            sources = sources.at[idx_inh, :].set(y[idx_inh, :])
            sources = sources.at[idx_ext, :].set(ax)

            mu = weights @ sources
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

        term = diffrax.ODETerm(vector_field)
        if solver_name == "tsit5":
            solver = diffrax.Tsit5()
        else:
            raise ValueError(f"Unsupported diffrax solver: {solver_name}")
        stepsize_controller = diffrax.ConstantStepSize()
        dt0 = time[1] - time[0]
        saveat = diffrax.SaveAt(ts=time)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=time[0],
            t1=time[-1],
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )

        y_all = sol.ys
        if store_trajectory:
            return y_all
        
        tail = y_all[int(time.shape[0] * 2 / 3) :, :, :]
        return jnp.mean(tail, axis=0), jnp.std(tail, axis=0)

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


def _make_jax_rk4(jax, jnp, *, store_trajectory: bool):
    def interp_phi(x, xp, fp):
        return jnp.interp(x, xp, fp, left=fp[0], right=fp[-1])

    def wc_rhs(
        y,
        ax,
        bg_e,
        bg_i,
        weights,
        idx_exc,
        idx_inh,
        idx_ext,
        phi_exc_x,
        phi_exc_y,
        phi_inh_x,
        phi_inh_y,
        tau_exc,
        tau_inh,
    ):
        sources = jnp.zeros((weights.shape[1], y.shape[1]), dtype=y.dtype)
        sources = sources.at[idx_exc, :].set(y[idx_exc, :])
        sources = sources.at[idx_inh, :].set(y[idx_inh, :])
        sources = sources.at[idx_ext, :].set(ax)

        mu = weights @ sources
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
        weights,
        idx_exc,
        idx_inh,
        idx_ext,
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
            weights,
            idx_exc,
            idx_inh,
            idx_ext,
            phi_exc_x,
            phi_exc_y,
            phi_inh_x,
            phi_inh_y,
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


def _prepare_jax_matrix(matrix, jnp, *, prefer_sparse: bool, dense_max_mb: float):
    if prefer_sparse and scipy_sparse.issparse(matrix):
        from jax.experimental import sparse as jax_sparse

        coo = matrix.tocoo()
        indices = np.column_stack([coo.row, coo.col]).astype(np.int32, copy=False)
        return jax_sparse.BCOO((jnp.asarray(coo.data), jnp.asarray(indices)), shape=coo.shape)

    dense_mb = np.prod(matrix.shape) * np.dtype(np.float64).itemsize / 1024.0**2
    if dense_mb > float(dense_max_mb):
        raise RuntimeError(f"Dense JAX weights fallback would require {dense_mb:.1f} MB.")
    return jnp.asarray(matrix.toarray() if scipy_sparse.issparse(matrix) else matrix)


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
