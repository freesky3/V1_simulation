from __future__ import annotations

import importlib.util

import numpy as np
from scipy import sparse as scipy_sparse

from v1_simulation.solvers.base import FloatArray

SUPPORTED_DIFFRAX_SOLVERS = ("tsit5", "heun")


def is_jax_available() -> bool:
    return importlib.util.find_spec("jax") is not None


def is_diffrax_available() -> bool:
    return importlib.util.find_spec("diffrax") is not None


def require_jax(backend_name: str = "jax-rk4"):
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


def require_diffrax():
    try:
        import diffrax
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "solver.backend='diffrax' requested, but diffrax is not installed.\n"
            "Diffrax backend requires installing the optional JAX dependencies.\n"
            "Try: pip install -e \".[jax]\""
        ) from exc
    return diffrax


def make_diffrax_solver(diffrax, solver_name: str):
    name = str(solver_name).lower()
    if name == "tsit5":
        return diffrax.Tsit5()
    if name == "heun":
        return diffrax.Heun()
    allowed = ", ".join(SUPPORTED_DIFFRAX_SOLVERS)
    raise ValueError(f"Unsupported diffrax solver: {solver_name!r}. Supported values: {allowed}.")


def slice_weight_blocks(
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
    """Pre-slice weights into dynamic E/I and external-source blocks before JIT."""
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
        prepare_jax_matrix(w_exc, jnp, prefer_sparse=prefer_sparse, dense_max_mb=dense_max_mb, dtype=dtype),
        prepare_jax_matrix(w_inh, jnp, prefer_sparse=prefer_sparse, dense_max_mb=dense_max_mb, dtype=dtype),
        prepare_jax_matrix(w_ext, jnp, prefer_sparse=prefer_sparse, dense_max_mb=dense_max_mb, dtype=dtype),
    )


def transfer_table_arrays(phi, name: str) -> tuple[FloatArray, FloatArray, float]:
    if hasattr(phi, "as_arrays"):
        x, y = phi.as_arrays()
    elif hasattr(phi, "mu") and hasattr(phi, "rate"):
        x, y = phi.mu, phi.rate
    else:
        raise ValueError(f"{name} must be a TransferTable-like object for JAX solvers.")
    rate_max = getattr(phi, "rate_max", None)
    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        float(rate_max) if rate_max is not None else float("inf"),
    )


def prepare_jax_matrix(matrix, jnp, *, prefer_sparse: bool, dense_max_mb: float, dtype=None):
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
