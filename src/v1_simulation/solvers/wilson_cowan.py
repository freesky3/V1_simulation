from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

from v1_simulation.solvers.base import (
    BatchODEResult,
    ExternalDrive,
    FloatArray,
    NetworkLayout,
    SolverOptions,
    TransferFunction,
    validate_background_trace,
    validate_external_drive_value,
    validate_solver_options,
)
from v1_simulation.stimuli.background import BackgroundTrace, validate_time_grid
from v1_simulation.transfer.siegert import TransferGrid, TransferTables, build_transfer_tables

if TYPE_CHECKING:
    from v1_simulation.config.schema import RootConfig, SolverConfig, TrainingBCMConfig, TransferConfig
    from v1_simulation.network.state import NetworkState


class WilsonCowanRHS:
    """Batched Wilson-Cowan right-hand side evaluator.

    Dynamic rates live on L2/3 rows. External L4 drives occupy their source
    columns in the connectivity-weight matrix.
    """

    def __init__(
        self,
        *,
        weights,
        layout: NetworkLayout,
        phi_exc: TransferFunction,
        phi_inh: TransferFunction,
        tau_exc: float,
        tau_inh: float,
        n_batch: int,
        background_trace: BackgroundTrace | None = None,
    ) -> None:
        self.weights = weights.tocsr(copy=False) if sparse.issparse(weights) else sparse.csr_matrix(weights)
        self.layout = layout
        self.phi_exc = phi_exc
        self.phi_inh = phi_inh
        self.tau_exc = _positive_float(tau_exc, "tau_exc")
        self.tau_inh = _positive_float(tau_inh, "tau_inh")
        self.n_batch = _positive_int(n_batch, "n_batch")
        self.background_trace = background_trace

        if self.weights.shape[0] != layout.n_rates:
            raise ValueError(
                f"weights row count {self.weights.shape[0]} must match layout.n_rates {layout.n_rates}."
            )
        if layout.idx_ext.size and int(layout.idx_ext.max()) >= self.weights.shape[1]:
            raise ValueError("layout.idx_ext contains source columns outside the weights matrix.")

        self._all_sources = np.zeros((self.weights.shape[1], self.n_batch), dtype=np.float64)
        self._dy = np.zeros((layout.n_rates, self.n_batch), dtype=np.float64)

    def __call__(
        self,
        t: float,
        y_flat: FloatArray,
        external_drive: ExternalDrive,
        *,
        background: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> FloatArray:
        y = np.asarray(y_flat, dtype=np.float64).reshape(self.layout.n_rates, self.n_batch)

        self._all_sources.fill(0.0)
        self._all_sources[self.layout.idx_exc, :] = y[self.layout.idx_exc, :]
        self._all_sources[self.layout.idx_inh, :] = y[self.layout.idx_inh, :]
        self._all_sources[self.layout.idx_ext, :] = validate_external_drive_value(
            external_drive(float(t)),
            n_ext=self.layout.n_ext,
            n_batch=self.n_batch,
        )

        mu = self.weights @ self._all_sources
        exc_drive = self.tau_exc * mu[self.layout.idx_exc, :]
        inh_drive = self.tau_inh * mu[self.layout.idx_inh, :]

        if background is not None:
            bg_exc, bg_inh = _background_stage(background, self.layout, self.n_batch)
            exc_drive = exc_drive + bg_exc
            inh_drive = inh_drive + bg_inh

        exc_rate = np.asarray(self.phi_exc(exc_drive), dtype=np.float64)
        inh_rate = np.asarray(self.phi_inh(inh_drive), dtype=np.float64)
        if exc_rate.shape != exc_drive.shape:
            raise ValueError(f"phi_exc returned shape {exc_rate.shape}, expected {exc_drive.shape}.")
        if inh_rate.shape != inh_drive.shape:
            raise ValueError(f"phi_inh returned shape {inh_rate.shape}, expected {inh_drive.shape}.")

        self._dy[self.layout.idx_exc, :] = (-y[self.layout.idx_exc, :] + exc_rate) / self.tau_exc
        self._dy[self.layout.idx_inh, :] = (-y[self.layout.idx_inh, :] + inh_rate) / self.tau_inh
        return self._dy.ravel().copy()


def solve_wilson_cowan_from_config(
    *,
    cfg: RootConfig,
    network: NetworkState,
    external_drive: ExternalDrive,
    time: ArrayLike,
    n_batch: int,
    background_trace: BackgroundTrace | None = None,
    transfer_tables: TransferTables | None = None,
    phi_exc: TransferFunction | None = None,
    phi_inh: TransferFunction | None = None,
    store_trajectory: bool = True,
    stop_at_steady_state: bool | None = None,
) -> BatchODEResult:
    """Solves Wilson-Cowan dynamics using configuration fields from RootConfig.

    Args:
        cfg: The parsed RootConfig configuration.
        network: The network layout and weights.
        external_drive: Continuous-time L4 stimulus drive.
        time: Target time grid points.
        n_batch: The batch size (number of stimuli).
        background_trace: Optional background trace.
        transfer_tables: Optional prebuilt Siegert transfer tables.
        phi_exc: Optional custom excitatory transfer function.
        phi_inh: Optional custom inhibitory transfer function.
        store_trajectory: Whether to store and return the full trajectories.
        stop_at_steady_state: Optional override for early steady-state stopping.

    Returns:
        The BatchODEResult containing trajectories or summarized statistics.
    """

    return solve_wilson_cowan_batch(
        network=network,
        external_drive=external_drive,
        time=time,
        n_batch=n_batch,
        solver_config=cfg.solver,
        transfer_config=cfg.solver.transfer,
        training_bcm=cfg.training.bcm if cfg.training.enabled else None,
        background_trace=background_trace,
        transfer_tables=transfer_tables,
        phi_exc=phi_exc,
        phi_inh=phi_inh,
        store_trajectory=store_trajectory,
        stop_at_steady_state=stop_at_steady_state,
    )


def solve_wilson_cowan_batch(
    *,
    network: NetworkState,
    external_drive: ExternalDrive,
    time: ArrayLike,
    n_batch: int,
    solver_config: SolverConfig,
    transfer_config: TransferConfig | None = None,
    training_bcm: TrainingBCMConfig | None = None,
    background_trace: BackgroundTrace | None = None,
    transfer_tables: TransferTables | None = None,
    phi_exc: TransferFunction | None = None,
    phi_inh: TransferFunction | None = None,
    options: SolverOptions | None = None,
    store_trajectory: bool = True,
    stop_at_steady_state: bool | None = None,
) -> BatchODEResult:
    """Solves a batch of Wilson-Cowan systems for one network state.

    Args:
        network: The network layout and weights.
        external_drive: Continuous-time L4 stimulus drive.
        time: Target time grid points.
        n_batch: The batch size (number of stimuli).
        solver_config: The parsed SolverConfig configuration.
        transfer_config: Optional transfer configuration to use.
        training_bcm: Optional TrainingBCMConfig settings for steady state.
        background_trace: Optional background trace.
        transfer_tables: Optional prebuilt Siegert transfer tables.
        phi_exc: Optional custom excitatory transfer function.
        phi_inh: Optional custom inhibitory transfer function.
        options: Optional prebuilt SolverOptions instance.
        store_trajectory: Whether to store and return the full trajectories.
        stop_at_steady_state: Optional override for early steady-state stopping.

    Returns:
        The BatchODEResult containing trajectories or summarized statistics.
    """

    time_grid = validate_time_grid(time, copy=True)
    layout = NetworkLayout.from_network_state(network)
    n_batch = _positive_int(n_batch, "n_batch")
    transfer_cfg = solver_config.transfer if transfer_config is None else transfer_config
    phi_E, phi_I = _resolve_transfer_functions(
        transfer_config=transfer_cfg,
        transfer_tables=transfer_tables,
        phi_exc=phi_exc,
        phi_inh=phi_inh,
    )

    run_options = options or SolverOptions.from_config(
        solver_config,
        training_bcm=training_bcm,
        store_trajectory=store_trajectory,
        stop_at_steady_state=stop_at_steady_state,
    )
    run_options = _apply_training_min_time(run_options, training_bcm, transfer_cfg)
    validate_solver_options(run_options)
    validate_background_trace(
        background_trace,
        n_exc=layout.n_exc,
        n_inh=layout.n_inh,
        n_batch=n_batch,
        time=time_grid,
    )

    rhs = WilsonCowanRHS(
        weights=network.weights,
        layout=layout,
        phi_exc=phi_E,
        phi_inh=phi_I,
        tau_exc=transfer_cfg.tau_e,
        tau_inh=transfer_cfg.tau_i,
        n_batch=n_batch,
        background_trace=background_trace,
    )

    if run_options.backend == "scipy":
        from v1_simulation.solvers.scipy_backend import solve_scipy

        return solve_scipy(rhs, external_drive, layout, n_batch, time_grid, run_options)

    if run_options.backend == "jax-rk4":
        from v1_simulation.solvers.jax_backend import solve_jax_rk4

        return solve_jax_rk4(
            rhs=rhs,
            external_drive=external_drive,
            layout=layout,
            n_batch=n_batch,
            time=time_grid,
            options=run_options,
        )

    if run_options.backend == "diffrax":
        from v1_simulation.solvers.jax_backend import solve_diffrax

        return solve_diffrax(
            rhs=rhs,
            external_drive=external_drive,
            layout=layout,
            n_batch=n_batch,
            time=time_grid,
            options=run_options,
        )

    raise ValueError(f"Unknown solver backend: {run_options.backend!r}")


def _resolve_transfer_functions(
    *,
    transfer_config: TransferConfig,
    transfer_tables: TransferTables | None,
    phi_exc: TransferFunction | None,
    phi_inh: TransferFunction | None,
) -> tuple[TransferFunction, TransferFunction]:
    if phi_exc is not None and phi_inh is not None:
        return phi_exc, phi_inh
    if (phi_exc is None) != (phi_inh is None):
        raise ValueError("phi_exc and phi_inh must be provided together.")

    tables = transfer_tables
    if tables is None:
        if transfer_config.kind != "siegert":
            raise ValueError(f"Unsupported transfer kind: {transfer_config.kind!r}")
        tables = build_transfer_tables(
            cfg_transfer=transfer_config,
            grid=TransferGrid.symmetric(transfer_config.mu_tab_max),
        )
    return tables.excitatory, tables.inhibitory


def _apply_training_min_time(
    options: SolverOptions,
    training_bcm: TrainingBCMConfig | None,
    transfer_config: TransferConfig,
) -> SolverOptions:
    if not options.stop_at_steady_state or options.steady_state_min_time is not None:
        return options
    if training_bcm is None:
        return options
    min_time = float(training_bcm.steady_state_min_tau) * max(
        float(transfer_config.tau_e),
        float(transfer_config.tau_i),
    )
    return replace(options, steady_state_min_time=min_time)


def _background_stage(
    background: tuple[ArrayLike, ArrayLike],
    layout: NetworkLayout,
    n_batch: int,
) -> tuple[FloatArray, FloatArray]:
    bg_exc = np.asarray(background[0], dtype=np.float64)
    bg_inh = np.asarray(background[1], dtype=np.float64)
    if bg_exc.shape != (n_batch, layout.n_exc):
        raise ValueError(f"background exc shape {bg_exc.shape} != ({n_batch}, {layout.n_exc}).")
    if bg_inh.shape != (n_batch, layout.n_inh):
        raise ValueError(f"background inh shape {bg_inh.shape} != ({n_batch}, {layout.n_inh}).")
    return bg_exc.T, bg_inh.T


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value
