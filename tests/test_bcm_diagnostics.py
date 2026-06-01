import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from v1_simulation.config.schema import RootConfig
from v1_simulation.network.geometry import SheetGeometry
from v1_simulation.network.state import NetworkState, PopulationLayout
from v1_simulation.solvers.base import BatchODEResult
from v1_simulation.training.trainer import BCMTrainer


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_bcm_training.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_bcm_training", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
diag = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = diag
_SPEC.loader.exec_module(diag)

DiagnosticsOptions = diag.DiagnosticsOptions
active_rate_stats = diag.active_rate_stats
cap_fraction = diag.cap_fraction
collect_update_metrics = diag.collect_update_metrics
connected_block_values = diag.connected_block_values
run_bcm_training_diagnostics = diag.run_bcm_training_diagnostics


def test_active_rate_stats_filters_small_rates() -> None:
    rates = np.array([[0.2, 1.0, 1.5], [2.0, 0.0, 3.0]])

    stats = active_rate_stats(rates, "probe_E", threshold=1.0)

    assert stats["probe_E_all_count"] == 6
    assert stats["probe_E_active_count"] == 3
    assert stats["probe_E_active_fraction"] == 0.5
    assert stats["probe_E_active_mean"] == 2.1666666666666665
    assert stats["probe_E_active_max"] == 3.0


def test_weight_block_and_cap_fraction_handles_none_wmax() -> None:
    network = _tiny_network()
    weights = network.weights.toarray()
    topology = network.connectivity.toarray().astype(bool)

    w_ee = connected_block_values(weights, topology, network.idx_E, network.idx_E, include_zero=False)
    cap = cap_fraction(w_ee, options=DiagnosticsOptions(w_max=None))

    np.testing.assert_allclose(np.sort(w_ee), np.array([0.1, 0.2, 0.3, 0.5]))
    assert cap["fraction"] == 0.0
    assert cap["count"] == 0


def test_first_bcm_batch_initializes_theta_without_delta() -> None:
    network = _tiny_network()
    trainer = BCMTrainer(_test_cfg().training.bcm, network)
    dynamics = _fake_dynamics(n_batch=2, n_e=network.layout.n_E, n_i=network.layout.n_I, level=1.0)
    before = trainer.state.network.weights.copy()

    log_row = trainer.train_batch(dynamics, epoch=1, batch_size=2, images="probe")
    after = trainer.state.network.weights
    delta = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    metrics = collect_update_metrics(
        network=trainer.state.network,
        topology=network.connectivity.toarray().astype(bool),
        weights_after=np.asarray(after, dtype=float),
        delta=delta,
        log_row=log_row,
        theta=trainer.state.theta,
        row_sum_limits=trainer.row_sum_limits,
        probe_dynamics=dynamics,
        options=DiagnosticsOptions(w_max=30.0),
    )

    assert log_row.updated == 0
    assert metrics["dW_EE_mean_abs"] == 0.0
    assert metrics["dW_IE_mean_abs"] == 0.0
    assert metrics["theta_E_median"] > 0.0


def test_run_bcm_training_diagnostics_smoke_generates_artifacts(tmp_path) -> None:
    cfg = _test_cfg()
    cfg.paths.run_root = tmp_path / "runs"
    network = _tiny_network()
    samples = (
        SimpleNamespace(path=Path("im1.iml"), crop=None),
        SimpleNamespace(path=Path("im2.iml"), crop=None),
    )
    sampler = _FakeSampler(samples)
    drive = _FakeDrive(network.layout.n_X)
    calls = []

    def solver(**kwargs):
        calls.append(kwargs)
        n_batch = kwargs["n_batch"]
        level = float(len(calls))
        return _fake_dynamics(
            n_batch=n_batch,
            n_e=network.layout.n_E,
            n_i=network.layout.n_I,
            level=level,
            time=np.asarray(kwargs["time"], dtype=float),
        )

    result = run_bcm_training_diagnostics(
        cfg,
        options=DiagnosticsOptions(
            probe_count=2,
            probe_every=1,
            w_max=cfg.training.bcm.w_max,
            show_progress=False,
        ),
        network=network,
        drive=drive,
        sampler=sampler,
        solver=solver,
        time=np.array([0.0, 0.01]),
    )

    assert result.steps == 2
    assert (result.diagnostics_dir / "per_update_metrics.csv").exists()
    assert (result.diagnostics_dir / "summary.json").exists()
    assert (result.diagnostics_dir / "input_l4_rate_distribution.npy").exists()
    assert (result.diagnostics_dir / "figures" / "update_000001.png").exists()

    rows = list(csv.DictReader((result.diagnostics_dir / "per_update_metrics.csv").open()))
    assert [row["step"] for row in rows] == ["1", "2"]
    assert rows[0]["updated"] == "0"
    assert rows[1]["updated"] == "1"
    assert len(calls) == 4  # two training batches plus two fixed-probe batches


class _FakeSampler:
    def __init__(self, samples):
        self.samples = samples

    def make_epoch(self, *, limit=None, shuffle_paths=True, shuffle_samples=True):
        items = self.samples if limit is None else self.samples[:limit]
        return tuple(items)


class _FakeDrive:
    def __init__(self, n_ext):
        self.n_ext = n_ext

    def make_static_batch_func(self, samples):
        n_batch = len(samples)
        base = np.arange(1, self.n_ext + 1, dtype=float)[:, np.newaxis]
        rates = np.repeat(base, n_batch, axis=1)
        return lambda _t: rates


def _test_cfg() -> RootConfig:
    cfg = RootConfig()
    cfg.mode = "train"
    cfg.job_name = "diag_test"
    cfg.training.enabled = True
    cfg.training.natural_image.dir = "unused"
    cfg.training.bcm.epochs = 1
    cfg.training.bcm.batch_size = 1
    cfg.training.bcm.eta = 0.01
    cfg.training.bcm.theta_init = 1.0
    cfg.training.bcm.theta_floor = 1.0e-4
    cfg.training.bcm.save_every = 1
    cfg.solver.method = "RK45"
    return cfg


def _fake_dynamics(
    *,
    n_batch: int,
    n_e: int,
    n_i: int,
    level: float,
    time=None,
) -> BatchODEResult:
    time = np.array([0.0, 0.01]) if time is None else np.asarray(time, dtype=float)
    exc = np.full((n_batch, n_e), level, dtype=float)
    inh = np.full((n_batch, n_i), level + 0.5, dtype=float)
    return BatchODEResult(
        exc=exc,
        inh=inh,
        exc_trajectory=None,
        inh_trajectory=None,
        time=time,
        exc_convergence=np.zeros(n_batch),
        inh_convergence=np.zeros(n_batch),
        steady_state_reached=True,
        steady_state_index=time.size - 1,
        steady_state_start_index=0,
        summary_start_index=0,
        summary_end_index=time.size,
        dy_max=0.0,
        dy_rms=0.0,
    )


def _tiny_network() -> NetworkState:
    l23 = SheetGeometry(2, 1.0, 0.1)
    l4 = SheetGeometry(1, 1.0, 0.0)
    layout = PopulationLayout(
        l23=l23,
        l4=l4,
        l23_types=np.array(["E", "E", "I", "I"]),
        l4_tunings=np.array(["T"]),
        l4_pref_dirs=np.array([0.0]),
    )
    weights = np.array(
        [
            [0.5, 0.2, -0.1, 0.0, 10.0],
            [0.1, 0.3, -0.2, 0.0, 10.0],
            [0.7, 0.4, -0.3, 0.0, 10.0],
            [0.0, 0.2, -0.2, 0.0, 10.0],
        ],
        dtype=float,
    )
    return NetworkState(
        layout=layout,
        connectivity=sparse.csr_matrix(weights != 0.0),
        weights=sparse.csr_matrix(weights),
    )
