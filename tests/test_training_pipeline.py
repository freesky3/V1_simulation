import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from v1_simulation.config.schema import JaxSolverConfig, RootConfig
from v1_simulation.io.artifacts import TrainingArtifacts
from v1_simulation.network.geometry import SheetGeometry
from v1_simulation.network.state import NetworkState, PopulationLayout
from v1_simulation.simulation.pipeline import run_bcm_training
from v1_simulation.solvers.base import BatchODEResult
from v1_simulation.training.checkpoints import load_checkpoint, save_checkpoint


def test_training_artifacts_save_sparse_checkpoints_and_append_logs(tmp_path):
    network = _tiny_network()
    artifacts = TrainingArtifacts(tmp_path / "run")

    save_checkpoint(artifacts.run_dir, "network_final", network, metadata={"step": 3})
    artifacts.append_log({"step": 1, "weight_stats": {"W_EE_mean": 0.5}})
    artifacts.append_log({"step": 2, "weight_stats": {"W_EE_mean": 0.6}})

    assert (artifacts.run_dir / "network_final" / "weights.npz").exists()
    assert not (artifacts.run_dir / "network_final.npz").exists()

    loaded = load_checkpoint(artifacts.run_dir, "network_final")
    assert sparse.isspmatrix_csr(loaded["weights"])
    assert loaded["weights"].shape == network.weights.shape
    assert loaded["metadata"]["step"] == 3

    rows = list(csv.DictReader((artifacts.run_dir / "training_log.csv").open(encoding="utf-8")))
    assert [row["step"] for row in rows] == ["1", "2"]
    assert [row["W_EE_mean"] for row in rows] == ["0.5", "0.6"]


def test_run_bcm_training_uses_schema_solver_and_writes_streamed_artifacts(tmp_path):
    cfg = RootConfig()
    cfg.mode = "train"
    cfg.job_name = "pipeline_test"
    cfg.paths.run_root = tmp_path / "runs"
    cfg.training.enabled = True
    cfg.training.natural_image.dir = str(tmp_path)
    cfg.training.bcm.epochs = 1
    cfg.training.bcm.batch_size = 1
    cfg.training.bcm.eta = 0.0
    cfg.training.bcm.save_every = 1
    cfg.solver.method = "RK45"

    network = _tiny_network()
    samples = (
        SimpleNamespace(path=Path("im1.iml")),
        SimpleNamespace(path=Path("im2.iml")),
    )
    sampler = _FakeSampler(samples)
    drive = _FakeDrive(network.layout.n_X)
    calls = []

    def solver(**kwargs):
        calls.append(kwargs)
        n_batch = kwargs["n_batch"]
        time = np.asarray(kwargs["time"], dtype=float)
        level = float(len(calls))
        return BatchODEResult(
            exc=np.full((n_batch, network.layout.n_E), level),
            inh=np.full((n_batch, network.layout.n_I), level + 0.5),
            exc_trajectory=None,
            inh_trajectory=None,
            time=time,
            exc_convergence=np.zeros(n_batch),
            inh_convergence=np.zeros(n_batch),
            steady_state_reached=True,
            steady_state_index=time.size - 1,
            steady_state_start_index=0,
        )

    result = run_bcm_training(
        cfg,
        network=network,
        drive=drive,
        sampler=sampler,
        solver=solver,
        time=np.array([0.0, 0.01]),
        show_progress=False,
    )

    assert result.steps == 2
    assert result.samples_seen == 2
    assert result.images_seen == 2
    assert [call["solver_config"].method for call in calls] == ["RK45", "RK45"]
    assert all(call["store_trajectory"] is False for call in calls)

    log_rows = list(csv.DictReader((result.run_dir / "training_log.csv").open(encoding="utf-8")))
    assert [row["updated"] for row in log_rows] == ["0", "1"]
    assert (result.run_dir / "network_initial" / "weights.npz").exists()
    assert (result.run_dir / "network_latest" / "weights.npz").exists()
    assert (result.run_dir / "network_final" / "weights.npz").exists()
    assert (result.run_dir / "theta_M.npz").exists()
    assert (result.run_dir / "run_config.json").exists()


def test_run_bcm_training_keeps_dense_jax_weights_between_batches(tmp_path):
    from v1_simulation.solvers.jax_utils import is_jax_available

    if not is_jax_available():
        pytest.skip("JAX not installed")

    cfg = RootConfig()
    cfg.mode = "train"
    cfg.job_name = "jax_bcm_pipeline_test"
    cfg.paths.run_root = tmp_path / "runs"
    cfg.training.enabled = True
    cfg.training.natural_image.dir = str(tmp_path)
    cfg.training.bcm.epochs = 1
    cfg.training.bcm.batch_size = 1
    cfg.training.bcm.eta = 0.01
    cfg.training.bcm.save_every = 1
    cfg.solver.backend = "jax-rk4"
    cfg.solver.method = "RK4"
    cfg.solver.jax = JaxSolverConfig(prefer_sparse=False, dtype="float32")

    network = _tiny_network()
    samples = (
        SimpleNamespace(path=Path("im1.iml")),
        SimpleNamespace(path=Path("im2.iml")),
    )
    seen_weight_modules = []

    def solver(**kwargs):
        seen_weight_modules.append(type(kwargs["network"].weights).__module__)
        n_batch = kwargs["n_batch"]
        time = np.asarray(kwargs["time"], dtype=float)
        level = float(len(seen_weight_modules))
        return BatchODEResult(
            exc=np.full((n_batch, network.layout.n_E), level),
            inh=np.full((n_batch, network.layout.n_I), level + 0.5),
            exc_trajectory=None,
            inh_trajectory=None,
            time=time,
            exc_convergence=np.zeros(n_batch),
            inh_convergence=np.zeros(n_batch),
            steady_state_reached=True,
            steady_state_index=time.size - 1,
            steady_state_start_index=0,
        )

    result = run_bcm_training(
        cfg,
        network=network,
        drive=_FakeDrive(network.layout.n_X),
        sampler=_FakeSampler(samples),
        solver=solver,
        time=np.array([0.0, 0.01]),
        show_progress=False,
    )

    assert result.steps == 2
    assert all(module.startswith("jaxlib") for module in seen_weight_modules)
    loaded = load_checkpoint(result.run_dir, "network_final")
    assert sparse.isspmatrix_csr(loaded["weights"])


class _FakeSampler:
    def __init__(self, samples):
        self.samples = samples

    def make_epoch(self, *, limit=None, shuffle_paths=True, shuffle_samples=True):
        return self.samples if limit is None else self.samples[:limit]


class _FakeDrive:
    def __init__(self, n_ext):
        self.n_ext = n_ext

    def make_static_batch_func(self, samples):
        n_batch = len(samples)
        return lambda _t: np.zeros((self.n_ext, n_batch), dtype=float)


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
