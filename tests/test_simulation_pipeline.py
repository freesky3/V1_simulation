import json
from dataclasses import asdict

import numpy as np
from scipy import sparse

from v1_simulation.config.schema import RootConfig
from v1_simulation.io.artifacts import SimulationArtifacts, json_ready
from v1_simulation.network import build_network_state
from v1_simulation.network.geometry import SheetGeometry
from v1_simulation.network.state import NetworkState, PopulationLayout
from v1_simulation.simulation import (
    default_simulation_time_grid,
    run_drifting_grating_pipeline,
    run_simulation,
)
from v1_simulation.solvers.base import BatchODEResult


def test_drifting_grating_pipeline_uses_schema_fields_and_returns_result_object():
    cfg = _tiny_cfg()
    network = _tiny_network()
    calls = {}
    time = np.array([0.0, 0.01, 0.02])

    result = run_drifting_grating_pipeline(
        cfg,
        network=network,
        time=time,
        solver=_fake_solver(calls),
    )

    assert calls["solver_config"] is cfg.solver
    assert calls["transfer_config"] is cfg.solver.transfer
    assert calls["n_batch"] == cfg.stimulus.n_theta
    assert calls["drive_shape"] == (network.layout.n_X, cfg.stimulus.n_theta)
    assert result.theta_angles.shape == (cfg.stimulus.n_theta,)
    assert result.exc_responses.shape == (network.layout.n_E, cfg.stimulus.n_theta, time.size)
    assert result.aE_all.shape == (cfg.stimulus.n_theta, network.layout.n_E, time.size)
    assert result.analysis_inputs().responses.shape == result.exc_responses.shape


def test_simulation_runner_writes_artifacts_without_pipeline_path_literals(tmp_path):
    cfg = _tiny_cfg()
    cfg.paths.run_root = tmp_path / "runs"
    network = _tiny_network()
    artifacts = SimulationArtifacts(tmp_path / "run")

    saved = run_simulation(
        cfg,
        network=network,
        time=np.array([0.0, 0.01, 0.02]),
        artifacts=artifacts,
        solver=_fake_solver({}),
    )

    assert saved.run_dir == artifacts.run_dir
    assert (saved.run_dir / "responses_exc.npy").exists()
    assert (saved.run_dir / "aE_all.npy").exists()
    assert (saved.run_dir / "run_config.json").exists()
    assert (saved.run_dir / "network" / "weights.npz").exists()
    assert np.load(saved.run_dir / "responses_exc.npy").shape == saved.result.exc_responses.shape


def test_default_simulation_time_grid_comes_from_simulation_schema():
    cfg = _tiny_cfg()
    cfg.solver.transfer.tau_e = 0.02
    cfg.solver.transfer.tau_i = 0.012
    cfg.simulation.duration_tau_e = 2.0
    cfg.simulation.dt_tau_i_fraction = 0.5

    time = default_simulation_time_grid(cfg)

    assert time[0] == 0.0
    assert np.isclose(time[1] - time[0], 0.006)
    assert time[-1] < 0.04


def test_network_builder_loads_trained_run_dir_and_records_source(tmp_path):
    from v1_simulation.training.checkpoints import save_checkpoint

    training_cfg = RootConfig()
    network = _tiny_network()
    save_checkpoint(tmp_path, "network_final", network, metadata={"step": 1})
    (tmp_path / "run_config.json").write_text(
        json.dumps(json_ready({"config": asdict(training_cfg)})),
        encoding="utf-8",
    )

    cfg = RootConfig()
    cfg.model.trained_network_path = str(tmp_path)
    cfg.model.connectivity.scales.ee = 2.0
    loaded = build_network_state(cfg)

    ee_block = loaded.weights.toarray()[np.ix_(loaded.idx_E, loaded.idx_E)]
    assert np.allclose(ee_block, 2.0)
    assert loaded.source["mode"] == "trained"
    assert loaded.source["path"].endswith("network_final")
    assert loaded.source["scale_ratios"]["EE"] == 2.0


def _fake_solver(calls):
    def solver(**kwargs):
        network = kwargs["network"]
        time = np.asarray(kwargs["time"], dtype=float)
        n_batch = int(kwargs["n_batch"])
        drive = np.asarray(kwargs["external_drive"](0.0), dtype=float)
        calls.update(
            solver_config=kwargs["solver_config"],
            transfer_config=kwargs["transfer_config"],
            n_batch=n_batch,
            drive_shape=drive.shape,
        )
        n_exc = network.layout.n_E
        n_inh = network.layout.n_I
        exc_t = np.ones((time.size, n_batch, n_exc), dtype=float)
        inh_t = np.full((time.size, n_batch, n_inh), 0.5, dtype=float)
        return BatchODEResult(
            exc=exc_t[-1],
            inh=inh_t[-1],
            exc_trajectory=exc_t,
            inh_trajectory=inh_t,
            time=time,
            exc_convergence=np.zeros(n_batch),
            inh_convergence=np.zeros(n_batch),
        )

    return solver


def _tiny_cfg() -> RootConfig:
    cfg = RootConfig()
    cfg.job_name = "simulation_test"
    cfg.stimulus.kind = "drifting_grating"
    cfg.stimulus.n_theta = 3
    cfg.stimulus.resolution = 8
    cfg.stimulus.visual_gain = 10.0
    cfg.simulation.store_trajectory = True
    return cfg


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
    weights = np.ones(layout.shape, dtype=float)
    weights[:, layout.idx_I] *= -1.0
    return NetworkState(
        layout=layout,
        connectivity=sparse.csr_matrix(weights != 0.0),
        weights=sparse.csr_matrix(weights),
        source={"mode": "test", "path": None},
    )
