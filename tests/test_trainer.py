import numpy as np
import pytest
from scipy import sparse
from types import SimpleNamespace

from v1_simulation.config.schema import JaxSolverConfig, SolverConfig, TrainingBCMConfig
from v1_simulation.network.geometry import SheetGeometry
from v1_simulation.network.state import NetworkState, PopulationLayout
from v1_simulation.solvers.base import BatchODEResult
from v1_simulation.training.trainer import BCMTrainer


def test_bcm_trainer_initialization_and_batch_update() -> None:
    # 1. Setup minimal layout and weights (1 E cell, 1 I cell)
    l23 = SheetGeometry(n_side=2, region_size=2.0, z_pos=0.1)
    l4 = SheetGeometry(n_side=1, region_size=2.0, z_pos=0.0)
    layout = PopulationLayout(
        l23=l23,
        l4=l4,
        l23_types=np.array(["E", "I", "E", "E"]),
        l4_tunings=np.array(["T"]),
        l4_pref_dirs=np.array([0.0]),
    )
    # Weights array shape: (N_l23, N_l23 + N_l4) -> (4, 5)
    weights = np.array([
        [0.5, -0.2, 0.0, 0.0, 1.0],
        [0.1, -0.3, 0.0, 0.0, 0.5],
        [0.0, -0.4, 0.0, 0.0, 0.0],
        [0.2, -0.1, 0.0, 0.0, 0.2],
    ])
    network = NetworkState(
        layout=layout,
        connectivity=sparse.csr_matrix(weights != 0.0),
        weights=sparse.csr_matrix(weights),
    )

    # 2. Setup training config
    config = TrainingBCMConfig(
        epochs=1,
        batch_size=2,
        eta=0.01,
        theta_beta=0.5,
        theta_update_order="post",
        theta_floor=0.1,
        w_max=1.5,
        row_sum_max_scale=1.1,
    )

    trainer = BCMTrainer(config, network)
    assert trainer.state.step == 0
    assert trainer.state.samples_seen == 0
    assert trainer.state.theta is None

    # 3. Setup mock dynamics result
    dynamics = BatchODEResult(
        exc=np.array([[0.5, 0.6, 0.2]]),  # shape: (n_batch, n_E) -> (1, 3)
        inh=np.array([[0.7]]),             # shape: (n_batch, n_I) -> (1, 1)
        exc_trajectory=None,
        inh_trajectory=None,
        time=np.array([0.0, 0.1]),
        exc_convergence=np.array([0.0]),
        inh_convergence=np.array([0.0]),
        steady_state_reached=True,
        steady_state_index=1,
        steady_state_start_index=0,
    )

    # 4. Train one batch
    weight_buffer_before = trainer.state.network.weights
    log = trainer.train_batch(dynamics, epoch=0, batch_size=2, images="img1.iml")

    # 5. Assertions on trainer state changes
    assert trainer.state.step == 1
    assert trainer.state.samples_seen == 2
    assert trainer.state.theta is not None
    assert trainer.state.network.weights is weight_buffer_before
    # Theta arrays should have length of E/I cells respectively
    assert trainer.state.theta.E.shape == (3,)
    assert trainer.state.theta.I.shape == (1,)

    # 6. Assertions on log structure
    assert log.step == 1
    assert log.epoch == 0
    assert log.batch_size == 2
    assert log.samples_seen == 2
    assert log.images == "img1.iml"
    assert log.aE_mean == pytest.approx(np.mean(dynamics.exc))
    assert log.aI_mean == pytest.approx(np.mean(dynamics.inh))
    assert log.aE_active_fraction == 0.0
    assert log.aI_active_fraction == 0.0
    assert log.steady_state_reached == 1
    assert log.t_final == 0.1

    # Verify weight stats dictionary is populated
    assert "W_EE_mean" in log.weight_stats
    assert "W_EE_row_sum_mean" in log.weight_stats
    assert "W_IE_mean" in log.weight_stats

    # Verify theta stats dictionary is populated
    assert "theta_E_median" in log.theta_stats
    assert "theta_I_median" in log.theta_stats
    assert "theta_E_floor_fraction" in log.theta_stats


def test_bcm_trainer_config_validation() -> None:
    l23 = SheetGeometry(n_side=2, region_size=2.0, z_pos=0.1)
    l4 = SheetGeometry(n_side=1, region_size=2.0, z_pos=0.0)
    layout = PopulationLayout(
        l23=l23,
        l4=l4,
        l23_types=np.array(["E", "I", "E", "E"]),
        l4_tunings=np.array(["T"]),
        l4_pref_dirs=np.array([0.0]),
    )
    network = NetworkState(
        layout=layout,
        connectivity=sparse.csr_matrix(np.zeros((4, 5), dtype=bool)),
        weights=sparse.csr_matrix(np.zeros((4, 5))),
    )

    # Bad config update order (throws ValueError)
    bad_config = TrainingBCMConfig(theta_update_order="bad-order")
    with pytest.raises(ValueError, match="theta_update_order"):
        BCMTrainer(bad_config, network)


def test_bcm_trainer_skips_batch_without_steady_state() -> None:
    network = _tiny_trainer_network()
    config = TrainingBCMConfig(max_consecutive_bad_batches=2)
    trainer = BCMTrainer(config, network)

    log = trainer.train_batch(
        BatchODEResult(
            exc=np.array([[0.5, 0.6, 0.2]]),
            inh=np.array([[0.7]]),
            exc_trajectory=None,
            inh_trajectory=None,
            time=np.array([0.0, 0.1]),
            exc_convergence=np.array([0.0]),
            inh_convergence=np.array([0.0]),
            steady_state_reached=False,
            steady_state_index=None,
            steady_state_start_index=None,
        ),
        epoch=0,
        batch_size=1,
    )

    assert log.skipped_bad_batch is True
    assert log.updated == 0
    assert trainer.state.consecutive_bad_batches == 1


def test_bcm_trainer_skips_capped_saturation_fraction() -> None:
    network = _tiny_trainer_network()
    solver_config = SolverConfig()
    solver_config.transfer.rate_max = 100.0
    config = TrainingBCMConfig(
        max_consecutive_bad_batches=2,
        saturation_fraction_threshold=0.05,
        rate_explosion_threshold=None,
    )
    trainer = BCMTrainer(config, network, solver_config=solver_config)

    log = trainer.train_batch(
        BatchODEResult(
            exc=np.array([[100.0, 99.5, 0.2]]),
            inh=np.array([[0.7]]),
            exc_trajectory=None,
            inh_trajectory=None,
            time=np.array([0.0, 0.1]),
            exc_convergence=np.array([0.0]),
            inh_convergence=np.array([0.0]),
            steady_state_reached=True,
            steady_state_index=1,
            steady_state_start_index=0,
            y_diff_max=0.0,
            dy_max=0.0,
        ),
        epoch=0,
        batch_size=1,
    )

    assert log.skipped_bad_batch is True
    assert log.aE_saturation_fraction == pytest.approx(2 / 3)
    assert log.updated == 0


def test_bcm_trainer_allows_high_rate_when_converged_and_not_saturated() -> None:
    network = _tiny_trainer_network()
    solver_config = SolverConfig()
    solver_config.transfer.rate_max = 100.0
    config = TrainingBCMConfig(
        eta=0.0,
        saturation_fraction_threshold=0.05,
        rate_explosion_threshold=None,
    )
    trainer = BCMTrainer(config, network, solver_config=solver_config)

    log = trainer.train_batch(
        BatchODEResult(
            exc=np.array([[90.0, 0.6, 0.2]]),
            inh=np.array([[70.0]]),
            exc_trajectory=None,
            inh_trajectory=None,
            time=np.array([0.0, 0.1]),
            exc_convergence=np.array([0.0]),
            inh_convergence=np.array([0.0]),
            steady_state_reached=True,
            steady_state_index=1,
            steady_state_start_index=0,
            y_diff_max=0.0,
            dy_max=0.0,
        ),
        epoch=0,
        batch_size=1,
    )

    assert log.skipped_bad_batch is False
    assert log.aE_saturation_fraction == 0.0
    assert log.aI_saturation_fraction == 0.0
    assert log.aE_active_fraction == pytest.approx(1 / 3)
    assert log.aE_active_mean == pytest.approx(90.0)
    assert log.aI_active_fraction == 1.0
    assert log.aI_active_mean == pytest.approx(70.0)


def test_bcm_trainer_preclips_initial_plastic_weights() -> None:
    network = _tiny_trainer_network()
    weights = network.weights.toarray()
    idx_e = network.idx_E
    idx_i = network.idx_I
    idx_x = network.idx_X
    weights[idx_e[0], idx_e[0]] = 5.0
    weights[idx_i[0], idx_e[1]] = 4.0
    weights[idx_e[0], idx_x[0]] = 9.0
    network = NetworkState(
        layout=network.layout,
        connectivity=sparse.csr_matrix(weights != 0.0),
        weights=sparse.csr_matrix(weights),
    )

    trainer = BCMTrainer(TrainingBCMConfig(w_max=1.5, clip_initial_weights=True), network)
    clipped = np.asarray(trainer.state.network.weights)

    assert clipped[idx_e[0], idx_e[0]] == pytest.approx(1.5)
    assert clipped[idx_i[0], idx_e[1]] == pytest.approx(1.5)
    assert clipped[idx_e[0], idx_x[0]] == pytest.approx(9.0)
    assert trainer.row_sum_limits.target_E_source_E[0] < 5.0


def test_bcm_trainer_uses_jax_bcm_for_dense_jax_solver() -> None:
    from v1_simulation.solvers.jax_utils import is_jax_available

    if not is_jax_available():
        pytest.skip("JAX not installed")

    network = _tiny_trainer_network()
    config = TrainingBCMConfig(
        epochs=1,
        batch_size=2,
        eta=0.01,
        theta_beta=0.5,
        theta_update_order="pre",
        theta_floor=0.1,
        w_max=1.5,
        row_sum_max_scale=1.1,
    )
    solver_config = SolverConfig(
        backend="jax-rk4",
        method="RK4",
        jax=JaxSolverConfig(prefer_sparse=False, dtype="float32"),
    )
    trainer = BCMTrainer(config, network, solver_config=solver_config)

    assert type(trainer.state.network.weights).__module__.startswith("jaxlib")

    trainer.train_batch(
        BatchODEResult(
            exc=np.array([[0.5, 0.6, 0.2], [0.4, 0.7, 0.3]]),
            inh=np.array([[0.7], [0.8]]),
            exc_trajectory=None,
            inh_trajectory=None,
            time=np.array([0.0, 0.1]),
            exc_convergence=np.array([0.0, 0.0]),
            inh_convergence=np.array([0.0, 0.0]),
            steady_state_reached=True,
            steady_state_index=1,
            steady_state_start_index=0,
        ),
        epoch=0,
        batch_size=2,
    )
    log = trainer.train_batch(
        BatchODEResult(
            exc=np.array([[0.6, 0.5, 0.25], [0.45, 0.75, 0.35]]),
            inh=np.array([[0.75], [0.85]]),
            exc_trajectory=None,
            inh_trajectory=None,
            time=np.array([0.0, 0.1]),
            exc_convergence=np.array([0.0, 0.0]),
            inh_convergence=np.array([0.0, 0.0]),
            steady_state_reached=True,
            steady_state_index=1,
            steady_state_start_index=0,
        ),
        epoch=0,
        batch_size=2,
    )

    assert type(trainer.state.network.weights).__module__.startswith("jaxlib")
    assert log.updated == 1
    assert "W_EE_mean" in log.weight_stats


def _tiny_trainer_network() -> NetworkState:
    l23 = SheetGeometry(n_side=2, region_size=2.0, z_pos=0.1)
    l4 = SheetGeometry(n_side=1, region_size=2.0, z_pos=0.0)
    layout = PopulationLayout(
        l23=l23,
        l4=l4,
        l23_types=np.array(["E", "I", "E", "E"]),
        l4_tunings=np.array(["T"]),
        l4_pref_dirs=np.array([0.0]),
    )
    weights = np.array([
        [0.5, -0.2, 0.1, 0.2, 1.0],
        [0.1, -0.3, 0.2, 0.1, 0.5],
        [0.2, -0.4, 0.3, 0.1, 0.0],
        [0.2, -0.1, 0.1, 0.2, 0.2],
    ])
    return NetworkState(
        layout=layout,
        connectivity=sparse.csr_matrix(weights != 0.0),
        weights=sparse.csr_matrix(weights),
    )
