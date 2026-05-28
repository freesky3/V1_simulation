import numpy as np
import pytest
from scipy import sparse
from types import SimpleNamespace

from v1_simulation.config.schema import TrainingBCMConfig
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
    log = trainer.train_batch(dynamics, epoch=0, batch_size=2, images="img1.iml")

    # 5. Assertions on trainer state changes
    assert trainer.state.step == 1
    assert trainer.state.samples_seen == 2
    assert trainer.state.theta is not None
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
