import unittest

import numpy as np
from scipy import sparse

from v1_simulation.config import load_config
from v1_simulation.config.schema import RootConfig, TrainingBCMConfig
from v1_simulation.config.validation import validate_config
from v1_simulation.network.geometry import SheetGeometry
from v1_simulation.network.state import NetworkState, PopulationLayout
from v1_simulation.simulation.pipeline import default_training_time_grid
from v1_simulation.training.bcm import BCMThetaState, update_theta
from v1_simulation.training.plasticity import (
    BCMRowSumLimits,
    bcm_training_step,
    make_bcm_efferent_update_index,
    update_efferent_excitatory_weights,
    update_excitatory_block,
)


class BCMPlasticityTests(unittest.TestCase):
    def test_update_theta_returns_new_readonly_state(self) -> None:
        config = TrainingBCMConfig(theta_beta=0.25, theta_floor=None)
        theta = BCMThetaState(E=np.array([1.0, 4.0]), I=np.array([9.0]))

        updated = update_theta(
            theta,
            y_E=np.array([3.0, 1.0]),
            y_I=np.array([2.0]),
            config=config,
        )

        self.assertIsNot(updated, theta)
        self.assertTrue(np.allclose(theta.E, np.array([1.0, 4.0])))
        self.assertTrue(np.allclose(theta.I, np.array([9.0])))
        self.assertTrue(np.allclose(updated.E, np.array([3.0, 3.25])))
        self.assertTrue(np.allclose(updated.I, np.array([7.75])))
        with self.assertRaises(ValueError):
            theta.E[0] = 0.0

    def test_update_excitatory_block_rejects_fractional_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "0/1"):
            update_excitatory_block(
                weights=np.array([[0.5, 0.2]]),
                connection_mask=np.array([[1.0, 0.5]]),
                x=np.array([1.0, 2.0]),
                y=np.array([0.5]),
                theta=np.array([0.25]),
                config=TrainingBCMConfig(eta=0.1),
            )

    def test_dense_and_sparse_updates_are_pure_and_equivalent(self) -> None:
        idx_E = np.array([0, 1])
        idx_I = np.array([2])
        W = np.array(
            [
                [0.5, 0.0, -0.4],
                [0.2, 0.3, -0.1],
                [0.7, 0.0, -0.2],
            ],
            dtype=float,
        )
        topology = W != 0.0
        theta = BCMThetaState(E=np.array([0.01, 0.02]), I=np.array([0.03]))
        config = TrainingBCMConfig(eta=0.01, w_max=None)

        dense = update_efferent_excitatory_weights(
            W,
            topology,
            idx_E,
            idx_I,
            x_E=np.array([[1.0, 2.0], [2.0, 3.0]]),
            y_E=np.array([[0.5, 0.6], [0.4, 0.7]]),
            y_I=np.array([[0.7], [0.8]]),
            theta=theta,
            config=config,
        )

        W_sparse = sparse.csr_matrix(W)
        sparse_updated = update_efferent_excitatory_weights(
            W_sparse,
            sparse.csr_matrix(topology),
            idx_E,
            idx_I,
            x_E=np.array([[1.0, 2.0], [2.0, 3.0]]),
            y_E=np.array([[0.5, 0.6], [0.4, 0.7]]),
            y_I=np.array([[0.7], [0.8]]),
            theta=theta,
            config=config,
        )

        self.assertTrue(sparse.isspmatrix_csr(sparse_updated))
        self.assertIsNot(sparse_updated, W_sparse)
        self.assertTrue(np.allclose(W_sparse.toarray(), W))
        self.assertTrue(np.allclose(sparse_updated.toarray(), dense))
        self.assertEqual(dense[0, 1], 0.0)
        self.assertEqual(dense[2, 1], 0.0)
        self.assertEqual(dense[0, 2], W[0, 2])

    def test_dense_update_can_mutate_in_place_for_training_hot_path(self) -> None:
        idx_E = np.array([0, 1])
        idx_I = np.array([2])
        W = np.array(
            [
                [0.5, 0.2, -0.4],
                [0.2, 0.3, -0.1],
                [0.7, 0.4, -0.2],
            ],
            dtype=float,
        )
        topology = W != 0.0
        theta = BCMThetaState(E=np.array([0.01, 0.02]), I=np.array([0.03]))

        updated = update_efferent_excitatory_weights(
            W,
            topology,
            idx_E,
            idx_I,
            x_E=np.array([[1.0, 2.0], [2.0, 3.0]]),
            y_E=np.array([[0.5, 0.6], [0.4, 0.7]]),
            y_I=np.array([[0.7], [0.8]]),
            theta=theta,
            config=TrainingBCMConfig(eta=0.01, w_max=None),
            _cached_topology=topology,
            copy_weights=False,
        )

        self.assertIs(updated, W)
        self.assertTrue(np.allclose(W, updated))

    def test_in_place_edge_update_matches_pure_dense_update(self) -> None:
        idx_E = np.array([0, 1])
        idx_I = np.array([2])
        W = np.array(
            [
                [0.5, 0.2, -0.4],
                [0.2, 0.3, -0.1],
                [0.7, 0.4, -0.2],
            ],
            dtype=float,
        )
        topology = W != 0.0
        theta = BCMThetaState(E=np.array([0.01, 0.02]), I=np.array([0.03]))
        config = TrainingBCMConfig(eta=0.01, w_max=1.0, row_sum_max_scale=None)
        row_sum_max_E = np.array([0.65, 0.45])
        row_sum_max_I = np.array([0.8])

        expected = update_efferent_excitatory_weights(
            W,
            topology,
            idx_E,
            idx_I,
            x_E=np.array([[1.0, 2.0], [2.0, 3.0]]),
            y_E=np.array([[0.5, 0.6], [0.4, 0.7]]),
            y_I=np.array([[0.7], [0.8]]),
            theta=theta,
            config=config,
            row_sum_max_E=row_sum_max_E,
            row_sum_max_I=row_sum_max_I,
        )

        actual_input = W.copy()
        actual = update_efferent_excitatory_weights(
            actual_input,
            topology,
            idx_E,
            idx_I,
            x_E=np.array([[1.0, 2.0], [2.0, 3.0]]),
            y_E=np.array([[0.5, 0.6], [0.4, 0.7]]),
            y_I=np.array([[0.7], [0.8]]),
            theta=theta,
            config=config,
            row_sum_max_E=row_sum_max_E,
            row_sum_max_I=row_sum_max_I,
            _cached_topology=topology,
            copy_weights=False,
            update_index=make_bcm_efferent_update_index(topology, idx_E, idx_I),
        )

        self.assertIs(actual, actual_input)
        self.assertTrue(np.allclose(actual, expected))

    def test_jax_bcm_training_step_matches_numpy_edge_update(self) -> None:
        from v1_simulation.solvers.jax_utils import is_jax_available

        if not is_jax_available():
            self.skipTest("JAX not installed")
        import jax

        jax.config.update("jax_enable_x64", True)
        from v1_simulation.training.jax_bcm import JAXBCMUpdater

        network = _tiny_network_for_bcm()
        topology = network.connectivity.toarray().astype(bool)
        update_index = make_bcm_efferent_update_index(topology, network.idx_E, network.idx_I)
        limits = BCMRowSumLimits(
            target_E_source_E=np.array([0.8, 0.75]),
            target_I_source_E=np.array([0.9, 0.85]),
        )
        theta = BCMThetaState(E=np.array([0.01, 0.02]), I=np.array([0.03, 0.04]))
        config = TrainingBCMConfig(
            eta=0.01,
            theta_beta=0.5,
            theta_update_order="pre",
            theta_floor=0.001,
            w_max=1.0,
        )
        x_E = np.array([[1.0, 2.0], [2.0, 3.0]])
        y_E = np.array([[0.5, 0.6], [0.4, 0.7]])
        y_I = np.array([[0.7, 0.8], [0.8, 0.9]])

        numpy_network = NetworkState(
            layout=network.layout,
            connectivity=network.connectivity,
            weights=network.weights.toarray(),
            source=network.source,
        )
        expected = bcm_training_step(
            network=numpy_network,
            x_E=x_E,
            y_E=y_E,
            y_I=y_I,
            theta=theta,
            config=config,
            row_sum_limits=limits,
            _cached_topology=topology,
            copy_weights=False,
            update_index=update_index,
        )

        updater = JAXBCMUpdater(
            config=config,
            row_sum_limits=limits,
            update_index=update_index,
            dtype="float64",
        )
        actual = updater.training_step(
            network=updater.to_device_network(network),
            x_E=x_E,
            y_E=y_E,
            y_I=y_I,
            theta=theta,
        )

        self.assertTrue(actual.updated)
        self.assertTrue(np.allclose(np.asarray(actual.network.weights), expected.network.weights))
        self.assertTrue(np.allclose(actual.theta.E, expected.theta.E))
        self.assertTrue(np.allclose(actual.theta.I, expected.theta.I))
        self.assertTrue(np.allclose(actual.theta_for_update.E, expected.theta_for_update.E))
        self.assertTrue(np.allclose(actual.theta_for_update.I, expected.theta_for_update.I))

    def test_training_step_post_order_uses_old_theta_for_weights(self) -> None:
        network = _tiny_network_for_bcm()
        theta = BCMThetaState(E=np.array([1.0, 1.0]), I=np.array([1.0, 1.0]))
        config = TrainingBCMConfig(
            eta=0.0,
            theta_beta=1.0,
            theta_update_order="post",
            theta_floor=None,
        )

        result = bcm_training_step(
            network=network,
            x_E=np.array([1.0, 2.0]),
            y_E=np.array([3.0, 4.0]),
            y_I=np.array([5.0, 6.0]),
            theta=theta,
            config=config,
        )

        self.assertTrue(result.updated)
        self.assertTrue(np.allclose(result.theta_for_update.E, np.array([1.0, 1.0])))
        self.assertTrue(np.allclose(result.theta.E, np.array([9.0, 16.0])))
        self.assertTrue(np.allclose(result.theta.I, np.array([25.0, 36.0])))
        self.assertTrue(np.allclose(theta.E, np.array([1.0, 1.0])))

    def test_yaml_bcm_config_loads_into_schema_used_by_plasticity(self) -> None:
        cfg = load_config(overrides=["+experiment=bcm_train"])

        self.assertEqual(cfg.mode, "train")
        self.assertTrue(cfg.training.enabled)
        self.assertEqual(cfg.training.bcm.theta_update_order, "pre")
        self.assertEqual(cfg.training.bcm.duration_tau_e, 30.0)
        self.assertEqual(cfg.training.bcm.dt_tau_i_fraction, 1.0 / 3.0)

        # 1. Test invalid theta_update_order
        bad = RootConfig()
        bad.training.enabled = True
        bad.training.natural_image.dir = "data/vanhateren_iml"
        bad.training.bcm.theta_update_order = "middle"
        with self.assertRaisesRegex(ValueError, "theta_update_order"):
            validate_config(bad)

        # 2. Test invalid duration_tau_e (<= 0.0)
        bad_duration = RootConfig()
        bad_duration.training.enabled = True
        bad_duration.training.natural_image.dir = "data/vanhateren_iml"
        bad_duration.training.bcm.duration_tau_e = 0.0
        with self.assertRaisesRegex(ValueError, "training.bcm.duration_tau_e must be positive"):
            validate_config(bad_duration)

        bad_duration.training.bcm.duration_tau_e = -5.0
        with self.assertRaisesRegex(ValueError, "training.bcm.duration_tau_e must be positive"):
            validate_config(bad_duration)

        bad_dt_fraction = RootConfig()
        bad_dt_fraction.training.enabled = True
        bad_dt_fraction.training.natural_image.dir = "data/vanhateren_iml"
        bad_dt_fraction.training.bcm.dt_tau_i_fraction = 0.0
        with self.assertRaisesRegex(ValueError, "training.bcm.dt_tau_i_fraction must be positive"):
            validate_config(bad_dt_fraction)

    def test_training_time_grid_uses_bcm_schema_dt_fraction(self) -> None:
        cfg = RootConfig()
        cfg.training.bcm.duration_tau_e = 2.0
        cfg.training.bcm.dt_tau_i_fraction = 0.5
        cfg.solver.transfer.tau_e = 0.02
        cfg.solver.transfer.tau_i = 0.01

        time = default_training_time_grid(cfg)

        np.testing.assert_allclose(time[:3], [0.0, 0.005, 0.01])
        self.assertLess(time[-1], 0.04)



def _tiny_network_for_bcm() -> NetworkState:
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


if __name__ == "__main__":
    unittest.main()
