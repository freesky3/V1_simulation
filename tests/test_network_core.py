import unittest

import numpy as np
from scipy import sparse

from v1_simulation.config import load_config
from v1_simulation.config.schema import RootConfig
from v1_simulation.config.validation import validate_config
from v1_simulation.network import (
    EmpiricalData,
    PopulationCounts,
    PopulationLayout,
    SheetGeometry,
    build_network_spec,
    build_network_state,
    build_population_layout,
    derive_connection_probabilities,
    make_network_rngs,
    probability_block,
    probability_matrix,
)


class NetworkCoreTests(unittest.TestCase):
    def test_yaml_schema_drives_final_counts_before_probability_derivation(self) -> None:
        cfg = load_config(overrides=["+experiment=smoke"])
        empirical = EmpiricalData.from_path(cfg.paths.sample_data_path)
        layout = build_population_layout(cfg.model, empirical, make_network_rngs(cfg.seed).layout)
        spec = build_network_spec(cfg.model, empirical, layout)

        self.assertEqual(layout.l23.n_side, 14)
        self.assertEqual(layout.n_E, 166)
        self.assertEqual(layout.n_I, 30)
        self.assertEqual(layout.n_X, 64)

        final_counts = PopulationCounts(l23_n_side=14, n_e=166, n_i=30, n_x=64)
        expected = derive_connection_probabilities(
            counts=final_counts,
            empirical=empirical,
            p_ee=cfg.model.connectivity.p_ee,
        )
        drifted_counts = PopulationCounts(l23_n_side=14, n_e=171, n_i=25, n_x=64)
        drifted = derive_connection_probabilities(
            counts=drifted_counts,
            empirical=empirical,
            p_ee=cfg.model.connectivity.p_ee,
        )

        self.assertEqual(spec.probabilities, expected)
        self.assertNotAlmostEqual(spec.probabilities.ei, drifted.ei)
        self.assertNotAlmostEqual(spec.probabilities.ii, drifted.ii)

    def test_probability_block_applies_valid_mask_before_row_equalization(self) -> None:
        score = np.ones((4, 4), dtype=float)
        valid = ~np.eye(4, dtype=bool)

        prob = probability_block(score, 0.25, valid_mask=valid, equalize_rows=True)

        self.assertTrue(np.all(prob.diagonal() == 0.0))
        self.assertTrue(np.allclose(prob.sum(axis=1) / valid.sum(axis=1), 0.25))

    def test_probability_matrix_has_no_self_connections_and_row_targets(self) -> None:
        cfg = load_config(overrides=["+experiment=smoke"])
        empirical = EmpiricalData.from_path(cfg.paths.sample_data_path)
        layout = build_population_layout(cfg.model, empirical, make_network_rngs(cfg.seed).layout)
        spec = build_network_spec(cfg.model, empirical, layout)

        probabilities = probability_matrix(layout, spec.kernel, spec.connectivity)

        ee = probabilities[np.ix_(layout.idx_E, layout.idx_E)]
        ii = probabilities[np.ix_(layout.idx_I, layout.idx_I)]
        self.assertTrue(np.all(ee.diagonal() == 0.0))
        self.assertTrue(np.all(ii.diagonal() == 0.0))
        self.assertTrue(np.allclose(ee.sum(axis=1) / (layout.n_E - 1), spec.connectivity.p_ee))
        self.assertTrue(np.allclose(ii.sum(axis=1) / (layout.n_I - 1), spec.connectivity.p_ii))

        ei = probabilities[np.ix_(layout.idx_E, layout.idx_I)]
        ie = probabilities[np.ix_(layout.idx_I, layout.idx_E)]
        ex = probabilities[np.ix_(layout.idx_E, layout.idx_X)]
        ix = probabilities[np.ix_(layout.idx_I, layout.idx_X)]
        self.assertTrue(np.allclose(ei.mean(axis=1), spec.connectivity.p_ei))
        self.assertTrue(np.allclose(ie.mean(axis=1), spec.connectivity.p_ie))
        self.assertTrue(np.allclose(ex.mean(axis=1), spec.connectivity.p_ex))
        self.assertTrue(np.allclose(ix.mean(axis=1), spec.connectivity.p_ix))

    def test_invalid_l23_type_fails_fast(self) -> None:
        l23 = SheetGeometry(2, 1.0, 0.1)
        l4 = SheetGeometry(2, 1.0, 0.0)

        with self.assertRaisesRegex(ValueError, "Unsupported L2/3 cell types"):
            PopulationLayout(l23=l23, l4=l4, l23_types=np.array(["E", "I", "X", "E"]))

    def test_build_network_seed_reproducibility_and_weight_signs(self) -> None:
        cfg = load_config(overrides=["+experiment=smoke"])

        net_a = build_network_state(cfg, seed=7)
        net_b = build_network_state(cfg, seed=7)
        net_c = build_network_state(cfg, seed=8)

        self.assertTrue(sparse.isspmatrix_csr(net_a.connectivity))
        self.assertTrue(sparse.isspmatrix_csr(net_a.weights))
        self.assertEqual((net_a.connectivity != net_b.connectivity).nnz, 0)
        self.assertEqual((net_a.weights != net_b.weights).nnz, 0)
        self.assertGreater((net_a.connectivity != net_c.connectivity).nnz + (net_a.weights != net_c.weights).nnz, 0)

        q = net_a.connectivity.toarray()
        weights = net_a.weights.toarray()
        active_e = weights[:, net_a.idx_E][q[:, net_a.idx_E]]
        active_i = weights[:, net_a.idx_I][q[:, net_a.idx_I]]
        active_x = weights[:, net_a.idx_X][q[:, net_a.idx_X]]

        self.assertTrue(np.all(active_e >= 0.0))
        self.assertTrue(np.all(active_i <= 0.0))
        self.assertTrue(np.all(active_x >= 0.0))

    def test_network_schema_validation_rejects_bad_new_fields(self) -> None:
        bad_n_side = RootConfig()
        bad_n_side.model.layers.l23.n_side = 0
        with self.assertRaisesRegex(ValueError, "model.layers.l23.n_side"):
            validate_config(bad_n_side)

        bad_kappa = RootConfig()
        bad_kappa.model.connectivity.kernel.kappa = 1.5
        with self.assertRaisesRegex(ValueError, "kappa"):
            validate_config(bad_kappa)


if __name__ == "__main__":
    unittest.main()
