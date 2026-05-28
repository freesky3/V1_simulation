import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import sparse

from v1_simulation.config import load_config
from v1_simulation.config.schema import RootConfig, SolverConfig, TransferConfig
from v1_simulation.config.validation import validate_config
from v1_simulation.network.state import NetworkState, PopulationLayout
from v1_simulation.network.geometry import SheetGeometry
from v1_simulation.solvers import solve_wilson_cowan_batch
from v1_simulation.solvers.base import NetworkLayout, SolverOptions
from v1_simulation.solvers.scipy_backend import solve_scipy
from v1_simulation.solvers.wilson_cowan import WilsonCowanRHS


class SolverConfigTests(unittest.TestCase):
    def test_yaml_solver_groups_match_schema_contract(self) -> None:
        jax_cfg = load_config(overrides=["solver=jax"])
        self.assertEqual(jax_cfg.solver.backend, "jax-rk4")
        self.assertEqual(jax_cfg.solver.method, "RK4")

        diffrax_cfg = load_config(overrides=["solver=diffrax_tsit5"])
        self.assertEqual(diffrax_cfg.solver.backend, "diffrax")
        self.assertEqual(diffrax_cfg.solver.method, "adaptive")

        smoke_cfg = load_config(overrides=["+experiment=smoke"])
        self.assertEqual(smoke_cfg.solver.transfer.mu_tab_max, 2.0)
        self.assertEqual(smoke_cfg.transfer.mu_tab_max, 100.0)

    def test_backend_method_mismatch_fails_fast(self) -> None:
        bad_jax = RootConfig()
        bad_jax.solver.backend = "jax-rk4"
        bad_jax.solver.method = "RK45"
        with self.assertRaisesRegex(ValueError, "jax-rk4"):
            validate_config(bad_jax)

        bad_diffrax = RootConfig()
        bad_diffrax.solver.backend = "diffrax"
        bad_diffrax.solver.method = "RK4"
        with self.assertRaisesRegex(ValueError, "diffrax"):
            validate_config(bad_diffrax)

    def test_solver_options_do_not_inherit_training_steady_state_unless_requested(self) -> None:
        cfg = RootConfig()

        simulate_options = SolverOptions.from_config(cfg.solver)
        training_options = SolverOptions.from_config(cfg.solver, training_bcm=cfg.training.bcm)

        self.assertFalse(simulate_options.stop_at_steady_state)
        self.assertTrue(training_options.stop_at_steady_state)


class WilsonCowanSolverTests(unittest.TestCase):
    def test_scipy_rk4_solves_batched_rates_with_schema_config(self) -> None:
        network = _tiny_network()
        time = np.linspace(0.0, 0.05, 11)

        result = solve_wilson_cowan_batch(
            network=network,
            external_drive=lambda _t: np.array([[1.0, 2.0]]),
            time=time,
            n_batch=2,
            solver_config=SolverConfig(backend="scipy", method="RK4"),
            transfer_config=TransferConfig(tau_e=0.02, tau_i=0.01),
            phi_exc=lambda x: np.maximum(x, 0.0),
            phi_inh=lambda x: np.maximum(x, 0.0),
        )

        self.assertEqual(result.exc.shape, (2, 3))
        self.assertEqual(result.inh.shape, (2, 1))
        self.assertEqual(result.exc_trajectory.shape, (11, 2, 3))
        self.assertEqual(result.inh_trajectory.shape, (11, 2, 1))
        self.assertTrue(np.all(result.exc[1] > result.exc[0]))
        self.assertTrue(np.all(result.exc_convergence >= 0.0))

    def test_scipy_solve_ivp_failure_raises_runtime_error(self) -> None:
        network = _tiny_network()
        layout = NetworkLayout.from_network_state(network)
        rhs = WilsonCowanRHS(
            weights=network.weights,
            layout=layout,
            phi_exc=lambda x: x,
            phi_inh=lambda x: x,
            tau_exc=0.02,
            tau_inh=0.01,
            n_batch=1,
        )

        with patch(
            "v1_simulation.solvers.scipy_backend.solve_ivp",
            return_value=SimpleNamespace(success=False, message="controlled failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "controlled failure"):
                solve_scipy(
                    rhs,
                    lambda _t: np.array([1.0]),
                    layout,
                    1,
                    np.array([0.0, 0.01]),
                    SolverOptions(backend="scipy", method="RK45"),
                )


class DiffraxSolverTests(unittest.TestCase):
    def test_missing_optional_dependency_raises_runtime_error(self) -> None:
        network = _tiny_network()
        with patch("importlib.import_module", side_effect=ModuleNotFoundError("No module named 'diffrax'")):
            with self.assertRaisesRegex(RuntimeError, "requires installing the optional JAX dependencies"):
                from v1_simulation.solvers.jax_backend import _require_diffrax
                _require_diffrax()

    def test_diffrax_shape_and_finite_smoke_test(self) -> None:
        from v1_simulation.solvers.jax_backend import is_diffrax_available
        if not is_diffrax_available():
            self.skipTest("Diffrax not installed")

        network = _tiny_network()
        time = np.linspace(0.0, 0.05, 11)

        result = solve_wilson_cowan_batch(
            network=network,
            external_drive=lambda _t: np.array([[1.0, 2.0]]),
            time=time,
            n_batch=2,
            solver_config=SolverConfig(backend="diffrax", method="adaptive"),
            transfer_config=TransferConfig(tau_e=0.02, tau_i=0.01),
            phi_exc=lambda x: np.maximum(x, 0.0),
            phi_inh=lambda x: np.maximum(x, 0.0),
        )

        self.assertEqual(result.exc.shape, (2, 3))
        self.assertEqual(result.inh.shape, (2, 1))
        self.assertEqual(result.exc_trajectory.shape, (11, 2, 3))
        self.assertEqual(result.inh_trajectory.shape, (11, 2, 1))
        
        self.assertTrue(np.isfinite(result.exc).all())
        self.assertTrue(np.isfinite(result.inh).all())

    def test_diffrax_determinism(self) -> None:
        from v1_simulation.solvers.jax_backend import is_diffrax_available
        if not is_diffrax_available():
            self.skipTest("Diffrax not installed")

        network = _tiny_network()
        time = np.linspace(0.0, 0.02, 5)

        def run():
            return solve_wilson_cowan_batch(
                network=network,
                external_drive=lambda _t: np.array([[1.0, 2.0]]),
                time=time,
                n_batch=2,
                solver_config=SolverConfig(backend="diffrax", method="adaptive"),
                transfer_config=TransferConfig(tau_e=0.02, tau_i=0.01),
                phi_exc=lambda x: np.maximum(x, 0.0),
                phi_inh=lambda x: np.maximum(x, 0.0),
            )

        res1 = run()
        res2 = run()

        np.testing.assert_allclose(res1.exc, res2.exc)
        np.testing.assert_allclose(res1.inh, res2.inh)

    def test_diffrax_loose_comparison_with_scipy(self) -> None:
        from v1_simulation.solvers.jax_backend import is_diffrax_available
        if not is_diffrax_available():
            self.skipTest("Diffrax not installed")

        network = _tiny_network()
        time = np.linspace(0.0, 0.02, 100)

        diffrax_res = solve_wilson_cowan_batch(
            network=network,
            external_drive=lambda _t: np.array([[1.0, 2.0]]),
            time=time,
            n_batch=2,
            solver_config=SolverConfig(backend="diffrax", method="adaptive"),
            transfer_config=TransferConfig(tau_e=0.02, tau_i=0.01),
            phi_exc=lambda x: np.maximum(x, 0.0),
            phi_inh=lambda x: np.maximum(x, 0.0),
        )

        scipy_res = solve_wilson_cowan_batch(
            network=network,
            external_drive=lambda _t: np.array([[1.0, 2.0]]),
            time=time,
            n_batch=2,
            solver_config=SolverConfig(backend="scipy", method="RK4"),
            transfer_config=TransferConfig(tau_e=0.02, tau_i=0.01),
            phi_exc=lambda x: np.maximum(x, 0.0),
            phi_inh=lambda x: np.maximum(x, 0.0),
        )

        # Loose comparison since inputs are linearly interpolated differently
        np.testing.assert_allclose(
            diffrax_res.exc,
            scipy_res.exc,
            rtol=1e-1,
            atol=1e-2,
        )


def _tiny_network() -> NetworkState:
    l23 = SheetGeometry(2, 1.0, 0.1)
    l4 = SheetGeometry(1, 1.0, 0.0)
    layout = PopulationLayout(
        l23=l23,
        l4=l4,
        l23_types=np.array(["E", "I", "E", "E"]),
        l4_tunings=np.array(["T"]),
        l4_pref_dirs=np.array([0.0]),
    )
    weights = np.zeros(layout.shape, dtype=float)
    weights[:, layout.idx_X[0]] = 50.0
    return NetworkState(
        layout=layout,
        connectivity=sparse.csr_matrix(weights != 0.0),
        weights=sparse.csr_matrix(weights),
    )


if __name__ == "__main__":
    unittest.main()
