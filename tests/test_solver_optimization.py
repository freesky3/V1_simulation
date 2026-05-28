import unittest
import numpy as np
from scipy import sparse

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None


from v1_simulation.config.schema import SolverConfig, TransferConfig
from v1_simulation.network.state import NetworkState, PopulationLayout
from v1_simulation.network.geometry import SheetGeometry
from v1_simulation.solvers import solve_wilson_cowan_batch
from v1_simulation.solvers.base import NetworkLayout, SolverOptions
from v1_simulation.solvers.jax_backend import (
    is_diffrax_available,
    is_jax_available,
    _make_diffrax_diffeqsolve,
    _make_jax_rk4,
    _prepare_jax_matrix,
    _precompute_diffrax_inputs,
    _precompute_diffrax_background,
    _transfer_table_arrays,
)
from v1_simulation.transfer.siegert import TransferTable


class SolverOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        if not is_jax_available():
            self.skipTest("JAX not installed")
        
        # Enable 64-bit precision in JAX for strict equivalence testing
        jax.config.update("jax_enable_x64", True)

        # Create a small network
        self.l23 = SheetGeometry(2, 1.0, 0.1)
        self.l4 = SheetGeometry(1, 1.0, 0.0)
        self.layout = PopulationLayout(
            l23=self.l23,
            l4=self.l4,
            l23_types=np.array(["E", "I", "E", "E"]),
            l4_tunings=np.array(["T"]),
            l4_pref_dirs=np.array([0.0]),
        )
        # Create dense random weights to ensure no structural zeros mask indexing bugs
        np.random.seed(42)
        self.raw_weights = np.random.uniform(0.1, 1.0, self.layout.shape)
        self.network = NetworkState(
            layout=self.layout,
            connectivity=sparse.csr_matrix(self.raw_weights != 0.0),
            weights=sparse.csr_matrix(self.raw_weights),
        )
        self.time = np.linspace(0.0, 0.02, 5)
        self.phi_table = TransferTable(
            np.linspace(-100.0, 100.0, 1000),
            np.maximum(np.linspace(-100.0, 100.0, 1000), 0.0)
        )
        self.n_batch = 2
        
    def test_jax_rk4_static_vs_dynamic_equivalence(self) -> None:
        # Run solver with static config
        res_static = solve_wilson_cowan_batch(
            network=self.network,
            external_drive=lambda _t: np.array([[1.0, 2.0]]),
            time=self.time,
            n_batch=self.n_batch,
            solver_config=SolverConfig(backend="jax-rk4", method="RK4"),
            transfer_config=TransferConfig(tau_e=0.02, tau_i=0.01),
            phi_exc=self.phi_table,
            phi_inh=self.phi_table,
        )

        # Run solver with slightly time-varying drive that is numerically identical for testing
        res_dynamic = solve_wilson_cowan_batch(
            network=self.network,
            # Non-static drive function to force dynamic compilation branch
            external_drive=lambda t: np.array([[1.0 + 0.0 * t, 2.0 + 0.0 * t]]),
            time=self.time,
            n_batch=self.n_batch,
            solver_config=SolverConfig(backend="jax-rk4", method="RK4"),
            transfer_config=TransferConfig(tau_e=0.02, tau_i=0.01),
            phi_exc=self.phi_table,
            phi_inh=self.phi_table,
        )

        np.testing.assert_allclose(res_static.exc, res_dynamic.exc, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res_static.inh, res_dynamic.inh, rtol=1e-12, atol=1e-12)

    def test_diffrax_static_vs_dynamic_equivalence(self) -> None:
        if not is_diffrax_available():
            self.skipTest("Diffrax not installed")

        # Run solver with static config
        res_static = solve_wilson_cowan_batch(
            network=self.network,
            external_drive=lambda _t: np.array([[1.0, 2.0]]),
            time=self.time,
            n_batch=self.n_batch,
            solver_config=SolverConfig(backend="diffrax", method="adaptive"),
            transfer_config=TransferConfig(tau_e=0.02, tau_i=0.01),
            phi_exc=self.phi_table,
            phi_inh=self.phi_table,
        )

        # Run solver with slightly time-varying drive
        res_dynamic = solve_wilson_cowan_batch(
            network=self.network,
            external_drive=lambda t: np.array([[1.0 + 0.0 * t, 2.0 + 0.0 * t]]),
            time=self.time,
            n_batch=self.n_batch,
            solver_config=SolverConfig(backend="diffrax", method="adaptive"),
            transfer_config=TransferConfig(tau_e=0.02, tau_i=0.01),
            phi_exc=self.phi_table,
            phi_inh=self.phi_table,
        )

        np.testing.assert_allclose(res_static.exc, res_dynamic.exc, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res_static.inh, res_dynamic.inh, rtol=1e-12, atol=1e-12)

    def test_gradients_equivalence(self) -> None:
        # Build RK4 runners directly to trace gradients
        y0 = jnp.zeros((self.layout.idx_E.size + self.layout.idx_I.size, self.n_batch), dtype=jnp.float64)
        weights_jax = _prepare_jax_matrix(self.network.weights, jnp, prefer_sparse=False, dense_max_mb=128.0)

        # Precompute inputs
        from v1_simulation.solvers.jax_backend import _precompute_rk4_inputs, _precompute_rk4_background
        ax_left, ax_mid, ax_right = _precompute_rk4_inputs(
            lambda _t: np.array([[1.0, 2.0]]), self.time, n_ext=self.layout.n_X.size, n_batch=self.n_batch
        )
        bg_left_e, bg_mid_e, bg_right_e, bg_left_i, bg_mid_i, bg_right_i = _precompute_rk4_background(
            None, layout=NetworkLayout.from_network_state(self.network), n_batch=self.n_batch, time=self.time
        )
        phi_exc_x, phi_exc_y = _transfer_table_arrays(self.phi_table, "phi_exc")
        phi_inh_x, phi_inh_y = _transfer_table_arrays(self.phi_table, "phi_inh")

        # Compile solvers
        run_static = _make_jax_rk4(jax, jnp, store_trajectory=True, is_static=True)
        run_dynamic = _make_jax_rk4(jax, jnp, store_trajectory=True, is_static=False)

        # Compute gradient functions
        def loss_static(w):
            out = run_static(
                y0, w,
                jnp.asarray(self.layout.idx_E, dtype=jnp.int32),
                jnp.asarray(self.layout.idx_I, dtype=jnp.int32),
                jnp.asarray(self.layout.idx_X, dtype=jnp.int32),
                jnp.asarray(self.time),
                jnp.asarray(ax_left), jnp.asarray(ax_mid), jnp.asarray(ax_right),
                jnp.asarray(bg_left_e), jnp.asarray(bg_mid_e), jnp.asarray(bg_right_e),
                jnp.asarray(bg_left_i), jnp.asarray(bg_mid_i), jnp.asarray(bg_right_i),
                jnp.asarray(phi_exc_x), jnp.asarray(phi_exc_y),
                jnp.asarray(phi_inh_x), jnp.asarray(phi_inh_y),
                jnp.asarray(0.02), jnp.asarray(0.01)
            )
            return jnp.sum(out)

        def loss_dynamic(w):
            out = run_dynamic(
                y0, w,
                jnp.asarray(self.layout.idx_E, dtype=jnp.int32),
                jnp.asarray(self.layout.idx_I, dtype=jnp.int32),
                jnp.asarray(self.layout.idx_X, dtype=jnp.int32),
                jnp.asarray(self.time),
                jnp.asarray(ax_left), jnp.asarray(ax_mid), jnp.asarray(ax_right),
                jnp.asarray(bg_left_e), jnp.asarray(bg_mid_e), jnp.asarray(bg_right_e),
                jnp.asarray(bg_left_i), jnp.asarray(bg_mid_i), jnp.asarray(bg_right_i),
                jnp.asarray(phi_exc_x), jnp.asarray(phi_exc_y),
                jnp.asarray(phi_inh_x), jnp.asarray(phi_inh_y),
                jnp.asarray(0.02), jnp.asarray(0.01)
            )
            return jnp.sum(out)

        grad_static = jax.grad(loss_static)(weights_jax)
        grad_dynamic = jax.grad(loss_dynamic)(weights_jax)

        np.testing.assert_allclose(grad_static, grad_dynamic, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
