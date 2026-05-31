import unittest
import numpy as np
from scipy import sparse

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None


from v1_simulation.config.schema import SolverConfig, TransferConfig, JaxSolverConfig
from v1_simulation.network.state import NetworkState, PopulationLayout
from v1_simulation.network.geometry import SheetGeometry
from v1_simulation.solvers import solve_wilson_cowan_batch
from v1_simulation.solvers.base import NetworkLayout, SolverOptions
from v1_simulation.solvers.jax_inputs import precompute_rk4_background, precompute_rk4_inputs
from v1_simulation.solvers.jax_rk4_backend import make_jax_rk4
from v1_simulation.solvers.jax_utils import (
    is_diffrax_available,
    is_jax_available,
    slice_weight_blocks,
    transfer_table_arrays,
)
from v1_simulation.transfer.siegert import TransferTable


def _make_test_network():
    """Helper: build a small network for testing."""
    l23 = SheetGeometry(2, 1.0, 0.1)
    l4 = SheetGeometry(1, 1.0, 0.0)
    layout = PopulationLayout(
        l23=l23,
        l4=l4,
        l23_types=np.array(["E", "I", "E", "E"]),
        l4_tunings=np.array(["T"]),
        l4_pref_dirs=np.array([0.0]),
    )
    np.random.seed(42)
    raw_weights = np.random.uniform(0.1, 1.0, layout.shape)
    network = NetworkState(
        layout=layout,
        connectivity=sparse.csr_matrix(raw_weights != 0.0),
        weights=sparse.csr_matrix(raw_weights),
    )
    return network, layout


def _dense_solver_config():
    """SolverConfig with sparse disabled so tests use dense JAX arrays."""
    return SolverConfig(
        backend="jax-rk4",
        method="RK4",
        jax=JaxSolverConfig(prefer_sparse=False),
    )


def _dense_diffrax_config():
    """SolverConfig with sparse disabled for diffrax tests."""
    return SolverConfig(
        backend="diffrax",
        method="adaptive",
        jax=JaxSolverConfig(prefer_sparse=False),
    )


class SolverOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        if not is_jax_available():
            self.skipTest("JAX not installed")

        # Enable 64-bit precision in JAX for strict equivalence testing
        jax.config.update("jax_enable_x64", True)

        self.network, self.pop_layout = _make_test_network()
        self.net_layout = NetworkLayout.from_network_state(self.network)
        self.time = np.linspace(0.0, 0.02, 5)
        self.phi_table = TransferTable(
            np.linspace(-100.0, 100.0, 1000),
            np.maximum(np.linspace(-100.0, 100.0, 1000), 0.0),
        )
        self.n_batch = 2

    # ------------------------------------------------------------------
    # Test 1: JAX-RK4 static vs dynamic branch equivalence
    # ------------------------------------------------------------------
    def test_jax_rk4_static_vs_dynamic_equivalence(self) -> None:
        transfer_cfg = TransferConfig(tau_e=0.02, tau_i=0.01)

        res_static = solve_wilson_cowan_batch(
            network=self.network,
            external_drive=lambda _t: np.array([[1.0, 2.0]]),
            time=self.time,
            n_batch=self.n_batch,
            solver_config=_dense_solver_config(),
            transfer_config=transfer_cfg,
            phi_exc=self.phi_table,
            phi_inh=self.phi_table,
        )

        # Non-static drive: same values but `is_static` detection returns False
        res_dynamic = solve_wilson_cowan_batch(
            network=self.network,
            external_drive=lambda t: np.array([[1.0 + 0.0 * t, 2.0 + 0.0 * t]]),
            time=self.time,
            n_batch=self.n_batch,
            solver_config=_dense_solver_config(),
            transfer_config=transfer_cfg,
            phi_exc=self.phi_table,
            phi_inh=self.phi_table,
        )

        np.testing.assert_allclose(res_static.exc, res_dynamic.exc, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res_static.inh, res_dynamic.inh, rtol=1e-12, atol=1e-12)

    # ------------------------------------------------------------------
    # Test 2: Diffrax static vs dynamic branch equivalence
    # ------------------------------------------------------------------
    def test_diffrax_static_vs_dynamic_equivalence(self) -> None:
        if not is_diffrax_available():
            self.skipTest("Diffrax not installed")

        transfer_cfg = TransferConfig(tau_e=0.02, tau_i=0.01)

        res_static = solve_wilson_cowan_batch(
            network=self.network,
            external_drive=lambda _t: np.array([[1.0, 2.0]]),
            time=self.time,
            n_batch=self.n_batch,
            solver_config=_dense_diffrax_config(),
            transfer_config=transfer_cfg,
            phi_exc=self.phi_table,
            phi_inh=self.phi_table,
        )

        res_dynamic = solve_wilson_cowan_batch(
            network=self.network,
            external_drive=lambda t: np.array([[1.0 + 0.0 * t, 2.0 + 0.0 * t]]),
            time=self.time,
            n_batch=self.n_batch,
            solver_config=_dense_diffrax_config(),
            transfer_config=transfer_cfg,
            phi_exc=self.phi_table,
            phi_inh=self.phi_table,
        )

        np.testing.assert_allclose(res_static.exc, res_dynamic.exc, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res_static.inh, res_dynamic.inh, rtol=1e-12, atol=1e-12)

    # ------------------------------------------------------------------
    # Test 3: Gradient equivalence between static and dynamic RK4 runners
    # ------------------------------------------------------------------
    def test_gradients_equivalence(self) -> None:
        net_layout = self.net_layout
        n_ext = net_layout.n_ext  # int

        y0 = jnp.zeros((net_layout.n_rates, self.n_batch), dtype=jnp.float64)

        # Pre-slice weight blocks (dense, for gradient tracing)
        W_exc, W_inh, W_ext = slice_weight_blocks(
            self.network.weights,
            net_layout.idx_exc,
            net_layout.idx_inh,
            net_layout.idx_ext,
            jnp,
            prefer_sparse=False,
            dense_max_mb=128.0,
        )

        # Precompute inputs
        ax_left, ax_mid, ax_right = precompute_rk4_inputs(
            lambda _t: np.array([[1.0, 2.0]]),
            self.time,
            n_ext=n_ext,
            n_batch=self.n_batch,
        )
        bg_left_e, bg_mid_e, bg_right_e, bg_left_i, bg_mid_i, bg_right_i = precompute_rk4_background(
            None,
            layout=net_layout,
            n_batch=self.n_batch,
            time=self.time,
        )
        phi_exc_x, phi_exc_y, phi_exc_rate_max = transfer_table_arrays(self.phi_table, "phi_exc")
        phi_inh_x, phi_inh_y, phi_inh_rate_max = transfer_table_arrays(self.phi_table, "phi_inh")

        # Static precomputed mu_ext
        mu_ext_static = W_ext @ jnp.asarray(ax_left[0])
        mu_ext_zero = jnp.zeros((net_layout.n_rates, self.n_batch), dtype=jnp.float64)

        # Compile both branches
        run_static = make_jax_rk4(jax, jnp, store_trajectory=True, is_static=True)
        run_dynamic = make_jax_rk4(jax, jnp, store_trajectory=True, is_static=False)

        common_args = dict(
            idx_exc=jnp.asarray(net_layout.idx_exc, dtype=jnp.int32),
            idx_inh=jnp.asarray(net_layout.idx_inh, dtype=jnp.int32),
            time=jnp.asarray(self.time),
            ax_left=jnp.asarray(ax_left),
            ax_mid=jnp.asarray(ax_mid),
            ax_right=jnp.asarray(ax_right),
            bg_left_e=jnp.asarray(bg_left_e),
            bg_mid_e=jnp.asarray(bg_mid_e),
            bg_right_e=jnp.asarray(bg_right_e),
            bg_left_i=jnp.asarray(bg_left_i),
            bg_mid_i=jnp.asarray(bg_mid_i),
            bg_right_i=jnp.asarray(bg_right_i),
            phi_exc_x=jnp.asarray(phi_exc_x),
            phi_exc_y=jnp.asarray(phi_exc_y),
            phi_exc_rate_max=jnp.asarray(phi_exc_rate_max),
            phi_inh_x=jnp.asarray(phi_inh_x),
            phi_inh_y=jnp.asarray(phi_inh_y),
            phi_inh_rate_max=jnp.asarray(phi_inh_rate_max),
            tau_exc=jnp.asarray(0.02),
            tau_inh=jnp.asarray(0.01),
        )

        def loss_static(w_exc):
            out = run_static(
                y0, w_exc, W_inh, W_ext, mu_ext_static,
                common_args["idx_exc"], common_args["idx_inh"],
                common_args["time"],
                common_args["ax_left"], common_args["ax_mid"], common_args["ax_right"],
                common_args["bg_left_e"], common_args["bg_mid_e"], common_args["bg_right_e"],
                common_args["bg_left_i"], common_args["bg_mid_i"], common_args["bg_right_i"],
                common_args["phi_exc_x"], common_args["phi_exc_y"], common_args["phi_exc_rate_max"],
                common_args["phi_inh_x"], common_args["phi_inh_y"], common_args["phi_inh_rate_max"],
                common_args["tau_exc"], common_args["tau_inh"],
            )
            return jnp.sum(out[0])

        def loss_dynamic(w_exc):
            out = run_dynamic(
                y0, w_exc, W_inh, W_ext, mu_ext_zero,
                common_args["idx_exc"], common_args["idx_inh"],
                common_args["time"],
                common_args["ax_left"], common_args["ax_mid"], common_args["ax_right"],
                common_args["bg_left_e"], common_args["bg_mid_e"], common_args["bg_right_e"],
                common_args["bg_left_i"], common_args["bg_mid_i"], common_args["bg_right_i"],
                common_args["phi_exc_x"], common_args["phi_exc_y"], common_args["phi_exc_rate_max"],
                common_args["phi_inh_x"], common_args["phi_inh_y"], common_args["phi_inh_rate_max"],
                common_args["tau_exc"], common_args["tau_inh"],
            )
            return jnp.sum(out[0])

        grad_static = jax.grad(loss_static)(W_exc)
        grad_dynamic = jax.grad(loss_dynamic)(W_exc)

        np.testing.assert_allclose(
            np.asarray(grad_static), np.asarray(grad_dynamic), rtol=1e-12, atol=1e-12
        )

    # ------------------------------------------------------------------
    # Test 4: Weight block slicing correctness
    # ------------------------------------------------------------------
    def test_weight_block_slicing(self) -> None:
        """Verify that pre-sliced W blocks reproduce the full weights @ sources product."""
        net_layout = self.net_layout
        W_exc, W_inh, W_ext = slice_weight_blocks(
            self.network.weights,
            net_layout.idx_exc,
            net_layout.idx_inh,
            net_layout.idx_ext,
            jnp,
            prefer_sparse=False,
            dense_max_mb=128.0,
        )

        np.random.seed(0)
        y = np.random.randn(net_layout.n_rates, self.n_batch)
        ax = np.random.randn(net_layout.n_ext, self.n_batch)

        # Build full source vector as the scipy solver does
        n_cols = self.network.weights.shape[1]
        sources = np.zeros((n_cols, self.n_batch))
        sources[net_layout.idx_exc, :] = y[net_layout.idx_exc, :]
        sources[net_layout.idx_inh, :] = y[net_layout.idx_inh, :]
        sources[net_layout.idx_ext, :] = ax

        mu_ref = self.network.weights @ sources
        mu_opt = (
            np.asarray(W_exc) @ y[net_layout.idx_exc, :]
            + np.asarray(W_inh) @ y[net_layout.idx_inh, :]
            + np.asarray(W_ext) @ ax
        )

        np.testing.assert_allclose(mu_opt, mu_ref, rtol=1e-12, atol=1e-14)

    def test_transfer_table_arrays_preserve_rate_cap(self) -> None:
        table = TransferTable(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 10.0, 20.0]),
            rate_max=7.5,
        )

        x, y, rate_max = transfer_table_arrays(table, "phi_exc")

        np.testing.assert_allclose(x, table.mu)
        np.testing.assert_allclose(y, table.rate)
        self.assertEqual(rate_max, 7.5)

    def test_jax_rk4_respects_transfer_table_rate_cap(self) -> None:
        capped_phi = TransferTable(
            np.linspace(-100.0, 100.0, 1000),
            np.full(1000, 50.0),
            rate_max=3.0,
        )

        result = solve_wilson_cowan_batch(
            network=self.network,
            external_drive=lambda _t: np.array([[100.0, 100.0]]),
            time=np.linspace(0.0, 0.2, 20),
            n_batch=self.n_batch,
            solver_config=_dense_solver_config(),
            transfer_config=TransferConfig(tau_e=0.02, tau_i=0.01),
            phi_exc=capped_phi,
            phi_inh=capped_phi,
        )

        self.assertLessEqual(float(np.max(result.exc)), 3.0 + 1e-6)
        self.assertLessEqual(float(np.max(result.inh)), 3.0 + 1e-6)

    def test_diffrax_respects_transfer_table_rate_cap(self) -> None:
        if not is_diffrax_available():
            self.skipTest("Diffrax not installed")

        capped_phi = TransferTable(
            np.linspace(-100.0, 100.0, 1000),
            np.full(1000, 50.0),
            rate_max=3.0,
        )

        result = solve_wilson_cowan_batch(
            network=self.network,
            external_drive=lambda _t: np.array([[100.0, 100.0]]),
            time=np.linspace(0.0, 0.2, 20),
            n_batch=self.n_batch,
            solver_config=_dense_diffrax_config(),
            transfer_config=TransferConfig(tau_e=0.02, tau_i=0.01),
            phi_exc=capped_phi,
            phi_inh=capped_phi,
        )

        self.assertLessEqual(float(np.max(result.exc)), 3.0 + 1e-6)
        self.assertLessEqual(float(np.max(result.inh)), 3.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
