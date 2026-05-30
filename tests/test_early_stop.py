import pytest
import jax
import jax.numpy as jnp
import numpy as np

# This test file should be run in an environment with JAX and Diffrax installed.

def mock_vector_field(t, y, args):
    """Linear analytic system: dy/dt = -lambda * y"""
    lam = args["lambda"]
    return -lam * y

def mock_fixed_point_field(t, y, args):
    """Nonzero fixed point: dy/dt = -(y - y_star)"""
    y_star = args["y_star"]
    return -(y - y_star)

def mock_damped_oscillator(t, y, args):
    """Damped oscillator: y'' + c y' + k y = 0
    Turned into a 1st order system:
    dy1/dt = y2
    dy2/dt = -k y1 - c y2
    """
    c = args["c"]
    k = args["k"]
    y1 = y[0]
    y2 = y[1]
    dy1 = y2
    dy2 = -k * y1 - c * y2
    return jnp.array([dy1, dy2])


def test_early_stop_linear_analytic():
    """Test Diffrax Early Stopping on linear analytic system."""
    try:
        import diffrax
    except ImportError:
        pytest.skip("Diffrax not installed.")
        
    lam = 10.0
    y0 = jnp.array([1.0])
    
    # We want early stop when norm(dy/dt) < f_atol
    # dy/dt = -10 * y(t) = -10 * exp(-10t)
    # Target: 10 * exp(-10t) < 1e-4 -> exp(-10t) < 1e-5 -> -10t < ln(1e-5) -> t > 1.15
    f_atol = 1e-4
    f_rtol = 0.0
    
    def cond_fn(state_or_t, y=None, args_in=None, **kwargs):
        if y is None:
            t = state_or_t.t
            y_val = state_or_t.y
        else:
            t = state_or_t
            y_val = y
            
        dy = mock_vector_field(t, y_val, {"lambda": lam})
        f_norm = jnp.max(jnp.abs(dy))
        y_norm = jnp.max(jnp.abs(y_val))
        is_steady = f_norm < f_atol + f_rtol * y_norm
        return jnp.logical_and(t >= 0.1, is_steady)

    if hasattr(diffrax, "Event"):
        event = diffrax.Event(cond_fn)
    else:
        event = diffrax.DiscreteTerminatingEvent(cond_fn)

    term = diffrax.ODETerm(mock_vector_field)
    solver = diffrax.Tsit5()
    
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=5.0, dt0=0.01, y0=y0,
        args={"lambda": lam}, event=event, throw=False, max_steps=4096
    )
    
    assert sol.result == diffrax.RESULTS.event_occurred
    # Check stopping time is roughly 1.15
    stopping_time = sol.ts[sol.stats["num_steps"]]
    assert 1.0 < stopping_time < 1.3

def test_early_stop_fixed_point():
    """Test Diffrax Early Stopping converges to a non-zero fixed point."""
    try:
        import diffrax
    except ImportError:
        pytest.skip("Diffrax not installed.")
        
    y_star = jnp.array([5.0])
    y0 = jnp.array([0.0])
    f_atol = 1e-4
    f_rtol = 1e-4
    
    def cond_fn(state_or_t, y=None, args_in=None, **kwargs):
        if y is None:
            t = state_or_t.t
            y_val = state_or_t.y
        else:
            t = state_or_t
            y_val = y
            
        dy = mock_fixed_point_field(t, y_val, {"y_star": y_star})
        f_norm = jnp.max(jnp.abs(dy))
        y_norm = jnp.max(jnp.abs(y_val))
        is_steady = f_norm < f_atol + f_rtol * y_norm
        return jnp.logical_and(t >= 0.1, is_steady)

    if hasattr(diffrax, "Event"):
        event = diffrax.Event(cond_fn)
    else:
        event = diffrax.DiscreteTerminatingEvent(cond_fn)

    term = diffrax.ODETerm(mock_fixed_point_field)
    solver = diffrax.Tsit5()
    
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=20.0, dt0=0.1, y0=y0,
        args={"y_star": y_star}, event=event, throw=False, max_steps=4096
    )
    
    assert sol.result == diffrax.RESULTS.event_occurred
    final_y = sol.ys[sol.stats["num_steps"]]
    # The final value should be very close to the fixed point (5.0)
    assert jnp.allclose(final_y, y_star, atol=1e-3)

def test_early_stop_damped_oscillator():
    """Test Diffrax Early Stopping does not falsely stop during damped oscillation."""
    try:
        import diffrax
    except ImportError:
        pytest.skip("Diffrax not installed.")
        
    y0 = jnp.array([1.0, 0.0]) # start at displacement=1, velocity=0
    c = 1.5  # damping (increased to decay faster and avoid max_steps limit)
    k = 20.0 # spring constant (creates fast oscillation)
    f_atol = 1e-4
    f_rtol = 1e-4
    
    def cond_fn(state_or_t, y=None, args_in=None, **kwargs):
        if y is None:
            t = state_or_t.t
            y_val = state_or_t.y
        else:
            t = state_or_t
            y_val = y
            
        dy = mock_damped_oscillator(t, y_val, {"c": c, "k": k})
        f_norm = jnp.max(jnp.abs(dy))
        y_norm = jnp.max(jnp.abs(y_val))
        is_steady = f_norm < f_atol + f_rtol * y_norm
        return jnp.logical_and(t >= 0.1, is_steady)

    if hasattr(diffrax, "Event"):
        event = diffrax.Event(cond_fn)
    else:
        event = diffrax.DiscreteTerminatingEvent(cond_fn)

    term = diffrax.ODETerm(mock_damped_oscillator)
    solver = diffrax.Tsit5()
    
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=50.0, dt0=0.01, y0=y0,
        args={"c": c, "k": k}, event=event, throw=False, max_steps=16384
    )
    
    assert sol.result == diffrax.RESULTS.event_occurred
    final_y = sol.ys[sol.stats["num_steps"]]
    
    # It should not have falsely stopped at a velocity crossing (where dy/dt might briefly be small)
    # The final displacement and velocity should both be near zero
    assert jnp.max(jnp.abs(final_y)) < 1e-2


def test_pack_trajectory_result_early_stop():
    from v1_simulation.solvers.base import pack_trajectory_result, NetworkLayout
    import numpy as np
    import pytest

    layout = NetworkLayout(
        idx_exc=np.array([0, 1]),
        idx_inh=np.array([2]),
        idx_ext=np.array([])
    )
    time = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # Case 1: early stop success with trailing NaN padding (shape: (6, 3, 1))
    trajectory = np.array([
        [[1.0], [2.0], [3.0]],  # t = 0.0
        [[2.0], [4.0], [6.0]],  # t = 0.1
        [[3.0], [6.0], [9.0]],  # t = 0.2
        [[4.0], [8.0], [12.0]], # t = 0.3
        [[np.nan], [np.nan], [np.nan]], # t = 0.4 (invalid/padded)
        [[np.nan], [np.nan], [np.nan]]  # t = 0.5 (invalid/padded)
    ])

    result = pack_trajectory_result(
        trajectory=trajectory,
        layout=layout,
        time=time,
        store_trajectory=True,
        steady_state_reached=True,
        steady_state_index=4,
        steady_state_start_index=2
    )

    # Valid summary window should be sliced from index 2 to 4:
    # exc_t[2:4, 0] = [[3.0, 6.0], [4.0, 8.0]] -> mean = [3.5, 7.0]
    # inh_t[2:4, 0] = [[9.0], [12.0]] -> mean = [10.5]
    np.testing.assert_allclose(result.exc[0], [3.5, 7.0])
    np.testing.assert_allclose(result.inh[0], [10.5])
    assert np.all(np.isfinite(result.exc))
    assert np.all(np.isfinite(result.inh))
    assert np.all(np.isfinite(result.exc_convergence))
    assert np.all(np.isfinite(result.inh_convergence))

    # Case 2: early stop not triggered, full finite trajectory
    trajectory_full = np.array([
        [[1.0], [2.0], [3.0]],
        [[2.0], [4.0], [6.0]],
        [[3.0], [6.0], [9.0]],
        [[4.0], [8.0], [12.0]],
        [[5.0], [10.0], [15.0]],
        [[6.0], [12.0], [18.0]]
    ])
    result_full = pack_trajectory_result(
        trajectory=trajectory_full,
        layout=layout,
        time=time,
        store_trajectory=True,
        steady_state_reached=False,
        steady_state_index=None,
        steady_state_start_index=None
    )
    # Defaults to last 1/3 of the grid: start = 6 * 2/3 = 4. Slice: 4 to 6.
    # exc_t[4:6] = [[5.0, 10.0], [6.0, 12.0]] -> mean = [5.5, 11.0]
    np.testing.assert_allclose(result_full.exc[0], [5.5, 11.0])

    # Case 3: early stop success, but steady_state_index <= 0 leading to error
    with pytest.raises(ValueError, match="Invalid steady-state end index"):
        pack_trajectory_result(
            trajectory=trajectory,
            layout=layout,
            time=time,
            store_trajectory=True,
            steady_state_reached=True,
            steady_state_index=0,
            steady_state_start_index=2
        )

    # Case 4: trajectory contains NaN in the active window (index 4)
    trajectory_nan = np.array([
        [[1.0], [2.0], [3.0]],
        [[2.0], [4.0], [6.0]],
        [[3.0], [6.0], [9.0]],
        [[4.0], [8.0], [12.0]],
        [[5.0], [np.nan], [15.0]],
        [[6.0], [12.0], [18.0]]
    ])
    with pytest.raises(ValueError, match="Non-finite values inside steady-state summary window"):
        pack_trajectory_result(
            trajectory=trajectory_nan,
            layout=layout,
            time=time,
            store_trajectory=True,
            steady_state_reached=False,
            steady_state_index=None,
            steady_state_start_index=None
        )


