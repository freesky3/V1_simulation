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
    
    def cond_fn(state, **kwargs):
        dy = mock_vector_field(state.t, state.y, {"lambda": lam})
        f_norm = jnp.max(jnp.abs(dy))
        y_norm = jnp.max(jnp.abs(state.y))
        is_steady = f_norm < f_atol + f_rtol * y_norm
        return jnp.logical_and(state.t >= 0.1, is_steady)

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
    
    def cond_fn(state, **kwargs):
        dy = mock_fixed_point_field(state.t, state.y, {"y_star": y_star})
        f_norm = jnp.max(jnp.abs(dy))
        y_norm = jnp.max(jnp.abs(state.y))
        is_steady = f_norm < f_atol + f_rtol * y_norm
        return jnp.logical_and(state.t >= 0.1, is_steady)

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
    c = 0.5  # damping
    k = 20.0 # spring constant (creates fast oscillation)
    f_atol = 1e-4
    f_rtol = 1e-4
    
    def cond_fn(state, **kwargs):
        dy = mock_damped_oscillator(state.t, state.y, {"c": c, "k": k})
        f_norm = jnp.max(jnp.abs(dy))
        y_norm = jnp.max(jnp.abs(state.y))
        is_steady = f_norm < f_atol + f_rtol * y_norm
        return jnp.logical_and(state.t >= 0.1, is_steady)

    if hasattr(diffrax, "Event"):
        event = diffrax.Event(cond_fn)
    else:
        event = diffrax.DiscreteTerminatingEvent(cond_fn)

    term = diffrax.ODETerm(mock_damped_oscillator)
    solver = diffrax.Tsit5()
    
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=50.0, dt0=0.01, y0=y0,
        args={"c": c, "k": k}, event=event, throw=False, max_steps=4096
    )
    
    assert sol.result == diffrax.RESULTS.event_occurred
    final_y = sol.ys[sol.stats["num_steps"]]
    
    # It should not have falsely stopped at a velocity crossing (where dy/dt might briefly be small)
    # The final displacement and velocity should both be near zero
    assert jnp.max(jnp.abs(final_y)) < 1e-2

