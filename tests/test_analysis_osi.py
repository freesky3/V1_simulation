import numpy as np
import pytest

from v1_simulation.analysis.osi import compute_osi


def test_compute_osi_typical() -> None:
    # 4 orientations: 0, 45, 90, 135 degrees (0, pi/4, pi/2, 3pi/4 radians)
    theta = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    # Neuron 1: Perfectly tuned to 0 rad (cos/sin coefficients sum to maximum)
    # Neuron 2: Completely flat response
    responses = np.array([
        [10.0, 0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0, 5.0],
    ])

    osi, pref_ori = compute_osi(responses, theta, min_osi=0.4)

    assert osi.shape == (2,)
    assert pref_ori.shape == (2,)
    # Perfectly tuned neuron should have OSI = 1.0, pref = 0.0
    assert osi[0] == pytest.approx(1.0)
    assert pref_ori[0] == pytest.approx(0.0)

    # Flat neuron should have OSI = 0.0, pref = NaN (since 0 < 0.4)
    assert osi[1] == pytest.approx(0.0)
    assert np.isnan(pref_ori[1])


def test_compute_osi_threshold_boundaries() -> None:
    theta = np.array([0.0, np.pi / 2])
    responses = np.array([
        [2.0, 1.0],  # OSI = 1/3 ~ 0.333
    ])

    # If min_osi is 0.3, it should be valid
    _, pref_03 = compute_osi(responses, theta, min_osi=0.3)
    assert not np.isnan(pref_03[0])

    # If min_osi is 0.4, it should be NaN
    _, pref_04 = compute_osi(responses, theta, min_osi=0.4)
    assert np.isnan(pref_04[0])


def test_compute_osi_zero_responses() -> None:
    theta = np.array([0.0, np.pi / 2])
    responses = np.array([
        [0.0, 0.0],
    ])
    osi, pref = compute_osi(responses, theta)
    assert osi[0] == 0.0
    assert np.isnan(pref[0])


def test_compute_osi_invalid_inputs() -> None:
    theta = np.array([0.0, np.pi / 2])
    responses = np.array([
        [1.0, 2.0],
    ])

    # Shape mismatch
    with pytest.raises(ValueError, match="theta_angles must have shape"):
        compute_osi(responses, np.array([0.0]))

    # Dimension mismatch
    with pytest.raises(ValueError, match="responses_mean must have shape"):
        compute_osi(np.array([1.0, 2.0]), theta)

    # Non-finite values in responses
    with pytest.raises(ValueError, match="responses_mean contains NaN or infinite values"):
        compute_osi(np.array([[1.0, np.nan]]), theta)

    # Non-finite values in theta
    with pytest.raises(ValueError, match="theta_angles contains NaN or infinite values"):
        compute_osi(responses, np.array([0.0, np.inf]))

    # Bad min_osi
    with pytest.raises(ValueError, match="min_osi must be in"):
        compute_osi(responses, theta, min_osi=-0.1)
