from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def compute_osi(
    responses_mean: ArrayLike,
    theta_angles: ArrayLike,
    *,
    min_osi: float = 0.4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes the Orientation Selectivity Index (OSI) and preferred orientation.

    The OSI is computed using circular variance where the stimulus orientations are mapped
    to a 2*theta angle to cover a full circular range [0, 2*pi).

    Args:
        responses_mean: Mean responses of shape (n_neurons, n_theta).
        theta_angles: Stimulus orientation angles in radians of shape (n_theta,).
        min_osi: Minimum OSI threshold below which the preferred orientation is set to NaN.

    Returns:
        A tuple of:
            - osi: The computed OSI for each neuron of shape (n_neurons,).
            - pref_ori: Preferred orientation in radians of shape (n_neurons,), or NaN
              if OSI is below min_osi.

    Raises:
        ValueError: If inputs have incorrect shapes, range, or contain invalid values (NaN/inf).
    """
    responses = np.asarray(responses_mean, dtype=float)
    theta = np.asarray(theta_angles, dtype=float)

    if responses.ndim != 2:
        raise ValueError("responses_mean must have shape (n_neurons, n_theta).")
    if theta.shape != (responses.shape[1],):
        raise ValueError(f"theta_angles must have shape ({responses.shape[1]},), got {theta.shape}.")
    if not 0.0 <= float(min_osi) <= 1.0:
        raise ValueError("min_osi must be in [0, 1].")
    if not np.all(np.isfinite(responses)):
        raise ValueError("responses_mean contains NaN or infinite values.")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta_angles contains NaN or infinite values.")

    vector = np.sum(responses * np.exp(2j * theta), axis=1)
    scalar = np.sum(responses, axis=1)
    osi = np.divide(np.abs(vector), scalar, out=np.zeros_like(scalar, dtype=float), where=scalar > 0.0)
    pref_ori = (np.angle(vector) / 2.0) % np.pi
    pref_ori[osi < float(min_osi)] = np.nan
    return osi.astype(float, copy=False), pref_ori.astype(float, copy=False)
