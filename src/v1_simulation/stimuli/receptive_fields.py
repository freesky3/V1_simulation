from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from v1_simulation.config.schema import StimulusConfig


@dataclass(frozen=True, slots=True)
class VisualGrid:
    """A 2D coordinate grid representing visual space.

    Attributes:
        x: 2D array of x-coordinates on the grid.
        y: 2D array of y-coordinates on the grid.
        dx: Grid spacing step size along the x-axis.
        dy: Grid spacing step size along the y-axis.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    dx: float
    dy: float

    @classmethod
    def centered_midpoint(cls, size: float, resolution: int) -> "VisualGrid":
        """Creates a centered midpoint visual grid.

        Args:
            size: The physical size (width and height) of the visual grid.
            resolution: The number of grid points along each axis.

        Returns:
            A centered VisualGrid instance.
        """
        dx = size / resolution
        axis = (np.arange(resolution, dtype=float) + 0.5) * dx
        axis -= size / 2.0
        x, y = np.meshgrid(axis, axis, indexing="xy")
        return cls(x=x, y=y, dx=dx, dy=dx)

    @property
    def area_element(self) -> float:
        """Returns the infinitesimal area element (dx * dy) for spatial integration."""
        return self.dx * self.dy


def gabor_kernel(
    grid: VisualGrid,
    cfg_drifting_grating: StimulusConfig,
    theta_pref: float,
    is_tuned: bool,
) -> NDArray[np.float64]:
    """Generates a 2D spatial Gabor filter representing a neuron's receptive field.

    Evaluates a Gabor kernel (elliptical/circular Gaussian envelope modulated by
    a cosine grating for tuned cells, or simple Gaussian envelope for untuned cells).

    Args:
        grid: The visual grid coordinates.
        cfg_drifting_grating: Stimulus configuration settings.
        theta_pref: The preferred orientation angle in radians.
        is_tuned: If True, modulates the Gaussian envelope with a cosine grating.

    Returns:
        A 2D numpy array representing the spatial receptive field profile.
    """
    if not is_tuned:
        theta_pref = 0.0

    x_prime = grid.x * np.cos(theta_pref) + grid.y * np.sin(theta_pref)
    y_prime = -grid.x * np.sin(theta_pref) + grid.y * np.cos(theta_pref)

    gamma = cfg_drifting_grating.gamma if is_tuned else 1.0
    gaussian = np.exp(
        -(x_prime**2 + gamma * y_prime**2) / (2.0 * cfg_drifting_grating.sigma**2)
    )

    if not is_tuned:
        return gaussian

    grating = np.cos(
        cfg_drifting_grating.spatial_frequency * x_prime - cfg_drifting_grating.phase
    )
    return gaussian * grating