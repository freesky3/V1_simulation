from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from v1_simulation.network.geometry import L4, SheetGeometry


@dataclass(frozen=True, slots=True)
class GaborConfig:
    """Parameters for a spatial Gabor receptive field kernel."""

    sigma: float
    gamma: float
    spatial_frequency: float
    phase: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")


@dataclass(frozen=True, slots=True)
class GaborRFConfig:
    """Gabor RF parameters together with the visual integration grid."""

    stimulus_size: float
    resolution: int
    gabor: GaborConfig

    def __post_init__(self) -> None:
        if self.stimulus_size <= 0:
            raise ValueError("stimulus_size must be positive.")
        if self.resolution <= 1:
            raise ValueError("resolution must be greater than 1.")


@dataclass(frozen=True, slots=True)
class VisualGrid:
    """A 2D coordinate grid representing visual space.

    Attributes:
        x_axis: 1D x-coordinate midpoint axis.
        y_axis: 1D y-coordinate midpoint axis.
        x: 2D array of x-coordinates on the grid.
        y: 2D array of y-coordinates on the grid.
        dx: Grid spacing step size along the x-axis.
        dy: Grid spacing step size along the y-axis.
    """

    x_axis: NDArray[np.float64]
    y_axis: NDArray[np.float64]
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
        if size <= 0:
            raise ValueError("size must be positive.")
        if resolution <= 1:
            raise ValueError("resolution must be greater than 1.")

        dx = size / resolution
        axis = (np.arange(resolution, dtype=float) + 0.5) * dx
        axis -= size / 2.0
        x, y = np.meshgrid(axis, axis, indexing="xy")
        return cls(x_axis=axis, y_axis=axis, x=x, y=y, dx=dx, dy=dx)

    @property
    def X(self) -> NDArray[np.float64]:
        """Backward-compatible alias for the 2D x-coordinate grid."""
        return self.x

    @property
    def Y(self) -> NDArray[np.float64]:
        """Backward-compatible alias for the 2D y-coordinate grid."""
        return self.y

    @property
    def area_element(self) -> float:
        """Returns the infinitesimal area element (dx * dy) for spatial integration."""
        return self.dx * self.dy


def gabor_kernel(
    grid: VisualGrid,
    cfg: GaborConfig,
    theta_pref: float,
    is_tuned: bool,
) -> NDArray[np.float64]:
    """Generates a 2D spatial Gabor filter representing a neuron's receptive field.

    Evaluates a Gabor kernel (elliptical/circular Gaussian envelope modulated by
    a cosine grating for tuned cells, or simple Gaussian envelope for untuned cells).
    The gamma parameter is applied linearly to y_prime**2.

    Args:
        grid: The visual grid coordinates.
        cfg: Gabor receptive field parameters.
        theta_pref: The preferred orientation angle in radians.
        is_tuned: If True, modulates the Gaussian envelope with a cosine grating.

    Returns:
        A 2D numpy array representing the spatial receptive field profile.
    """
    if not is_tuned:
        theta_pref = 0.0

    x_prime = grid.x * np.cos(theta_pref) + grid.y * np.sin(theta_pref)
    y_prime = -grid.x * np.sin(theta_pref) + grid.y * np.cos(theta_pref)

    gamma = cfg.gamma if is_tuned else 1.0
    gaussian = np.exp(-(x_prime**2 + gamma * y_prime**2) / (2.0 * cfg.sigma**2))

    if not is_tuned:
        return gaussian

    grating = np.cos(cfg.spatial_frequency * x_prime - cfg.phase)
    return gaussian * grating


def gabor_bank(
    grid: VisualGrid,
    cfg: GaborConfig,
    theta_pref: NDArray[np.float64],
    is_tuned: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Builds a stack of Gabor kernels for a population of L4 neurons.

    Args:
        grid: The visual grid coordinates common to all neurons.
        cfg: Common parameters for the spatial Gabor profile.
        theta_pref: A 1D array of preferred orientation angles (in radians)
            for each neuron.
        is_tuned: A 1D boolean array indicating whether each neuron is tuned.

    Returns:
        A 3D array of shape (n_neurons, resolution, resolution) containing
        the spatial filters for all L4 neurons.
    """
    kernels = [
        gabor_kernel(grid, cfg, float(theta), bool(tuned))
        for theta, tuned in zip(theta_pref, is_tuned, strict=True)
    ]
    return np.stack(kernels, axis=0)


class L4GaborBank:
    """Lazy Gabor RF bank for an L4 population."""

    def __init__(
        self,
        cfg: GaborRFConfig,
        l4_layer: SheetGeometry | L4,
        *,
        l4_tunings: ArrayLike | None = None,
        l4_pref_dirs: ArrayLike | None = None,
    ) -> None:
        self.cfg = cfg
        self.l4 = l4_layer
        self.grid = VisualGrid.centered_midpoint(
            size=cfg.stimulus_size,
            resolution=cfg.resolution,
        )

        if hasattr(l4_layer, "tunings") and hasattr(l4_layer, "pref_dirs"):
            tunings_array = np.asarray(l4_layer.tunings)
            pref_dirs_array = np.asarray(l4_layer.pref_dirs)
        elif hasattr(l4_layer, "is_tuned") and hasattr(l4_layer, "preferred_orientations"):
            tunings_array = np.where(l4_layer.is_tuned, "T", "U")
            pref_dirs_array = np.asarray(l4_layer.preferred_orientations)
        else:
            if l4_tunings is None or l4_pref_dirs is None:
                raise ValueError("l4_tunings and l4_pref_dirs are required when l4_layer does not contain tuning information.")
            tunings_array = np.asarray(l4_tunings)
            pref_dirs_array = np.asarray(l4_pref_dirs)

        self.is_tuned = np.asarray(tunings_array) == "T"
        self.theta = np.nan_to_num(np.asarray(pref_dirs_array, dtype=float), nan=0.0)

        if np.asarray(l4_layer.coords).shape[0] != len(self.theta):
            raise ValueError("l4_layer coords and pref_dirs must have the same length.")

        self._filters: NDArray[np.float64] | None = None

    @property
    def filters(self) -> NDArray[np.float64]:
        """Lazy loads and returns the Gabor kernels stack for L4 population."""
        if self._filters is None:
            self._filters = gabor_bank(self.grid, self.cfg.gabor, self.theta, self.is_tuned)
            self._filters.setflags(write=False)
        return self._filters

