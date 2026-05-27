from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from v1_simulation.stimuli.receptive_fields import (
    VisualGrid,
    gabor_kernel,
)

if TYPE_CHECKING:
    from v1_simulation.config.schema import StimulusConfig
    from v1_simulation.network.geometry import L4


class DriftingGratingInput:
    """Computes the external input drive to L4 neurons from drifting gratings.

    Calculates spatial integration of visual stimulus frames with neurons' Gabor receptive
    fields analytically, allowing very fast continuous-time drive calculations.
    """

    def __init__(self, cfg_drifting_grating: StimulusConfig, l4: L4):
        """Initializes the DriftingGratingInput generator.

        Args:
            cfg_drifting_grating: Stimulus configuration.
            l4: The L4 layer geometry.
        """
        self.cfg = cfg_drifting_grating
        self.l4 = l4
        self.grid = VisualGrid.centered_midpoint(
            self.cfg.receptive_field.stimulus_size,
            self.cfg.receptive_field.resolution,
        )
        self._integral_cache: Dict[float, Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = {}

    def external_drive(self, theta_stim: float, t: float) -> NDArray[np.float64]:
        """Calculates the firing rate drive for L4 neurons at a specific time.

        Args:
            theta_stim: The orientation of the stimulus grating in radians.
            t: Time point in seconds.

        Returns:
            A 1D array of input rates for each L4 neuron.
        """
        base, cos_coeff, sin_coeff = self._precompute_integrals(theta_stim)

        wt = self.cfg.temporal_frequency * t
        integral = base + cos_coeff * np.cos(wt) + sin_coeff * np.sin(wt)

        rates = np.maximum(0.0, self.cfg.baseline_rate + integral)
        return rates * self.cfg.visual_gain

    def make_drive_func(self, theta_stim: float) -> Callable[[float], NDArray[np.float64]]:
        """Creates a continuous-time drive function for a single stimulus orientation.

        Args:
            theta_stim: Stimulus orientation angle in radians.

        Returns:
            A function mapping time `t` to a 1D array of input rates.
        """
        def drive(t: float) -> NDArray[np.float64]:
            return self.external_drive(theta_stim, t)

        return drive

    def make_batched_drive_func(
        self, theta_stims: ArrayLike
    ) -> Callable[[float], NDArray[np.float64]]:
        """Creates a continuous-time drive function for a batch of stimulus orientations.

        Args:
            theta_stims: Array-like of stimulus orientation angles in radians.

        Returns:
            A function mapping time `t` to a 2D array of input rates (neurons x orientations).
        """
        theta_stims_arr = np.asarray(theta_stims, dtype=float)
        terms = [self._precompute_integrals(theta) for theta in theta_stims_arr]

        base = np.column_stack([item[0] for item in terms])
        cos_coeff = np.column_stack([item[1] for item in terms])
        sin_coeff = np.column_stack([item[2] for item in terms])

        def drive(t: float) -> NDArray[np.float64]:
            wt = self.cfg.temporal_frequency * t
            integral = base + cos_coeff * np.cos(wt) + sin_coeff * np.sin(wt)
            rates = np.maximum(0.0, self.cfg.baseline_rate + integral)
            return rates * self.cfg.visual_gain

        return drive

    def stimulus_frame(self, theta_stim: float, t: float) -> NDArray[np.float64]:
        """Generates the 2D visual stimulus pixel values (luminance) for rendering.

        Args:
            theta_stim: The orientation of the grating in radians.
            t: Time point in seconds.

        Returns:
            A 2D array of stimulus pixel values.
        """
        rf = self.cfg.receptive_field
        x = self.l4.coords[:, 0, None, None] + self.grid.x[None, :, :]
        y = self.l4.coords[:, 1, None, None] + self.grid.y[None, :, :]

        phase = rf.spatial_frequency * (
            x * np.cos(theta_stim) + y * np.sin(theta_stim)
        )
        phase -= self.cfg.temporal_frequency * t

        return self.cfg.luminance * (1.0 + self.cfg.contrast * np.cos(phase))

    def _precompute_integrals(
        self, theta_stim: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Precomputes spatial integrals of the Gabor receptive field and grating.

        Analytically factors out time by computing the phase-independent base,
        cosine, and sine coefficients of the projection.

        Args:
            theta_stim: Stimulus orientation angle in radians.

        Returns:
            A tuple of three 1D arrays: (base, cos_coeff, sin_coeff).
        """
        key = float(theta_stim)
        if key in self._integral_cache:
            return self._integral_cache[key]

        n = self.l4.N
        base = np.zeros(n, dtype=float)
        cos_coeff = np.zeros(n, dtype=float)
        sin_coeff = np.zeros(n, dtype=float)

        area = self.grid.area_element
        k = self.cfg.spatial_frequency

        cos_theta = np.cos(theta_stim)
        sin_theta = np.sin(theta_stim)

        grid_phase = k * (self.grid.x * cos_theta + self.grid.y * sin_theta)
        grid_cos = np.cos(grid_phase)
        grid_sin = np.sin(grid_phase)

        x_i = self.l4.coords[:, 0]
        y_i = self.l4.coords[:, 1]
        cell_phase = k * (x_i * cos_theta + y_i * sin_theta)
        cell_cos = np.cos(cell_phase)
        cell_sin = np.sin(cell_phase)

        for tuned in (False, True):
            tuned_mask = self.l4.is_tuned == tuned
            if not np.any(tuned_mask):
                continue

            theta_values = (
                np.unique(self.l4.preferred_orientations[tuned_mask])
                if tuned
                else np.array([0.0])
            )

            for theta_pref in theta_values:
                group = tuned_mask & (self.l4.preferred_orientations == theta_pref)
                kernel = gabor_kernel(self.grid, self.cfg, float(theta_pref), tuned)

                group_base = np.sum(kernel * self.cfg.luminance) * area
                group_cos = np.sum(kernel * self.cfg.luminance * self.cfg.contrast * grid_cos) * area
                group_sin = np.sum(kernel * self.cfg.luminance * self.cfg.contrast * grid_sin) * area

                base[group] = group_base
                cos_coeff[group] = cell_cos[group] * group_cos - cell_sin[group] * group_sin
                sin_coeff[group] = cell_sin[group] * group_cos + cell_cos[group] * group_sin

        self._integral_cache[key] = (base, cos_coeff, sin_coeff)
        return self._integral_cache[key]