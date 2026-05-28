from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AnalysisInputs:
    """Input data required for performing network simulation analysis.

    Attributes:
        responses: Firing rate responses of shape (n_neurons, n_theta, n_time).
        coords: Spatial coordinates of shape (n_neurons, 2).
        distance: Pairwise distances between neurons of shape (n_neurons, n_neurons).
        theta_angles: Stimulus orientations (angles in degrees) of shape (n_theta,).
    """
    responses: NDArray[np.float64]
    coords: NDArray[np.float64]
    distance: NDArray[np.float64]
    theta_angles: NDArray[np.float64]

    def validate(self) -> None:
        responses = np.asarray(self.responses, dtype=float)
        coords = np.asarray(self.coords, dtype=float)
        distance = np.asarray(self.distance, dtype=float)
        theta_angles = np.asarray(self.theta_angles, dtype=float)

        if responses.ndim != 3:
            raise ValueError("responses must have shape (n_neurons, n_theta, n_time).")
        n_neurons, n_theta, n_time = responses.shape
        if n_neurons < 1:
            raise ValueError("responses must contain at least one neuron.")
        if n_theta < 1 or n_time < 1:
            raise ValueError("responses must contain at least one orientation and one time step.")
        if coords.shape != (n_neurons, 2):
            raise ValueError(f"coords must have shape ({n_neurons}, 2), got {coords.shape}.")
        if distance.shape != (n_neurons, n_neurons):
            raise ValueError(
                f"distance must have shape ({n_neurons}, {n_neurons}), got {distance.shape}."
            )
        if theta_angles.shape != (n_theta,):
            raise ValueError(f"theta_angles must have shape ({n_theta},), got {theta_angles.shape}.")
        for name, values in (
            ("responses", responses),
            ("coords", coords),
            ("distance", distance),
            ("theta_angles", theta_angles),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains NaN or infinite values.")


@dataclass(frozen=True, slots=True)
class CommunityResult:
    """Result of community detection analysis on network responses.

    Attributes:
        labels: Assigned community ID for each neuron of shape (n_neurons,).
            0 denotes unclassified.
        similarity: Pairwise similarity matrix of shape (n_neurons, n_neurons).
        agreement: Optional consensus agreement matrix of shape (n_neurons, n_neurons).
        diagnostics: Additional metric summaries or algorithm details.
    """
    labels: NDArray[np.int64]
    similarity: NDArray[np.float64]
    agreement: NDArray[np.float64] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        similarity = np.asarray(self.similarity, dtype=float)
        if similarity.shape != (labels.size, labels.size):
            raise ValueError(
                f"similarity shape {similarity.shape} does not match labels length {labels.size}."
            )
        if np.any(labels < 0):
            raise ValueError("community labels must be non-negative; 0 is unclassified.")
        if not np.all(np.isfinite(similarity)):
            raise ValueError("similarity contains NaN or infinite values.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "similarity", similarity)
        if self.agreement is not None:
            agreement = np.asarray(self.agreement, dtype=float)
            if agreement.shape != similarity.shape:
                raise ValueError(
                    f"agreement shape {agreement.shape} does not match similarity shape {similarity.shape}."
                )
            if not np.all(np.isfinite(agreement)):
                raise ValueError("agreement contains NaN or infinite values.")
            object.__setattr__(self, "agreement", agreement)

    @property
    def n_ensembles(self) -> int:
        return int(np.unique(self.labels[self.labels != 0]).size)

    @property
    def classified_neurons(self) -> int:
        return int(np.sum(self.labels != 0))


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Aggregated analysis results from a network simulation run.

    Attributes:
        status: Status string (e.g. 'success').
        selected_indices: Original indices of neurons selected for analysis.
        osi: Orientation Selectivity Index (OSI) of selected neurons.
        pref_ori: Preferred orientation of selected neurons in degrees.
        responses_mean: Mean responses across orientations of selected neurons.
        steady_state_responses: Firing rates at steady state of shape (selected_neurons, n_theta).
        coords: Coordinates of selected neurons.
        distance: Pairwise distances of selected neurons.
        communities: Optional community detection results.
        diagnostics: Diagnostic properties or metrics.
    """
    status: str
    selected_indices: NDArray[np.int64]
    osi: NDArray[np.float64]
    pref_ori: NDArray[np.float64]
    responses_mean: NDArray[np.float64]
    steady_state_responses: NDArray[np.float64]
    coords: NDArray[np.float64]
    distance: NDArray[np.float64]
    communities: CommunityResult | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_indices", np.asarray(self.selected_indices, dtype=np.int64))
        object.__setattr__(self, "osi", np.asarray(self.osi, dtype=float))
        object.__setattr__(self, "pref_ori", np.asarray(self.pref_ori, dtype=float))
        object.__setattr__(self, "responses_mean", np.asarray(self.responses_mean, dtype=float))
        object.__setattr__(self, "steady_state_responses", np.asarray(self.steady_state_responses, dtype=float))
        object.__setattr__(self, "coords", np.asarray(self.coords, dtype=float))
        object.__setattr__(self, "distance", np.asarray(self.distance, dtype=float))
