from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

from v1_simulation.analysis.communities import identify_communities
from v1_simulation.analysis.metrics import activity_health_metrics, summarize_communities
from v1_simulation.analysis.osi import compute_osi
from v1_simulation.analysis.types import AnalysisInputs, AnalysisResult
from v1_simulation.config.schema import AnalysisConfig


def run_analysis(
    config: AnalysisConfig,
    inputs: AnalysisInputs,
    *,
    rng: np.random.Generator | None = None,
) -> AnalysisResult:
    """Executes the complete post-simulation analysis pipeline.

    Extracts steady-state responses, calculates OSI, filters and samples neurons,
    downsamples (decimates) activity traces, detects ensembles using consensus Louvain,
    and computes global and cluster-specific metrics.

    Args:
        config: Configuration containing analysis parameters (thresholds, fractions, etc.).
        inputs: Input data containing firing rates, coordinates, distances, and orientations.
        rng: Optional random number generator for reproducible sampling.

    Returns:
        An AnalysisResult object containing filtered states, detected communities, and diagnostics.
    """
    inputs.validate()
    local_rng = _analysis_rng(config, rng)

    responses = np.asarray(inputs.responses, dtype=float)
    steady_start = int(responses.shape[2] * 2 / 3)
    steady_state = responses[:, :, steady_start:]
    responses_mean = np.mean(steady_state, axis=2)

    osi, pref_ori = compute_osi(responses_mean, inputs.theta_angles, min_osi=config.osi_threshold)
    selected = select_analysis_neuron_indices(osi, config=config, rng=local_rng)

    diagnostics = {
        "seed": None if config.seed is None else int(config.seed),
        "steady_state_start": int(steady_start),
        "candidate_neurons_before_osi_filter": int(osi.size),
        "candidate_neurons_after_osi_filter": int(np.sum(np.isfinite(osi) & (osi >= config.osi_threshold))),
        "selected_neurons_for_analysis": int(selected.size),
        "osi_threshold": float(config.osi_threshold),
        "random_sample_fraction": float(config.random_sample_fraction),
    }
    diagnostics.update(activity_health_metrics(responses_mean, active_threshold=config.active_threshold))

    if selected.size < 2:
        summary, rows = summarize_communities(
            np.zeros(selected.size, dtype=np.int64),
            distance=np.asarray(inputs.distance, dtype=float)[np.ix_(selected, selected)],
            coords=np.asarray(inputs.coords, dtype=float)[selected],
            osi=osi[selected],
            pref_ori=pref_ori[selected],
        )
        diagnostics["metrics_summary"] = summary
        diagnostics["ensemble_metrics"] = rows
        return AnalysisResult(
            status="not_enough_neurons",
            selected_indices=selected,
            osi=osi[selected],
            pref_ori=pref_ori[selected],
            responses_mean=responses_mean[selected],
            steady_state_responses=steady_state[selected],
            coords=np.asarray(inputs.coords, dtype=float)[selected],
            distance=np.asarray(inputs.distance, dtype=float)[np.ix_(selected, selected)],
            communities=None,
            diagnostics=diagnostics,
        )

    selected_steady = steady_state[selected]
    activity_trace, decimation_factor = ensemble_activity_trace(selected_steady)
    diagnostics["activity_decimation_factor"] = int(decimation_factor)

    communities = identify_communities(
        activity_trace,
        config.louvain,
        rng=local_rng,
    )
    selected_distance = np.asarray(inputs.distance, dtype=float)[np.ix_(selected, selected)]
    selected_coords = np.asarray(inputs.coords, dtype=float)[selected]
    summary, rows = summarize_communities(
        communities.labels,
        similarity=communities.similarity,
        distance=selected_distance,
        coords=selected_coords,
        osi=osi[selected],
        pref_ori=pref_ori[selected],
    )
    diagnostics["metrics_summary"] = summary
    diagnostics["ensemble_metrics"] = rows

    return AnalysisResult(
        status="ok",
        selected_indices=selected,
        osi=osi[selected],
        pref_ori=pref_ori[selected],
        responses_mean=responses_mean[selected],
        steady_state_responses=selected_steady,
        coords=selected_coords,
        distance=selected_distance,
        communities=communities,
        diagnostics=diagnostics,
    )


def select_analysis_neuron_indices(
    osi: ArrayLike,
    *,
    config: AnalysisConfig,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Selects neuron indices based on an OSI threshold and a random sampling fraction.

    Args:
        osi: Orientation Selectivity Index values for all neurons.
        config: Analysis config holding osi_threshold and random_sample_fraction.
        rng: Optional random number generator.

    Returns:
        An array of selected indices.

    Raises:
        ValueError: If config parameters are out of range.
    """
    if not 0.0 <= float(config.osi_threshold) <= 1.0:
        raise ValueError("config.osi_threshold must be in [0, 1].")
    if not 0.0 < float(config.random_sample_fraction) <= 1.0:
        raise ValueError("config.random_sample_fraction must be in (0, 1].")

    values = np.asarray(osi, dtype=float)
    candidate_indices = np.flatnonzero(np.isfinite(values) & (values >= float(config.osi_threshold)))
    if candidate_indices.size == 0:
        return candidate_indices.astype(np.int64, copy=False)

    sample_size = max(1, int(round(candidate_indices.size * float(config.random_sample_fraction))))
    sample_size = min(sample_size, candidate_indices.size)
    if sample_size == candidate_indices.size:
        return candidate_indices.astype(np.int64, copy=True)

    local_rng = _analysis_rng(config, rng)
    selected = local_rng.choice(candidate_indices, size=sample_size, replace=False)
    selected.sort()
    return selected.astype(np.int64, copy=False)


def ensemble_activity_trace(
    steady_state_responses: ArrayLike,
    *,
    source_hz: float = 100.0,
    target_hz: float = 4.0,
) -> tuple[NDArray[np.float64], int]:
    """Decimates and reshapes firing rate responses for community detection.

    Reduces the temporal sampling rate from source_hz to target_hz along the time axis,
    and then flattens the orientation and time axes into a single dimension per neuron.

    Args:
        steady_state_responses: Firing rates of shape (n_neurons, n_theta, n_time).
        source_hz: Original sampling frequency in Hz.
        target_hz: Target downsampled frequency in Hz.

    Returns:
        A tuple of:
            - concatenated_trace: Firing rates of shape (n_neurons, n_theta * downsampled_time).
            - factor: The decimation downsampling factor used.

    Raises:
        ValueError: If steady_state_responses is not 3D, or frequency values are not positive.
    """
    responses = np.asarray(steady_state_responses, dtype=float)
    if responses.ndim != 3:
        raise ValueError("steady_state_responses must have shape (n_neurons, n_theta, n_time).")
    if source_hz <= 0.0 or target_hz <= 0.0:
        raise ValueError("source_hz and target_hz must be positive.")

    factor = max(1, int(round(float(source_hz) / float(target_hz))))
    if factor > 1 and responses.shape[2] > 3 * factor:
        filtered = signal.decimate(responses, factor, axis=2)
        return filtered.reshape(responses.shape[0], -1), factor
    return responses.reshape(responses.shape[0], -1), 1


def _analysis_rng(config: AnalysisConfig, rng: np.random.Generator | None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(config.seed)
