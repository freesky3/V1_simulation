from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v1_simulation.analysis.metrics import write_analysis_metrics
from v1_simulation.analysis.types import AnalysisInputs, AnalysisResult
from v1_simulation.io.artifacts import json_ready
from v1_simulation.network.state import load_trained_network_state


def load_analysis_inputs_from_run(
    run_dir: str | Path,
    *,
    network_path: str | Path | None = None,
) -> AnalysisInputs:
    """Load ``AnalysisInputs`` from a persisted simulation run directory."""

    root = Path(run_dir)
    responses_path = root / "responses_exc.npy"
    theta_path = root / "theta_angles.npy"
    if not responses_path.exists():
        raise FileNotFoundError(f"Missing excitatory response trajectory: {responses_path}")
    if not theta_path.exists():
        raise FileNotFoundError(f"Missing stimulus orientation array: {theta_path}")

    checkpoint_path = Path(network_path) if network_path is not None else root / "network"
    network = load_trained_network_state(checkpoint_path).network
    responses = np.load(responses_path)
    theta_angles = np.load(theta_path)

    l23 = network.layout.l23
    exc_idx = network.idx_E
    coords = l23.coords[exc_idx]
    distance = l23.distance_matrix()[np.ix_(exc_idx, exc_idx)]

    run_config_path = root / "run_config.json"
    center_side_fraction = 1.0
    if run_config_path.exists():
        try:
            with run_config_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
                center_side_fraction = float(meta.get("config", {}).get("analysis", {}).get("center_side_fraction", 1.0))
        except Exception:
            pass

    if center_side_fraction < 1.0:
        half_side = (l23.region_size * center_side_fraction) / 2.0
        in_center_E = (np.abs(coords[:, 0]) <= half_side) & (np.abs(coords[:, 1]) <= half_side)
        coords = coords[in_center_E]
        distance = distance[np.ix_(in_center_E, in_center_E)]

    return AnalysisInputs(
        responses=np.asarray(responses, dtype=float),
        coords=coords,
        distance=distance,
        theta_angles=np.asarray(theta_angles, dtype=float),
    )


def write_analysis_result_artifacts(
    result: AnalysisResult,
    output_dir: str | Path,
    *,
    save_plots: bool = True,
    num_surrogates: int = 10000,
) -> Path:
    """Persist arrays, diagnostics, and tabular metrics from an analysis result."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    np.save(target / "selected_indices.npy", result.selected_indices)
    np.save(target / "osi.npy", result.osi)
    np.save(target / "pref_ori.npy", result.pref_ori)
    np.save(target / "responses_mean.npy", result.responses_mean)
    np.save(target / "steady_state_responses.npy", result.steady_state_responses)
    np.save(target / "coords.npy", result.coords)
    np.save(target / "distance.npy", result.distance)

    if result.communities is not None:
        np.save(target / "community_labels.npy", result.communities.labels)
        np.save(target / "similarity.npy", result.communities.similarity)
        if result.communities.agreement is not None:
            np.save(target / "agreement.npy", result.communities.agreement)
        with (target / "community_diagnostics.json").open("w", encoding="utf-8") as f:
            json.dump(json_ready(result.communities.diagnostics), f, indent=2)

    diagnostics = json_ready(result.diagnostics)
    with (target / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    summary = diagnostics.get("metrics_summary", {})
    rows = diagnostics.get("ensemble_metrics", [])
    if isinstance(summary, dict) and isinstance(rows, list):
        write_analysis_metrics(summary, rows, target)

    if save_plots:
        from v1_simulation.analysis.plotting import generate_and_save_all_analysis_plots
        generate_and_save_all_analysis_plots(result, target, num_surrogates=num_surrogates)

    return target


__all__ = ["load_analysis_inputs_from_run", "write_analysis_result_artifacts"]
