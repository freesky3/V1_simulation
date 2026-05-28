from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike

from v1_simulation.analysis.clusters import cluster_members, labels_array
from v1_simulation.analysis.spatial import cluster_spatial_metrics
from v1_simulation.io.artifacts import json_ready

METRIC_SCHEMA_VERSION = 2


def activity_health_metrics(responses: ArrayLike, *, active_threshold: float = 1.0e-6) -> dict[str, Any]:
    """Computes rate-based activity health statistics across neurons.

    Args:
        responses: Neuron responses as 1D, 2D, or 3D ArrayLike (e.g., shape [neurons, orientations, time]).
        active_threshold: Firing rate threshold above which a neuron is classified as active.

    Returns:
        A dictionary containing stats such as active neuron counts, fraction, silent fraction,
        and mean/median/percentile firing rates.

    Raises:
        ValueError: If responses has an unsupported shape.
    """
    values = np.asarray(responses, dtype=float)
    if values.ndim == 1:
        per_neuron = values
    elif values.ndim == 2:
        per_neuron = np.nanmean(values, axis=1)
    elif values.ndim == 3:
        per_neuron = np.nanmean(values, axis=(1, 2))
    else:
        raise ValueError("responses must be 1D, 2D, or 3D.")

    per_neuron = np.nan_to_num(per_neuron, nan=0.0, posinf=0.0, neginf=0.0)
    active = per_neuron > float(active_threshold)
    total_activity = float(np.sum(np.maximum(per_neuron, 0.0)))
    sorted_activity = np.sort(np.maximum(per_neuron, 0.0))[::-1]
    top1 = float(sorted_activity[0] / total_activity) if total_activity > 0.0 else None
    top5 = (
        float(np.sum(sorted_activity[: min(5, sorted_activity.size)]) / total_activity)
        if total_activity > 0.0
        else None
    )

    return json_ready(
        {
            "active_threshold": float(active_threshold),
            "active_neuron_count": int(np.sum(active)),
            "active_fraction": _finite_float(np.mean(active)) if per_neuron.size else None,
            "silent_fraction": _finite_float(np.mean(~active)) if per_neuron.size else None,
            "rate_mean": _safe_mean(per_neuron),
            "rate_median": _safe_median(per_neuron),
            "rate_p95": _finite_float(np.percentile(per_neuron, 95)) if per_neuron.size else None,
            "rate_max": _finite_float(np.max(per_neuron)) if per_neuron.size else None,
            "top1_activity_fraction": top1,
            "top5_activity_fraction": top5,
        }
    )


def osi_distribution_metrics(osi: ArrayLike) -> dict[str, Any]:
    """Computes distribution metrics (mean, median, std, fractions above thresholds) of OSI.

    Args:
        osi: Orientation Selectivity Index values for a population of neurons.

    Returns:
        A dictionary containing population-level OSI statistics.
    """
    values = np.asarray(osi, dtype=float)
    finite = values[np.isfinite(values)]
    return json_ready(
        {
            "n_neurons": int(values.size),
            "osi_finite_count": int(finite.size),
            "osi_finite_fraction": _finite_float(finite.size / values.size) if values.size else None,
            "osi_mean": _safe_mean(finite),
            "osi_median": _safe_median(finite),
            "osi_std": _safe_std(finite),
            "osi_count_gt_0_2": int(np.sum(finite > 0.2)),
            "osi_count_gt_0_4": int(np.sum(finite > 0.4)),
            "osi_count_gt_0_5": int(np.sum(finite > 0.5)),
            "osi_count_gt_0_6": int(np.sum(finite > 0.6)),
            "osi_fraction_gt_0_2": _finite_float(np.mean(finite > 0.2)) if finite.size else None,
            "osi_fraction_gt_0_4": _finite_float(np.mean(finite > 0.4)) if finite.size else None,
            "osi_fraction_gt_0_5": _finite_float(np.mean(finite > 0.5)) if finite.size else None,
            "osi_fraction_gt_0_6": _finite_float(np.mean(finite > 0.6)) if finite.size else None,
        }
    )


def summarize_communities(
    labels: ArrayLike,
    *,
    similarity: ArrayLike | None = None,
    distance: ArrayLike | None = None,
    coords: ArrayLike | None = None,
    osi: ArrayLike | None = None,
    pref_ori: ArrayLike | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generates global and ensemble-specific summary metrics.

    Calculates size, similarity profiles, centroid positions, spatial metrics,
    and preferred orientation coherence for detected neuron ensembles.

    Args:
        labels: Assigned community labels for all neurons.
        similarity: Optional pairwise similarity matrix of shape (n_neurons, n_neurons).
        distance: Optional pairwise distance matrix of shape (n_neurons, n_neurons).
        coords: Optional spatial coordinates of shape (n_neurons, 2).
        osi: Optional Orientation Selectivity Index values of shape (n_neurons,).
        pref_ori: Optional preferred orientation in radians of shape (n_neurons,).

    Returns:
        A tuple of:
            - summary: Global community statistics.
            - rows: A list of dictionaries representing metrics for each individual ensemble.
    """
    label_values = labels_array(labels)
    n_neurons = label_values.size
    clusters = cluster_members(label_values)
    classified = label_values != 0
    rows: list[dict[str, Any]] = []

    similarity_values = None if similarity is None else np.asarray(similarity, dtype=float)
    distance_values = None if distance is None else np.asarray(distance, dtype=float)
    coords_values = None if coords is None else np.asarray(coords, dtype=float)
    osi_values = None if osi is None else np.asarray(osi, dtype=float)
    pref_values = None if pref_ori is None else np.asarray(pref_ori, dtype=float)

    spatial = cluster_spatial_metrics(label_values, distance_values) if distance_values is not None else {}

    for c_id, members in clusters.items():
        row: dict[str, Any] = {
            "ensemble_id": int(c_id),
            "size": int(members.size),
            "member_fraction": _finite_float(members.size / n_neurons) if n_neurons else None,
        }
        if similarity_values is not None:
            within = _offdiag_values(similarity_values[np.ix_(members, members)])
            outside = np.flatnonzero((label_values != c_id) & classified)
            between = (
                similarity_values[np.ix_(members, outside)].ravel()
                if outside.size
                else np.array([], dtype=float)
            )
            row["within_similarity_mean"] = _safe_mean(within)
            row["outside_similarity_mean"] = _safe_mean(between[np.isfinite(between)])
        if coords_values is not None:
            centroid = np.mean(coords_values[members], axis=0)
            row["centroid_x"] = _finite_float(centroid[0])
            row["centroid_y"] = _finite_float(centroid[1])
        row.update(spatial.get(c_id, {}))
        if osi_values is not None:
            row["member_osi_mean"] = _safe_mean(osi_values[members])
            row["member_osi_median"] = _safe_median(osi_values[members])
        if pref_values is not None:
            prefs = pref_values[members]
            prefs = prefs[np.isfinite(prefs)]
            row["member_pref_ori_coherence"] = (
                _finite_float(np.abs(np.mean(np.exp(2j * prefs)))) if prefs.size else None
            )
        rows.append(row)

    sizes = [row["size"] for row in rows]
    summary = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "n_neurons": int(n_neurons),
        "n_ensembles": int(len(rows)),
        "classified_neurons": int(np.sum(classified)),
        "unclassified_neurons": int(np.sum(~classified)),
        "classified_fraction": _finite_float(np.mean(classified)) if n_neurons else None,
        "ensemble_size_mean": _safe_mean(sizes),
        "ensemble_size_median": _safe_median(sizes),
        "ensemble_size_min": int(np.min(sizes)) if sizes else None,
        "ensemble_size_max": int(np.max(sizes)) if sizes else None,
    }
    if similarity_values is not None:
        upper = np.triu_indices(n_neurons, k=1)
        left = label_values[upper[0]]
        right = label_values[upper[1]]
        valid = (left != 0) & (right != 0)
        within = valid & (left == right)
        between = valid & (left != right)
        upper_similarity = similarity_values[upper]
        summary["within_similarity_mean"] = _safe_mean(upper_similarity[within])
        summary["between_similarity_mean"] = _safe_mean(upper_similarity[between])
    if osi_values is not None:
        summary.update(osi_distribution_metrics(osi_values))
    return json_ready(summary), json_ready(rows)


def write_analysis_metrics(
    summary: dict[str, Any],
    ensemble_rows: Iterable[dict[str, Any]],
    save_dir: str | Path,
) -> tuple[Path, Path]:
    """Writes global summary metrics to a JSON file and ensemble metrics to a CSV file.

    Args:
        summary: Global analysis metrics dictionary.
        ensemble_rows: Sequence of ensemble metric dictionaries.
        save_dir: Directory where the output files should be written.

    Returns:
        A tuple of:
            - summary_path: Path to the written JSON file.
            - ensemble_path: Path to the written CSV file.
    """
    target = Path(save_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary_metrics.json"
    ensemble_path = target / "ensemble_metrics.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(summary), f, indent=2)

    rows = [json_ready(row) for row in ensemble_rows]
    fieldnames = _ordered_fieldnames(rows)
    with ensemble_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path, ensemble_path


def _ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "ensemble_id",
        "size",
        "member_fraction",
        "within_similarity_mean",
        "outside_similarity_mean",
        "mean_pairwise_distance",
        "nearest_neighbor_distance",
        "centroid_x",
        "centroid_y",
        "member_osi_mean",
        "member_osi_median",
        "member_pref_ori_coherence",
    ]
    present = {key for row in rows for key in row}
    return [key for key in preferred if key in present] + sorted(present - set(preferred))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _safe_mean(values: Iterable[Any]) -> float | None:
    finite = _finite_values(values)
    return float(np.mean(finite)) if finite.size else None


def _safe_median(values: Iterable[Any]) -> float | None:
    finite = _finite_values(values)
    return float(np.median(finite)) if finite.size else None


def _safe_std(values: Iterable[Any]) -> float | None:
    finite = _finite_values(values)
    return float(np.std(finite, ddof=1)) if finite.size > 1 else None


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    return arr[np.isfinite(arr)]


def _offdiag_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.array([], dtype=float)
    values = matrix[np.triu_indices_from(matrix, k=1)]
    return values[np.isfinite(values)]
