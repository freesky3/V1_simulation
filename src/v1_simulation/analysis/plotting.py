from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# Safely handle Colab/Jupyter inherited MPLBACKEND environment variable
if os.environ.get('MPLBACKEND') == 'module://matplotlib_inline.backend_inline':
    try:
        import matplotlib_inline
    except ImportError:
        os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
try:
    import matplotlib.pyplot as plt
except ValueError:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import numpy as np

from v1_simulation.analysis.types import AnalysisResult

logger = logging.getLogger(__name__)

# Centralized Filename Constants
OSI_HISTOGRAM_FILENAME = "osi_histogram.png"
OSI_SPATIAL_FILENAME = "osi_spatial_distribution.png"
PREF_ORI_SPATIAL_FILENAME = "pref_ori_spatial_distribution.png"
ORI_CENTERS_SPATIAL_FILENAME = "ori_centers_spatial.png"
ENSEMBLE_CORRELATION_FILENAME = "ensemble_correlation.png"
ENSEMBLE_ACTIVITY_TRACE_FILENAME = "ensemble_activity_trace.png"
ENSEMBLE_ACTIVITY_TRACE_NORM_VAR_FILENAME = "ensemble_activity_trace_normalized_variance.png"

# Prefixes and suffixes for dynamic names
ENSEMBLE_SPATIAL_PREFIX = "ensemble_spatial_"
SPATIAL_SURROGATE_PREFIX = "spatial_metrics_cluster_"
SPATIAL_SURROGATE_NND_SUFFIX = "_NND.png"
SPATIAL_SURROGATE_MEANDIST_SUFFIX = "_MeanDist.png"


def _finite_axis_max(values: np.ndarray, floor: float, scale: float = 1.1) -> float:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return floor
    return max(floor, float(np.max(finite_values)) * scale)


def _set_three_axis_ticks(ax: plt.Axes, axis_max: float, formatter: Any) -> None:
    ticks = [0.0, axis_max / 2.0, axis_max]
    ax.set_yticks(ticks)
    ax.set_yticklabels([formatter(v) for v in ticks])


def ensemble_clusters(partition: dict[int, int], *, include_unclassified: bool = False) -> dict[int, list[int]]:
    """Group neuron ids by ensemble id.

    Cluster id 0 is treated as unclassified and excluded by default.
    """
    clusters: dict[int, list[int]] = {}
    for n_id, c_id in partition.items():
        if c_id == 0 and not include_unclassified:
            continue
        clusters.setdefault(int(c_id), []).append(int(n_id))

    grouped = sorted(clusters.items())
    return {c_id: members for c_id, members in grouped}


def _cluster_spatial_values(
    dist_matrix: np.ndarray,
    members: list[int] | np.ndarray,
) -> tuple[float, np.ndarray]:
    """Helper to compute mean distance and NND values for a set of members."""
    sub_dist = np.array(dist_matrix[np.ix_(members, members)], copy=True)
    if sub_dist.shape[0] < 2:
        return np.nan, np.array([], dtype=float)

    # Pairwise mean distance (triangular upper indices without diagonal)
    pairwise = sub_dist[np.triu_indices_from(sub_dist, k=1)]
    pairwise = pairwise[np.isfinite(pairwise)]
    if pairwise.size == 0:
        mean_dist = np.nan
    else:
        mean_dist = float(np.mean(pairwise))

    np.fill_diagonal(sub_dist, np.inf)
    nnd_values = np.min(sub_dist, axis=1)
    return mean_dist, nnd_values[np.isfinite(nnd_values)]


def prepare_ensemble_activity_trace_plot_data(result: AnalysisResult, max_clusters: int = 6) -> dict[str, Any]:
    """Prepares and structures the trace data for plotting from AnalysisResult."""
    if result.communities is None:
        return {
            "N_theta": 0,
            "T_steps": 0,
            "x": np.array([], dtype=int),
            "traces": [],
        }

    steady_state_responses = result.steady_state_responses
    if steady_state_responses.ndim != 3:
        raise ValueError("steady_state_responses must have shape (N_neurons, N_theta, T_steps).")

    labels = result.communities.labels
    partition = {i: int(label) for i, label in enumerate(labels)}
    clusters = ensemble_clusters(partition)
    unique_clusters = sorted(clusters.keys())

    N_theta = steady_state_responses.shape[1]
    T_steps = steady_state_responses.shape[2]
    total_frames = N_theta * T_steps
    x = np.arange(total_frames)
    traces = []

    clusters_to_plot = unique_clusters[:min(len(unique_clusters), max_clusters)]
    for c_id in clusters_to_plot:
        members = clusters[c_id]
        ensemble_trace = steady_state_responses[members, :, :]
        flat_members = ensemble_trace.reshape(len(members), -1)
        member_norms = np.linalg.norm(flat_members, axis=1, keepdims=True)
        normalized_members = np.divide(
            flat_members,
            member_norms,
            out=np.zeros_like(flat_members, dtype=float),
            where=member_norms > 0.0,
        )

        mean_trace = np.mean(ensemble_trace, axis=0)
        variance_trace = np.var(ensemble_trace, axis=0)
        normalized_mean = np.mean(normalized_members, axis=0)
        normalized_variance = np.var(normalized_members, axis=0)

        flat_trace = mean_trace.flatten()
        flat_variance = variance_trace.flatten()
        flat_std = np.sqrt(np.maximum(0.0, flat_variance))

        traces.append({
            "cluster_id": int(c_id),
            "mean": flat_trace,
            "variance": flat_variance,
            "std": flat_std,
            "normalized_mean": normalized_mean.flatten(),
            "normalized_variance": normalized_variance.flatten(),
            "normalized_std": np.sqrt(np.maximum(0.0, normalized_variance)).flatten(),
        })

    return {
        "N_theta": int(N_theta),
        "T_steps": int(T_steps),
        "x": x,
        "traces": traces,
    }


def prepare_spatial_surrogate_plot_data(
    result: AnalysisResult,
    num_surrogates: int = 10000,
    target_cluster: int = 1,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Generates the spatial surrogate distributions comparison data by shuffling coordinates."""
    if num_surrogates < 1:
        raise ValueError("num_surrogates must be at least 1.")

    if result.communities is None:
        return {
            "has_valid_ensembles": False,
            "target_cluster": int(target_cluster),
            "target_cluster_available": False,
            "actual_NNDs": np.array([], dtype=float),
            "mean_surrogate_NND": np.array([], dtype=float),
            "percentile_2_5": np.array([], dtype=float),
            "percentile_97_5": np.array([], dtype=float),
            "target_cluster_random_mean_dists": np.array([], dtype=float),
            "actual_target_mean_dist": None,
            "target_threshold_p05": None,
        }

    labels = result.communities.labels
    partition = {i: int(label) for i, label in enumerate(labels)}
    clusters = ensemble_clusters(partition)
    cluster_sizes = {c_id: len(members) for c_id, members in clusters.items() if len(members) >= 2}

    dist_matrix = result.distance
    all_neurons = np.arange(dist_matrix.shape[0])

    actual_NNDs: list[float] = []
    actual_mean_dist: dict[int, float] = {}

    for c_id, members in clusters.items():
        if len(members) < 2:
            continue
        mean_dist, nnd_values = _cluster_spatial_values(dist_matrix, members)
        actual_mean_dist[c_id] = mean_dist
        actual_NNDs.extend(nnd_values)

    actual_NNDs_sorted = np.sort(actual_NNDs)

    if len(actual_NNDs_sorted) == 0 or len(cluster_sizes) == 0:
        return {
            "has_valid_ensembles": False,
            "target_cluster": int(target_cluster),
            "target_cluster_available": False,
            "actual_NNDs": actual_NNDs_sorted,
            "mean_surrogate_NND": np.array([], dtype=float),
            "percentile_2_5": np.array([], dtype=float),
            "percentile_97_5": np.array([], dtype=float),
            "target_cluster_random_mean_dists": np.array([], dtype=float),
            "actual_target_mean_dist": None,
            "target_threshold_p05": None,
        }

    local_rng = np.random.default_rng() if rng is None else rng

    surrogate_NNDs_matrix = []
    target_cluster_random_mean_dists = []

    for _ in range(num_surrogates):
        shuffled_neurons = local_rng.permutation(all_neurons)
        idx = 0
        surr_NNDs_this_round = []
        for c_id, size in cluster_sizes.items():
            members = shuffled_neurons[idx:idx+size]
            idx += size

            mean_dist, nnd_values = _cluster_spatial_values(dist_matrix, members)
            if c_id == target_cluster:
                target_cluster_random_mean_dists.append(mean_dist)

            surr_NNDs_this_round.extend(nnd_values)

        surrogate_NNDs_matrix.append(np.sort(surr_NNDs_this_round))

    surrogate_NNDs_matrix = np.array(surrogate_NNDs_matrix)

    mean_surrogate_NND = np.mean(surrogate_NNDs_matrix, axis=0)
    percentile_2_5 = np.percentile(surrogate_NNDs_matrix, 2.5, axis=0)
    percentile_97_5 = np.percentile(surrogate_NNDs_matrix, 97.5, axis=0)

    target_cluster_random_mean_dists = np.asarray(target_cluster_random_mean_dists, dtype=float)
    target_cluster_available = target_cluster in cluster_sizes

    if target_cluster_available and target_cluster_random_mean_dists.size > 0:
        target_threshold_p05 = np.percentile(target_cluster_random_mean_dists, 5)
    else:
        target_threshold_p05 = None

    return {
        "has_valid_ensembles": True,
        "target_cluster": int(target_cluster),
        "target_cluster_available": bool(target_cluster_available),
        "actual_NNDs": actual_NNDs_sorted,
        "mean_surrogate_NND": mean_surrogate_NND,
        "percentile_2_5": percentile_2_5,
        "percentile_97_5": percentile_97_5,
        "target_cluster_random_mean_dists": target_cluster_random_mean_dists,
        "actual_target_mean_dist": actual_mean_dist.get(target_cluster),
        "target_threshold_p05": target_threshold_p05,
    }


def spatial_surrogate_summary(
    result: AnalysisResult,
    *,
    num_surrogates: int = 10000,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    """Summarizes spatial compactness and component separation against random labels."""
    if num_surrogates < 1:
        raise ValueError("num_surrogates must be at least 1.")
    if result.communities is None:
        return {
            "num_surrogates": int(num_surrogates),
            "has_valid_ensembles": False,
            "clusters": [],
            "component_centroid_distance": None,
        }

    labels = result.communities.labels
    partition = {i: int(label) for i, label in enumerate(labels)}
    clusters = ensemble_clusters(partition)
    valid_clusters = {c_id: members for c_id, members in clusters.items() if len(members) >= 2}
    rng = np.random.default_rng(rng_seed)

    cluster_rows: list[dict[str, Any]] = []
    for c_id in sorted(valid_clusters):
        surrogate_data = prepare_spatial_surrogate_plot_data(
            result,
            num_surrogates=num_surrogates,
            target_cluster=c_id,
            rng=rng,
        )
        random_mean_dists = np.asarray(surrogate_data["target_cluster_random_mean_dists"], dtype=float)
        actual_mean_dist = surrogate_data["actual_target_mean_dist"]
        actual_nnd = np.asarray(surrogate_data["actual_NNDs"], dtype=float)
        surrogate_nnd = np.asarray(surrogate_data["mean_surrogate_NND"], dtype=float)

        cluster_rows.append(
            {
                "ensemble_id": int(c_id),
                "size": int(len(valid_clusters[c_id])),
                "actual_mean_pairwise_distance": _optional_float(actual_mean_dist),
                "surrogate_mean_pairwise_distance_mean": _safe_array_mean(random_mean_dists),
                "surrogate_mean_pairwise_distance_p05": _safe_array_percentile(random_mean_dists, 5),
                "surrogate_mean_pairwise_distance_p50": _safe_array_percentile(random_mean_dists, 50),
                "surrogate_mean_pairwise_distance_p95": _safe_array_percentile(random_mean_dists, 95),
                "p_random_mean_distance_le_actual": (
                    None
                    if actual_mean_dist is None or random_mean_dists.size == 0
                    else float(np.mean(random_mean_dists <= float(actual_mean_dist)))
                ),
                "actual_nnd_mean": _safe_array_mean(actual_nnd),
                "surrogate_nnd_mean": _safe_array_mean(surrogate_nnd),
                "fraction_actual_nnd_below_surrogate_mean": (
                    None
                    if actual_nnd.size == 0 or surrogate_nnd.size == 0
                    else float(np.mean(actual_nnd < surrogate_nnd))
                ),
            }
        )

    return {
        "num_surrogates": int(num_surrogates),
        "has_valid_ensembles": bool(valid_clusters),
        "clusters": cluster_rows,
        "component_centroid_distance": _component_centroid_surrogate_summary(
            result,
            valid_clusters,
            num_surrogates=num_surrogates,
            rng=rng,
        ),
    }


def _component_centroid_surrogate_summary(
    result: AnalysisResult,
    clusters: dict[int, list[int]],
    *,
    num_surrogates: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    if len(clusters) < 2:
        return None
    coords = np.asarray(result.coords, dtype=float)
    all_neurons = np.arange(coords.shape[0])
    cluster_sizes = [len(members) for _, members in sorted(clusters.items())]

    actual_mean, actual_min = _centroid_pairwise_distances(coords, [clusters[c] for c in sorted(clusters)])
    random_mean_values: list[float] = []
    random_min_values: list[float] = []
    for _ in range(num_surrogates):
        shuffled = rng.permutation(all_neurons)
        idx = 0
        random_members = []
        for size in cluster_sizes:
            random_members.append(shuffled[idx : idx + size])
            idx += size
        mean_dist, min_dist = _centroid_pairwise_distances(coords, random_members)
        if np.isfinite(mean_dist):
            random_mean_values.append(mean_dist)
        if np.isfinite(min_dist):
            random_min_values.append(min_dist)

    random_mean = np.asarray(random_mean_values, dtype=float)
    random_min = np.asarray(random_min_values, dtype=float)
    return {
        "actual_centroid_pairwise_mean": _optional_float(actual_mean),
        "surrogate_centroid_pairwise_mean_mean": _safe_array_mean(random_mean),
        "surrogate_centroid_pairwise_mean_p05": _safe_array_percentile(random_mean, 5),
        "surrogate_centroid_pairwise_mean_p50": _safe_array_percentile(random_mean, 50),
        "surrogate_centroid_pairwise_mean_p95": _safe_array_percentile(random_mean, 95),
        "p_random_centroid_pairwise_mean_ge_actual": (
            None if random_mean.size == 0 else float(np.mean(random_mean >= actual_mean))
        ),
        "percentile_random_centroid_pairwise_mean_le_actual": (
            None if random_mean.size == 0 else float(np.mean(random_mean <= actual_mean))
        ),
        "actual_centroid_pairwise_min": _optional_float(actual_min),
        "surrogate_centroid_pairwise_min_mean": _safe_array_mean(random_min),
        "p_random_centroid_pairwise_min_ge_actual": (
            None if random_min.size == 0 else float(np.mean(random_min >= actual_min))
        ),
        "percentile_random_centroid_pairwise_min_le_actual": (
            None if random_min.size == 0 else float(np.mean(random_min <= actual_min))
        ),
    }


def _centroid_pairwise_distances(coords: np.ndarray, cluster_members: list[list[int] | np.ndarray]) -> tuple[float, float]:
    centroids = np.asarray([np.mean(coords[np.asarray(members, dtype=int)], axis=0) for members in cluster_members])
    if centroids.shape[0] < 2:
        return float("nan"), float("nan")
    distances = []
    for i in range(centroids.shape[0]):
        for j in range(i + 1, centroids.shape[0]):
            distances.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    values = np.asarray(distances, dtype=float)
    return float(np.mean(values)), float(np.min(values))


def _safe_array_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else None


def _safe_array_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def plot_osi_results(result: AnalysisResult, output_dir: Path) -> list[Path]:
    """Plots and saves OSI histogram, spatial distribution, preferred orientation, and orientation centers."""
    os.makedirs(output_dir, exist_ok=True)
    generated_paths: list[Path] = []

    osi = result.osi
    pref_ori = result.pref_ori
    neuron_coords = result.coords

    if osi.size == 0 or neuron_coords.size == 0:
        logger.warning("Empty OSI or coordinates data. Skipping OSI plots.")
        return []

    # ---- 1. Histogram of OSI ----
    try:
        plt.figure(figsize=(6, 4))
        valid_osi = osi[np.isfinite(osi)]
        if valid_osi.size > 0:
            plt.hist(valid_osi, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel("OSI", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title("Distribution of Global OSI", fontsize=14)
        plt.tight_layout()
        hist_path = output_dir / OSI_HISTOGRAM_FILENAME
        plt.savefig(hist_path, dpi=300)
        generated_paths.append(hist_path)
    except Exception as e:
        logger.error("Failed to plot OSI histogram: %s", e)
    finally:
        plt.close()

    # ---- 2. Spatial Distribution of OSI ----
    try:
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(neuron_coords[:, 0], neuron_coords[:, 1], c=osi, cmap='viridis', s=20)
        plt.colorbar(sc, label='OSI')
        plt.title("Spatial Distribution of OSI", fontsize=14)
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        spatial_path = output_dir / OSI_SPATIAL_FILENAME
        plt.savefig(spatial_path, dpi=300)
        generated_paths.append(spatial_path)
    except Exception as e:
        logger.error("Failed to plot OSI spatial distribution: %s", e)
    finally:
        plt.close()

    # ---- 3. Spatial Distribution of Preferred Orientation ----
    try:
        plt.figure(figsize=(6, 5))
        valid = ~np.isnan(pref_ori) & np.isfinite(pref_ori)
        plt.scatter(
            neuron_coords[~valid, 0],
            neuron_coords[~valid, 1],
            color='lightgray',
            s=10,
            alpha=0.5,
            label='Low OSI',
        )
        if np.sum(valid) > 0:
            sc = plt.scatter(
                neuron_coords[valid, 0],
                neuron_coords[valid, 1],
                c=pref_ori[valid],
                cmap='hsv',
                s=20,
            )
            plt.colorbar(sc, label='Preferred Orientation (rad)')
        plt.title("Preferred Orientation", fontsize=14)
        plt.axis('equal')
        plt.axis('off')
        plt.legend(frameon=False)
        plt.tight_layout()
        pref_path = output_dir / PREF_ORI_SPATIAL_FILENAME
        plt.savefig(pref_path, dpi=300)
        generated_paths.append(pref_path)
    except Exception as e:
        logger.error("Failed to plot preferred orientation spatial distribution: %s", e)
    finally:
        plt.close()

    # ---- 4. Centers of Preferred Orientations ----
    try:
        plt.figure(figsize=(5, 5))
        plt.scatter(
            neuron_coords[~valid, 0],
            neuron_coords[~valid, 1],
            facecolors='none',
            edgecolors='lightgray',
            s=30,
            alpha=0.8,
        )
        if np.sum(valid) > 0:
            pref_ori_mod = np.mod(pref_ori[valid], np.pi)
            n_bins = 4
            bins = np.linspace(0, np.pi, n_bins + 1)
            colors = ['#e377c2', '#1f77b4', '#8A2BE2', '#DAA520']
            valid_coords = neuron_coords[valid]
            for i in range(n_bins):
                if i == n_bins - 1:
                    idx = (pref_ori_mod >= bins[i]) & (pref_ori_mod <= bins[i + 1])
                else:
                    idx = (pref_ori_mod >= bins[i]) & (pref_ori_mod < bins[i + 1])
                group_coords = valid_coords[idx]
                if len(group_coords) > 0:
                    plt.scatter(
                        group_coords[:, 0],
                        group_coords[:, 1],
                        color=colors[i],
                        s=50,
                        alpha=0.9,
                        edgecolors='none',
                    )
                    mean_pos = np.mean(group_coords, axis=0)
                    plt.scatter(
                        mean_pos[0],
                        mean_pos[1],
                        color=colors[i],
                        marker='x',
                        s=400,
                        linewidths=4,
                        zorder=10,
                    )
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        centers_path = output_dir / ORI_CENTERS_SPATIAL_FILENAME
        plt.savefig(centers_path, dpi=300, transparent=False)
        generated_paths.append(centers_path)
    except Exception as e:
        logger.error("Failed to plot orientation centers: %s", e)
    finally:
        plt.close()

    return generated_paths


def plot_louvain_results(
    result: AnalysisResult,
    output_dir: Path,
    *,
    max_ensembles_to_plot: int | None = None,
) -> list[Path]:
    """Plots and saves the Louvain cluster similarity matrix and spatial distribution per ensemble."""
    if result.communities is None:
        logger.warning("No communities found in AnalysisResult. Skipping Louvain plots.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    generated_paths: list[Path] = []

    labels = result.communities.labels
    corr_matrix = result.communities.similarity
    neuron_coords = result.coords

    partition = {i: int(label) for i, label in enumerate(labels)}
    clusters = ensemble_clusters(partition, include_unclassified=True)

    # ---- 1. Plot Sorted Similarity Matrix ----
    try:
        sorted_neurons = []
        ticks = []
        tick_labels = []
        current_idx = 0
        unique_clusters = sorted([c for c in clusters.keys() if c != 0])
        lines = []

        for c_id in unique_clusters:
            members = clusters[c_id]
            sorted_neurons.extend(members)
            length = len(members)
            ticks.append(current_idx + length / 2)
            tick_labels.append(str(c_id))
            current_idx += length
            lines.append(current_idx)

        if 0 in clusters:
            members = clusters[0]
            sorted_neurons.extend(members)
            length = len(members)
            ticks.append(current_idx + length / 2)
            tick_labels.append("others")
            current_idx += length

        if len(sorted_neurons) > 0:
            sorted_neurons_arr = np.array(sorted_neurons)
            sorted_sim = corr_matrix[np.ix_(sorted_neurons_arr, sorted_neurons_arr)]

            plt.figure(figsize=(6, 5))
            valid_sims = sorted_sim[sorted_sim > 0]
            vmax = np.percentile(valid_sims, 95) if valid_sims.size > 0 else 0.5
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 0.5

            im = plt.imshow(
                sorted_sim,
                cmap='viridis',
                aspect='auto',
                interpolation='none',
                vmin=0,
                vmax=vmax,
            )
            plt.colorbar(im, label='Correlation (r)')

            for line in lines:
                plt.axhline(line - 0.5, color='white', linewidth=1)
                plt.axvline(line - 0.5, color='white', linewidth=1)

            plt.xticks(ticks, tick_labels, fontsize=12)
            plt.yticks(ticks, tick_labels, fontsize=12)
            plt.xlabel('Neurons of each ensemble', fontsize=14)
            plt.ylabel('Neurons of each ensemble', fontsize=14)
            plt.title('Ensembles', fontsize=16)
            plt.tight_layout()

            corr_path = output_dir / ENSEMBLE_CORRELATION_FILENAME
            plt.savefig(corr_path, dpi=300)
            generated_paths.append(corr_path)
    except Exception as e:
        logger.error("Failed to plot ensemble correlation matrix: %s", e)
    finally:
        plt.close()

    # ---- 2. Plot Spatial Distribution for Each Ensemble ----
    if max_ensembles_to_plot is not None:
        clusters_to_plot = unique_clusters[:max_ensembles_to_plot]
    else:
        clusters_to_plot = unique_clusters

    for c_id in clusters_to_plot:
        try:
            plt.figure(figsize=(4, 4))
            plt.scatter(
                neuron_coords[:, 0],
                neuron_coords[:, 1],
                c='none',
                edgecolors='lightgrey',
                s=20,
                label='GCaMP6s-active neurons',
            )
            c_members = clusters[c_id]
            if len(c_members) > 0:
                c_coords = neuron_coords[np.array(c_members)]
                plt.scatter(c_coords[:, 0], c_coords[:, 1], c='black', s=20, label='Ensembles')

            plt.text(
                0.95,
                0.05,
                f"#{c_id}",
                color='red',
                fontsize=16,
                transform=plt.gca().transAxes,
                ha='right',
                va='bottom',
            )
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()

            c_path = output_dir / f"{ENSEMBLE_SPATIAL_PREFIX}{c_id}.png"
            plt.savefig(c_path, dpi=300)
            generated_paths.append(c_path)
        except Exception as e:
            logger.error("Failed to plot ensemble spatial map for #%d: %s", c_id, e)
        finally:
            plt.close()

    return generated_paths


def plot_ensemble_activity_trace(
    result: AnalysisResult,
    output_dir: Path,
) -> list[Path]:
    """Plots and saves ensemble activity traces (raw and normalized). Excludes with_variance plot."""
    if result.communities is None:
        logger.warning("No communities found in AnalysisResult. Skipping activity trace plots.")
        return []

    if result.steady_state_responses.ndim != 3:
        logger.warning("steady_state_responses is not 3D. Skipping activity trace plots.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    generated_paths: list[Path] = []

    try:
        trace_data = prepare_ensemble_activity_trace_plot_data(result)
    except Exception as e:
        logger.error("Failed to prepare activity trace plot data: %s", e)
        return []

    traces = trace_data["traces"]
    if len(traces) == 0:
        logger.warning("No valid clusters found for trace plot.")
        return []

    N_theta = trace_data["N_theta"]
    T_steps = trace_data["T_steps"]
    x = trace_data["x"]
    num_plots = len(traces)
    colors = ['#8A2BE2', '#DAA520', '#1f77b4', '#e377c2', '#2ca02c', '#d62728']

    # ---- Plot 1: ensemble_activity_trace.png ----
    try:
        fig, axes = plt.subplots(num_plots, 1, figsize=(6, 1.5 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]

        for i, trace in enumerate(traces):
            ax = axes[i]
            c_id = trace["cluster_id"]
            flat_trace = trace["mean"]
            color = colors[i % len(colors)]

            ax.plot(x, flat_trace, color=color, linewidth=1.5)
            for j in range(1, N_theta):
                ax.axvline(j * T_steps, color='k', linestyle='--', alpha=0.5)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            activity_max = _finite_axis_max(flat_trace, 0.8)
            ax.set_ylim(-0.02, activity_max)

            _set_three_axis_ticks(ax, activity_max, lambda v: f"{v:.1f}")
            ax.text(1.02, 0.5, f"#{c_id}", transform=ax.transAxes, fontsize=14, va='center')

            if i == num_plots // 2:
                ax.set_ylabel('Ensembles activity', fontsize=14)

        axes[-1].set_xlabel('Frames (sorted)', fontsize=14)
        axes[-1].set_xticks([])
        plt.tight_layout()

        trace_path = output_dir / ENSEMBLE_ACTIVITY_TRACE_FILENAME
        plt.savefig(trace_path, dpi=300)
        generated_paths.append(trace_path)
    except Exception as e:
        logger.error("Failed to plot ensemble activity trace: %s", e)
    finally:
        plt.close()

    # ---- Plot 2: ensemble_activity_trace_normalized_variance.png ----
    try:
        fig_norm, axes_norm = plt.subplots(num_plots, 1, figsize=(6, 1.5 * num_plots), sharex=True)
        if num_plots == 1:
            axes_norm = [axes_norm]

        for i, trace in enumerate(traces):
            ax = axes_norm[i]
            variance_ax = ax.twinx()
            c_id = trace["cluster_id"]
            normalized_mean = trace.get("normalized_mean", trace["mean"])
            normalized_variance = trace.get("normalized_variance", trace["variance"])
            normalized_std = trace.get("normalized_std", trace["std"])
            color = colors[i % len(colors)]

            # Prevent zero variance issues
            denom = float(np.nanmax(normalized_variance))
            if denom <= 0.0 or not np.isfinite(denom):
                normalized_variance = np.zeros_like(normalized_variance)
                normalized_std = np.zeros_like(normalized_std)

            lower = normalized_mean - normalized_std
            upper = normalized_mean + normalized_std
            ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, normalized_mean, color=color, linewidth=1.5)
            variance_ax.plot(
                x,
                normalized_variance,
                color='0.25',
                linewidth=0.8,
                alpha=0.6,
                linestyle=':',
            )
            for j in range(1, N_theta):
                ax.axvline(j * T_steps, color='k', linestyle='--', alpha=0.5)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            variance_ax.spines['top'].set_visible(False)
            variance_ax.spines['left'].set_visible(False)

            activity_max = _finite_axis_max(upper, 1e-3)
            activity_min = min(0.0, float(np.nanmin(lower)) if np.any(np.isfinite(lower)) else 0.0)
            ax.set_ylim(activity_min, activity_max)

            variance_axis_max = _finite_axis_max(normalized_variance, 1e-12)
            variance_ax.set_ylim(0, variance_axis_max)

            _set_three_axis_ticks(ax, activity_max, lambda v: f"{v:.3f}")
            _set_three_axis_ticks(variance_ax, variance_axis_max, lambda v: f"{v:.1e}")
            ax.tick_params(axis='y', labelsize=7)
            variance_ax.tick_params(axis='y', labelsize=7, colors='0.25')
            ax.text(
                0.98,
                0.82,
                f"#{c_id}",
                transform=ax.transAxes,
                fontsize=11,
                fontweight='bold',
                ha='right',
                va='center',
                color=color,
            )

            if i == num_plots // 2:
                ax.set_ylabel('L2-normalized activity', fontsize=12)
                variance_ax.set_ylabel('Normalized variance', fontsize=10, color='0.25')

        axes_norm[-1].set_xlabel('Frames (sorted)', fontsize=14)
        axes_norm[-1].set_xticks([])
        fig_norm.suptitle('L2-normalized ensemble activity with across-neuron variance', fontsize=12)
        fig_norm.tight_layout(rect=[0.0, 0.0, 0.92, 0.97])

        norm_var_path = output_dir / ENSEMBLE_ACTIVITY_TRACE_NORM_VAR_FILENAME
        plt.savefig(norm_var_path, dpi=300)
        generated_paths.append(norm_var_path)
    except Exception as e:
        logger.error("Failed to plot normalized activity trace: %s", e)
    finally:
        plt.close()

    return generated_paths


def plot_spatial_surrogate_results(
    result: AnalysisResult,
    output_dir: Path,
    *,
    num_surrogates: int = 10000,
    rng_seed: int | None = None,
) -> list[Path]:
    """Plots and saves spatial metric surrogate comparison graphs (NND and MeanDist) for valid clusters."""
    if result.communities is None:
        logger.warning("No communities found in AnalysisResult. Skipping spatial surrogate plots.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    generated_paths: list[Path] = []

    labels = result.communities.labels
    partition = {i: int(label) for i, label in enumerate(labels)}
    clusters = ensemble_clusters(partition)

    # We plot surrogates for all clusters with at least 2 members.
    valid_clusters = sorted([c_id for c_id, members in clusters.items() if len(members) >= 2])

    if not valid_clusters:
        logger.warning("No valid ensembles with >= 2 members found. Plotting dummy comparison.")
        # Create a single placeholder file for cluster 1 to represent missing ensembles
        for target_cluster in [1]:
            nnd_path = output_dir / f"{SPATIAL_SURROGATE_PREFIX}{target_cluster}{SPATIAL_SURROGATE_NND_SUFFIX}"
            md_path = output_dir / f"{SPATIAL_SURROGATE_PREFIX}{target_cluster}{SPATIAL_SURROGATE_MEANDIST_SUFFIX}"

            try:
                plt.figure(figsize=(5, 4.5))
                plt.text(
                    0.5, 0.5,
                    "No valid ensembles with >=2 members.",
                    ha='center', va='center',
                    transform=plt.gca().transAxes,
                )
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(nnd_path, dpi=300)
                generated_paths.append(nnd_path)
            except Exception as e:
                logger.error("Failed to write dummy NND plot: %s", e)
            finally:
                plt.close()

            try:
                plt.figure(figsize=(5, 4.5))
                plt.text(
                    0.5, 0.5,
                    f"Cluster #{target_cluster} not found\nor has <2 members.",
                    ha='center', va='center',
                    transform=plt.gca().transAxes,
                )
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(md_path, dpi=300)
                generated_paths.append(md_path)
            except Exception as e:
                logger.error("Failed to write dummy MeanDist plot: %s", e)
            finally:
                plt.close()
        return generated_paths

    rng = np.random.default_rng(rng_seed)

    for c_id in valid_clusters:
        try:
            surrogate_data = prepare_spatial_surrogate_plot_data(
                result,
                num_surrogates=num_surrogates,
                target_cluster=c_id,
                rng=rng,
            )
        except Exception as e:
            logger.error("Failed to prepare spatial surrogate data for cluster %d: %s", c_id, e)
            continue

        nnd_path = output_dir / f"{SPATIAL_SURROGATE_PREFIX}{c_id}{SPATIAL_SURROGATE_NND_SUFFIX}"
        md_path = output_dir / f"{SPATIAL_SURROGATE_PREFIX}{c_id}{SPATIAL_SURROGATE_MEANDIST_SUFFIX}"

        # ---- Panel 1: NND cumulative probability ----
        try:
            plt.figure(figsize=(5, 4.5))
            if surrogate_data["has_valid_ensembles"]:
                actual_NNDs = surrogate_data["actual_NNDs"]
                y_vals = np.linspace(0, 1, len(actual_NNDs))

                plt.plot(actual_NNDs, y_vals, color='black', linewidth=2, label='Ensembles')
                plt.plot(
                    surrogate_data["mean_surrogate_NND"],
                    y_vals,
                    color='darkgray',
                    linewidth=1.5,
                    label='Surrogates',
                )
                plt.plot(
                    surrogate_data["percentile_2_5"],
                    y_vals,
                    color='gray',
                    linestyle='--',
                    linewidth=1,
                    label='95% confidence',
                )
                plt.plot(
                    surrogate_data["percentile_97_5"],
                    y_vals,
                    color='gray',
                    linestyle='--',
                    linewidth=1,
                )
                plt.xlabel('Nearest neighbor\ndistance (NND, μm)', fontsize=12)
                plt.ylabel('Cumulative probability', fontsize=12)
                plt.ylim(-0.02, 1.02)
                plt.legend(frameon=False, loc='upper left')
            else:
                plt.text(
                    0.5, 0.5,
                    "No valid ensembles with >=2 members.",
                    ha='center', va='center',
                    transform=plt.gca().transAxes,
                )
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(nnd_path, dpi=300)
            generated_paths.append(nnd_path)
        except Exception as e:
            logger.error("Failed to plot surrogate NND cumulative probability: %s", e)
        finally:
            plt.close()

        # ---- Panel 2: target-cluster random mean-distance distribution ----
        try:
            plt.figure(figsize=(5, 4.5))
            if surrogate_data["target_cluster_available"]:
                sorted_rand_md = np.sort(surrogate_data["target_cluster_random_mean_dists"])
                y_rand = np.linspace(0, 100, len(sorted_rand_md))
                actual_md_val = surrogate_data["actual_target_mean_dist"]
                threshold_val = surrogate_data["target_threshold_p05"]

                plt.plot(sorted_rand_md, y_rand, color='darkgray', linewidth=3, label='Random')
                if actual_md_val is not None:
                    plt.axvline(x=actual_md_val, color='coral', linestyle='--', linewidth=2, label='Ensemble')
                if threshold_val is not None:
                    plt.axvline(
                        x=threshold_val,
                        color='cornflowerblue',
                        linestyle='--',
                        linewidth=2,
                        label='Threshold',
                    )
                plt.xlabel('Mean distance (μm)', fontsize=12)
                plt.ylabel('Cum. % of mean distance', fontsize=12)
                plt.title(f'#{c_id} ensemble', fontsize=14)
                plt.legend(frameon=False, loc='lower right')
                plt.ylim(-2, 102)
            else:
                plt.text(
                    0.5, 0.5,
                    f"Cluster #{c_id} not found\nor has <2 members.",
                    ha='center', va='center',
                    transform=plt.gca().transAxes,
                )
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(md_path, dpi=300)
            generated_paths.append(md_path)
        except Exception as e:
            logger.error("Failed to plot surrogate MeanDist: %s", e)
        finally:
            plt.close()

    return generated_paths


def generate_and_save_all_analysis_plots(
    result: AnalysisResult,
    output_dir: Path,
    *,
    num_surrogates: int = 10000,
    rng_seed: int | None = None,
    max_ensembles_to_plot: int | None = None,
) -> list[Path]:
    """Generates all analysis-related plots and saves them to output_dir.

    Returns:
        A list of Path objects pointing to all successfully generated images.
    """
    logger.info("Generating and saving analysis plots to %s", output_dir)
    generated_paths: list[Path] = []

    # 1. OSI plots
    generated_paths.extend(plot_osi_results(result, output_dir))

    if result.communities is None:
        logger.warning("Skipping ensemble and spatial surrogate plots because no ensembles were detected.")
        return generated_paths

    # 2. Louvain plots
    generated_paths.extend(
        plot_louvain_results(
            result,
            output_dir,
            max_ensembles_to_plot=max_ensembles_to_plot,
        )
    )

    # 3. Ensemble activity trace plots
    generated_paths.extend(plot_ensemble_activity_trace(result, output_dir))

    # 4. Spatial surrogate comparison plots
    generated_paths.extend(
        plot_spatial_surrogate_results(
            result,
            output_dir,
            num_surrogates=num_surrogates,
            rng_seed=rng_seed,
        )
    )

    return generated_paths
