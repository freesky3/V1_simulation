#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from v1_simulation.network.state import load_trained_network_state


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = compare_analysis(
        initial_dir=args.initial_run,
        final_dir=args.final_run,
        analysis_dir=args.analysis_dir,
        output=args.output,
        num_surrogates=args.num_surrogates,
        seed=args.seed,
    )
    print(f"Wrote {args.output}")
    print(
        "Final-label similarity delta: "
        f"initial={result['initial']['within_minus_between']:.6g}, "
        f"final={result['final']['within_minus_between']:.6g}, "
        f"delta={result['delta']['within_minus_between']:.6g}, "
        f"surrogate_p_ge={result['delta_surrogate']['p_ge_observed']:.6g}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one analysis folder's selected neurons and labels to matched "
            "initial/final simulation responses, then test whether final improves "
            "cluster functional separation beyond size-matched random labels."
        )
    )
    parser.add_argument("--initial-run", type=Path, required=True)
    parser.add_argument("--final-run", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-surrogates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.num_surrogates < 1:
        raise ValueError("--num-surrogates must be positive.")
    return args


def compare_analysis(
    *,
    initial_dir: Path,
    final_dir: Path,
    analysis_dir: Path,
    output: Path,
    num_surrogates: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.load(analysis_dir / "community_labels.npy").astype(np.int64, copy=False)
    coords = np.load(analysis_dir / "coords.npy")
    distance = np.load(analysis_dir / "distance.npy")

    initial_trace = _trace_for_analysis_coords(initial_dir, coords)
    final_trace = _trace_for_analysis_coords(final_dir, coords)
    if initial_trace.shape[0] != labels.size or final_trace.shape[0] != labels.size:
        raise ValueError(
            "Selected response traces and labels have inconsistent neuron counts: "
            f"initial={initial_trace.shape}, final={final_trace.shape}, labels={labels.shape}."
        )
    final_saved_trace = _analysis_saved_trace(analysis_dir)
    final_trace_max_abs_diff = (
        _finite_float(np.max(np.abs(final_trace - final_saved_trace)))
        if final_saved_trace.shape == final_trace.shape and final_trace.size
        else None
    )

    rng = np.random.default_rng(seed)
    initial_similarity = _cosine_similarity(initial_trace)
    final_similarity = _cosine_similarity(final_trace)
    initial_metrics = _partition_metrics(labels, initial_similarity, distance)
    final_metrics = _partition_metrics(labels, final_similarity, distance)
    delta_metrics = {
        "within_similarity_mean": _finite_delta(
            final_metrics["within_similarity_mean"],
            initial_metrics["within_similarity_mean"],
        ),
        "between_similarity_mean": _finite_delta(
            final_metrics["between_similarity_mean"],
            initial_metrics["between_similarity_mean"],
        ),
        "within_minus_between": _finite_delta(
            final_metrics["within_minus_between"],
            initial_metrics["within_minus_between"],
        ),
    }
    surrogate = _delta_surrogate(
        labels,
        initial_similarity,
        final_similarity,
        num_surrogates=num_surrogates,
        rng=rng,
    )

    result = {
        "schema_version": 1,
        "initial_run": str(initial_dir),
        "final_run": str(final_dir),
        "analysis_dir": str(analysis_dir),
        "seed": int(seed),
        "num_surrogates": int(num_surrogates),
        "n_selected": int(labels.size),
        "n_classified": int(np.sum(labels != 0)),
        "n_ensembles": int(np.unique(labels[labels != 0]).size),
        "mapping_validation": {
            "final_trace_matches_analysis_saved": (
                None if final_trace_max_abs_diff is None else bool(final_trace_max_abs_diff <= 1.0e-8)
            ),
            "final_trace_max_abs_diff": final_trace_max_abs_diff,
        },
        "initial": initial_metrics,
        "final": final_metrics,
        "delta": delta_metrics,
        "delta_surrogate": surrogate,
        "spatial": _spatial_metrics(labels, distance),
        "coords_bbox": {
            "x_min": _finite_float(np.min(coords[:, 0])) if coords.size else None,
            "x_max": _finite_float(np.max(coords[:, 0])) if coords.size else None,
            "y_min": _finite_float(np.min(coords[:, 1])) if coords.size else None,
            "y_max": _finite_float(np.max(coords[:, 1])) if coords.size else None,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(result), f, indent=2)
    return result


def _trace_for_analysis_coords(run_dir: Path, analysis_coords: np.ndarray) -> np.ndarray:
    responses = np.load(run_dir / "responses_exc.npy")
    if responses.ndim != 3:
        raise ValueError(f"{run_dir / 'responses_exc.npy'} must be 3D.")
    row_indices = _match_response_rows(run_dir, responses.shape[0], analysis_coords)
    steady_start = int(responses.shape[2] * 2 / 3)
    steady = responses[row_indices, :, steady_start:]
    return steady.reshape(row_indices.size, -1)


def _analysis_saved_trace(analysis_dir: Path) -> np.ndarray:
    steady = np.load(analysis_dir / "steady_state_responses.npy")
    if steady.ndim != 3:
        raise ValueError(f"{analysis_dir / 'steady_state_responses.npy'} must be 3D.")
    return steady.reshape(steady.shape[0], -1)


def _match_response_rows(run_dir: Path, n_response_rows: int, analysis_coords: np.ndarray) -> np.ndarray:
    response_coords = _response_coords(run_dir, n_response_rows)
    targets = np.asarray(analysis_coords, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("analysis coords must have shape (n_neurons, 2).")

    indices = np.empty(targets.shape[0], dtype=np.int64)
    used: set[int] = set()
    for idx, coord in enumerate(targets):
        diff = np.max(np.abs(response_coords - coord[np.newaxis, :]), axis=1)
        row = int(np.argmin(diff))
        if float(diff[row]) > 1.0e-10:
            raise ValueError(
                f"Could not match analysis coord {coord.tolist()} to response rows in {run_dir}; "
                f"nearest max abs diff is {float(diff[row]):.6g}."
            )
        if row in used:
            raise ValueError(f"Duplicate coordinate match for response row {row} in {run_dir}.")
        indices[idx] = row
        used.add(row)
    return indices


def _response_coords(run_dir: Path, n_response_rows: int) -> np.ndarray:
    network = load_trained_network_state(run_dir / "network").network
    l23 = network.layout.l23
    full_coords = np.asarray(l23.coords[network.idx_E], dtype=float)
    if full_coords.shape[0] == int(n_response_rows):
        return full_coords

    run_center_side_fraction = _run_center_side_fraction(run_dir)
    if run_center_side_fraction < 1.0:
        mask = _center_mask(full_coords, l23.region_size, run_center_side_fraction)
        coords = full_coords[mask]
        if coords.shape[0] == int(n_response_rows):
            return coords

    raise ValueError(
        f"Could not reconstruct response coordinates for {run_dir}: "
        f"full E count={full_coords.shape[0]}, response rows={n_response_rows}, "
        f"run analysis.center_side_fraction={run_center_side_fraction}."
    )


def _run_center_side_fraction(run_dir: Path) -> float:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return 1.0
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    try:
        return float(config.get("config", {}).get("analysis", {}).get("center_side_fraction", 1.0))
    except Exception:
        return 1.0


def _center_mask(coords: np.ndarray, region_size: float, center_side_fraction: float) -> np.ndarray:
    half_side = (float(region_size) * float(center_side_fraction)) / 2.0
    return (np.abs(coords[:, 0]) <= half_side) & (np.abs(coords[:, 1]) <= half_side)


def _cosine_similarity(trace: np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=float)
    norms = np.linalg.norm(values, axis=1)
    denom = norms[:, None] * norms[None, :]
    sim = np.divide(
        values @ values.T,
        denom,
        out=np.zeros((values.shape[0], values.shape[0]), dtype=float),
        where=denom > 0.0,
    )
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(sim, 0.0)
    return sim


def _partition_metrics(labels: np.ndarray, similarity: np.ndarray, distance: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    classified = labels != 0
    upper = np.triu_indices(labels.size, k=1)
    left = labels[upper[0]]
    right = labels[upper[1]]
    valid = (left != 0) & (right != 0)
    within = valid & (left == right)
    between = valid & (left != right)
    within_sim = similarity[upper][within]
    between_sim = similarity[upper][between]
    within_dist = np.asarray(distance, dtype=float)[upper][within]
    between_dist = np.asarray(distance, dtype=float)[upper][between]
    within_mean = _safe_mean(within_sim)
    between_mean = _safe_mean(between_sim)
    return {
        "classified_fraction": _finite_float(np.mean(classified)) if labels.size else None,
        "within_pair_count": int(np.sum(within)),
        "between_pair_count": int(np.sum(between)),
        "within_similarity_mean": within_mean,
        "between_similarity_mean": between_mean,
        "within_minus_between": _finite_delta(within_mean, between_mean),
        "within_distance_mean": _safe_mean(within_dist),
        "between_distance_mean": _safe_mean(between_dist),
        "between_minus_within_distance": _finite_delta(_safe_mean(between_dist), _safe_mean(within_dist)),
        "per_cluster": _per_cluster_metrics(labels, similarity, distance),
    }


def _per_cluster_metrics(labels: np.ndarray, similarity: np.ndarray, distance: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    classified = labels != 0
    for c_id in [int(x) for x in np.unique(labels) if x != 0]:
        members = np.flatnonzero(labels == c_id)
        outside = np.flatnonzero(classified & (labels != c_id))
        sub_sim = similarity[np.ix_(members, members)]
        sub_dist = distance[np.ix_(members, members)]
        rows.append(
            {
                "cluster_id": c_id,
                "size": int(members.size),
                "within_similarity_mean": _safe_mean(_upper_values(sub_sim)),
                "outside_similarity_mean": _safe_mean(similarity[np.ix_(members, outside)].ravel())
                if outside.size
                else None,
                "mean_pairwise_distance": _safe_mean(_upper_values(sub_dist)),
            }
        )
    return rows


def _spatial_metrics(labels: np.ndarray, distance: np.ndarray) -> dict[str, Any]:
    metrics = _partition_metrics(labels, np.zeros_like(distance, dtype=float), distance)
    return {
        "within_distance_mean": metrics["within_distance_mean"],
        "between_distance_mean": metrics["between_distance_mean"],
        "between_minus_within_distance": metrics["between_minus_within_distance"],
        "per_cluster": [
            {
                "cluster_id": row["cluster_id"],
                "size": row["size"],
                "mean_pairwise_distance": row["mean_pairwise_distance"],
            }
            for row in metrics["per_cluster"]
        ],
    }


def _delta_surrogate(
    labels: np.ndarray,
    initial_similarity: np.ndarray,
    final_similarity: np.ndarray,
    *,
    num_surrogates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    observed_initial = _partition_metrics(labels, initial_similarity, np.zeros_like(initial_similarity))[
        "within_minus_between"
    ]
    observed_final = _partition_metrics(labels, final_similarity, np.zeros_like(final_similarity))[
        "within_minus_between"
    ]
    observed_delta = _finite_delta(observed_final, observed_initial)
    if observed_delta is None:
        return {
            "observed_delta": None,
            "p_ge_observed": None,
            "surrogate_mean": None,
            "surrogate_p95": None,
        }

    classified = np.flatnonzero(labels != 0)
    cluster_sizes = [int(np.sum(labels == c_id)) for c_id in np.unique(labels[labels != 0])]
    values = np.empty(int(num_surrogates), dtype=float)
    for idx in range(int(num_surrogates)):
        shuffled = np.zeros_like(labels)
        order = rng.permutation(classified)
        start = 0
        for c_idx, size in enumerate(cluster_sizes, start=1):
            members = order[start : start + size]
            shuffled[members] = c_idx
            start += size
        s_initial = _partition_metrics(shuffled, initial_similarity, np.zeros_like(initial_similarity))[
            "within_minus_between"
        ]
        s_final = _partition_metrics(shuffled, final_similarity, np.zeros_like(final_similarity))[
            "within_minus_between"
        ]
        values[idx] = np.nan if s_initial is None or s_final is None else float(s_final) - float(s_initial)

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "observed_delta": observed_delta,
            "p_ge_observed": None,
            "surrogate_mean": None,
            "surrogate_p95": None,
        }
    return {
        "observed_delta": observed_delta,
        "p_ge_observed": _finite_float((np.sum(finite >= float(observed_delta)) + 1.0) / (finite.size + 1.0)),
        "surrogate_mean": _finite_float(np.mean(finite)),
        "surrogate_p05": _finite_float(np.percentile(finite, 5)),
        "surrogate_p50": _finite_float(np.percentile(finite, 50)),
        "surrogate_p95": _finite_float(np.percentile(finite, 95)),
        "surrogate_max": _finite_float(np.max(finite)),
    }


def _upper_values(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.shape[0] < 2:
        return np.array([], dtype=float)
    return values[np.triu_indices(values.shape[0], k=1)]


def _safe_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return _finite_float(np.mean(finite))


def _finite_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    out = float(left) - float(right)
    return out if np.isfinite(out) else None


def _finite_float(value: Any) -> float | None:
    out = float(value)
    return out if np.isfinite(out) else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == "__main__":
    main()
