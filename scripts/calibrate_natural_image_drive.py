#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from v1_simulation.cli import has_group_override
from v1_simulation.config import RootConfig, load_config
from v1_simulation.io.artifacts import create_unique_run_dir, json_ready
from v1_simulation.network.builder import build_network_state
from v1_simulation.network.state import NetworkState, load_trained_network_state
from v1_simulation.simulation.pipeline import build_theta_angles
from v1_simulation.stimuli.grating import DriftingGratingInput
from v1_simulation.stimuli.natural import NaturalImagePreprocessor
from v1_simulation.training.natural_inputs import build_natural_image_l4_drive


@dataclass(frozen=True, slots=True)
class NaturalProjectionBasis:
    base_integrals: np.ndarray
    offset_integral: np.ndarray
    baseline_rate: float
    visual_gain: float


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    if not has_group_override(overrides, "experiment"):
        overrides = ["+experiment=bcm_train", *overrides]

    cfg = load_config(overrides=overrides)
    cfg.mode = "train"
    cfg.training.enabled = True

    network = _load_or_build_network(cfg, args.network)
    output_dir = args.output_dir or create_unique_run_dir(cfg.paths.run_root, job_name=args.job_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_drive, base_sampler = build_natural_image_l4_drive(
        cfg=cfg.training.natural_image,
        stimulus_cfg=cfg.stimulus,
        model_cfg=cfg.model,
        layers_cfg=cfg.model.layers,
        l4_layer=network.layout.l4,
        l4_tunings=network.layout.l4_tunings,
        l4_pref_dirs=network.layout.l4_pref_dirs,
    )
    probe_samples = _make_probe_samples(cfg, base_sampler, probe_count=args.probe_count, probe_seed=args.probe_seed)

    reference_grating = _drifting_grating_rates(cfg, network, n_phases=args.grating_phases)
    reference_stats = _rate_stats(reference_grating, active_threshold=args.active_threshold)
    projection_basis = _natural_projection_basis(base_drive, probe_samples)
    base_natural = _natural_rates_from_basis(
        projection_basis,
        frame_scale=float(cfg.training.natural_image.frame_scale),
        frame_offset=float(cfg.training.natural_image.frame_offset),
    )
    base_stats = _rate_stats(base_natural, active_threshold=args.active_threshold)

    frame_scales = _parse_float_list(args.frame_scales)
    frame_offsets = _parse_float_list(args.frame_offsets)
    if any(value < 0.0 for value in frame_scales):
        raise ValueError("--frame-scales values must be non-negative.")
    max_allowed = float(reference_stats["max"]) * float(args.max_ratio)

    rows: list[dict[str, Any]] = []
    for frame_offset in frame_offsets:
        for frame_scale in frame_scales:
            rates = _natural_rates_from_basis(
                projection_basis,
                frame_scale=frame_scale,
                frame_offset=frame_offset,
            )
            stats = _rate_stats(rates, active_threshold=args.active_threshold)
            row = {
                "frame_scale": frame_scale,
                "frame_offset": frame_offset,
                **stats,
                "score": _candidate_score(stats, reference_stats, max_allowed=max_allowed),
                "max_allowed": max_allowed,
            }
            rows.append(row)

    rows.sort(key=lambda row: float(row["score"]))
    best = rows[0] if rows else {}

    _write_candidates(output_dir / "candidates.csv", rows)
    _write_json(
        output_dir / "summary.json",
        {
            "best": best,
            "reference_grating": reference_stats,
            "base_natural": base_stats,
            "active_threshold": float(args.active_threshold),
            "probe_count": int(args.probe_count),
            "grating_phases": int(args.grating_phases),
            "max_ratio": float(args.max_ratio),
            "overrides": overrides,
            "config": asdict(cfg),
        },
    )
    _save_distribution_plot(
        output_dir / "rate_distributions.png",
        reference_grating=reference_grating,
        base_natural=base_natural,
        best_natural=_natural_rates_from_basis(
            projection_basis,
            frame_scale=float(best["frame_scale"]),
            frame_offset=float(best["frame_offset"]),
        )
        if best
        else base_natural,
        active_threshold=args.active_threshold,
    )

    print(f"Saved calibration: {output_dir}")
    if best:
        print(
            "Best natural image preprocessing: "
            f"training.natural_image.frame_scale={best['frame_scale']} "
            f"training.natural_image.frame_offset={best['frame_offset']} "
            f"(mean={best['mean']:.4g}, active_fraction={best['active_fraction']:.4g}, "
            f"p95={best['p95']:.4g}, max={best['max']:.4g})"
        )
    print(
        "Reference drifting grating: "
        f"mean={reference_stats['mean']:.4g}, "
        f"active_fraction={reference_stats['active_fraction']:.4g}, "
        f"p95={reference_stats['p95']:.4g}, max={reference_stats['max']:.4g}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan natural-image preprocessing parameters against drifting-grating L4 input statistics.",
    )
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--probe-seed", type=int, default=None)
    parser.add_argument("--active-threshold", type=float, default=1.0)
    parser.add_argument("--grating-phases", type=int, default=16)
    parser.add_argument("--frame-scales", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--frame-offsets", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--max-ratio", type=float, default=1.2)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.gabor_cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--network", type=Path, default=None, help="Optional checkpoint to reuse for the L4 layout.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--job-name", default="natural_image_calibration")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.probe_count <= 0:
        raise ValueError("--probe-count must be positive.")
    if args.grating_phases <= 0:
        raise ValueError("--grating-phases must be positive.")
    if args.max_ratio <= 0.0:
        raise ValueError("--max-ratio must be positive.")
    return args


def _load_or_build_network(cfg: RootConfig, checkpoint: Path | None) -> NetworkState:
    if checkpoint is not None:
        return load_trained_network_state(checkpoint, model_cfg=cfg.model).network
    return build_network_state(cfg)


def _make_probe_samples(cfg: RootConfig, natural_sampler: Any, *, probe_count: int, probe_seed: int | None) -> tuple[Any, ...]:
    seed = probe_seed
    if seed is None:
        seed = (
            int(cfg.training.natural_image.seed)
            if cfg.training.natural_image.seed is not None
            else int(cfg.seed) + 12345
        )
    dataset = getattr(natural_sampler, "dataset", None)
    crop_size = getattr(natural_sampler, "crop_size", cfg.training.natural_image.crop_size)
    patches_per_image = int(getattr(natural_sampler, "patches_per_image", cfg.training.natural_image.patches_per_image))
    if dataset is not None:
        from v1_simulation.data.natural_images import NaturalImageSampler

        probe_sampler = NaturalImageSampler(
            dataset,
            crop_size=crop_size,
            patches_per_image=patches_per_image,
            seed=seed,
        )
    else:
        probe_sampler = natural_sampler

    path_limit = max(1, ceil(int(probe_count) / max(1, patches_per_image))) if dataset is not None else int(probe_count)
    samples = tuple(
        probe_sampler.make_epoch(
            limit=path_limit,
            shuffle_paths=False,
            shuffle_samples=False,
        )
    )
    if len(samples) < int(probe_count):
        raise RuntimeError(f"Only {len(samples)} probe samples are available; requested {probe_count}.")
    return tuple(samples[: int(probe_count)])


def _natural_projection_basis(drive: Any, probe_samples: Sequence[Any]) -> NaturalProjectionBasis:
    neutral_preprocessor = NaturalImagePreprocessor(
        replace(drive.preprocessor.cfg, frame_scale=1.0, frame_offset=0.0)
    )
    base_integrals: list[np.ndarray] = []
    offset_integral: np.ndarray | None = None

    for sample in probe_samples:
        image = drive.dataset.read(sample.path)
        frame = neutral_preprocessor.transform(image, sample)
        projection_matrix = drive.projector._get_projection_matrix(*frame.shape)
        if offset_integral is None:
            offset_integral = projection_matrix @ np.ones(frame.size, dtype=float)
        base_integrals.append(projection_matrix @ frame.ravel())

    if offset_integral is None:
        raise RuntimeError("No probe samples are available for natural image calibration.")

    return NaturalProjectionBasis(
        base_integrals=np.column_stack(base_integrals),
        offset_integral=np.asarray(offset_integral, dtype=float),
        baseline_rate=float(drive.projector.drive_cfg.baseline_rate),
        visual_gain=float(drive.projector.drive_cfg.visual_gain),
    )


def _natural_rates_from_basis(
    basis: NaturalProjectionBasis,
    *,
    frame_scale: float,
    frame_offset: float,
) -> np.ndarray:
    integrals = (
        float(frame_scale) * basis.base_integrals
        + float(frame_offset) * basis.offset_integral[:, np.newaxis]
    )
    return np.maximum(0.0, basis.baseline_rate + integrals) * basis.visual_gain


def _drifting_grating_rates(cfg: RootConfig, network: NetworkState, *, n_phases: int) -> np.ndarray:
    if cfg.stimulus.kind != "drifting_grating":
        raise ValueError("Calibration reference requires cfg.stimulus.kind='drifting_grating'.")
    if float(cfg.stimulus.temporal_frequency) <= 0.0:
        raise ValueError("Calibration reference requires positive stimulus.temporal_frequency.")
    grating = DriftingGratingInput(
        cfg.stimulus,
        network.layout.l4,
        l4_tunings=network.layout.l4_tunings,
        l4_pref_dirs=network.layout.l4_pref_dirs,
    )
    theta_angles = build_theta_angles(cfg)
    drive = grating.make_batched_drive_func(theta_angles)
    period = 2.0 * np.pi / float(cfg.stimulus.temporal_frequency)
    times = np.linspace(0.0, period, int(n_phases), endpoint=False)
    return np.concatenate([np.asarray(drive(float(t)), dtype=float) for t in times], axis=1)


def _rate_stats(rates: np.ndarray, *, active_threshold: float) -> dict[str, float]:
    arr = np.asarray(rates, dtype=float)
    flat = arr.ravel()
    active = flat[flat > float(active_threshold)]
    by_x = np.mean(arr, axis=1) if arr.ndim == 2 and arr.shape[0] else np.array([], dtype=float)
    return {
        "count": int(flat.size),
        "mean": _safe_mean(flat),
        "p50": _safe_percentile(flat, 50),
        "p95": _safe_percentile(flat, 95),
        "p99": _safe_percentile(flat, 99),
        "max": _safe_max(flat),
        "active_count": int(active.size),
        "active_fraction": float(active.size / flat.size) if flat.size else 0.0,
        "active_mean": _safe_mean(active),
        "mean_by_x_p50": _safe_percentile(by_x, 50),
        "mean_by_x_p95": _safe_percentile(by_x, 95),
        "mean_by_x_max": _safe_max(by_x),
    }


def _candidate_score(stats: dict[str, float], reference: dict[str, float], *, max_allowed: float) -> float:
    ref_mean = max(float(reference["mean"]), 1.0e-12)
    ref_p95 = max(float(reference["p95"]), 1.0e-12)
    mean_error = abs(float(stats["mean"]) / ref_mean - 1.0)
    p95_error = abs(float(stats["p95"]) / ref_p95 - 1.0)
    active_error = abs(float(stats["active_fraction"]) - float(reference["active_fraction"]))
    max_penalty = max(0.0, float(stats["max"]) / float(max_allowed) - 1.0)
    return float(mean_error + active_error + 0.5 * p95_error + 2.0 * max_penalty)


def _safe_mean(values: Iterable[float] | np.ndarray) -> float:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_percentile(values: Iterable[float] | np.ndarray, q: float) -> float:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _safe_max(values: Iterable[float] | np.ndarray) -> float:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return float(np.max(arr)) if arr.size else float("nan")


def _parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Float list must not be empty.")
    return values


def _write_candidates(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def _save_distribution_plot(
    path: Path,
    *,
    reference_grating: np.ndarray,
    base_natural: np.ndarray,
    best_natural: np.ndarray,
    active_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    for label, rates, color in (
        ("drifting grating", reference_grating, "#3568a8"),
        ("natural base", base_natural, "#a86045"),
        ("natural best", best_natural, "#4b8f6b"),
    ):
        axs[0].hist(np.asarray(rates, dtype=float).ravel(), bins=60, alpha=0.42, label=label, color=color)
        axs[1].hist(np.mean(np.asarray(rates, dtype=float), axis=1), bins=50, alpha=0.42, label=label, color=color)
    axs[0].axvline(float(active_threshold), color="#222222", linestyle="--", linewidth=1.0)
    axs[0].set_title("All X rates")
    axs[0].set_xlabel("Hz")
    axs[1].set_title("Mean by X neuron")
    axs[1].set_xlabel("Hz")
    for ax in axs:
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
