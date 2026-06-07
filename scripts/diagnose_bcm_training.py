#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import csv
import json
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from tqdm.auto import tqdm

from v1_simulation.cli import has_group_override
from v1_simulation.config import RootConfig, load_config, validate_config
from v1_simulation.io.artifacts import TrainingArtifacts, json_ready
from v1_simulation.network.builder import build_network_state
from v1_simulation.network.state import NetworkState, load_trained_network_state
from v1_simulation.simulation.pipeline import (
    _checkpoint_metadata,
    _count_unique_paths,
    _iter_batches,
    _make_background_trace,
    _sample_paths_for_log,
    _steady_state_enabled,
    default_training_time_grid,
)
from v1_simulation.solvers.base import BatchODEResult
from v1_simulation.solvers.wilson_cowan import solve_wilson_cowan_batch
from v1_simulation.training.checkpoints import save_checkpoint, save_theta
from v1_simulation.training.natural_inputs import build_natural_image_l4_drive
from v1_simulation.training.trainer import BCMTrainer


SolverCallable = Callable[..., BatchODEResult]


@dataclass(frozen=True, slots=True)
class DiagnosticsOptions:
    probe_count: int = 8
    probe_every: int = 1
    active_rate_threshold: float = 1.0
    save_per_step_figures: bool = True
    w_max: float | None = None
    cap_warn_fraction: float = 0.01
    cap_bad_fraction: float = 0.05
    cap_atol: float = 1.0e-8
    slope_rtol: float = 1.0e-3
    probe_seed: int | None = None
    job_name: str = "bcm_diagnostics"
    show_progress: bool = True
    preload_cache: str = "batch"
    cache_prefetch: bool = True
    network_cache_dir: Path | None = Path("data/.network_cache")
    use_network_cache: bool = True
    stage_logging: bool = True


@dataclass(frozen=True, slots=True)
class DiagnosticsResult:
    run_dir: Path
    diagnostics_dir: Path
    steps: int
    summary: dict[str, Any]


class CSVAppendLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def append(self, row: Mapping[str, Any]) -> None:
        flat = {str(key): _csv_value(value) for key, value in row.items()}
        if self._fieldnames is None:
            self._fieldnames = list(flat)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow({key: flat.get(key, "") for key in self._fieldnames})


class StageLogger:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.timings: list[dict[str, Any]] = []

    @contextmanager
    def stage(self, name: str):
        start = perf_counter()
        if self.enabled:
            print(f"[diagnostics] {name} ...", flush=True)
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.timings.append({"stage": name, "seconds": elapsed})
            if self.enabled:
                print(f"[diagnostics] {name} done in {elapsed:.2f}s", flush=True)

    def info(self, message: str) -> None:
        if self.enabled:
            print(f"[diagnostics] {message}", flush=True)


class BatchCachePrefetcher:
    def __init__(self, natural_drive: Any, *, enabled: bool) -> None:
        self.natural_drive = natural_drive
        self.enabled = bool(enabled) and hasattr(natural_drive, "preload_cache")
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[Any] | None = None
        if self.enabled:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gabor-cache-prefetch")

    def prefetch(self, samples: Sequence[Any]) -> None:
        if not samples or not hasattr(self.natural_drive, "preload_cache"):
            return
        if not self.enabled:
            self.natural_drive.preload_cache(samples)
            return
        self.wait()
        assert self._executor is not None
        self._future = self._executor.submit(self.natural_drive.preload_cache, tuple(samples))

    def wait(self) -> None:
        if self._future is None:
            return
        self._future.result()
        self._future = None

    def close(self) -> None:
        try:
            self.wait()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None


def run_bcm_training_diagnostics(
    cfg: RootConfig,
    *,
    options: DiagnosticsOptions | None = None,
    network: NetworkState | None = None,
    drive: Any | None = None,
    sampler: Any | None = None,
    solver: SolverCallable | None = None,
    time: Sequence[float] | np.ndarray | None = None,
    run_root: str | Path | None = None,
    artifacts: TrainingArtifacts | None = None,
) -> DiagnosticsResult:
    """Run BCM training and save per-update diagnostics without changing the main pipeline."""

    opts = options or DiagnosticsOptions()
    stage_log = StageLogger(enabled=opts.stage_logging)
    with stage_log.stage("validate config"):
        validate_config(cfg)
        if not cfg.training.enabled:
            raise ValueError("cfg.training.enabled must be true for BCM diagnostics.")
        if opts.w_max is None and cfg.training.bcm.w_max is not None:
            opts = replace(opts, w_max=cfg.training.bcm.w_max)
        _validate_options(opts)

    with stage_log.stage("build/load network"):
        run_network = network if network is not None else _get_or_build_network(cfg, opts, stage_log=stage_log)

    natural_drive = drive
    natural_sampler = sampler
    with stage_log.stage("build natural image drive"):
        if natural_drive is None or natural_sampler is None:
            natural_drive, natural_sampler = build_natural_image_l4_drive(
                cfg=cfg.training.natural_image,
                stimulus_cfg=cfg.stimulus,
                model_cfg=cfg.model,
                layers_cfg=cfg.model.layers,
                l4_layer=run_network.layout.l4,
                l4_tunings=run_network.layout.l4_tunings,
                l4_pref_dirs=run_network.layout.l4_pref_dirs,
            )
            _warn_if_unseeded_natural_images(cfg, stage_log)

    with stage_log.stage("prepare time grid/artifacts"):
        time_grid = (
            default_training_time_grid(cfg)
            if time is None
            else np.asarray(time, dtype=float).copy()
        )
        run_artifacts = artifacts or TrainingArtifacts.create(
            Path(cfg.paths.run_root) if run_root is None else run_root,
            job_name=opts.job_name,
        )
    diagnostics_dir = run_artifacts.run_dir / "diagnostics"
    figures_dir = diagnostics_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with stage_log.stage("initialize trainer"):
        solver_fn = solve_wilson_cowan_batch if solver is None else solver
        trainer = BCMTrainer(cfg.training.bcm, run_network)
        topology = _dense_topology(trainer.state.network.connectivity)

    with stage_log.stage("prepare probe diagnostics"):
        probe_samples = _make_probe_samples(
            cfg,
            natural_sampler,
            probe_count=opts.probe_count,
            probe_seed=opts.probe_seed,
        )
        _preload_samples(natural_drive, probe_samples, mode="probe", stage_log=stage_log, enabled=True)
        probe_drive_func = natural_drive.make_static_batch_func(probe_samples)
        input_l4_rates = np.asarray(probe_drive_func(0.0), dtype=float)
        np.save(diagnostics_dir / "input_l4_rate_distribution.npy", input_l4_rates)
        _write_json(diagnostics_dir / "probe_samples.json", [_sample_json(sample) for sample in probe_samples])
        _save_input_l4_distribution(input_l4_rates, diagnostics_dir / "input_l4_rate_distribution.png")

    with stage_log.stage("save initial network checkpoint"):
        save_checkpoint(run_artifacts.run_dir, "network_initial", trainer.state.network, metadata={"step": 0})

    diag_log = CSVAppendLogger(diagnostics_dir / "per_update_metrics.csv")
    history: list[dict[str, Any]] = []
    image_count = 0

    for epoch in range(1, int(cfg.training.bcm.epochs) + 1):
        with stage_log.stage(f"prepare epoch {epoch} samples"):
            epoch_samples = tuple(
                natural_sampler.make_epoch(
                    limit=cfg.training.natural_image.limit,
                    shuffle_paths=True,
                    shuffle_samples=True,
                )
            )
        if opts.preload_cache == "epoch":
            _preload_samples(natural_drive, epoch_samples, mode=f"epoch {epoch}", stage_log=stage_log, enabled=True)
        image_count += _count_unique_paths(epoch_samples)
        batch_size = int(cfg.training.bcm.batch_size)
        batches = tuple(_iter_batches(epoch_samples, batch_size))
        n_batches = len(batches)
        progress = tqdm(
            batches,
            total=n_batches,
            desc=f"BCM diagnostics epoch {epoch}/{cfg.training.bcm.epochs}",
            unit="batch",
            dynamic_ncols=True,
            disable=not opts.show_progress,
        )

        prefetcher = BatchCachePrefetcher(natural_drive, enabled=opts.preload_cache == "batch" and opts.cache_prefetch)
        try:
            if opts.preload_cache == "batch":
                prefetcher.prefetch(batches[0] if batches else ())

            for batch_index, batch in enumerate(progress):
                if opts.preload_cache == "batch":
                    prefetcher.wait()
                    next_index = batch_index + 1
                    if next_index < n_batches:
                        prefetcher.prefetch(batches[next_index])
                    elif not opts.cache_prefetch:
                        _preload_samples(
                            natural_drive,
                            batch,
                            mode=f"batch {trainer.state.step + 1}",
                            stage_log=stage_log,
                            enabled=False,
                        )
                background_trace = _make_background_trace(
                    cfg,
                    network=trainer.state.network,
                    n_batch=len(batch),
                    time=time_grid,
                    step=trainer.state.step,
                )
                dynamics = solver_fn(
                    network=trainer.state.network,
                    external_drive=natural_drive.make_static_batch_func(batch),
                    time=time_grid,
                    n_batch=len(batch),
                    solver_config=cfg.solver,
                    transfer_config=cfg.solver.transfer,
                    training_bcm=cfg.training.bcm,
                    background_trace=background_trace,
                    store_trajectory=False,
                    stop_at_steady_state=_steady_state_enabled(cfg),
                )

                weights_before = _dense_weights(trainer.state.network.weights).copy()
                log_row = trainer.train_batch(
                    dynamics,
                    epoch=epoch,
                    batch_size=len(batch),
                    images=_sample_paths_for_log(batch),
                )
                weights_after = _dense_weights(trainer.state.network.weights)
                delta = weights_after - weights_before
                run_artifacts.append_log(log_row)

                probe_dynamics = None
                if log_row.step % opts.probe_every == 0:
                    probe_dynamics = _run_probe_batch(
                        cfg,
                        network=trainer.state.network,
                        probe_drive_func=probe_drive_func,
                        probe_count=len(probe_samples),
                        time_grid=time_grid,
                        solver_fn=solver_fn,
                    )

                metrics = collect_update_metrics(
                    network=trainer.state.network,
                    topology=topology,
                    weights_after=weights_after,
                    delta=delta,
                    log_row=log_row,
                    theta=trainer.state.theta,
                    row_sum_limits=trainer.row_sum_limits,
                    probe_dynamics=probe_dynamics,
                    options=opts,
                )
                history.append(metrics)
                diag_log.append(metrics)

                if opts.save_per_step_figures and probe_dynamics is not None:
                    _save_update_figure(
                        figures_dir / f"update_{log_row.step:06d}.png",
                        network=trainer.state.network,
                        topology=topology,
                        weights_after=weights_after,
                        delta=delta,
                        probe_dynamics=probe_dynamics,
                        history=history,
                        metrics=metrics,
                        options=opts,
                    )

                progress.set_postfix(
                    samples=trainer.state.samples_seen,
                    aE=f"{log_row.aE_mean:.3g}",
                    aI=f"{log_row.aI_mean:.3g}",
                    cap=metrics["w_cap_status"],
                    updated=log_row.updated,
                )

                if trainer.state.step % int(cfg.training.bcm.save_every) == 0:
                    save_checkpoint(
                        run_artifacts.run_dir,
                        "network_latest",
                        trainer.state.network,
                        metadata=_checkpoint_metadata(trainer, image_count),
                    )
        finally:
            prefetcher.close()

    if trainer.state.theta is None:
        raise RuntimeError("BCM diagnostics did not process any natural-image samples.")

    save_checkpoint(
        run_artifacts.run_dir,
        "network_final",
        trainer.state.network,
        metadata=_checkpoint_metadata(trainer, image_count),
    )
    save_theta(run_artifacts.run_dir, trainer.state.theta)

    final_weights = _dense_weights(trainer.state.network.weights)
    _save_training_overview(history, diagnostics_dir / "training_overview.png")
    _save_final_weight_distribution(
        trainer.state.network,
        topology,
        final_weights,
        diagnostics_dir / "final_weight_distribution.png",
        w_max=cfg.training.bcm.w_max,
    )
    _save_probe_response_stability(history, diagnostics_dir / "probe_response_stability.png")

    summary = _build_summary(
        cfg,
        options=opts,
        history=history,
        input_l4_rates=input_l4_rates,
        stage_timings=stage_log.timings,
        run_dir=run_artifacts.run_dir,
        diagnostics_dir=diagnostics_dir,
        steps=trainer.state.step,
        samples_seen=trainer.state.samples_seen,
        images_seen=image_count,
    )
    _write_json(diagnostics_dir / "summary.json", summary)
    run_artifacts.save_json(
        "run_config.json",
        {
            "config": asdict(cfg),
            "diagnostics": asdict(opts),
            "stage_timings": stage_log.timings,
            "training": {
                "batches": trainer.state.step,
                "images": image_count,
                "samples": trainer.state.samples_seen,
                "time_steps": int(time_grid.size),
                "t_final": float(time_grid[-1]),
                "steady_state_enabled": _steady_state_enabled(cfg),
                "probe_count": len(probe_samples),
            },
        },
    )

    return DiagnosticsResult(
        run_dir=run_artifacts.run_dir,
        diagnostics_dir=diagnostics_dir,
        steps=trainer.state.step,
        summary=summary,
    )


def collect_update_metrics(
    *,
    network: NetworkState,
    topology: NDArray[np.bool_],
    weights_after: NDArray[np.float64],
    delta: NDArray[np.float64],
    log_row: Any,
    theta: Any | None,
    row_sum_limits: Any,
    probe_dynamics: BatchODEResult | None,
    options: DiagnosticsOptions,
) -> dict[str, Any]:
    idx_E = network.idx_E
    idx_I = network.idx_I
    idx_X = network.idx_X

    metrics: dict[str, Any] = {
        "step": int(log_row.step),
        "epoch": int(log_row.epoch),
        "batch_size": int(log_row.batch_size),
        "samples_seen": int(log_row.samples_seen),
        "updated": int(log_row.updated),
        "skipped_bad_batch": int(bool(log_row.skipped_bad_batch)),
        "train_aE_mean": float(log_row.aE_mean),
        "train_aI_mean": float(log_row.aI_mean),
        "train_aE_max": float(log_row.aE_max),
        "train_aI_max": float(log_row.aI_max),
        "steady_state_reached": int(log_row.steady_state_reached),
        "steady_state_index": int(log_row.steady_state_index),
        "dy_max": float(log_row.dy_max),
        "dy_rms": float(log_row.dy_rms),
    }

    for name, targets, sources in (
        ("W_EE", idx_E, idx_E),
        ("W_IE", idx_I, idx_E),
        ("W_EX", idx_E, idx_X),
        ("W_IX", idx_I, idx_X),
    ):
        values = connected_block_values(weights_after, topology, targets, sources, include_zero=False)
        metrics.update(_basic_stats(values, name))

    for name, targets, sources in (
        ("dW_EE", idx_E, idx_E),
        ("dW_IE", idx_I, idx_E),
    ):
        values = delta_block_values(delta, topology, targets, sources)
        metrics.update(_delta_stats(values, name))

    cap_EE = cap_fraction(
        connected_block_values(weights_after, topology, idx_E, idx_E, include_zero=True),
        options=options,
    )
    cap_IE = cap_fraction(
        connected_block_values(weights_after, topology, idx_I, idx_E, include_zero=True),
        options=options,
    )
    max_cap = max(cap_EE["fraction"], cap_IE["fraction"])
    metrics.update({f"W_EE_cap_{key}": value for key, value in cap_EE.items()})
    metrics.update({f"W_IE_cap_{key}": value for key, value in cap_IE.items()})
    metrics["w_cap_max_fraction"] = float(max_cap)
    metrics["w_cap_status"] = cap_status(max_cap, options)

    metrics.update(
        {
            f"W_EE_row_sum_{key}": value
            for key, value in row_sum_pressure(
                weights_after,
                idx_E,
                idx_E,
                getattr(row_sum_limits, "target_E_source_E", None),
            ).items()
        }
    )
    metrics.update(
        {
            f"W_IE_row_sum_{key}": value
            for key, value in row_sum_pressure(
                weights_after,
                idx_I,
                idx_E,
                getattr(row_sum_limits, "target_I_source_E", None),
            ).items()
        }
    )

    if theta is not None:
        metrics.update(_theta_stats("theta_E", theta.E))
        metrics.update(_theta_stats("theta_I", theta.I))
    else:
        metrics.update(_empty_theta_stats("theta_E"))
        metrics.update(_empty_theta_stats("theta_I"))

    if probe_dynamics is not None:
        metrics.update(active_rate_stats(probe_dynamics.exc, "probe_E", options.active_rate_threshold))
        metrics.update(active_rate_stats(probe_dynamics.inh, "probe_I", options.active_rate_threshold))
        metrics["probe_steady_state_reached"] = int(probe_dynamics.steady_state_reached)
        metrics["probe_dy_max"] = float(probe_dynamics.dy_max)
        metrics["probe_dy_rms"] = float(probe_dynamics.dy_rms)
    else:
        metrics.update(empty_active_rate_stats("probe_E"))
        metrics.update(empty_active_rate_stats("probe_I"))
        metrics["probe_steady_state_reached"] = 0
        metrics["probe_dy_max"] = float("nan")
        metrics["probe_dy_rms"] = float("nan")

    return metrics


def active_rate_stats(
    rates: Any,
    prefix: str,
    threshold: float,
) -> dict[str, float]:
    arr = np.asarray(rates, dtype=float)
    flat = arr.ravel()
    active = flat[flat > float(threshold)]
    stats = {
        f"{prefix}_all_count": int(flat.size),
        f"{prefix}_all_mean": _safe_mean(flat),
        f"{prefix}_all_p95": _safe_percentile(flat, 95),
        f"{prefix}_all_max": _safe_max(flat),
        f"{prefix}_active_threshold": float(threshold),
        f"{prefix}_active_count": int(active.size),
        f"{prefix}_active_fraction": float(active.size / flat.size) if flat.size else 0.0,
        f"{prefix}_active_mean": _safe_mean(active),
        f"{prefix}_active_p50": _safe_percentile(active, 50),
        f"{prefix}_active_p95": _safe_percentile(active, 95),
        f"{prefix}_active_max": _safe_max(active),
    }
    if arr.ndim == 2 and arr.shape[0] > 0:
        per_probe_mean = np.mean(arr > float(threshold), axis=1)
        stats[f"{prefix}_per_probe_active_fraction_mean"] = _safe_mean(per_probe_mean)
        stats[f"{prefix}_per_probe_active_fraction_max"] = _safe_max(per_probe_mean)
    else:
        stats[f"{prefix}_per_probe_active_fraction_mean"] = 0.0
        stats[f"{prefix}_per_probe_active_fraction_max"] = 0.0
    return stats


def empty_active_rate_stats(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_all_count": 0,
        f"{prefix}_all_mean": float("nan"),
        f"{prefix}_all_p95": float("nan"),
        f"{prefix}_all_max": float("nan"),
        f"{prefix}_active_threshold": float("nan"),
        f"{prefix}_active_count": 0,
        f"{prefix}_active_fraction": float("nan"),
        f"{prefix}_active_mean": float("nan"),
        f"{prefix}_active_p50": float("nan"),
        f"{prefix}_active_p95": float("nan"),
        f"{prefix}_active_max": float("nan"),
        f"{prefix}_per_probe_active_fraction_mean": float("nan"),
        f"{prefix}_per_probe_active_fraction_max": float("nan"),
    }


def connected_block_values(
    weights: NDArray[np.float64],
    topology: NDArray[np.bool_],
    targets: Sequence[int],
    sources: Sequence[int],
    *,
    include_zero: bool,
) -> NDArray[np.float64]:
    if len(targets) == 0 or len(sources) == 0:
        return np.array([], dtype=float)
    block = np.asarray(weights[np.ix_(targets, sources)], dtype=float)
    mask = np.asarray(topology[np.ix_(targets, sources)], dtype=bool)
    values = block[mask]
    if not include_zero:
        values = values[values != 0.0]
    return np.asarray(values, dtype=float)


def delta_block_values(
    delta: NDArray[np.float64],
    topology: NDArray[np.bool_],
    targets: Sequence[int],
    sources: Sequence[int],
) -> NDArray[np.float64]:
    if len(targets) == 0 or len(sources) == 0:
        return np.array([], dtype=float)
    values = np.asarray(delta[np.ix_(targets, sources)], dtype=float)
    mask = np.asarray(topology[np.ix_(targets, sources)], dtype=bool)
    return np.asarray(values[mask], dtype=float)


def cap_fraction(values: NDArray[np.float64], *, options: DiagnosticsOptions) -> dict[str, Any]:
    w_max = options.w_max
    if w_max is None or values.size == 0:
        return {"fraction": 0.0, "count": 0, "total": int(values.size)}
    capped = values >= (float(w_max) - float(options.cap_atol))
    return {
        "fraction": float(np.mean(capped)),
        "count": int(np.sum(capped)),
        "total": int(values.size),
    }


def cap_status(fraction: float, options: DiagnosticsOptions) -> str:
    if fraction > float(options.cap_bad_fraction):
        return "bad"
    if fraction > float(options.cap_warn_fraction):
        return "warning"
    return "ok"


def row_sum_pressure(
    weights: NDArray[np.float64],
    targets: Sequence[int],
    sources: Sequence[int],
    limits: Any | None,
) -> dict[str, float]:
    if len(targets) == 0 or len(sources) == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0, "pressure_max": float("nan"), "pressure_p95": float("nan")}
    block = np.asarray(weights[np.ix_(targets, sources)], dtype=float)
    row_sums = np.sum(block, axis=1)
    stats = {
        "mean": _safe_mean(row_sums),
        "p95": _safe_percentile(row_sums, 95),
        "max": _safe_max(row_sums),
    }
    if limits is None:
        stats["pressure_max"] = float("nan")
        stats["pressure_p95"] = float("nan")
        return stats
    limits_arr = np.asarray(limits, dtype=float)
    valid = limits_arr > 0.0
    if limits_arr.shape != row_sums.shape or not np.any(valid):
        stats["pressure_max"] = float("nan")
        stats["pressure_p95"] = float("nan")
        return stats
    pressure = row_sums[valid] / limits_arr[valid]
    stats["pressure_max"] = _safe_max(pressure)
    stats["pressure_p95"] = _safe_percentile(pressure, 95)
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BCM training with fixed-input firing-rate and weight diagnostics.",
    )
    parser.add_argument("--probe-count", type=int, default=8)
    parser.add_argument("--probe-every", type=int, default=1)
    parser.add_argument("--active-rate-threshold", type=float, default=1.0)
    parser.add_argument("--probe-seed", type=int, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--job-name", default="bcm_diagnostics")
    parser.add_argument("--preload-cache", choices=("epoch", "batch", "none"), default="batch")
    parser.add_argument("--cache-prefetch", dest="cache_prefetch", action="store_true", default=True)
    parser.add_argument("--no-cache-prefetch", dest="cache_prefetch", action="store_false")
    parser.add_argument("--network-cache-dir", type=Path, default=Path("data/.network_cache"))
    parser.add_argument("--no-network-cache", action="store_true")
    parser.add_argument("--no-stage-log", action="store_true")
    parser.add_argument("--no-per-step-figures", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.overrides and args.overrides[0] == "--":
        args.overrides = args.overrides[1:]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = list(args.overrides)
    if not has_group_override(overrides, "experiment"):
        overrides = ["+experiment=bcm_train", *overrides]

    cfg = load_config(overrides=overrides)
    cfg.mode = "train"
    cfg.training.enabled = True
    if args.run_root is not None:
        cfg.paths.run_root = args.run_root
    validate_config(cfg)

    options = DiagnosticsOptions(
        probe_count=int(args.probe_count),
        probe_every=int(args.probe_every),
        active_rate_threshold=float(args.active_rate_threshold),
        save_per_step_figures=not bool(args.no_per_step_figures),
        w_max=cfg.training.bcm.w_max,
        probe_seed=args.probe_seed,
        job_name=str(args.job_name),
        show_progress=not bool(args.no_progress),
        preload_cache=str(args.preload_cache),
        cache_prefetch=bool(args.cache_prefetch),
        network_cache_dir=args.network_cache_dir,
        use_network_cache=not bool(args.no_network_cache),
        stage_logging=not bool(args.no_stage_log),
    )

    result = run_bcm_training_diagnostics(cfg, options=options)
    print(f"Saved BCM diagnostics run: {result.run_dir}")
    print(f"Diagnostics directory: {result.diagnostics_dir}")
    print(f"Steps: {result.steps}")
    print(f"Final cap status: {result.summary.get('final_w_cap_status', 'unknown')}")
    print(f"Final stability status: {result.summary.get('final_weight_stability_status', 'unknown')}")


def _run_probe_batch(
    cfg: RootConfig,
    *,
    network: NetworkState,
    probe_drive_func: Callable[[float], Any],
    probe_count: int,
    time_grid: NDArray[np.float64],
    solver_fn: SolverCallable,
) -> BatchODEResult:
    return solver_fn(
        network=network,
        external_drive=probe_drive_func,
        time=time_grid,
        n_batch=int(probe_count),
        solver_config=cfg.solver,
        transfer_config=cfg.solver.transfer,
        training_bcm=cfg.training.bcm,
        background_trace=None,
        store_trajectory=False,
        stop_at_steady_state=_steady_state_enabled(cfg),
    )


def _get_or_build_network(
    cfg: RootConfig,
    options: DiagnosticsOptions,
    *,
    stage_log: StageLogger,
) -> NetworkState:
    if not options.use_network_cache or options.network_cache_dir is None:
        return build_network_state(cfg)
    if getattr(cfg.model, "trained_network_path", None):
        stage_log.info("network cache skipped because model.trained_network_path is set")
        return build_network_state(cfg)

    cache_key = _network_cache_key(cfg)
    cache_dir = Path(options.network_cache_dir) / cache_key
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open(encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata.get("cache_key") == cache_key:
                network = load_trained_network_state(cache_dir, model_cfg=cfg.model).network
                network.source.update({"mode": "diagnostics_cache", "path": str(cache_dir), "cache_key": cache_key})
                stage_log.info(f"network cache hit: {cache_dir}")
                return network
        except Exception as exc:
            stage_log.info(f"network cache ignored after load failure: {exc}")

    stage_log.info(f"network cache miss: {cache_dir}")
    network = build_network_state(cfg)
    save_checkpoint(
        cache_dir.parent,
        cache_dir.name,
        network,
        metadata={
            "cache_key": cache_key,
            "seed": cfg.seed,
            "sample_data_path": str(cfg.paths.sample_data_path),
            "model": asdict(cfg.model),
        },
    )
    metadata_path = cache_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            json_ready(
                {
                    "cache_key": cache_key,
                    "seed": cfg.seed,
                    "sample_data_path": str(cfg.paths.sample_data_path),
                    "model": asdict(cfg.model),
                }
            ),
            f,
            indent=2,
        )
    stage_log.info(f"network cache saved: {cache_dir}")
    return network


def _network_cache_key(cfg: RootConfig) -> str:
    payload = {
        "seed": cfg.seed,
        "sample_data_path": str(Path(cfg.paths.sample_data_path)),
        "sample_data_file": _file_fingerprint(Path(cfg.paths.sample_data_path)),
        "model": asdict(cfg.model),
    }
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"exists": False, "path": str(path)}
    return {
        "exists": True,
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _preload_samples(
    natural_drive: Any,
    samples: Sequence[Any],
    *,
    mode: str,
    stage_log: StageLogger,
    enabled: bool,
) -> None:
    if not hasattr(natural_drive, "preload_cache"):
        return
    if not samples:
        return
    if enabled:
        with stage_log.stage(f"preload Gabor cache ({mode}, {len(samples)} samples)"):
            natural_drive.preload_cache(samples)
        return
    natural_drive.preload_cache(samples)


def _warn_if_unseeded_natural_images(cfg: RootConfig, stage_log: StageLogger) -> None:
    if cfg.training.natural_image.seed is None:
        stage_log.info(
            "training.natural_image.seed is null; random crops reduce Gabor cache reuse across runs"
        )


def _make_probe_samples(
    cfg: RootConfig,
    natural_sampler: Any,
    *,
    probe_count: int,
    probe_seed: int | None,
) -> tuple[Any, ...]:
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

    path_limit = (
        max(1, ceil(int(probe_count) / max(1, patches_per_image)))
        if dataset is not None
        else int(probe_count)
    )
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


def _save_input_l4_distribution(input_l4_rates: NDArray[np.float64], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rates = np.asarray(input_l4_rates, dtype=float)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), dpi=150)
    axs[0].hist(rates.ravel(), bins=50, color="#3568a8", alpha=0.85)
    axs[0].set_title("All L4 X rates")
    axs[0].set_xlabel("rate (Hz)")
    axs[0].set_ylabel("count")
    axs[1].hist(np.mean(rates, axis=1), bins=40, color="#4b8f6b", alpha=0.85)
    axs[1].set_title("Mean by X neuron")
    axs[1].set_xlabel("mean rate (Hz)")
    axs[2].hist(np.mean(rates, axis=0), bins=min(20, max(3, rates.shape[1])), color="#a86045", alpha=0.85)
    axs[2].set_title("Mean by probe image")
    axs[2].set_xlabel("mean rate (Hz)")
    for ax in axs:
        ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_update_figure(
    path: Path,
    *,
    network: NetworkState,
    topology: NDArray[np.bool_],
    weights_after: NDArray[np.float64],
    delta: NDArray[np.float64],
    probe_dynamics: BatchODEResult,
    history: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    options: DiagnosticsOptions,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx_E = network.idx_E
    idx_I = network.idx_I
    idx_X = network.idx_X
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), dpi=140)
    fig.suptitle(f"BCM diagnostics update {metrics['step']}", fontsize=14, fontweight="bold")

    exc = np.asarray(probe_dynamics.exc, dtype=float).ravel()
    inh = np.asarray(probe_dynamics.inh, dtype=float).ravel()
    threshold = float(options.active_rate_threshold)
    axs[0, 0].hist(exc[exc > threshold], bins=35, alpha=0.75, label="E active", color="#3568a8")
    axs[0, 0].hist(inh[inh > threshold], bins=35, alpha=0.65, label="I active", color="#b6504a")
    axs[0, 0].set_title("Fixed-probe active rates")
    axs[0, 0].set_xlabel("rate (Hz)")
    axs[0, 0].legend()

    d_ee = delta_block_values(delta, topology, idx_E, idx_E)
    d_ie = delta_block_values(delta, topology, idx_I, idx_E)
    axs[0, 1].hist(d_ee, bins=50, alpha=0.75, label="dW_EE", color="#3568a8")
    axs[0, 1].hist(d_ie, bins=50, alpha=0.65, label="dW_IE", color="#4b8f6b")
    axs[0, 1].set_title("BCM weight changes")
    axs[0, 1].set_xlabel("delta weight")
    axs[0, 1].legend()

    for name, targets, sources, color in (
        ("W_EE", idx_E, idx_E, "#3568a8"),
        ("W_IE", idx_I, idx_E, "#4b8f6b"),
        ("W_EX", idx_E, idx_X, "#a86045"),
        ("W_IX", idx_I, idx_X, "#7d5ea8"),
    ):
        values = connected_block_values(weights_after, topology, targets, sources, include_zero=False)
        if values.size:
            axs[0, 2].hist(values, bins=45, alpha=0.45, label=name, color=color)
    axs[0, 2].set_title("Connected weight distributions")
    axs[0, 2].set_xlabel("weight")
    axs[0, 2].legend(fontsize=8)

    cap_vals = [
        float(metrics["W_EE_cap_fraction"]),
        float(metrics["W_IE_cap_fraction"]),
    ]
    axs[1, 0].bar(["W_EE", "W_IE"], cap_vals, color=["#3568a8", "#4b8f6b"])
    axs[1, 0].axhline(options.cap_warn_fraction, color="#c59b30", linestyle="--", linewidth=1, label="warn")
    axs[1, 0].axhline(options.cap_bad_fraction, color="#b6504a", linestyle="--", linewidth=1, label="bad")
    axs[1, 0].set_ylim(0.0, max(options.cap_bad_fraction * 1.4, max(cap_vals) * 1.2, 0.01))
    axs[1, 0].set_title(f"w_max pressure: {metrics['w_cap_status']}")
    axs[1, 0].set_ylabel("fraction near cap")
    axs[1, 0].legend(fontsize=8)

    steps = np.array([float(row["step"]) for row in history], dtype=float)
    axs[1, 1].plot(steps, [float(row["W_EE_mean"]) for row in history], label="W_EE mean", color="#3568a8")
    axs[1, 1].plot(steps, [float(row["W_IE_mean"]) for row in history], label="W_IE mean", color="#4b8f6b")
    axs[1, 1].set_title("Mean trained weights")
    axs[1, 1].set_xlabel("update")
    axs[1, 1].legend(fontsize=8)

    axs[1, 2].plot(steps, [float(row["theta_E_median"]) for row in history], label="theta_E median", color="#3568a8")
    axs[1, 2].plot(steps, [float(row["theta_I_median"]) for row in history], label="theta_I median", color="#b6504a")
    axs[1, 2].plot(steps, [float(row["probe_E_active_mean"]) for row in history], label="probe E active mean", color="#4b8f6b")
    axs[1, 2].plot(steps, [float(row["probe_I_active_mean"]) for row in history], label="probe I active mean", color="#a86045")
    axs[1, 2].set_title("Theta and fixed-probe activity")
    axs[1, 2].set_xlabel("update")
    axs[1, 2].legend(fontsize=8)

    for ax in axs.ravel():
        ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_training_overview(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    if not history:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return
    steps = np.array([float(row["step"]) for row in history], dtype=float)
    axs[0, 0].plot(steps, [float(row["W_EE_mean"]) for row in history], label="W_EE")
    axs[0, 0].plot(steps, [float(row["W_IE_mean"]) for row in history], label="W_IE")
    axs[0, 0].set_title("Mean trained weights")
    axs[0, 0].legend()
    axs[0, 1].plot(steps, [float(row["w_cap_max_fraction"]) for row in history], color="#b6504a")
    axs[0, 1].set_title("Max fraction near w_max")
    axs[1, 0].plot(steps, [float(row["probe_E_active_mean"]) for row in history], label="E active mean")
    axs[1, 0].plot(steps, [float(row["probe_I_active_mean"]) for row in history], label="I active mean")
    axs[1, 0].set_title("Fixed-probe active rates")
    axs[1, 0].legend()
    axs[1, 1].plot(steps, [float(row["theta_E_median"]) for row in history], label="theta_E")
    axs[1, 1].plot(steps, [float(row["theta_I_median"]) for row in history], label="theta_I")
    axs[1, 1].set_title("BCM theta medians")
    axs[1, 1].legend()
    for ax in axs.ravel():
        ax.set_xlabel("update")
        ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_final_weight_distribution(
    network: NetworkState,
    topology: NDArray[np.bool_],
    weights: NDArray[np.float64],
    path: Path,
    *,
    w_max: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idx_E = network.idx_E
    idx_I = network.idx_I
    idx_X = network.idx_X
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for name, targets, sources, color in (
        ("W_EE", idx_E, idx_E, "#3568a8"),
        ("W_IE", idx_I, idx_E, "#4b8f6b"),
        ("W_EX", idx_E, idx_X, "#a86045"),
        ("W_IX", idx_I, idx_X, "#7d5ea8"),
    ):
        values = connected_block_values(weights, topology, targets, sources, include_zero=False)
        if values.size:
            ax.hist(values, bins=60, alpha=0.48, label=name, color=color)
    if w_max is not None:
        ax.axvline(float(w_max), color="#b6504a", linestyle="--", linewidth=1.5, label="w_max")
    ax.set_title("Final connected weight distributions")
    ax.set_xlabel("weight")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_probe_response_stability(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    if history:
        steps = np.array([float(row["step"]) for row in history], dtype=float)
        axs[0].plot(steps, [float(row["probe_E_active_mean"]) for row in history], label="E")
        axs[0].plot(steps, [float(row["probe_I_active_mean"]) for row in history], label="I")
        axs[0].set_title("Active firing rate mean")
        axs[0].set_xlabel("update")
        axs[0].set_ylabel("Hz")
        axs[0].legend()
        axs[1].plot(steps, [float(row["probe_E_active_fraction"]) for row in history], label="E")
        axs[1].plot(steps, [float(row["probe_I_active_fraction"]) for row in history], label="I")
        axs[1].set_title("Active neuron fraction")
        axs[1].set_xlabel("update")
        axs[1].legend()
    for ax in axs:
        ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _build_summary(
    cfg: RootConfig,
    *,
    options: DiagnosticsOptions,
    history: Sequence[Mapping[str, Any]],
    input_l4_rates: NDArray[np.float64],
    stage_timings: Sequence[Mapping[str, Any]] | None = None,
    run_dir: Path,
    diagnostics_dir: Path,
    steps: int,
    samples_seen: int,
    images_seen: int,
) -> dict[str, Any]:
    if history:
        final = dict(history[-1])
        slope_info = _weight_stability(history, options.slope_rtol)
        skipped = int(sum(int(row["skipped_bad_batch"]) for row in history))
        bad_cap = int(sum(1 for row in history if row["w_cap_status"] == "bad"))
        warn_cap = int(sum(1 for row in history if row["w_cap_status"] == "warning"))
    else:
        final = {}
        slope_info = {"status": "unknown"}
        skipped = 0
        bad_cap = 0
        warn_cap = 0
    return {
        "run_dir": str(run_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "steps": int(steps),
        "samples_seen": int(samples_seen),
        "images_seen": int(images_seen),
        "epochs": int(cfg.training.bcm.epochs),
        "probe_count": int(options.probe_count),
        "preload_cache": options.preload_cache,
        "cache_prefetch": bool(options.cache_prefetch),
        "network_cache_dir": str(options.network_cache_dir) if options.network_cache_dir is not None else None,
        "use_network_cache": bool(options.use_network_cache),
        "stage_timings": list(stage_timings or ()),
        "active_rate_threshold": float(options.active_rate_threshold),
        "w_max": cfg.training.bcm.w_max,
        "l4_input": {
            "shape": list(input_l4_rates.shape),
            "mean": _safe_mean(input_l4_rates.ravel()),
            "p50": _safe_percentile(input_l4_rates.ravel(), 50),
            "p95": _safe_percentile(input_l4_rates.ravel(), 95),
            "max": _safe_max(input_l4_rates.ravel()),
        },
        "skipped_bad_batches": skipped,
        "cap_warning_steps": warn_cap,
        "cap_bad_steps": bad_cap,
        "final_w_cap_status": final.get("w_cap_status", "unknown"),
        "final_w_cap_max_fraction": final.get("w_cap_max_fraction", float("nan")),
        "final_probe_E_active_mean": final.get("probe_E_active_mean", float("nan")),
        "final_probe_I_active_mean": final.get("probe_I_active_mean", float("nan")),
        "final_weight_stability_status": slope_info.get("status", "unknown"),
        "weight_stability": slope_info,
    }


def _weight_stability(history: Sequence[Mapping[str, Any]], slope_rtol: float) -> dict[str, Any]:
    n = len(history)
    if n < 3:
        return {"status": "unknown", "reason": "fewer than 3 updates"}
    window = max(3, int(ceil(n * 0.25)))
    tail = history[-window:]
    x = np.array([float(row["step"]) for row in tail], dtype=float)
    result: dict[str, Any] = {"window": int(window)}
    statuses = []
    for key in ("W_EE_mean", "W_IE_mean"):
        y = np.array([float(row[key]) for row in tail], dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        scale = max(abs(float(np.mean(y))), 1.0e-12)
        rel = abs(slope) / scale
        result[f"{key}_slope"] = slope
        result[f"{key}_relative_slope"] = rel
        statuses.append(rel <= float(slope_rtol))
    result["status"] = "plateau_like" if all(statuses) else "drifting"
    return result


def _basic_stats(values: NDArray[np.float64], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": _safe_mean(values),
        f"{prefix}_p50": _safe_percentile(values, 50),
        f"{prefix}_p95": _safe_percentile(values, 95),
        f"{prefix}_p99": _safe_percentile(values, 99),
        f"{prefix}_max": _safe_max(values),
    }


def _delta_stats(values: NDArray[np.float64], prefix: str) -> dict[str, float]:
    abs_values = np.abs(np.asarray(values, dtype=float))
    changed = abs_values > 0.0
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_changed_fraction": float(np.mean(changed)) if values.size else 0.0,
        f"{prefix}_mean": _safe_mean(values),
        f"{prefix}_mean_abs": _safe_mean(abs_values),
        f"{prefix}_p95_abs": _safe_percentile(abs_values, 95),
        f"{prefix}_max_abs": _safe_max(abs_values),
    }


def _theta_stats(prefix: str, values: Any) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).ravel()
    return {
        f"{prefix}_min": _safe_min(arr),
        f"{prefix}_p05": _safe_percentile(arr, 5),
        f"{prefix}_median": _safe_percentile(arr, 50),
        f"{prefix}_p95": _safe_percentile(arr, 95),
        f"{prefix}_max": _safe_max(arr),
    }


def _empty_theta_stats(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_min": float("nan"),
        f"{prefix}_p05": float("nan"),
        f"{prefix}_median": float("nan"),
        f"{prefix}_p95": float("nan"),
        f"{prefix}_max": float("nan"),
    }


def _safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    return 0.0 if arr.size == 0 else float(np.mean(arr))


def _safe_min(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    return 0.0 if arr.size == 0 else float(np.min(arr))


def _safe_max(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    return 0.0 if arr.size == 0 else float(np.max(arr))


def _safe_percentile(values: Any, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    return 0.0 if arr.size == 0 else float(np.percentile(arr, q))


def _dense_weights(weights: Any) -> NDArray[np.float64]:
    return weights.toarray() if sparse.issparse(weights) else np.asarray(weights, dtype=float)


def _dense_topology(connectivity: Any) -> NDArray[np.bool_]:
    return connectivity.toarray().astype(bool) if sparse.issparse(connectivity) else np.asarray(connectivity, dtype=bool)


def _csv_value(value: Any) -> Any:
    value = json_ready(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2)


def _sample_json(sample: Any) -> dict[str, Any]:
    crop = getattr(sample, "crop", None)
    crop_payload = None
    if crop is not None:
        crop_payload = {
            "top": int(getattr(crop, "top")),
            "left": int(getattr(crop, "left")),
            "height": int(getattr(crop, "height")),
            "width": int(getattr(crop, "width")),
        }
    return {"path": str(getattr(sample, "path", sample)), "crop": crop_payload}


def _validate_options(options: DiagnosticsOptions) -> None:
    if int(options.probe_count) <= 0:
        raise ValueError("probe_count must be positive.")
    if int(options.probe_every) <= 0:
        raise ValueError("probe_every must be positive.")
    if options.preload_cache not in {"epoch", "batch", "none"}:
        raise ValueError("preload_cache must be one of: epoch, batch, none.")
    if float(options.active_rate_threshold) < 0.0:
        raise ValueError("active_rate_threshold must be non-negative.")
    if float(options.cap_warn_fraction) < 0.0 or float(options.cap_bad_fraction) < 0.0:
        raise ValueError("cap fractions must be non-negative.")
    if float(options.cap_warn_fraction) > float(options.cap_bad_fraction):
        raise ValueError("cap_warn_fraction must be <= cap_bad_fraction.")


if __name__ == "__main__":
    main()
