from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import sparse


def create_unique_run_dir(run_root: str | Path, *, job_name: str = "training") -> Path:
    """Creates a timestamped run directory without overwriting existing artifacts.

    Args:
        run_root: The root path where all job directories are located.
        job_name: The name of the current job, which will be a folder under run_root.

    Returns:
        The Path to the newly created, unique timestamped run directory.
    """

    root = Path(run_root) / job_name
    root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{suffix:02d}"
        suffix += 1

    candidate.mkdir()
    return candidate


class TrainingCSVLogger:
    """Append-only CSV logger that writes the header once and flushes each row.

    Attributes:
        path: Path to the target CSV file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def append(self, row: Any) -> None:
        """Appends a new training log entry to the CSV file.

        The log row is flattened, and if it is the first entry, the CSV header is written.

        Args:
            row: A dataclass or mapping containing the logging statistics.
        """
        flat = flatten_log_row(row)

        if self._fieldnames is None:
            self._fieldnames = list(flat)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()

        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow({key: _csv_value(flat.get(key)) for key in self._fieldnames})
            f.flush()


class TrainingArtifacts:
    """Filesystem helper for saving and loading artifacts from a single training run.

    Attributes:
        run_dir: The unique directory for this specific training run.
        log: A TrainingCSVLogger instance for recording batch-level statistics.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = TrainingCSVLogger(self.run_dir / "training_log.csv")

    @classmethod
    def create(cls, run_root: str | Path, *, job_name: str = "training") -> "TrainingArtifacts":
        """Creates a new TrainingArtifacts instance with a unique, timestamped run directory.

        Args:
            run_root: The root path where run directories are created.
            job_name: The sub-folder job name under run_root.

        Returns:
            A new TrainingArtifacts instance configured with the unique directory.
        """
        return cls(create_unique_run_dir(run_root, job_name=job_name))

    def append_log(self, row: Any) -> None:
        self.log.append(row)

    def save_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(json_ready(payload), f, indent=2)
        return path

    def save_npz(self, name: str, **arrays: Any) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{key: json_ready(value) for key, value in arrays.items()})
        return path

    def save_sparse(self, name: str, matrix: Any) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(path, _as_csr(matrix))
        return path


class SimulationArtifacts:
    """Filesystem helper for saving and organizing simulation run artifacts.

    Attributes:
        run_dir: The unique directory path for this simulation run.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, run_root: str | Path, *, job_name: str = "simulation") -> "SimulationArtifacts":
        """Creates a SimulationArtifacts instance with a unique, timestamped run directory.

        Args:
            run_root: The root path under which the job directory will be created.
            job_name: The name of the job directory.

        Returns:
            A new SimulationArtifacts instance.
        """
        return cls(create_unique_run_dir(run_root, job_name=job_name))

    def save_result(self, result, *, save_network: bool = True) -> Path:
        """Persists the simulation results (trajectories, metadata, network state) to disk.

        Args:
            result: The SimulationResult object.
            save_network: If True, saves the network state as a checkpoint.

        Returns:
            The Path to the directory where results were saved.
        """
        return write_simulation_artifacts(result, self.run_dir, save_network=save_network)


def write_simulation_artifacts(result, output_dir: str | Path, *, save_network: bool = True) -> Path:
    """Writes simulation arrays, metadata, and an optional network checkpoint to disk.

    Args:
        result: The SimulationResult object containing trajectories and network state.
        output_dir: The target output directory path.
        save_network: If True, saves the network state as a checkpoint under output_dir/network.

    Returns:
        The Path to the output directory.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "theta_angles.npy", result.theta_angles)
    np.save(output_dir / "time.npy", result.time)
    np.save(output_dir / "exc_mean.npy", result.ode.exc)
    np.save(output_dir / "inh_mean.npy", result.ode.inh)

    if result.ode.exc_trajectory is not None:
        np.save(output_dir / "responses_exc.npy", result.exc_responses)
        np.save(output_dir / "aE_all.npy", result.aE_all)
    if result.ode.inh_trajectory is not None:
        np.save(output_dir / "responses_inh.npy", result.inh_responses)
        np.save(output_dir / "aI_all.npy", result.aI_all)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(result.metadata), f, indent=2)

    if save_network:
        from v1_simulation.training.checkpoints import save_checkpoint

        save_checkpoint(
            output_dir,
            "network",
            result.network,
            metadata={"network_source": dict(result.network.source)},
        )
    return output_dir


def flatten_log_row(row: Any) -> dict[str, Any]:
    """Flattens a dataclass or mapping into CSV-friendly top-level columns."""

    data = asdict(row) if is_dataclass(row) else dict(row)
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            flat.update({str(inner_key): inner_value for inner_key, inner_value in value.items()})
        else:
            flat[str(key)] = value
    return flat


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if sparse.issparse(value):
        return {
            "format": value.getformat(),
            "shape": list(value.shape),
            "nnz": int(value.nnz),
        }
    return value


def _as_csr(matrix: Any) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr(copy=True)
    return sparse.csr_matrix(np.asarray(matrix, dtype=float))


def _csv_value(value: Any) -> Any:
    value = json_ready(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value
