from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from v1_simulation.io.artifacts import json_ready
from v1_simulation.network.state import NetworkState
from v1_simulation.training.bcm import BCMThetaState


def save_checkpoint(
    run_dir: str | Path,
    name: str,
    network: NetworkState,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Saves a sparse network checkpoint under ``run_dir/name``.

    The weight and connectivity matrices are stored with ``scipy.sparse.save_npz``
    so large networks are not materialized as dense arrays for artifacts.

    Args:
        run_dir: The directory path where the run artifacts are located.
        name: The subdirectory name for this specific checkpoint (e.g. 'network_initial').
        network: The NetworkState object containing weights, connectivity, and layout.
        metadata: Optional metadata dictionary to be saved as JSON alongside the checkpoint.

    Returns:
        The Path to the created checkpoint subdirectory.
    """

    checkpoint_dir = Path(run_dir) / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    weights = sparse.csr_matrix(np.asarray(network.weights)) if not sparse.issparse(network.weights) else network.weights.tocsr()
    sparse.save_npz(checkpoint_dir / "weights.npz", weights)
    sparse.save_npz(checkpoint_dir / "connectivity.npz", network.connectivity.tocsr())
    np.savez_compressed(checkpoint_dir / "layout.npz", **_layout_arrays(network))

    if metadata is not None:
        with (checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(json_ready(metadata), f, indent=2)

    return checkpoint_dir


def load_checkpoint(run_dir: str | Path, name: str) -> dict[str, Any]:
    """Loads a sparse checkpoint saved by ``save_checkpoint``.

    Args:
        run_dir: The directory path containing the run artifacts.
        name: The subdirectory name of the checkpoint to load.

    Returns:
        A dictionary containing the reconstructed sparse weights, connectivity,
        layout configuration, and metadata.
    """

    checkpoint_dir = Path(run_dir) / name
    layout = np.load(checkpoint_dir / "layout.npz", allow_pickle=False)
    metadata_path = checkpoint_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "weights": sparse.load_npz(checkpoint_dir / "weights.npz").tocsr(),
        "connectivity": sparse.load_npz(checkpoint_dir / "connectivity.npz").tocsr(),
        "layout": {key: layout[key] for key in layout.files},
        "metadata": metadata,
    }


def save_theta(run_dir: str | Path, theta: BCMThetaState, *, name: str = "theta_M.npz") -> Path:
    """Saves the BCM sliding threshold state to a compressed npz file.

    Args:
        run_dir: The directory path where the run artifacts are stored.
        theta: The BCMThetaState object containing the sliding threshold arrays.
        name: The filename of the saved numpy archive. Defaults to "theta_M.npz".

    Returns:
        The Path to the saved file.
    """
    path = Path(run_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, theta_E=theta.E, theta_I=theta.I)
    return path


def _layout_arrays(network: NetworkState) -> dict[str, Any]:
    layout = network.layout
    arrays: dict[str, Any] = {
        "l23_n_side": np.array(layout.l23.n_side, dtype=int),
        "l23_region_size": np.array(layout.l23.region_size, dtype=float),
        "l23_z_pos": np.array(layout.l23.z_pos, dtype=float),
        "l4_n_side": np.array(layout.l4.n_side, dtype=int),
        "l4_region_size": np.array(layout.l4.region_size, dtype=float),
        "l4_z_pos": np.array(layout.l4.z_pos, dtype=float),
        "l23_types": np.asarray(layout.l23_types),
        "idx_E": np.asarray(layout.idx_E, dtype=int),
        "idx_I": np.asarray(layout.idx_I, dtype=int),
        "idx_X": np.asarray(layout.idx_X, dtype=int),
    }
    if layout.l4_tunings is not None:
        arrays["l4_tunings"] = np.asarray(layout.l4_tunings)
    if layout.l4_pref_dirs is not None:
        arrays["l4_pref_dirs"] = np.asarray(layout.l4_pref_dirs, dtype=float)
    return arrays
