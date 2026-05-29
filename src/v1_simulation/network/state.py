from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from v1_simulation.network.geometry import SheetGeometry


def _readonly_1d_str(values: NDArray[np.str_] | list[str], *, name: str, length: int) -> NDArray[np.str_]:
    arr = np.asarray(values, dtype="<U1").reshape(-1).copy()
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {arr.shape}.")
    arr.setflags(write=False)
    return arr


def _readonly_1d_float(
    values: NDArray[np.float64] | list[float] | None,
    *,
    name: str,
    length: int,
) -> NDArray[np.float64] | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1).copy()
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {arr.shape}.")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, slots=True)
class PopulationLayout:
    """Geometry plus immutable population labels.

    Block suffixes in this package are target/source labels. For example, ``ei`` means
    target E neurons receiving from source I neurons.
    """

    l23: SheetGeometry
    l4: SheetGeometry
    l23_types: NDArray[np.str_]
    l4_tunings: NDArray[np.str_] | None = None
    l4_pref_dirs: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        l23_types = _readonly_1d_str(self.l23_types, name="l23_types", length=self.l23.n_cells)
        invalid = set(np.unique(l23_types)) - {"E", "I"}
        if invalid:
            raise ValueError(f"Unsupported L2/3 cell types: {sorted(invalid)}.")

        object.__setattr__(self, "l23_types", l23_types)
        if self.l4_tunings is not None:
            tunings = _readonly_1d_str(self.l4_tunings, name="l4_tunings", length=self.l4.n_cells)
            invalid_tunings = set(np.unique(tunings)) - {"T", "U"}
            if invalid_tunings:
                raise ValueError(f"Unsupported L4 tuning labels: {sorted(invalid_tunings)}.")
            object.__setattr__(self, "l4_tunings", tunings)
        object.__setattr__(
            self,
            "l4_pref_dirs",
            _readonly_1d_float(self.l4_pref_dirs, name="l4_pref_dirs", length=self.l4.n_cells),
        )

    @property
    def idx_E(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.l23_types == "E")

    @property
    def idx_I(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.l23_types == "I")

    @property
    def idx_X(self) -> NDArray[np.int64]:
        return np.arange(self.l23.n_cells, self.l23.n_cells + self.l4.n_cells, dtype=np.int64)

    @property
    def n_E(self) -> int:
        return int(np.sum(self.l23_types == "E"))

    @property
    def n_I(self) -> int:
        return int(np.sum(self.l23_types == "I"))

    @property
    def n_X(self) -> int:
        return self.l4.n_cells

    @property
    def shape(self) -> tuple[int, int]:
        return self.l23.n_cells, self.l23.n_cells + self.l4.n_cells


@dataclass(frozen=True, slots=True)
class NetworkState:
    layout: PopulationLayout
    connectivity: sparse.csr_matrix
    weights: sparse.csr_matrix | NDArray[np.float64]
    source: Mapping[str, Any] = field(default_factory=lambda: {"mode": "sampled", "path": None})

    def __post_init__(self) -> None:
        connectivity = sparse.csr_matrix(self.connectivity, dtype=bool)
        # Accept both sparse and dense weights to avoid costly CSR↔dense
        # round trips during training loops.
        if sparse.issparse(self.weights):
            weights = sparse.csr_matrix(self.weights, dtype=float)
        else:
            weights = np.asarray(self.weights, dtype=float)
            if weights.ndim != 2:
                raise ValueError("weights must be a 2D matrix.")
        if connectivity.shape != self.layout.shape:
            raise ValueError(
                f"connectivity shape {connectivity.shape} does not match layout shape {self.layout.shape}."
            )
        if weights.shape != self.layout.shape:
            raise ValueError(f"weights shape {weights.shape} does not match layout shape {self.layout.shape}.")
        object.__setattr__(self, "connectivity", connectivity)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "source", dict(self.source))

    @property
    def idx_E(self) -> NDArray[np.int64]:
        return self.layout.idx_E

    @property
    def idx_I(self) -> NDArray[np.int64]:
        return self.layout.idx_I

    @property
    def idx_X(self) -> NDArray[np.int64]:
        return self.layout.idx_X

    @property
    def Q_sparse(self) -> sparse.csr_matrix:
        return self.connectivity

    @property
    def QJ_ij(self) -> sparse.csr_matrix:
        return self.weights

    @property
    def Q(self) -> NDArray[np.float64]:
        return self.connectivity.toarray().astype(float)

    @property
    def J_ij(self) -> NDArray[np.float64]:
        return self.weights.toarray()


@dataclass(frozen=True, slots=True)
class TrainedNetworkState:
    """Loaded network snapshot plus optional training-run configuration."""

    path: Path
    network: NetworkState
    training_cfg: dict[str, Any] | None = None

    @property
    def weights(self) -> NDArray[np.float64]:
        return self.network.weights.toarray()

    @property
    def l4_tunings(self) -> NDArray[np.str_]:
        tunings = self.network.layout.l4_tunings
        if tunings is None:
            raise ValueError("Trained network snapshot is missing L4 tuning labels.")
        return tunings

    @property
    def l4_pref_dirs(self) -> NDArray[np.float64]:
        pref_dirs = self.network.layout.l4_pref_dirs
        if pref_dirs is None:
            raise ValueError("Trained network snapshot is missing L4 preferred directions.")
        return pref_dirs

    @property
    def l23_types(self) -> NDArray[np.str_]:
        return self.network.layout.l23_types

    @property
    def idx_E(self) -> NDArray[np.int64]:
        return self.network.idx_E

    @property
    def idx_I(self) -> NDArray[np.int64]:
        return self.network.idx_I

    @property
    def idx_X(self) -> NDArray[np.int64]:
        return self.network.idx_X


REQUIRED_LEGACY_NETWORK_FIELDS = (
    "W",
    "l4_tunings",
    "l4_pref_dirs",
    "l23_types",
    "idx_E",
    "idx_I",
    "idx_X",
)


def load_trained_network_state(
    path: str | Path,
    *,
    model_cfg: Any | None = None,
    expected_shape: tuple[int, int] | None = None,
) -> TrainedNetworkState:
    """Loads a trained network state from a checkpoint directory or legacy NPZ.

    Args:
        path: Path to the checkpoint directory or legacy NPZ file.
        model_cfg: Optional model configuration block for default layer settings if using legacy NPZ.
        expected_shape: Optional expected shape of the weights matrix to validate.

    Returns:
        The loaded TrainedNetworkState object.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If checkpoint files are missing or validation checks fail.
    """
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Trained network path does not exist: {snapshot_path}")
    if snapshot_path.is_dir() and not (snapshot_path / "layout.npz").exists():
        final_checkpoint = snapshot_path / "network_final"
        if final_checkpoint.exists():
            snapshot_path = final_checkpoint

    if snapshot_path.is_dir():
        state = _load_checkpoint_dir(snapshot_path)
    else:
        state = _load_legacy_npz(snapshot_path, model_cfg=model_cfg)

    if expected_shape is not None and state.network.weights.shape != tuple(expected_shape):
        raise ValueError(
            f"Trained network shape {state.network.weights.shape} does not match "
            f"expected shape {tuple(expected_shape)}."
        )
    return state


def _load_checkpoint_dir(path: Path) -> TrainedNetworkState:
    layout_path = path / "layout.npz"
    weights_path = path / "weights.npz"
    connectivity_path = path / "connectivity.npz"
    for required in (layout_path, weights_path, connectivity_path):
        if not required.exists():
            raise ValueError(f"{path} is missing checkpoint file: {required.name}")

    layout_data = np.load(layout_path, allow_pickle=False)
    try:
        layout = _layout_from_npz(layout_data)
    finally:
        layout_data.close()

    weights = sparse.load_npz(weights_path).tocsr()
    connectivity = sparse.load_npz(connectivity_path).tocsr()
    source = {"mode": "trained", "path": str(path)}
    network = NetworkState(layout=layout, connectivity=connectivity, weights=weights, source=source)
    return TrainedNetworkState(path=path, network=network, training_cfg=_load_training_cfg(path))


def _load_legacy_npz(path: Path, *, model_cfg: Any | None) -> TrainedNetworkState:
    loaded = np.load(path, allow_pickle=False)
    try:
        missing = [field_name for field_name in REQUIRED_LEGACY_NETWORK_FIELDS if field_name not in loaded.files]
        if missing:
            raise ValueError(f"{path} is missing required network fields: {missing}.")

        weights = np.asarray(loaded["W"], dtype=float)
        if weights.ndim != 2:
            raise ValueError(f"Trained network W must be a 2D matrix, got shape {weights.shape}.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("Trained network W contains NaN or infinite values.")

        l23_types = np.asarray(loaded["l23_types"]).astype(str, copy=False)
        l4_tunings = np.asarray(loaded["l4_tunings"]).astype(str, copy=False)
        l4_pref_dirs = np.asarray(loaded["l4_pref_dirs"], dtype=float)
        layout = _layout_from_arrays(
            l23_types=l23_types,
            l4_tunings=l4_tunings,
            l4_pref_dirs=l4_pref_dirs,
            model_cfg=model_cfg,
        )
        _validate_saved_indices("idx_E", np.asarray(loaded["idx_E"], dtype=int), layout.idx_E)
        _validate_saved_indices("idx_I", np.asarray(loaded["idx_I"], dtype=int), layout.idx_I)
        _validate_saved_indices("idx_X", np.asarray(loaded["idx_X"], dtype=int), layout.idx_X)
    finally:
        loaded.close()

    connectivity = sparse.csr_matrix(weights != 0.0)
    source = {"mode": "trained", "path": str(path), "format": "legacy_npz"}
    network = NetworkState(layout=layout, connectivity=connectivity, weights=sparse.csr_matrix(weights), source=source)
    return TrainedNetworkState(path=path, network=network, training_cfg=_load_training_cfg(path))


def _layout_from_npz(layout_data: np.lib.npyio.NpzFile) -> PopulationLayout:
    required = (
        "l23_n_side",
        "l23_region_size",
        "l23_z_pos",
        "l4_n_side",
        "l4_region_size",
        "l4_z_pos",
        "l23_types",
    )
    missing = [field_name for field_name in required if field_name not in layout_data.files]
    if missing:
        raise ValueError(f"layout.npz is missing required fields: {missing}.")

    return PopulationLayout(
        l23=SheetGeometry(
            int(layout_data["l23_n_side"]),
            float(layout_data["l23_region_size"]),
            float(layout_data["l23_z_pos"]),
        ),
        l4=SheetGeometry(
            int(layout_data["l4_n_side"]),
            float(layout_data["l4_region_size"]),
            float(layout_data["l4_z_pos"]),
        ),
        l23_types=np.asarray(layout_data["l23_types"]).astype(str, copy=False),
        l4_tunings=(
            np.asarray(layout_data["l4_tunings"]).astype(str, copy=False)
            if "l4_tunings" in layout_data.files
            else None
        ),
        l4_pref_dirs=(
            np.asarray(layout_data["l4_pref_dirs"], dtype=float)
            if "l4_pref_dirs" in layout_data.files
            else None
        ),
    )


def _layout_from_arrays(
    *,
    l23_types: NDArray[np.str_],
    l4_tunings: NDArray[np.str_],
    l4_pref_dirs: NDArray[np.float64],
    model_cfg: Any | None,
) -> PopulationLayout:
    l23_side = _square_side(l23_types.size, "l23_types")
    l4_side = _square_side(l4_tunings.size, "l4_tunings")
    layers = getattr(model_cfg, "layers", None)
    l23_cfg = getattr(layers, "l23", None)
    l4_cfg = getattr(layers, "l4", None)
    return PopulationLayout(
        l23=SheetGeometry(
            l23_side,
            _cfg_value(l23_cfg, "region_size", 2.0),
            _cfg_value(l23_cfg, "z_pos", 0.1),
        ),
        l4=SheetGeometry(
            l4_side,
            _cfg_value(l4_cfg, "region_size", 2.0),
            _cfg_value(l4_cfg, "z_pos", 0.0),
        ),
        l23_types=l23_types,
        l4_tunings=l4_tunings,
        l4_pref_dirs=l4_pref_dirs,
    )


def _load_training_cfg(path: Path) -> dict[str, Any] | None:
    candidates = []
    if path.is_dir():
        candidates.extend([path / "run_config.json", path.parent / "run_config.json"])
    else:
        candidates.append(path.parent / "run_config.json")

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open(encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        config = payload.get("config", payload.get("cfg"))
        if isinstance(config, dict):
            return config
    return None


def _validate_saved_indices(name: str, saved: NDArray[np.int64], current: NDArray[np.int64]) -> None:
    if not np.array_equal(saved, current):
        raise ValueError(f"Trained {name} does not match the saved network layout.")


def _square_side(size: int, name: str) -> int:
    side = int(round(np.sqrt(int(size))))
    if side * side != int(size):
        raise ValueError(f"{name} length {size} is not a square grid.")
    return side


def _cfg_value(cfg: Any, key: str, default: float) -> float:
    if cfg is None:
        return float(default)
    if isinstance(cfg, Mapping):
        return float(cfg.get(key, default))
    return float(getattr(cfg, key, default))
