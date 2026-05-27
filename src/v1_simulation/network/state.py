from __future__ import annotations

from dataclasses import dataclass

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
    weights: sparse.csr_matrix

    def __post_init__(self) -> None:
        connectivity = sparse.csr_matrix(self.connectivity, dtype=bool)
        weights = sparse.csr_matrix(self.weights, dtype=float)
        if connectivity.shape != self.layout.shape:
            raise ValueError(
                f"connectivity shape {connectivity.shape} does not match layout shape {self.layout.shape}."
            )
        if weights.shape != self.layout.shape:
            raise ValueError(f"weights shape {weights.shape} does not match layout shape {self.layout.shape}.")
        object.__setattr__(self, "connectivity", connectivity)
        object.__setattr__(self, "weights", weights)

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
