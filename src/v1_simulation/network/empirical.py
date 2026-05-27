from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


def _scalar_float(data: Mapping[str, Any], key: str) -> float:
    try:
        value = data[key]
    except KeyError as exc:
        raise KeyError(f"Missing empirical data key {key!r}.") from exc
    return float(np.asarray(value).item())


def _readonly_samples(data: Mapping[str, Any], key: str) -> NDArray[np.float64]:
    try:
        value = data[key]
    except KeyError as exc:
        raise KeyError(f"Missing empirical weight sample key {key!r}.") from exc
    arr = np.asarray(value, dtype=float).reshape(-1).copy()
    if arr.size == 0:
        raise ValueError(f"Empirical weight sample {key!r} must be non-empty.")
    arr.setflags(write=False)
    return arr


def _require_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value}.")


@dataclass(frozen=True, slots=True)
class EmpiricalWeightSamples:
    """Read-only empirical synaptic weight samples by target/source block."""

    ee: NDArray[np.float64]
    ei: NDArray[np.float64]
    ex: NDArray[np.float64]
    ie: NDArray[np.float64]
    ii: NDArray[np.float64]
    ix: NDArray[np.float64]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EmpiricalWeightSamples":
        return cls(
            ee=_readonly_samples(data, "sampled_J_EE"),
            ei=_readonly_samples(data, "sampled_J_EI"),
            ex=_readonly_samples(data, "sampled_J_EX"),
            ie=_readonly_samples(data, "sampled_J_IE"),
            ii=_readonly_samples(data, "sampled_J_II"),
            ix=_readonly_samples(data, "sampled_J_IX"),
        )


@dataclass(frozen=True, slots=True)
class EmpiricalData:
    """Immutable empirical ratios used to derive network counts and probabilities."""

    eta_i: float
    eta_x: float
    gamma_ee: float
    gamma_ei: float
    gamma_ex: float
    gamma_ie: float
    gamma_ii: float
    gamma_ix: float
    chi: float
    eta_t_e: float
    eta_t_x: float
    weights: EmpiricalWeightSamples

    def __post_init__(self) -> None:
        if self.eta_i < 0.0:
            raise ValueError(f"eta_i must be non-negative, got {self.eta_i}.")
        if self.eta_x <= 0.0:
            raise ValueError(f"eta_x must be positive, got {self.eta_x}.")
        if self.gamma_ee <= 0.0:
            raise ValueError(f"gamma_ee must be positive, got {self.gamma_ee}.")
        for name in ("gamma_ei", "gamma_ex", "gamma_ie", "gamma_ii", "gamma_ix", "chi"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        _require_probability(self.eta_t_e, "eta_t_e")
        _require_probability(self.eta_t_x, "eta_t_x")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EmpiricalData":
        return cls(
            eta_i=_scalar_float(data, "eta_I"),
            eta_x=_scalar_float(data, "eta_X"),
            gamma_ee=_scalar_float(data, "gamma_EE"),
            gamma_ei=_scalar_float(data, "gamma_EI"),
            gamma_ex=_scalar_float(data, "gamma_EX"),
            gamma_ie=_scalar_float(data, "gamma_IE"),
            gamma_ii=_scalar_float(data, "gamma_II"),
            gamma_ix=_scalar_float(data, "gamma_IX"),
            chi=_scalar_float(data, "chi"),
            eta_t_e=_scalar_float(data, "etaT_E"),
            eta_t_x=_scalar_float(data, "etaT_X"),
            weights=EmpiricalWeightSamples.from_mapping(data),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "EmpiricalData":
        loaded = np.load(Path(path), allow_pickle=True)
        if not isinstance(loaded, Mapping):
            raise TypeError(f"Expected empirical data at {path!s} to load as a mapping.")
        return cls.from_mapping(loaded)


@dataclass(frozen=True, slots=True)
class PopulationCounts:
    l23_n_side: int
    n_e: int
    n_i: int
    n_x: int

    @property
    def n_l23(self) -> int:
        return self.l23_n_side * self.l23_n_side

    def __post_init__(self) -> None:
        if self.l23_n_side <= 0:
            raise ValueError(f"l23_n_side must be positive, got {self.l23_n_side}.")
        if self.n_x <= 0:
            raise ValueError(f"n_x must be positive, got {self.n_x}.")
        if self.n_e < 0 or self.n_i < 0:
            raise ValueError(f"n_e and n_i must be non-negative, got {self.n_e}, {self.n_i}.")
        if self.n_e + self.n_i != self.n_l23:
            raise ValueError(
                "n_e + n_i must match L2/3 sheet size, "
                f"got n_e={self.n_e}, n_i={self.n_i}, n_l23={self.n_l23}."
            )


@dataclass(frozen=True, slots=True)
class ConnectionProbabilities:
    """Connection probabilities with target/source suffixes, e.g. ei = target E, source I."""

    ee: float
    ei: float
    ex: float
    ie: float
    ii: float
    ix: float

    def __post_init__(self) -> None:
        for name in ("ee", "ei", "ex", "ie", "ii", "ix"):
            _require_probability(getattr(self, name), f"p_{name}")


def derive_population_counts(
    *,
    n_x: int,
    empirical: EmpiricalData,
    l23_n_side: int | None = None,
    inhibitory_fraction: float | None = None,
) -> PopulationCounts:
    """Derives final L2/3 neuron counts before connectivity probabilities are calculated.

    Args:
        n_x: The number of external input neurons (L4 cells).
        empirical: The empirical ratios and data.
        l23_n_side: Optional number of neurons along one side of L2/3 sheet.
            If None, derived from n_x and empirical.eta_x.
        inhibitory_fraction: Optional fraction of L2/3 neurons that are inhibitory.
            If None, derived using empirical.eta_i ratio.

    Returns:
        The derived PopulationCounts specifying counts for all populations.
    """

    n_x = int(n_x)
    if l23_n_side is None:
        exact_n_e = n_x / empirical.eta_x
        exact_n_i = exact_n_e * empirical.eta_i
        l23_n_side = int(np.ceil(np.sqrt(exact_n_e + exact_n_i)))
    else:
        l23_n_side = int(l23_n_side)

    n_l23 = l23_n_side * l23_n_side
    if inhibitory_fraction is None:
        n_e = int(n_l23 / (1.0 + empirical.eta_i))
        n_i = n_l23 - n_e
    else:
        _require_probability(float(inhibitory_fraction), "inhibitory_fraction")
        n_i = int(round(n_l23 * float(inhibitory_fraction)))
        n_e = n_l23 - n_i

    return PopulationCounts(l23_n_side=l23_n_side, n_e=n_e, n_i=n_i, n_x=n_x)


def derive_connection_probabilities(
    *,
    counts: PopulationCounts,
    empirical: EmpiricalData,
    p_ee: float,
) -> ConnectionProbabilities:
    """Derives block-wise connection probabilities from final population counts and ratios.

    Args:
        counts: Derived PopulationCounts.
        empirical: The empirical ratios and constraints.
        p_ee: Base recurrent excitatory-to-excitatory connection probability.

    Returns:
        ConnectionProbabilities containing connection probabilities for all blocks (ee, ei, ex, ie, ii, ix).
    """

    _require_probability(float(p_ee), "p_ee")
    k_ee = float(p_ee) * counts.n_e
    k_e_total = k_ee / empirical.gamma_ee

    k_ei = empirical.gamma_ei * k_e_total
    k_ex = empirical.gamma_ex * k_e_total

    k_i_total = empirical.chi * k_e_total
    k_ie = empirical.gamma_ie * k_i_total
    k_ii = empirical.gamma_ii * k_i_total
    k_ix = empirical.gamma_ix * k_i_total

    return ConnectionProbabilities(
        ee=float(p_ee),
        ei=_divide_expected_count(k_ei, counts.n_i, "p_ei"),
        ex=_divide_expected_count(k_ex, counts.n_x, "p_ex"),
        ie=_divide_expected_count(k_ie, counts.n_e, "p_ie"),
        ii=_divide_expected_count(k_ii, counts.n_i, "p_ii"),
        ix=_divide_expected_count(k_ix, counts.n_x, "p_ix"),
    )


def _divide_expected_count(expected_count: float, n_source: int, name: str) -> float:
    if n_source <= 0:
        if expected_count == 0.0:
            return 0.0
        raise ValueError(f"Cannot derive {name}: source population has size {n_source}.")
    probability = float(expected_count) / float(n_source)
    _require_probability(probability, name)
    return probability
