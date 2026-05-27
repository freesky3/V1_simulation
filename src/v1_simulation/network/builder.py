from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from v1_simulation.config.schema import ModelConfig, RootConfig
from v1_simulation.network.connectivity import ConnectivitySpec, SpatialKernel, sample_connectivity
from v1_simulation.network.empirical import (
    ConnectionProbabilities,
    EmpiricalData,
    PopulationCounts,
    derive_connection_probabilities,
    derive_population_counts,
)
from v1_simulation.network.geometry import SheetGeometry
from v1_simulation.network.state import NetworkState, PopulationLayout
from v1_simulation.network.weights import WeightSpec, sample_weights


@dataclass(frozen=True, slots=True)
class NetworkRNGs:
    layout: np.random.Generator
    connectivity: np.random.Generator
    weights: np.random.Generator


@dataclass(frozen=True, slots=True)
class NetworkBuildSpec:
    counts: PopulationCounts
    probabilities: ConnectionProbabilities
    kernel: SpatialKernel
    connectivity: ConnectivitySpec
    weights: WeightSpec


def make_network_rngs(seed: int | np.random.SeedSequence | None) -> NetworkRNGs:
    """Create independent RNG streams for layout, connectivity, and weights."""

    seed_sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    layout_seed, connectivity_seed, weights_seed = seed_sequence.spawn(3)
    return NetworkRNGs(
        layout=np.random.default_rng(layout_seed),
        connectivity=np.random.default_rng(connectivity_seed),
        weights=np.random.default_rng(weights_seed),
    )


def build_network_state(
    cfg: RootConfig | ModelConfig,
    *,
    empirical: EmpiricalData | None = None,
    sample_data_path: str | Path | None = None,
    seed: int | np.random.SeedSequence | None = None,
    rngs: NetworkRNGs | None = None,
) -> NetworkState:
    """Builds a full network state from the configuration and empirical data.

    Args:
        cfg: The root or model configuration object.
        empirical: Optional pre-loaded empirical data. If None, loaded from path.
        sample_data_path: Optional path override for the empirical data file.
        seed: Optional random seed or SeedSequence to initialize generators.
        rngs: Optional pre-constructed NetworkRNGs streams.

    Returns:
        The fully built NetworkState containing layout, connectivity, and weights.
    """

    model_cfg, resolved_path, resolved_seed = _resolve_config_inputs(cfg, sample_data_path, seed)
    empirical_data = empirical if empirical is not None else EmpiricalData.from_path(resolved_path)
    streams = rngs if rngs is not None else make_network_rngs(resolved_seed)

    layout = build_population_layout(model_cfg, empirical_data, streams.layout)
    spec = build_network_spec(model_cfg, empirical_data, layout)
    connectivity = sample_connectivity(layout, spec.kernel, spec.connectivity, streams.connectivity)
    return sample_weights(layout, connectivity, spec.weights, empirical_data.weights, streams.weights)


def build_population_layout(
    model_cfg: ModelConfig,
    empirical: EmpiricalData,
    rng: np.random.Generator,
) -> PopulationLayout:
    """Builds the 2D sheet geometries and assigns cell types and tuning orientations.

    Args:
        model_cfg: The model configuration containing layer settings.
        empirical: The empirical ratios and data.
        rng: Generator stream for layout randomization.

    Returns:
        The PopulationLayout with Layer 2/3 and Layer 4 geometries and properties.
    """
    layers = model_cfg.layers
    l4 = SheetGeometry(layers.l4.n_side, layers.l4.region_size, layers.l4.z_pos)
    counts = derive_population_counts(
        n_x=l4.n_cells,
        empirical=empirical,
        l23_n_side=layers.l23.n_side,
        inhibitory_fraction=layers.l23.inhibitory_fraction,
    )
    l23 = SheetGeometry(counts.l23_n_side, layers.l23.region_size, layers.l23.z_pos)
    l23_types = assign_l23_types(
        n_side=counts.l23_n_side,
        n_inhibitory=counts.n_i,
        random_inhibitory=layers.l23.random_inhibitory,
        rng=rng,
    )
    l4_tunings, l4_pref_dirs = assign_l4_tuning(
        n_cells=l4.n_cells,
        all_tuned=layers.l4.all_tuned,
        tuned_fraction=empirical.eta_t_x,
        n_theta=layers.l4.N_theta,
        rng=rng,
    )
    return PopulationLayout(
        l23=l23,
        l4=l4,
        l23_types=l23_types,
        l4_tunings=l4_tunings,
        l4_pref_dirs=l4_pref_dirs,
    )


def build_network_spec(
    model_cfg: ModelConfig,
    empirical: EmpiricalData,
    layout: PopulationLayout,
) -> NetworkBuildSpec:
    """Derives population counts, connectivity spec, and weight spec.

    Args:
        model_cfg: The model configuration containing connectivity parameters.
        empirical: The empirical ratios and data.
        layout: The generated PopulationLayout.

    Returns:
        The NetworkBuildSpec specifying counts, connection probabilities, and scales.
    """
    counts = PopulationCounts(
        l23_n_side=layout.l23.n_side,
        n_e=layout.n_E,
        n_i=layout.n_I,
        n_x=layout.n_X,
    )
    probabilities = derive_connection_probabilities(
        counts=counts,
        empirical=empirical,
        p_ee=model_cfg.connectivity.p_ee,
    )
    kernel = SpatialKernel(
        sigma_narrow=model_cfg.connectivity.kernel.sigma_narrow,
        sigma_broad=model_cfg.connectivity.kernel.sigma_broad,
        kappa=model_cfg.connectivity.kernel.kappa,
    )
    connectivity = ConnectivitySpec.from_probabilities(
        probabilities,
        periodic=model_cfg.layers.periodic,
        equalize_indegree=model_cfg.connectivity.equalize_indegree,
    )
    weights = WeightSpec(
        j=model_cfg.connectivity.j,
        g=model_cfg.connectivity.g,
        ee_scale=model_cfg.connectivity.scales.ee,
        ei_scale=model_cfg.connectivity.scales.ei,
        ex_scale=model_cfg.connectivity.scales.ex,
        ie_scale=model_cfg.connectivity.scales.ie,
        ii_scale=model_cfg.connectivity.scales.ii,
        ix_scale=model_cfg.connectivity.scales.ix,
    )
    return NetworkBuildSpec(
        counts=counts,
        probabilities=probabilities,
        kernel=kernel,
        connectivity=connectivity,
        weights=weights,
    )


def assign_l23_types(
    *,
    n_side: int,
    n_inhibitory: int,
    random_inhibitory: bool,
    rng: np.random.Generator,
) -> NDArray[np.str_]:
    """Assigns cell types (Excitatory vs Inhibitory) to L2/3 neurons.

    Args:
        n_side: The number of neurons along one side of the L2/3 sheet.
        n_inhibitory: The target number of inhibitory neurons.
        random_inhibitory: If True, randomly samples positions. Otherwise,
            places them as uniformly as possible on a grid.
        rng: Generator stream for random selection.

    Returns:
        A 1D array of labels ("E" or "I") for each neuron in L2/3.
    """
    n_cells = int(n_side) * int(n_side)
    n_inhibitory = _bounded_count(n_inhibitory, n_cells, "n_inhibitory")
    types = np.full(n_cells, "E", dtype="<U1")
    if n_inhibitory == 0:
        return types
    if random_inhibitory:
        inhibitory_idx = rng.choice(n_cells, size=n_inhibitory, replace=False)
    else:
        inhibitory_idx = uniform_grid_indices(n_side=n_side, count=n_inhibitory)
    types[inhibitory_idx] = "I"
    return types


def uniform_grid_indices(*, n_side: int, count: int) -> NDArray[np.int64]:
    """Generates neuron indices that are as uniformly spaced as possible on a 2D grid.

    Args:
        n_side: The number of neurons along one side of the square grid.
        count: The target number of indices to select.

    Returns:
        A 1D array of unique integer indices representing the selected grid points.
    """
    n_side = int(n_side)
    n_cells = n_side * n_side
    count = _bounded_count(count, n_cells, "count")
    if count == 0:
        return np.array([], dtype=np.int64)
    if count == n_cells:
        return np.arange(n_cells, dtype=np.int64)

    n_cols = min(n_side, int(np.ceil(np.sqrt(count))))
    n_rows = min(n_side, int(np.ceil(count / n_cols)))
    while n_rows * n_cols < count:
        if n_cols < n_side:
            n_cols += 1
        elif n_rows < n_side:
            n_rows += 1
        else:
            break

    rows = np.rint(np.linspace(0, n_side - 1, n_rows)).astype(int)
    cols = np.rint(np.linspace(0, n_side - 1, n_cols)).astype(int)
    candidates = np.array([r * n_side + c for r in rows for c in cols], dtype=np.int64)
    candidates = np.unique(candidates)

    if candidates.size > count:
        keep = np.linspace(0, candidates.size - 1, count, dtype=int)
        candidates = candidates[keep]

    if candidates.size < count:
        all_indices = np.arange(n_cells, dtype=np.int64)
        remaining = np.setdiff1d(all_indices, candidates, assume_unique=True)
        extra = remaining[np.linspace(0, remaining.size - 1, count - candidates.size, dtype=int)]
        candidates = np.concatenate([candidates, extra])

    return candidates.astype(np.int64, copy=False)


def assign_l4_tuning(
    *,
    n_cells: int,
    all_tuned: bool,
    tuned_fraction: float,
    n_theta: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.str_], NDArray[np.float64]]:
    """Assigns orientation tuning labels and preferred directions to L4 neurons.

    Args:
        n_cells: Total number of cells in the L4 layer.
        all_tuned: If True, forces all cells to be orientation-tuned.
        tuned_fraction: Fraction of cells to tune if all_tuned is False.
        n_theta: Number of discrete orientation angles.
        rng: Generator stream for shuffling and choosing tuned cells.

    Returns:
        A tuple of:
            - A 1D array of tuning labels ("T" for Tuned, "U" for Untuned).
            - A 1D array of preferred orientations in radians (NaN if untuned).
    """
    n_cells = int(n_cells)
    if n_cells <= 0:
        raise ValueError(f"n_cells must be positive, got {n_cells}.")
    if n_theta <= 0:
        raise ValueError(f"n_theta must be positive, got {n_theta}.")
    if not 0.0 <= tuned_fraction <= 1.0:
        raise ValueError(f"tuned_fraction must be in [0, 1], got {tuned_fraction}.")

    n_tuned = n_cells if all_tuned else int(round(n_cells * tuned_fraction))
    n_tuned = _bounded_count(n_tuned, n_cells, "n_tuned")
    tunings = np.full(n_cells, "U", dtype="<U1")
    pref_dirs = np.full(n_cells, np.nan, dtype=float)
    if n_tuned == 0:
        return tunings, pref_dirs

    tuned_idx = rng.choice(n_cells, size=n_tuned, replace=False)
    tunings[tuned_idx] = "T"
    theta_values = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    assigned = np.resize(theta_values, n_tuned)
    rng.shuffle(assigned)
    pref_dirs[tuned_idx] = assigned
    return tunings, pref_dirs


def _bounded_count(count: int, total: int, name: str) -> int:
    count = int(count)
    if count < 0 or count > total:
        raise ValueError(f"{name} must be between 0 and {total}, got {count}.")
    return count


def _resolve_config_inputs(
    cfg: RootConfig | ModelConfig,
    sample_data_path: str | Path | None,
    seed: int | np.random.SeedSequence | None,
) -> tuple[ModelConfig, Path, int | np.random.SeedSequence | None]:
    if isinstance(cfg, RootConfig):
        model_cfg = cfg.model
        resolved_path = Path(sample_data_path) if sample_data_path is not None else Path(cfg.paths.sample_data_path)
        resolved_seed = seed if seed is not None else cfg.seed
        return model_cfg, resolved_path, resolved_seed

    if isinstance(cfg, ModelConfig):
        resolved_path = Path(sample_data_path) if sample_data_path is not None else Path("data/sample_data.pkl")
        return cfg, resolved_path, seed

    raise TypeError(f"cfg must be RootConfig or ModelConfig, got {type(cfg).__name__}.")
