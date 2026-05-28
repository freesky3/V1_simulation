import numpy as np
import pytest

from v1_simulation.analysis import AnalysisInputs, agreement_matrix, run_analysis, select_analysis_neuron_indices
from v1_simulation.config import load_config
from v1_simulation.config.schema import AnalysisConfig, LouvainConfig


def test_yaml_analysis_config_exposes_louvain_scientific_assumptions():
    cfg = load_config(
        overrides=[
            "+experiment=smoke",
            "analysis.louvain.consensus_tau=0.55",
            "analysis.louvain.min_module_degree=2.0",
            "analysis.louvain.min_cluster_size=3",
        ]
    )

    assert cfg.analysis.seed == cfg.seed
    assert cfg.analysis.louvain.consensus_tau == pytest.approx(0.55)
    assert cfg.analysis.louvain.min_module_degree == pytest.approx(2.0)
    assert cfg.analysis.louvain.min_cluster_size == 3


def test_select_analysis_neuron_indices_filters_osi_then_samples_reproducibly():
    osi = np.array([0.2, 0.39, 0.4, 0.55, np.nan, 0.8, 0.9])
    cfg_all = AnalysisConfig(seed=3, osi_threshold=0.4, random_sample_fraction=1.0)
    cfg_sample = AnalysisConfig(seed=3, osi_threshold=0.4, random_sample_fraction=0.5)

    all_candidates = select_analysis_neuron_indices(osi, config=cfg_all)
    sampled_once = select_analysis_neuron_indices(osi, config=cfg_sample)
    sampled_twice = select_analysis_neuron_indices(osi, config=cfg_sample)

    assert np.array_equal(all_candidates, np.array([2, 3, 5, 6]))
    assert np.array_equal(sampled_once, sampled_twice)
    assert sampled_once.shape == (2,)
    assert set(sampled_once).issubset(set(all_candidates))


def test_agreement_matrix_accumulates_partitions_without_large_temp():
    partitions = np.array(
        [
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 1],
        ]
    )

    agreement = agreement_matrix(partitions)

    assert np.array_equal(
        agreement,
        np.array(
            [
                [3.0, 2.0, 0.0],
                [2.0, 3.0, 1.0],
                [0.0, 1.0, 3.0],
            ]
        ),
    )


def test_run_analysis_uses_schema_louvain_config_and_returns_label_array(monkeypatch):
    from v1_simulation.analysis import communities

    calls = {}

    def threshold_proportional(matrix, thr_prop):
        calls["thr_prop"] = thr_prop
        return np.ones_like(matrix) - np.eye(matrix.shape[0])

    def community_louvain(matrix, gamma=1.0, seed=None):
        calls.setdefault("community_seeds", []).append(seed)
        calls["gamma"] = gamma
        return np.array([1, 1, 2, 2]), 0.0

    def consensus_und(matrix, tau, reps=1000, seed=None):
        calls["tau"] = tau
        calls["reps"] = reps
        calls["consensus_seed"] = seed
        return np.array([1, 1, 2, 2])

    monkeypatch.setattr(communities.bct, "threshold_proportional", threshold_proportional)
    monkeypatch.setattr(communities.bct, "weight_conversion", lambda matrix, _mode: matrix)
    monkeypatch.setattr(communities.bct, "community_louvain", community_louvain)
    monkeypatch.setattr(communities.bct, "consensus_und", consensus_und)

    cfg = AnalysisConfig(
        seed=7,
        osi_threshold=0.0,
        random_sample_fraction=1.0,
        louvain=LouvainConfig(
            thr_prop=0.3,
            gamma=0.8,
            num_runs=2,
            consensus_tau=0.42,
            consensus_reps=5,
            min_module_degree=0.0,
            min_cluster_size=2,
        ),
    )
    responses = np.ones((4, 4, 12), dtype=float)
    coords = np.column_stack((np.arange(4, dtype=float), np.zeros(4)))
    distance = np.abs(coords[:, 0, None] - coords[None, :, 0])
    inputs = AnalysisInputs(
        responses=responses,
        coords=coords,
        distance=distance,
        theta_angles=np.linspace(0.0, np.pi, 4, endpoint=False),
    )

    result = run_analysis(cfg, inputs)

    assert result.status == "ok"
    assert np.array_equal(result.communities.labels, np.array([1, 1, 2, 2]))
    assert calls["thr_prop"] == pytest.approx(0.3)
    assert calls["gamma"] == pytest.approx(0.8)
    assert calls["tau"] == pytest.approx(0.42)
    assert calls["reps"] == 5
    assert len(calls["community_seeds"]) == 2
    assert result.communities.diagnostics["min_cluster_size"] == 2


def test_analysis_inputs_validate_shape_contract():
    inputs = AnalysisInputs(
        responses=np.ones((3, 2, 4)),
        coords=np.ones((2, 2)),
        distance=np.eye(3),
        theta_angles=np.linspace(0.0, np.pi, 2, endpoint=False),
    )

    with pytest.raises(ValueError, match="coords"):
        inputs.validate()
