import csv
import json
import numpy as np
import pytest

from v1_simulation.analysis.metrics import (
    activity_health_metrics,
    osi_distribution_metrics,
    summarize_communities,
    write_analysis_metrics,
)


def test_activity_health_metrics() -> None:
    # 1. 1D response input
    resp_1d = np.array([0.0, 1.0, 2.0])
    metrics_1d = activity_health_metrics(resp_1d, active_threshold=0.5)
    assert metrics_1d["active_neuron_count"] == 2
    assert metrics_1d["active_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics_1d["silent_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics_1d["rate_mean"] == pytest.approx(1.0)
    assert metrics_1d["rate_median"] == pytest.approx(1.0)
    assert metrics_1d["rate_max"] == pytest.approx(2.0)
    assert metrics_1d["top1_activity_fraction"] == pytest.approx(2.0 / 3.0)

    # 2. 2D response input (e.g. n_neurons x n_orientations)
    resp_2d = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    metrics_2d = activity_health_metrics(resp_2d, active_threshold=0.5)
    assert metrics_2d["active_neuron_count"] == 2
    assert metrics_2d["rate_mean"] == pytest.approx(1.0)

    # 3. 3D response input (e.g. n_neurons x n_orientations x time)
    resp_3d = np.ones((3, 2, 4))
    metrics_3d = activity_health_metrics(resp_3d, active_threshold=0.5)
    assert metrics_3d["active_neuron_count"] == 3

    # 4. Error check (4D input)
    with pytest.raises(ValueError, match="responses must be 1D, 2D, or 3D"):
        activity_health_metrics(np.ones((2, 2, 2, 2)))


def test_osi_distribution_metrics() -> None:
    osi = np.array([0.1, 0.3, np.nan, 0.5, 0.7])
    metrics = osi_distribution_metrics(osi)

    assert metrics["n_neurons"] == 5
    assert metrics["osi_finite_count"] == 4
    assert metrics["osi_finite_fraction"] == pytest.approx(0.8)
    assert metrics["osi_mean"] == pytest.approx(0.4)
    assert metrics["osi_median"] == pytest.approx(0.4)
    assert metrics["osi_count_gt_0_4"] == 2
    assert metrics["osi_fraction_gt_0_4"] == pytest.approx(0.5)


def test_summarize_communities() -> None:
    labels = np.array([1, 1, 2, 2, 0])
    similarity = np.array([
        [1.0, 0.8, 0.1, 0.1, 0.0],
        [0.8, 1.0, 0.1, 0.1, 0.0],
        [0.1, 0.1, 1.0, 0.9, 0.0],
        [0.1, 0.1, 0.9, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [11.0, 10.0],
        [0.0, 10.0],
    ])
    osi = np.array([0.5, 0.6, 0.7, 0.8, np.nan])
    pref_ori = np.array([0.0, 0.0, np.pi / 2, np.pi / 2, np.nan])

    summary, rows = summarize_communities(
        labels,
        similarity=similarity,
        distance=None,  # skip distance matrices
        coords=coords,
        osi=osi,
        pref_ori=pref_ori,
    )

    assert summary["n_neurons"] == 5
    assert summary["n_ensembles"] == 2
    assert summary["classified_neurons"] == 4
    assert summary["unclassified_neurons"] == 1
    assert summary["classified_fraction"] == pytest.approx(0.8)

    # Within/between similarity at global level
    assert "within_similarity_mean" in summary
    assert "between_similarity_mean" in summary

    # Rows assertions (for each community ID)
    assert len(rows) == 2
    # Ensemble 1: members [0, 1]
    row1 = rows[0]
    assert row1["ensemble_id"] == 1
    assert row1["size"] == 2
    assert row1["member_fraction"] == pytest.approx(0.4)
    assert row1["within_similarity_mean"] == pytest.approx(0.8)
    assert row1["centroid_x"] == pytest.approx(0.5)
    assert row1["centroid_y"] == pytest.approx(0.0)
    assert row1["member_osi_mean"] == pytest.approx(0.55)
    assert row1["member_pref_ori_coherence"] == pytest.approx(1.0)  # perfect agreement (both 0.0)


def test_write_analysis_metrics(tmp_path) -> None:
    summary = {"n_neurons": 100, "classified_fraction": 0.8}
    rows = [
        {"ensemble_id": 1, "size": 10, "centroid_x": 0.5},
        {"ensemble_id": 2, "size": 20, "centroid_x": 1.5},
    ]

    sum_path, ens_path = write_analysis_metrics(summary, rows, tmp_path)

    # Verify JSON summary structure
    assert sum_path.exists()
    with sum_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["n_neurons"] == 100
    assert data["classified_fraction"] == 0.8

    # Verify CSV rows structure
    assert ens_path.exists()
    with ens_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        results = list(reader)

    assert header == ["ensemble_id", "size", "centroid_x"]
    assert len(results) == 2
    assert results[0]["ensemble_id"] == "1"
    assert results[1]["size"] == "20"
