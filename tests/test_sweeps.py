import json
import math
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from v1_simulation.analysis import AnalysisResult
from v1_simulation.sweeps import (
    SweepPoint,
    best_row,
    expand_grid,
    format_override_value,
    grid_size,
    normalize_sweep_key,
    ordered_fieldnames,
    parameter_key,
    parameters_to_overrides,
    quality_score,
    read_completed_parameter_keys,
    read_sweep_rows,
    run_grid_sweep,
    run_sweep_trial,
    score_analysis_result,
    write_sweep_csv,
    write_sweep_summary,
)


def test_grid_utilities() -> None:
    grid = {
        "stimulus.sigma": [0.1, 0.2],
        "model.connectivity.p_ee": np.array([0.15]),
        "model.stimulus.spatial_frequency": 10.0,
    }

    # Normalize sweep keys (checking aliases conversion)
    assert normalize_sweep_key("stimulus.sigma") == "stimulus.gabor.sigma"
    assert normalize_sweep_key("model.stimulus.spatial_frequency") == "stimulus.gabor.spatial_frequency"
    assert normalize_sweep_key("model.connectivity.p_ee") == "model.connectivity.p_ee"

    # Grid combinations size
    assert grid_size(grid) == 2

    # Expansion
    points = expand_grid(grid)
    assert len(points) == 2
    assert isinstance(points[0], SweepPoint)
    assert points[0].index == 0
    assert points[0].parameters["stimulus.gabor.sigma"] == 0.1
    assert points[1].parameters["stimulus.gabor.sigma"] == 0.2

    # Overrides list
    assert points[0].overrides == [
        "stimulus.gabor.sigma=0.1",
        "model.connectivity.p_ee=0.15",
        "stimulus.gabor.spatial_frequency=10.0",
    ]

    # Deterministic JSON key check (sorted by parameters keys)
    expected_key = json.dumps(
        {
            "model.connectivity.p_ee": 0.15,
            "stimulus.gabor.sigma": 0.1,
            "stimulus.gabor.spatial_frequency": 10.0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert points[0].key == expected_key

    # Format override values
    assert format_override_value(None) == "null"
    assert format_override_value(True) == "true"
    assert format_override_value(False) == "false"
    assert format_override_value("str_val") == "str_val"
    assert format_override_value([1, 2]) == "1,2" or format_override_value([1, 2]) == "[1,2]"


def test_scoring_heuristic() -> None:
    summary = {
        "classified_fraction": 0.8,
        "n_ensembles": 3,
        "osi_mean": 0.4,
    }
    diagnostics = {
        "active_fraction": 0.9,
        "silent_fraction": 0.1,
        "top1_activity_fraction": 0.25,
    }

    score = quality_score(summary, diagnostics)
    # Check diversity factor: log1p(3) ~ 1.3863
    # penalty: 1.0 - 0.5*0.1 - 0.25*0.25 = 1.0 - 0.05 - 0.0625 = 0.8875
    # score = 0.8 * 0.9 * (1.0 + 0.4) * (1.0 + 1.3863) * 0.8875 > 0
    assert score > 0.0

    # Test score mapping function
    mock_res = AnalysisResult(
        status="ok",
        selected_indices=np.array([1, 2, 3]),
        osi=np.array([0.4]),
        pref_ori=np.array([0.0]),
        responses_mean=np.array([1.0]),
        steady_state_responses=np.array([[1.0]]),
        coords=np.array([[0.0, 0.0]]),
        distance=np.array([[0.0]]),
        communities=None,
        diagnostics={
            "active_fraction": 0.5,
            "metrics_summary": {"osi_mean": 0.3},
        },
    )

    scored = score_analysis_result(mock_res)
    assert scored["analysis_status"] == "ok"
    assert scored["selected_neurons"] == 3
    assert scored["score"] >= 0.0
    assert scored["active_fraction"] == 0.5
    assert scored["osi_mean"] == 0.3


def test_summary_writing_and_reading(tmp_path) -> None:
    csv_path = tmp_path / "sweep.csv"
    summary_path = tmp_path / "sweep.summary.json"

    rows = [
        {
            "trial": 0,
            "status": "completed",
            "score": 1.2,
            "parameter_key": "k1",
            "overrides": "o1",
            "param.p_ee": 0.1,
        },
        {
            "trial": 1,
            "status": "failed",
            "score": None,
            "parameter_key": "k2",
            "overrides": "o2",
            "param.p_ee": 0.2,
            "error": "ValueError: error",
        },
        {
            "trial": 2,
            "status": "completed",
            "score": 2.5,
            "parameter_key": "k3",
            "overrides": "o3",
            "param.p_ee": 0.3,
        },
    ]

    # Write CSV
    write_sweep_csv(rows, csv_path)
    assert csv_path.exists()

    # Read CSV back
    read_rows = read_sweep_rows(csv_path)
    assert len(read_rows) == 3
    # Check string conversions of DictReader
    assert read_rows[0]["trial"] == "0"
    assert read_rows[0]["status"] == "completed"
    assert read_rows[0]["score"] == "1.2"

    # Completed keys selection
    completed = read_completed_parameter_keys(csv_path)
    assert completed == {"k1", "k3"}

    # Best row selection
    best = best_row(rows)
    assert best is not None
    assert best["trial"] == 2
    assert best["score"] == 2.5

    # Write JSON summary
    write_sweep_summary(rows, summary_path)
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert summary_data["total_rows"] == 3
    assert summary_data["completed"] == 2
    assert summary_data["failed"] == 1
    assert summary_data["best"]["trial"] == 2


def test_ordered_fieldnames() -> None:
    rows = [
        {"trial": 1, "param.p_ee": 0.1, "extra_metric": 5.0},
    ]
    headers = ordered_fieldnames(rows)
    assert headers[0] == "trial"
    assert headers[-2] == "param.p_ee"
    assert headers[-1] == "extra_metric"


@patch("v1_simulation.sweeps.runner.run_simulation")
@patch("v1_simulation.sweeps.runner.run_analysis")
@patch("v1_simulation.sweeps.runner.write_analysis_result_artifacts")
@patch("v1_simulation.sweeps.runner.validate_config")
def test_run_sweep_trial(
    mock_validate: MagicMock,
    mock_write: MagicMock,
    mock_analyze: MagicMock,
    mock_run: MagicMock,
) -> None:
    point = SweepPoint(index=1, parameters={"p_ee": 0.1})
    cfg = SimpleNamespace(analysis=SimpleNamespace())

    # Successful simulation run
    mock_run.return_value = SimpleNamespace(run_dir=Path("/dummy/run"), result=SimpleNamespace(analysis_inputs=lambda: None))
    mock_analyze.return_value = SimpleNamespace(
        status="ok",
        selected_indices=np.array([1]),
        communities=None,
        diagnostics={},
    )
    mock_write.return_value = Path("/dummy/run/analysis")

    row = run_sweep_trial(point, cfg)  # type: ignore
    assert row["trial"] == 1
    assert row["status"] == "completed"
    assert row["run_dir"] == "/dummy/run" or row["run_dir"] == "\\dummy\\run"
    assert row["analysis_dir"] == "/dummy/run/analysis" or row["analysis_dir"] == "\\dummy\\run\\analysis"


@patch("v1_simulation.sweeps.runner.load_config")
@patch("v1_simulation.sweeps.runner._run_trials")
def test_run_grid_sweep(mock_run_trials: MagicMock, mock_load_config: MagicMock, tmp_path) -> None:
    csv_path = tmp_path / "sweep.csv"

    # Base configuration mock
    base_cfg = SimpleNamespace(
        sweep=SimpleNamespace(
            grid={"param1": [1.0, 2.0]},
            max_workers=2,
            resume=True,
            output_csv=str(csv_path),
        ),
        paths=SimpleNamespace(run_root=str(tmp_path)),
        job_name="v1",
    )
    mock_load_config.return_value = base_cfg

    # Mock trial execution return rows
    mock_run_trials.return_value = [
        {"trial": 0, "status": "completed", "parameter_key": '{"param1":1.0}', "score": 1.0},
        {"trial": 1, "status": "completed", "parameter_key": '{"param1":2.0}', "score": 2.0},
    ]

    result = run_grid_sweep(
        config_path=None,
        config_name="config",
        output_csv=csv_path,
        resume=True,
    )

    assert len(result.rows) == 2
    assert result.output_csv == csv_path
    assert result.summary_json == csv_path.with_suffix(".summary.json")
    assert result.summary_json.exists()
