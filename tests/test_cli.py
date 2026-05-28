import numpy as np
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from v1_simulation.cli import has_group_override, merge_overrides
from v1_simulation.cli.main import app


def test_merge_overrides() -> None:
    # Typical merging
    res = merge_overrides(["a=1", "b=2"], ["c=3"])
    assert res == ["c=3", "a=1", "b=2"]

    # None handling
    assert merge_overrides(None, ["x=1"]) == ["x=1"]
    assert merge_overrides(["y=2"], None) == ["y=2"]
    assert merge_overrides(None, None) == []


def test_has_group_override() -> None:
    # Match prefixes
    assert has_group_override(["experiment=smoke"], "experiment") is True
    assert has_group_override(["+experiment=smoke"], "experiment") is True
    assert has_group_override(["experiment.name=smoke"], "experiment") is True

    # No match
    assert has_group_override(["solver=jax"], "experiment") is False
    assert has_group_override(["+solver.method=RK4"], "experiment") is False
    assert has_group_override([], "experiment") is False


@patch("v1_simulation.cli.run.run_simulation")
def test_run_command(mock_run: MagicMock) -> None:
    runner = CliRunner()
    mock_run.return_value = SimpleNamespace(
        run_dir=Path("/dummy/run"),
        result=SimpleNamespace(
            ode=SimpleNamespace(
                exc=np.ones((2, 3)),
                inh=np.ones((2, 1)),
            ),
            time=np.array([0.0, 0.1]),
            theta_angles=np.array([0.0, 45.0, 90.0]),
        ),
    )

    result = runner.invoke(app, ["run", "+experiment=smoke"])
    assert result.exit_code == 0
    assert "Saved simulation: \\dummy\\run" in result.stdout or "Saved simulation: /dummy/run" in result.stdout
    assert "Shapes: exc=(2, 3), inh=(2, 1), time_steps=2, orientations=3" in result.stdout
    mock_run.assert_called_once()


@patch("v1_simulation.cli.train.run_bcm_training")
def test_train_command(mock_train: MagicMock) -> None:
    runner = CliRunner()
    mock_train.return_value = SimpleNamespace(
        run_dir=Path("/dummy/train"),
        steps=5,
        samples_seen=20,
        images_seen=20,
    )

    # Set mock dir for natural image folder to avoid validation checks
    result = runner.invoke(
        app,
        [
            "train",
            "+experiment=bcm_train",
            "--natural-image-dir",
            "dummy_path",
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "Saved training run: \\dummy\\train" in result.stdout or "Saved training run: /dummy/train" in result.stdout
    assert "steps=5, samples_seen=20, images_seen=20" in result.stdout
    mock_train.assert_called_once()


@patch("v1_simulation.cli.analyze.load_analysis_inputs_from_run")
@patch("v1_simulation.cli.analyze.run_analysis")
@patch("v1_simulation.cli.analyze.write_analysis_result_artifacts")
def test_analyze_command(
    mock_write: MagicMock,
    mock_analyze: MagicMock,
    mock_load: MagicMock,
) -> None:
    runner = CliRunner()
    mock_load.return_value = SimpleNamespace()
    mock_analyze.return_value = SimpleNamespace(
        status="ok",
        selected_indices=np.array([0, 1]),
        communities=SimpleNamespace(n_ensembles=2),
        diagnostics={},
    )
    mock_write.return_value = Path("/dummy/run/analysis")

    result = runner.invoke(app, ["analyze", "/dummy/run"])
    assert result.exit_code == 0
    assert "Saved analysis: \\dummy\\run\\analysis" in result.stdout or "Saved analysis: /dummy/run/analysis" in result.stdout
    assert "status=ok, selected=2, ensembles=2" in result.stdout
    mock_load.assert_called_once()
    mock_analyze.assert_called_once()
    mock_write.assert_called_once()


@patch("v1_simulation.cli.sweep.run_grid_sweep")
def test_sweep_command(mock_sweep: MagicMock) -> None:
    runner = CliRunner()
    mock_sweep.return_value = SimpleNamespace(
        output_csv=Path("/dummy/sweep.csv"),
        summary_json=Path("/dummy/sweep.summary.json"),
        rows=[{}],
        skipped=0,
    )

    # Command sweep execution
    result = runner.invoke(app, ["sweep", "+experiment=smoke"])
    assert result.exit_code == 0
    assert "Saved sweep CSV: \\dummy\\sweep.csv" in result.stdout or "Saved sweep CSV: /dummy/sweep.csv" in result.stdout
    mock_sweep.assert_called_once()

    # Dry-run execution
    result_dry = runner.invoke(app, ["sweep", "+experiment=smoke", "--dry-run"])
    assert result_dry.exit_code == 0
    assert "Expanded" in result_dry.stdout
