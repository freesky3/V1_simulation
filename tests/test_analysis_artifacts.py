import json
import numpy as np
import pytest
from scipy import sparse

from v1_simulation.analysis.artifacts import (
    load_analysis_inputs_from_run,
    write_analysis_result_artifacts,
)
from v1_simulation.analysis.types import AnalysisResult, CommunityResult
from v1_simulation.network import SheetGeometry, PopulationLayout, NetworkState
from v1_simulation.training.checkpoints import save_checkpoint


def test_load_analysis_inputs_from_run(tmp_path) -> None:
    # 1. Setup mock run directory
    run_dir = tmp_path / "trial_run"
    run_dir.mkdir()

    # 2. Save mock network layout using save_checkpoint
    l23 = SheetGeometry(n_side=2, region_size=2.0, z_pos=0.1)
    l4 = SheetGeometry(n_side=1, region_size=2.0, z_pos=0.0)
    layout = PopulationLayout(
        l23=l23,
        l4=l4,
        l23_types=np.array(["E", "E", "I", "I"]),
        l4_tunings=np.array(["T"]),
        l4_pref_dirs=np.array([0.0]),
    )
    weights = np.ones((4, 5))
    network = NetworkState(
        layout=layout,
        connectivity=sparse.csr_matrix(weights != 0.0),
        weights=sparse.csr_matrix(weights),
    )
    save_checkpoint(run_dir, "network", network)

    # 3. Save expected response trajectories
    np.save(run_dir / "responses_exc.npy", np.ones((2, 3, 4)))  # 2 E-cells, 3 angles, 4 time points
    np.save(run_dir / "theta_angles.npy", np.array([0.0, 45.0, 90.0]))

    # 4. Load and verify
    inputs = load_analysis_inputs_from_run(run_dir)
    assert inputs.responses.shape == (2, 3, 4)
    assert inputs.theta_angles.shape == (3,)
    assert inputs.coords.shape == (2, 2)
    assert inputs.distance.shape == (2, 2)


def test_load_analysis_inputs_missing_files(tmp_path) -> None:
    run_dir = tmp_path / "trial_run_bad"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing excitatory response trajectory"):
        load_analysis_inputs_from_run(run_dir)

    np.save(run_dir / "responses_exc.npy", np.ones((2, 3, 4)))
    with pytest.raises(FileNotFoundError, match="Missing stimulus orientation array"):
        load_analysis_inputs_from_run(run_dir)


def test_write_analysis_result_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "analysis_output"

    comm = CommunityResult(
        labels=np.array([1, 2]),
        similarity=np.array([[1.0, 0.5], [0.5, 1.0]]),
        agreement=np.array([[1.0, 0.4], [0.4, 1.0]]),
        diagnostics={"consensus_tau": 0.5},
    )
    result = AnalysisResult(
        status="ok",
        selected_indices=np.array([0, 1]),
        osi=np.array([0.4, 0.6]),
        pref_ori=np.array([0.0, 45.0]),
        responses_mean=np.array([1.5, 2.0]),
        steady_state_responses=np.array([[1.5, 1.6], [2.0, 2.1]]),
        coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
        distance=np.array([[0.0, 1.0], [1.0, 0.0]]),
        communities=comm,
        diagnostics={
            "metrics_summary": {"n_neurons": 2, "n_ensembles": 2, "classified_fraction": 1.0},
            "ensemble_metrics": [{"ensemble_id": 1, "size": 1}],
        },
    )

    write_analysis_result_artifacts(result, output_dir)

    # Check numpy outputs
    assert (output_dir / "selected_indices.npy").exists()
    assert (output_dir / "osi.npy").exists()
    assert (output_dir / "pref_ori.npy").exists()
    assert (output_dir / "responses_mean.npy").exists()
    assert (output_dir / "steady_state_responses.npy").exists()
    assert (output_dir / "coords.npy").exists()
    assert (output_dir / "distance.npy").exists()
    assert (output_dir / "community_labels.npy").exists()
    assert (output_dir / "similarity.npy").exists()
    assert (output_dir / "agreement.npy").exists()

    # Check json diagnostics
    assert (output_dir / "community_diagnostics.json").exists()
    with (output_dir / "community_diagnostics.json").open("r", encoding="utf-8") as f:
        comm_diag = json.load(f)
    assert comm_diag["consensus_tau"] == 0.5

    assert (output_dir / "diagnostics.json").exists()
    with (output_dir / "diagnostics.json").open("r", encoding="utf-8") as f:
        diag = json.load(f)
    assert diag["metrics_summary"]["n_neurons"] == 2

    # Check tabular summary written by write_analysis_metrics
    assert (output_dir / "summary_metrics.json").exists()
    assert (output_dir / "ensemble_metrics.csv").exists()
