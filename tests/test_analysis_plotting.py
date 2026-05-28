import matplotlib
matplotlib.use("Agg")

import os
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from v1_simulation.analysis.types import AnalysisResult, CommunityResult
from v1_simulation.analysis.plotting import (
    generate_and_save_all_analysis_plots,
    OSI_HISTOGRAM_FILENAME,
    OSI_SPATIAL_FILENAME,
    PREF_ORI_SPATIAL_FILENAME,
    ORI_CENTERS_SPATIAL_FILENAME,
    ENSEMBLE_CORRELATION_FILENAME,
    ENSEMBLE_ACTIVITY_TRACE_FILENAME,
    ENSEMBLE_ACTIVITY_TRACE_NORM_VAR_FILENAME,
    ENSEMBLE_SPATIAL_PREFIX,
    SPATIAL_SURROGATE_PREFIX,
    SPATIAL_SURROGATE_NND_SUFFIX,
    SPATIAL_SURROGATE_MEANDIST_SUFFIX,
)


def test_plotting_normal(tmp_path: Path) -> None:
    # 10 neurons, 2 communities (labels 1 and 2), and some unclassified (0)
    labels = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0, 0], dtype=np.int64)
    similarity = np.eye(10) * 0.8 + 0.1
    np.fill_diagonal(similarity, 0.0)
    agreement = np.eye(10) * 0.9
    comm = CommunityResult(labels=labels, similarity=similarity, agreement=agreement)

    # Construct steady state responses (10 neurons, 4 theta orientations, 15 time points)
    steady_state = np.random.default_rng(42).random((10, 4, 15))

    result = AnalysisResult(
        status="ok",
        selected_indices=np.arange(10),
        osi=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2, 0.1, 0.0]),
        pref_ori=np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, 0.0, np.pi/6, np.nan, np.nan, np.nan, np.nan]),
        responses_mean=np.ones(10) * 1.5,
        steady_state_responses=steady_state,
        coords=np.random.default_rng(42).random((10, 2)) * 100.0,
        distance=np.random.default_rng(42).random((10, 10)) * 50.0,
        communities=comm,
        diagnostics={},
    )

    # Generate plots (low surrogate count for speed)
    output_dir = tmp_path / "plots"
    generated_paths = generate_and_save_all_analysis_plots(
        result,
        output_dir,
        num_surrogates=5,
        rng_seed=123,
    )

    # Verify output list
    assert len(generated_paths) > 0

    # Expected files
    expected_filenames = {
        OSI_HISTOGRAM_FILENAME,
        OSI_SPATIAL_FILENAME,
        PREF_ORI_SPATIAL_FILENAME,
        ORI_CENTERS_SPATIAL_FILENAME,
        ENSEMBLE_CORRELATION_FILENAME,
        ENSEMBLE_ACTIVITY_TRACE_FILENAME,
        ENSEMBLE_ACTIVITY_TRACE_NORM_VAR_FILENAME,
        f"{ENSEMBLE_SPATIAL_PREFIX}1.png",
        f"{ENSEMBLE_SPATIAL_PREFIX}2.png",
        f"{SPATIAL_SURROGATE_PREFIX}1{SPATIAL_SURROGATE_NND_SUFFIX}",
        f"{SPATIAL_SURROGATE_PREFIX}1{SPATIAL_SURROGATE_MEANDIST_SUFFIX}",
        f"{SPATIAL_SURROGATE_PREFIX}2{SPATIAL_SURROGATE_NND_SUFFIX}",
        f"{SPATIAL_SURROGATE_PREFIX}2{SPATIAL_SURROGATE_MEANDIST_SUFFIX}",
    }

    for fname in expected_filenames:
        path = output_dir / fname
        assert path.exists(), f"Missing file: {fname}"
        assert path in generated_paths
        assert path.stat().st_size > 0
        
        # Check image validity using PIL
        with Image.open(path) as img:
            img.verify()

    # Explicitly verify ensemble_activity_trace_with_variance.png does NOT exist
    assert not (output_dir / "ensemble_activity_trace_with_variance.png").exists()


def test_plotting_no_communities(tmp_path: Path) -> None:
    result = AnalysisResult(
        status="not_enough_neurons",
        selected_indices=np.array([0]),
        osi=np.array([0.5]),
        pref_ori=np.array([0.0]),
        responses_mean=np.array([1.5]),
        steady_state_responses=np.ones((1, 2, 5)),
        coords=np.array([[0.0, 0.0]]),
        distance=np.array([[0.0]]),
        communities=None,
        diagnostics={},
    )

    output_dir = tmp_path / "plots"
    generated_paths = generate_and_save_all_analysis_plots(result, output_dir)

    # Only OSI-related plots should be created
    expected_filenames = {
        OSI_HISTOGRAM_FILENAME,
        OSI_SPATIAL_FILENAME,
        PREF_ORI_SPATIAL_FILENAME,
        ORI_CENTERS_SPATIAL_FILENAME,
    }

    for fname in expected_filenames:
        path = output_dir / fname
        assert path.exists(), f"Missing file: {fname}"
        assert path in generated_paths
        assert path.stat().st_size > 0
        with Image.open(path) as img:
            img.verify()

    # No ensemble-related files should exist
    unexpected_filenames = [
        ENSEMBLE_CORRELATION_FILENAME,
        ENSEMBLE_ACTIVITY_TRACE_FILENAME,
        ENSEMBLE_ACTIVITY_TRACE_NORM_VAR_FILENAME,
        "ensemble_activity_trace_with_variance.png",
    ]
    for fname in unexpected_filenames:
        assert not (output_dir / fname).exists()


def test_plotting_single_ensemble_boundary(tmp_path: Path) -> None:
    # Cluster 1 has 1 member (< 2 size boundary)
    # Cluster 2 has 3 members (>= 2 size boundary)
    labels = np.array([1, 2, 2, 2], dtype=np.int64)
    similarity = np.eye(4)
    comm = CommunityResult(labels=labels, similarity=similarity)

    result = AnalysisResult(
        status="ok",
        selected_indices=np.arange(4),
        osi=np.array([0.5, 0.6, 0.7, 0.8]),
        pref_ori=np.array([0.0, 0.1, 0.2, 0.3]),
        responses_mean=np.ones(4) * 1.5,
        steady_state_responses=np.ones((4, 2, 5)),
        coords=np.random.default_rng(42).random((4, 2)) * 100.0,
        distance=np.random.default_rng(42).random((4, 4)) * 50.0,
        communities=comm,
        diagnostics={},
    )

    output_dir = tmp_path / "plots"
    generated_paths = generate_and_save_all_analysis_plots(result, output_dir, num_surrogates=5)

    # Spatial surrogate should be generated for cluster 2 (>= 2 members)
    # But for cluster 1 (< 2 members), it should be skipped.
    assert (output_dir / f"{SPATIAL_SURROGATE_PREFIX}2{SPATIAL_SURROGATE_NND_SUFFIX}").exists()
    assert not (output_dir / f"{SPATIAL_SURROGATE_PREFIX}1{SPATIAL_SURROGATE_NND_SUFFIX}").exists()
