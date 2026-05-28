import numpy as np
import pytest
from types import SimpleNamespace

from v1_simulation.stimuli.grating import DriftingGratingInput


def test_drifting_grating_input_generation() -> None:
    # 1. Setup mock config
    cfg = SimpleNamespace(
        receptive_field=SimpleNamespace(
            stimulus_size=2.0,
            resolution=5,
            gabor=SimpleNamespace(spatial_frequency=1.0),
        ),
        gabor=SimpleNamespace(
            sigma=0.5,
            gamma=1.0,
            spatial_frequency=1.0,
            phase=0.0,
        ),
        baseline_rate=2.0,
        visual_gain=1.5,
        temporal_frequency=2.0 * np.pi,  # 1 Hz temporal frequency
        luminance=1.0,
        contrast=0.8,
    )

    # 2. Setup mock L4 layer (2 neurons, one tuned, one untuned)
    l4_layer = SimpleNamespace(
        coords=np.array([[-0.5, 0.0], [0.5, 0.0]]),
        N=2,
    )
    l4_tunings = np.array(["T", "U"])
    l4_pref_dirs = np.array([0.0, np.nan])

    # 3. Instantiate
    grating_input = DriftingGratingInput(
        cfg,
        l4_layer,
        l4_tunings=l4_tunings,
        l4_pref_dirs=l4_pref_dirs,
    )

    # 4. Verify external drive
    # Compute drive at t=0.0, theta_stim = 0.0
    drive_0 = grating_input.external_drive(theta_stim=0.0, t=0.0)
    assert drive_0.shape == (2,)
    assert np.all(drive_0 >= 0.0)

    # Make sure cache worked (second call returns same object or identical values)
    drive_0_cached = grating_input.external_drive(theta_stim=0.0, t=0.0)
    assert np.allclose(drive_0, drive_0_cached)

    # 5. Make single drive function
    drive_func = grating_input.make_drive_func(theta_stim=0.0)
    val_t0 = drive_func(0.0)
    val_t1 = drive_func(0.5)  # 1/2 period shift
    assert val_t0.shape == (2,)
    assert val_t1.shape == (2,)

    # 6. Make batched drive function
    batched_func = grating_input.make_batched_drive_func([0.0, np.pi / 2])
    batched_val = batched_func(0.1)
    # Output shape: (n_neurons, n_thetas) -> (2, 2)
    assert batched_val.shape == (2, 2)

    # 7. Render stimulus frame
    # Output shape: (n_neurons, resolution, resolution) -> (2, 5, 5)
    frame = grating_input.stimulus_frame(theta_stim=0.0, t=0.0)
    assert frame.shape == (2, 5, 5)
    # Luminance limits check: baseline 1.0, contrast 0.8 -> max 1.8, min 0.2
    assert np.all(frame >= 0.19)
    assert np.all(frame <= 1.81)


def test_drifting_grating_generic_geometry_validation() -> None:
    cfg = SimpleNamespace(
        receptive_field=SimpleNamespace(
            stimulus_size=2.0,
            resolution=5,
            gabor=SimpleNamespace(spatial_frequency=1.0),
        ),
        gabor=SimpleNamespace(
            sigma=0.5,
            gamma=1.0,
            spatial_frequency=1.0,
            phase=0.0,
        ),
    )
    l4_layer = SimpleNamespace(
        coords=np.array([[0.0, 0.0]]),
        N=1,
    )

    # Missing tuning fields
    with pytest.raises(ValueError, match="required for generic L4 geometries"):
        DriftingGratingInput(cfg, l4_layer)

    # Shape mismatch validation
    with pytest.raises(ValueError, match="l4_tunings must have shape"):
        DriftingGratingInput(
            cfg,
            l4_layer,
            l4_tunings=np.array(["T", "U"]),
            l4_pref_dirs=np.array([0.0]),
        )
