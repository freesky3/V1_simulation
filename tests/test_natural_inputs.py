import numpy as np
import pytest
from types import SimpleNamespace

from v1_simulation.training.natural_inputs import (
    _resolve_periodic,
    _resolve_stimulus_config,
    build_natural_image_l4_drive,
)


def test_resolve_stimulus_config() -> None:
    # 1. Direct stimulus config
    stim_cfg = SimpleNamespace(kind="grating")
    assert _resolve_stimulus_config(stim_cfg, None) is stim_cfg

    # 2. Fallback to model_cfg attributes
    model_cfg_attr = SimpleNamespace(stimulus=SimpleNamespace(kind="grating-fallback"))
    assert _resolve_stimulus_config(None, model_cfg_attr).kind == "grating-fallback"

    # 3. Fallback to model_cfg itself
    model_cfg_self = SimpleNamespace(kind="grating-self")
    assert _resolve_stimulus_config(None, model_cfg_self).kind == "grating-self"

    # 4. Raises error when none provided
    with pytest.raises(ValueError, match="stimulus_cfg or model_cfg must be provided"):
        _resolve_stimulus_config(None, None)


def test_resolve_periodic() -> None:
    # 1. Direct layers config
    layers_cfg = SimpleNamespace(periodic=True)
    assert _resolve_periodic(layers_cfg, None) is True

    # 2. Fallback to model_cfg.model.layers.periodic
    model_cfg_nested = SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(periodic=False)))
    assert _resolve_periodic(None, model_cfg_nested) is False

    # 3. Fallback to model_cfg.layers.periodic
    model_cfg_flat = SimpleNamespace(layers=SimpleNamespace(periodic=True))
    assert _resolve_periodic(None, model_cfg_flat) is True

    # 4. Fallback to model_cfg.periodic
    model_cfg_direct = SimpleNamespace(periodic=False)
    assert _resolve_periodic(None, model_cfg_direct) is False

    # 5. Default is True
    assert _resolve_periodic(None, None) is True


def test_build_natural_image_l4_drive_pipeline(tmp_path) -> None:
    # 1. Directory is None throws error
    cfg_bad = SimpleNamespace(dir=None)
    with pytest.raises(ValueError, match="training.natural_image.dir must be set"):
        build_natural_image_l4_drive(cfg=cfg_bad, l4_layer=None)  # type: ignore

    # 2. Setup mock directory with a mock image
    image_dir = tmp_path / "vanhateren"
    image_dir.mkdir()
    # Write a small 1024x1536 image = 1572864 pixels * 2 bytes = 3145728 bytes
    (image_dir / "im_test.iml").write_bytes(b"\x00" * 3145728)

    cfg = SimpleNamespace(
        dir=str(image_dir),
        limit=None,
        seed=42,
        crop_size=256,
        patches_per_image=1,
        res=64,
        normalization="log-zscore",
        clip_zscore=3.0,
        frame_scale=0.5,
        frame_offset=0.25,
        projection_chunk_size=32,
    )
    stimulus_cfg = SimpleNamespace(
        stimulus_size=2.0,
        visual_gain=2.0,
        baseline_rate=1.0,
        gabor=SimpleNamespace(sigma=0.085, gamma=1.0, spatial_frequency=14.0, phase=0.0),
    )
    layers_cfg = SimpleNamespace(periodic=False)
    l4_layer = SimpleNamespace(
        coords=np.array([[0.0, 0.0]]),
        tunings=["U"],
        pref_dirs=[np.nan],
        N=1,
    )

    drive, sampler = build_natural_image_l4_drive(
        cfg=cfg,  # type: ignore
        stimulus_cfg=stimulus_cfg,  # type: ignore
        layers_cfg=layers_cfg,  # type: ignore
        l4_layer=l4_layer,  # type: ignore
    )

    assert sampler.crop_size == 256
    assert drive.preprocessor.cfg.resolution == 64
    assert drive.preprocessor.cfg.frame_scale == 0.5
    assert drive.preprocessor.cfg.frame_offset == 0.25
    assert drive.projector.drive_cfg.periodic is False
    assert drive.projector.drive_cfg.projection_chunk_size == 32
