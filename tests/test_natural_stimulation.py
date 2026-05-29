import numpy as np
import pytest
from types import SimpleNamespace

from v1_simulation.data.natural_images import CropBox, NaturalImageSample
from v1_simulation.stimuli.natural import (
    L4NaturalImageProjector,
    NaturalImageDriveConfig,
    NaturalImageL4Drive,
    NaturalImagePreprocessConfig,
    NaturalImagePreprocessor,
)
from v1_simulation.stimuli.receptive_fields import GaborConfig, GaborRFConfig


def test_natural_image_preprocessor_resizing() -> None:
    # 1. Antialiasing resize (downsample from 4x4 to 2x2)
    cfg_aa = NaturalImagePreprocessConfig(resolution=2, antialias=True, normalization="maxscale")
    preproc_aa = NaturalImagePreprocessor(cfg_aa)
    img_large = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
    ])
    sample = NaturalImageSample(path="dummy.iml", crop=None)

    out_aa = preproc_aa.transform(img_large, sample)
    assert out_aa.shape == (2, 2)

    # 2. No antialiasing resize
    cfg_no_aa = NaturalImagePreprocessConfig(resolution=2, antialias=False, normalization="maxscale")
    preproc_no_aa = NaturalImagePreprocessor(cfg_no_aa)
    out_no_aa = preproc_no_aa.transform(img_large, sample)
    assert out_no_aa.shape == (2, 2)


def test_natural_image_preprocessor_normalization() -> None:
    # 1. Maxscale
    preproc_max = NaturalImagePreprocessor(NaturalImagePreprocessConfig(resolution=3, normalization="maxscale"))
    img = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out_max = preproc_max.transform(img, NaturalImageSample(path="dummy.iml"))
    assert np.max(out_max) == pytest.approx(1.0)
    assert out_max[0, 1] == pytest.approx(2.0 / 3.0)

    # 2. Zscore
    preproc_z = NaturalImagePreprocessor(NaturalImagePreprocessConfig(resolution=3, normalization="zscore", clip_zscore=None))
    out_z = preproc_z.transform(img, NaturalImageSample(path="dummy.iml"))
    assert np.isclose(np.mean(out_z), 0.0)
    assert np.isclose(np.std(out_z), 1.0)

    # 3. Log-Zscore
    preproc_log_z = NaturalImagePreprocessor(NaturalImagePreprocessConfig(resolution=3, normalization="log-zscore", clip_zscore=None))
    out_log_z = preproc_log_z.transform(img, NaturalImageSample(path="dummy.iml"))
    assert np.isclose(np.mean(out_log_z), 0.0)
    assert np.isclose(np.std(out_log_z), 1.0)

    # 4. Zscore clipping
    preproc_clip = NaturalImagePreprocessor(NaturalImagePreprocessConfig(resolution=3, normalization="zscore", clip_zscore=0.5))
    out_clip = preproc_clip.transform(img, NaturalImageSample(path="dummy.iml"))
    assert np.max(out_clip) <= 0.5
    assert np.min(out_clip) >= -0.5

    # 5. Flat image (zero variance -> outputs zeros)
    flat_img = np.ones((3, 3))
    out_flat = preproc_max.transform(flat_img, NaturalImageSample(path="dummy.iml"))
    # Max value of constant 1.0 is 1.0, so maxscale normalizes to ones
    assert np.allclose(out_flat, 1.0)

    out_flat_z = preproc_z.transform(flat_img, NaturalImageSample(path="dummy.iml"))
    # Standard deviation is 0.0, so zscore returns zeros to avoid div by zero
    assert np.allclose(out_flat_z, 0.0)

    # Validation errors
    with pytest.raises(ValueError, match="Unsupported natural image"):
        preproc_bad = NaturalImagePreprocessor(
            SimpleNamespace(resolution=3, normalization="bad-mode", zscore_eps=1e-8, clip_zscore=None)  # type: ignore
        )
        preproc_bad.transform(img, NaturalImageSample(path="dummy.iml"))


def test_l4_natural_image_projector() -> None:
    l4_layer = SimpleNamespace(
        coords=np.array([[-0.5, -0.5], [0.5, 0.5]]),
        tunings=["T", "U"],
        pref_dirs=[0.0, np.nan],
        N=2,
    )
    rf_cfg = GaborRFConfig(
        stimulus_size=2.0,
        resolution=3,
        gabor=GaborConfig(sigma=0.5, gamma=1.0, spatial_frequency=1.0, phase=0.0),
    )
    drive_cfg = NaturalImageDriveConfig(
        visual_gain=2.0,
        baseline_rate=1.0,
        periodic=True,
        projection_chunk_size=1,
    )

    projector = L4NaturalImageProjector(
        l4_layer=l4_layer,
        rf_cfg=rf_cfg,
        drive_cfg=drive_cfg,
    )

    frame = np.ones((3, 3), dtype=float)
    rates = projector.project(frame)
    assert rates.shape == (2,)
    assert np.all(rates >= 0.0)

    # Shape error check
    with pytest.raises(ValueError, match="two-dimensional image"):
        projector.project(np.ones(9))


def test_natural_image_l4_drive(tmp_path) -> None:
    # 1. Setup mock classes
    class MockDataset:
        def read(self, path):
            return np.ones((3, 3))

    preprocessor = NaturalImagePreprocessor(NaturalImagePreprocessConfig(resolution=3, normalization="maxscale"))

    l4_layer = SimpleNamespace(
        coords=np.array([[0.0, 0.0]]),
        tunings=["U"],
        pref_dirs=[np.nan],
        N=1,
    )
    rf_cfg = GaborRFConfig(
        stimulus_size=2.0,
        resolution=3,
        gabor=GaborConfig(sigma=0.5, gamma=1.0, spatial_frequency=1.0, phase=0.0),
    )
    drive_cfg = NaturalImageDriveConfig(visual_gain=1.0, baseline_rate=0.0)
    projector = L4NaturalImageProjector(l4_layer=l4_layer, rf_cfg=rf_cfg, drive_cfg=drive_cfg)

    # 2. Instantiate drive
    drive = NaturalImageL4Drive(
        dataset=MockDataset(),  # type: ignore
        preprocessor=preprocessor,
        projector=projector,
    )

    sample1 = NaturalImageSample(path="dummy1.iml", crop=None)
    sample2 = NaturalImageSample(path="dummy2.iml", crop=None)

    # 3. Test single rate
    rates = drive.rates_for_sample(sample1)
    assert rates.shape == (1,)

    # 4. Test static drive function mapping
    func = drive.make_static_func(sample1)
    assert np.array_equal(func(0.0), rates)
    assert not func(0.0).flags.writeable

    # 5. Test static batch drive function mapping
    batch_func = drive.make_static_batch_func([sample1, sample2])
    batch_rates = batch_func(0.0)
    # Output shape: (n_neurons, n_batch) -> (1, 2)
    assert batch_rates.shape == (1, 2)
    assert not batch_rates.flags.writeable


def test_gabor_projection_cache(tmp_path) -> None:
    from v1_simulation.training.gabor_cache import GaborProjectionCache

    # Create dummy datasets, preprocessor, projector, samples
    class MockDataset:
        def __init__(self):
            self.calls = 0
        def read(self, path):
            self.calls += 1
            return np.ones((3, 3))

    preprocessor = NaturalImagePreprocessor(
        NaturalImagePreprocessConfig(resolution=3, normalization="maxscale")
    )

    l4_layer = SimpleNamespace(
        coords=np.array([[0.0, 0.0]]),
        tunings=["U"],
        pref_dirs=[np.nan],
        N=1,
    )
    rf_cfg = GaborRFConfig(
        stimulus_size=2.0,
        resolution=3,
        gabor=GaborConfig(sigma=0.5, gamma=1.0, spatial_frequency=1.0, phase=0.0),
    )
    drive_cfg = NaturalImageDriveConfig(visual_gain=1.0, baseline_rate=0.0)
    projector = L4NaturalImageProjector(l4_layer=l4_layer, rf_cfg=rf_cfg, drive_cfg=drive_cfg)

    dataset = MockDataset()
    samples = [
        NaturalImageSample(path="dummy1.iml", crop=None),
        NaturalImageSample(path="dummy2.iml", crop=None),
    ]

    cache = GaborProjectionCache(cache_dir=tmp_path)

    # First call - cache miss, should build cache and read dataset
    res_miss = cache.load_or_build(projector, preprocessor, dataset, samples)
    assert len(res_miss) == 2
    assert dataset.calls == 2

    # Second call - cache hit, should load from file and not read dataset again
    dataset_hit = MockDataset()
    res_hit = cache.load_or_build(projector, preprocessor, dataset_hit, samples)
    assert len(res_hit) == 2
    assert dataset_hit.calls == 0

    # Verify keys and values are equal
    for s in samples:
        assert np.array_equal(res_miss[s], res_hit[s])

