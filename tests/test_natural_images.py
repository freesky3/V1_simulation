import numpy as np
import pytest

from v1_simulation.data.natural_images import (
    CropBox,
    NaturalImageSample,
    NaturalImageSampler,
    VanHaterenImageDataset,
    apply_crop,
    read_van_hateren_iml,
)


def test_read_van_hateren_iml(tmp_path) -> None:
    shape = (2, 3)
    file_path = tmp_path / "test_im.iml"
    # Write 6 big-endian uint16 values: [0, 1, 2, 3, 4, 5]
    data = np.array([0, 1, 2, 3, 4, 5], dtype=">u2")
    data.tofile(file_path)

    img = read_van_hateren_iml(file_path, shape=shape)
    assert img.shape == (2, 3)
    assert np.array_equal(img, np.arange(6).reshape(2, 3))

    # Incorrect size validation
    bad_file = tmp_path / "bad.iml"
    np.array([1, 2], dtype=">u2").tofile(bad_file)
    with pytest.raises(ValueError, match="pixels, expected"):
        read_van_hateren_iml(bad_file, shape=shape)


def test_van_hateren_dataset_scanning(tmp_path) -> None:
    # Empty directory
    with pytest.raises(FileNotFoundError, match="No Van Hateren .iml files found"):
        VanHaterenImageDataset(tmp_path, shape=(2, 3))

    # Write files
    (tmp_path / "im1.iml").write_bytes(b"\x00" * 12)
    (tmp_path / "im2.iml").write_bytes(b"\x00" * 12)

    dataset = VanHaterenImageDataset(tmp_path, shape=(2, 3))
    assert len(dataset.paths) == 2

    # iter_paths check
    paths = dataset.iter_paths(shuffle=False)
    assert len(paths) == 2
    assert paths[0].name == "im1.iml"

    # limit and shuffle reproducible
    paths_limit = dataset.iter_paths(limit=1, shuffle=True, rng=np.random.default_rng(42))
    assert len(paths_limit) == 1

    with pytest.raises(ValueError, match="limit must be non-negative"):
        dataset.iter_paths(limit=-1)


def test_van_hateren_dataset_real() -> None:
    from pathlib import Path
    from v1_simulation.data.natural_images import VAN_HATEREN_SHAPE
    real_path = Path("data/vanhateren_iml")
    if not real_path.exists() or not list(real_path.glob("*.iml")):
        pytest.skip("Real Van Hateren dataset not found at data/vanhateren_iml")

    dataset = VanHaterenImageDataset(real_path)
    assert len(dataset.paths) > 0
    img = dataset.read(dataset.paths[0])
    assert img.shape == VAN_HATEREN_SHAPE
    assert np.issubdtype(img.dtype, np.uint16)
    assert np.any(img > 0)  # Real image should not be completely empty/all-zero


def test_natural_image_sampler(tmp_path) -> None:
    (tmp_path / "im1.iml").write_bytes(b"\x00" * 24)
    (tmp_path / "im2.iml").write_bytes(b"\x00" * 24)

    # dataset size: 3x4 pixels (total 12 elements * 2 bytes = 24 bytes)
    dataset = VanHaterenImageDataset(tmp_path, shape=(3, 4))

    # Setup validation errors
    with pytest.raises(ValueError, match="patches_per_image must be positive"):
        NaturalImageSampler(dataset, crop_size=2, patches_per_image=0)
    with pytest.raises(ValueError, match="crop_size must be positive"):
        NaturalImageSampler(dataset, crop_size=-1)

    sampler = NaturalImageSampler(dataset, crop_size=2, patches_per_image=2, seed=123)
    epoch = sampler.make_epoch(limit=1)

    # 1 image * 2 patches = 2 samples
    assert len(epoch) == 2
    for sample in epoch:
        assert isinstance(sample, NaturalImageSample)
        assert sample.crop is not None
        assert sample.crop.height == 2
        assert sample.crop.width == 2
        assert 0 <= sample.crop.top <= 1
        assert 0 <= sample.crop.left <= 2

    # Crop size too large check
    bad_sampler = NaturalImageSampler(dataset, crop_size=5)
    with pytest.raises(ValueError, match="exceeds image shape"):
        bad_sampler.make_epoch()


def test_apply_crop() -> None:
    image = np.arange(12).reshape(3, 4)
    # None crop returns identical image
    assert np.array_equal(apply_crop(image, None), image)

    # Valid crop
    crop = CropBox(top=1, left=1, height=2, width=2)
    cropped = apply_crop(image, crop)
    expected = np.array([
        [5, 6],
        [9, 10],
    ])
    assert np.array_equal(cropped, expected)
