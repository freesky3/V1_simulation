from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


VAN_HATEREN_SHAPE = (1024, 1536)


@dataclass(frozen=True, slots=True)
class CropBox:
    top: int
    left: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class NaturalImageSample:
    path: Path
    crop: CropBox | None = None


def read_van_hateren_iml(
    path: str | Path,
    shape: tuple[int, int] = VAN_HATEREN_SHAPE,
) -> NDArray[np.uint16]:
    path = Path(path)
    image = np.fromfile(path, dtype=">u2")

    expected_size = int(shape[0]) * int(shape[1])
    if image.size != expected_size:
        raise ValueError(f"{path} has {image.size} pixels, expected {expected_size}.")

    return image.reshape(shape)


class VanHaterenImageDataset:
    def __init__(
        self,
        image_dir: str | Path,
        *,
        shape: tuple[int, int] = VAN_HATEREN_SHAPE,
        pattern: str = "*.iml",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.shape = tuple(shape)
        self.paths = tuple(sorted(self.image_dir.glob(pattern)))

        if not self.paths:
            raise FileNotFoundError(f"No Van Hateren .iml files found in {self.image_dir}.")

    def read(self, path: str | Path) -> NDArray[np.uint16]:
        return read_van_hateren_iml(path, self.shape)

    def iter_paths(
        self,
        *,
        limit: int | None = None,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[Path, ...]:
        paths = np.array(self.paths, dtype=object)

        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            paths = paths[rng.permutation(paths.size)]

        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative.")
            paths = paths[:limit]

        return tuple(Path(path) for path in paths)


class NaturalImageSampler:
    def __init__(
        self,
        dataset: VanHaterenImageDataset,
        *,
        crop_size: int | None,
        patches_per_image: int = 1,
        seed: int | None = None,
    ) -> None:
        if patches_per_image <= 0:
            raise ValueError("patches_per_image must be positive.")

        if crop_size is not None and crop_size <= 0:
            raise ValueError("crop_size must be positive when provided.")

        self.dataset = dataset
        self.crop_size = crop_size
        self.patches_per_image = int(patches_per_image)
        self.rng = np.random.default_rng(seed)

    def make_epoch(
        self,
        *,
        limit: int | None = None,
        shuffle_paths: bool = True,
        shuffle_samples: bool = True,
    ) -> tuple[NaturalImageSample, ...]:
        paths = self.dataset.iter_paths(limit=limit, shuffle=shuffle_paths, rng=self.rng)
        samples: list[NaturalImageSample] = []

        for path in paths:
            for _ in range(self.patches_per_image):
                samples.append(NaturalImageSample(path=path, crop=self._sample_crop()))

        if shuffle_samples and samples:
            order = self.rng.permutation(len(samples))
            samples = [samples[i] for i in order]

        return tuple(samples)

    def _sample_crop(self) -> CropBox | None:
        if self.crop_size is None:
            return None

        height, width = self.dataset.shape
        if self.crop_size > height or self.crop_size > width:
            raise ValueError(f"crop_size={self.crop_size} exceeds image shape {self.dataset.shape}.")

        top = int(self.rng.integers(0, height - self.crop_size + 1))
        left = int(self.rng.integers(0, width - self.crop_size + 1))

        return CropBox(
            top=top,
            left=left,
            height=self.crop_size,
            width=self.crop_size,
        )


def apply_crop(image: NDArray[np.generic], crop: CropBox | None) -> NDArray[np.generic]:
    if crop is None:
        return image

    return image[
        crop.top : crop.top + crop.height,
        crop.left : crop.left + crop.width,
    ]