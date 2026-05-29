from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

from v1_simulation.data.natural_images import NaturalImageSample, apply_crop
from v1_simulation.stimuli.receptive_fields import GaborRFConfig, L4GaborBank

if TYPE_CHECKING:
    from v1_simulation.data.natural_images import VanHaterenImageDataset
    from v1_simulation.network.geometry import L4, SheetGeometry


NormalizationMode = Literal["log-zscore", "zscore", "maxscale"]


@dataclass(frozen=True, slots=True)
class NaturalImagePreprocessConfig:
    """Configuration for pre-processing natural images before model projection.

    Attributes:
        resolution: Target resolution (height and width) of the preprocessed image.
        normalization: Normalization strategy to apply ("log-zscore", "zscore", or "maxscale").
        clip_zscore: Optional value to clip extreme z-score values.
        antialias: If True, applies Gaussian filtering to prevent aliasing before downsampling.
        zscore_eps: A small constant to prevent division by zero during normalization.
    """
    resolution: int
    normalization: NormalizationMode = "log-zscore"
    clip_zscore: float | None = 3.0
    antialias: bool = True
    zscore_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.resolution <= 1:
            raise ValueError("resolution must be greater than 1.")
        if self.clip_zscore is not None and self.clip_zscore <= 0:
            raise ValueError("clip_zscore must be positive when provided.")


@dataclass(frozen=True, slots=True)
class NaturalImageDriveConfig:
    """Configuration for mapping visual stimulus coordinates to receptive fields.

    Attributes:
        visual_gain: Overall scaling factor (multiplier) for the firing rate drive.
        baseline_rate: The baseline firing rate (in Hz) added to the neuron input.
        periodic: If True, uses wrapped/periodic boundary conditions during visual sampling.
        projection_chunk_size: Number of neurons to process concurrently in vector operations.
    """
    visual_gain: float
    baseline_rate: float
    periodic: bool = True
    projection_chunk_size: int = 64

    def __post_init__(self) -> None:
        if self.projection_chunk_size <= 0:
            raise ValueError("projection_chunk_size must be positive.")


class NaturalImagePreprocessor:
    """Preprocesses raw natural images (cropping, resizing, normalizing)."""

    def __init__(self, cfg: NaturalImagePreprocessConfig) -> None:
        self.cfg = cfg

    def transform(
        self,
        image: NDArray[np.generic],
        sample: NaturalImageSample,
    ) -> NDArray[np.float64]:
        """Transforms a raw image sample into a normalized preprocessed frame.

        Args:
            image: 2D array of raw image pixel intensities.
            sample: The image sample metadata (defines the crop boundaries).

        Returns:
            A 2D array of normalized float64 pixel intensities.
        """
        image = apply_crop(np.asarray(image), sample.crop)
        image = self._resize(np.asarray(image, dtype=float))
        image = self._normalize(image)
        return image.astype(float, copy=False)

    def _resize(self, image: NDArray[np.float64]) -> NDArray[np.float64]:
        target = self.cfg.resolution

        if image.shape == (target, target):
            return image

        if self.cfg.antialias:
            image = self._antialias_before_downsample(image, target)

        factors = (target / image.shape[0], target / image.shape[1])
        return zoom(image, factors, order=1)

    def _antialias_before_downsample(
        self,
        image: NDArray[np.float64],
        target: int,
    ) -> NDArray[np.float64]:
        downsample_factor = max(image.shape[0] / target, image.shape[1] / target)

        if downsample_factor <= 1.0:
            return image

        sigma = max(0.0, (downsample_factor - 1.0) / 2.0)
        return gaussian_filter(image, sigma=sigma, mode="nearest")

    def _normalize(self, image: NDArray[np.float64]) -> NDArray[np.float64]:
        mode = self.cfg.normalization

        if mode == "log-zscore":
            return self._zscore(np.log1p(np.maximum(image, 0.0)))

        if mode == "zscore":
            return self._zscore(image)

        if mode == "maxscale":
            max_value = float(np.max(image))
            if max_value <= self.cfg.zscore_eps:
                return np.zeros_like(image, dtype=float)
            return image / max_value

        raise ValueError(f"Unsupported natural image normalization: {mode}")

    def _zscore(self, image: NDArray[np.float64]) -> NDArray[np.float64]:
        std = float(np.std(image))

        if std <= self.cfg.zscore_eps:
            return np.zeros_like(image, dtype=float)

        normalized = (image - float(np.mean(image))) / std

        if self.cfg.clip_zscore is not None:
            clip = float(self.cfg.clip_zscore)
            normalized = np.clip(normalized, -clip, clip)

        return normalized


class L4NaturalImageProjector:
    """Projects 2D visual frames onto Layer 4 neurons based on their receptive fields."""

    def __init__(
        self,
        *,
        l4_layer: SheetGeometry | L4,
        rf_cfg: GaborRFConfig,
        drive_cfg: NaturalImageDriveConfig,
        l4_tunings: ArrayLike | None = None,
        l4_pref_dirs: ArrayLike | None = None,
    ) -> None:
        self.l4 = l4_layer
        self.rf_bank = L4GaborBank(
            rf_cfg,
            l4_layer,
            l4_tunings=l4_tunings,
            l4_pref_dirs=l4_pref_dirs,
        )
        self.drive_cfg = drive_cfg
        self._M = None

        coords = np.asarray(l4_layer.coords, dtype=float)
        self.x_i = coords[:, 0]
        self.y_i = coords[:, 1]

        grid = self.rf_bank.grid
        self.image_x_min = float(np.min(self.x_i) + grid.x_axis[0])
        self.image_x_max = float(np.max(self.x_i) + grid.x_axis[-1])
        self.image_y_min = float(np.min(self.y_i) + grid.y_axis[0])
        self.image_y_max = float(np.max(self.y_i) + grid.y_axis[-1])

    def _get_projection_matrix(self, H: int, W: int) -> NDArray[np.float64]:
        if self._M is not None and self._M.shape == (int(self.l4.N), H * W):
            return self._M

        n_l4 = int(self.l4.N)
        M = np.zeros((n_l4, H * W), dtype=float)
        grid = self.rf_bank.grid
        filters = self.rf_bank.filters
        chunk_size = self.drive_cfg.projection_chunk_size
        periodic = self.drive_cfg.periodic

        dx_dy = grid.dx * grid.dy

        for start in range(0, n_l4, chunk_size):
            stop = min(start + chunk_size, n_l4)
            chunk_len = stop - start

            x = self.x_i[start:stop, np.newaxis, np.newaxis] + grid.x[np.newaxis, :, :]
            y = self.y_i[start:stop, np.newaxis, np.newaxis] + grid.y[np.newaxis, :, :]

            cols = self._coord_to_pixel(x, self.image_x_min, self.image_x_max, W)
            rows = self._coord_to_pixel(y, self.image_y_min, self.image_y_max, H)

            r0 = np.floor(rows).astype(np.int32)
            c0 = np.floor(cols).astype(np.int32)
            dr = rows - r0
            dc = cols - c0
            r1 = r0 + 1
            c1 = c0 + 1

            if periodic:
                r0 = r0 % H
                r1 = r1 % H
                c0 = c0 % W
                c1 = c1 % W
            else:
                r0 = np.clip(r0, 0, H - 1)
                r1 = np.clip(r1, 0, H - 1)
                c0 = np.clip(c0, 0, W - 1)
                c1 = np.clip(c1, 0, W - 1)

            w00 = (1.0 - dr) * (1.0 - dc)
            w01 = (1.0 - dr) * dc
            w10 = dr * (1.0 - dc)
            w11 = dr * dc

            K = filters[start:stop] * dx_dy

            coeff00 = K * w00
            coeff01 = K * w01
            coeff10 = K * w10
            coeff11 = K * w11

            idx00 = r0 * W + c0
            idx01 = r0 * W + c1
            idx10 = r1 * W + c0
            idx11 = r1 * W + c1

            for j in range(chunk_len):
                row_idx = start + j
                np.add.at(M[row_idx], idx00[j].ravel(), coeff00[j].ravel())
                np.add.at(M[row_idx], idx01[j].ravel(), coeff01[j].ravel())
                np.add.at(M[row_idx], idx10[j].ravel(), coeff10[j].ravel())
                np.add.at(M[row_idx], idx11[j].ravel(), coeff11[j].ravel())

        self._M = M
        return self._M

    def project(self, frame: NDArray[np.float64]) -> NDArray[np.float64]:
        """Projects a preprocessed 2D visual frame to calculate L4 neuron input rates.

        Computes the spatial overlap (dot product) between each neuron's receptive
        field filter and the image frame sampled at that neuron's visual position.

        Args:
            frame: A preprocessed 2D visual frame of shape (resolution, resolution).

        Returns:
            A 1D array of size (n_neurons,) representing the visual firing rate drive.
        """
        frame = np.asarray(frame, dtype=float)

        if frame.ndim != 2:
            raise ValueError("frame must be a two-dimensional image.")

        H, W = frame.shape
        M = self._get_projection_matrix(H, W)
        integral = M @ frame.ravel()

        rates = (
            np.maximum(0.0, self.drive_cfg.baseline_rate + integral)
            * self.drive_cfg.visual_gain
        )

        return rates

    def _sample_frame_at_rf_positions(
        self,
        frame: NDArray[np.float64],
        start: int,
        stop: int,
    ) -> NDArray[np.float64]:
        """Samples pixel values from the frame centered at receptive field coordinates."""
        grid = self.rf_bank.grid

        x = self.x_i[start:stop, np.newaxis, np.newaxis] + grid.x[np.newaxis, :, :]
        y = self.y_i[start:stop, np.newaxis, np.newaxis] + grid.y[np.newaxis, :, :]

        cols = self._coord_to_pixel(x, self.image_x_min, self.image_x_max, frame.shape[1])
        rows = self._coord_to_pixel(y, self.image_y_min, self.image_y_max, frame.shape[0])

        sampled = map_coordinates(
            frame,
            [rows.ravel(), cols.ravel()],
            order=1,
            mode="wrap" if self.drive_cfg.periodic else "nearest",
        )

        return sampled.reshape(stop - start, grid.x.shape[0], grid.x.shape[1])

    @staticmethod
    def _coord_to_pixel(
        coord: NDArray[np.float64],
        coord_min: float,
        coord_max: float,
        n_pixels: int,
    ) -> NDArray[np.float64]:
        """Maps physical visual coordinates to fractional pixel indices."""
        if coord_max <= coord_min:
            return np.zeros_like(coord, dtype=float)

        return (coord - coord_min) * (n_pixels - 1) / (coord_max - coord_min)


class NaturalImageL4Drive:
    """Manages the dataset, preprocessing, and projection pipeline for L4 natural image drive."""

    def __init__(
        self,
        *,
        dataset: VanHaterenImageDataset,
        preprocessor: NaturalImagePreprocessor,
        projector: L4NaturalImageProjector,
    ) -> None:
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.projector = projector
        self._cached_rates = {}

    def preload_cache(self, samples: Sequence[NaturalImageSample], cache_dir: str | Path = "data/.gabor_cache") -> None:
        from v1_simulation.training.gabor_cache import GaborProjectionCache
        cache_manager = GaborProjectionCache(cache_dir)
        self._cached_rates.update(
            cache_manager.load_or_build(
                self.projector,
                self.preprocessor,
                self.dataset,
                samples,
            )
        )

    def rates_for_sample(self, sample: NaturalImageSample) -> NDArray[np.float64]:
        """Computes the visual input rates for a single natural image sample."""
        if sample in self._cached_rates:
            return self._cached_rates[sample]
        image = self.dataset.read(sample.path)
        frame = self.preprocessor.transform(image, sample)
        return self.projector.project(frame)

    def make_static_func(self, sample: NaturalImageSample) -> Callable[[float], NDArray[np.float64]]:
        """Returns a time-invariant visual drive function for a single natural image crop."""
        rates = self.rates_for_sample(sample)
        rates.setflags(write=False)

        def aX_func(_t: float) -> NDArray[np.float64]:
            return rates

        aX_func.is_time_dependent = False
        return aX_func

    def make_static_batch_func(
        self,
        samples: tuple[NaturalImageSample, ...] | list[NaturalImageSample],
    ) -> Callable[[float], NDArray[np.float64]]:
        """Returns a time-invariant visual drive function for a batch of natural image crops."""
        rates = []
        for sample in samples:
            rates.append(self.rates_for_sample(sample))

        rate_matrix = np.column_stack(rates)
        rate_matrix.setflags(write=False)

        def aX_array_func(_t: float) -> NDArray[np.float64]:
            return rate_matrix

        aX_array_func.is_time_dependent = False
        return aX_array_func
