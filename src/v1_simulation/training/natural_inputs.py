from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from v1_simulation.data.natural_images import NaturalImageSampler, VanHaterenImageDataset
from v1_simulation.stimuli.natural import (
    L4NaturalImageProjector,
    NaturalImageDriveConfig,
    NaturalImageL4Drive,
    NaturalImagePreprocessConfig,
    NaturalImagePreprocessor,
)
from v1_simulation.stimuli.receptive_fields import GaborConfig, GaborRFConfig

if TYPE_CHECKING:
    from v1_simulation.config.schema import LayersConfig, ModelConfig, StimulusConfig, TrainingNaturalImageConfig
    from v1_simulation.network.geometry import L4, SheetGeometry


def build_natural_image_l4_drive(
    *,
    cfg: TrainingNaturalImageConfig,
    stimulus_cfg: StimulusConfig | None = None,
    model_cfg: ModelConfig | StimulusConfig | None = None,
    layers_cfg: LayersConfig | None = None,
    l4_layer: SheetGeometry | L4,
    l4_tunings: ArrayLike | None = None,
    l4_pref_dirs: ArrayLike | None = None,
) -> tuple[NaturalImageL4Drive, NaturalImageSampler]:
    """Builds the natural image L4 visual drive and sampler for model training.

    Args:
        cfg: Configuration parameters for natural image training.
        stimulus_cfg: Optional stimulus configuration override.
        model_cfg: Optional model or stimulus configuration. Used to resolve the
            Gabor receptive field parameters and visual gain if `stimulus_cfg`
            is not provided.
        layers_cfg: Optional layers configuration. Used to resolve boundary periodicity.
        l4_layer: The Layer 4 geometry and tuning properties.

    Returns:
        A tuple of (drive, sampler):
            - drive: The NaturalImageL4Drive instance used to project images.
            - sampler: The NaturalImageSampler for drawing image patches/crops.

    Raises:
        ValueError: If `cfg.dir` is not set, or if stimulus Gabor parameters
            cannot be resolved from the provided configurations.
    """
    image_dir = cfg.dir
    if image_dir is None:
        raise ValueError("training.natural_image.dir must be set.")

    stimulus = _resolve_stimulus_config(stimulus_cfg, model_cfg)
    periodic = _resolve_periodic(layers_cfg, model_cfg)

    dataset = VanHaterenImageDataset(image_dir)

    sampler = NaturalImageSampler(
        dataset,
        crop_size=cfg.crop_size,
        patches_per_image=cfg.patches_per_image,
        seed=cfg.seed,
    )

    preprocessor = NaturalImagePreprocessor(
        NaturalImagePreprocessConfig(
            resolution=cfg.res,
            normalization=cfg.normalization,
            clip_zscore=cfg.clip_zscore,
        )
    )

    projector = L4NaturalImageProjector(
        l4_layer=l4_layer,
        rf_cfg=GaborRFConfig(
            stimulus_size=stimulus.stimulus_size,
            resolution=cfg.res,
            gabor=GaborConfig(
                sigma=stimulus.gabor.sigma,
                gamma=stimulus.gabor.gamma,
                spatial_frequency=stimulus.gabor.spatial_frequency,
                phase=stimulus.gabor.phase,
            ),
        ),
        drive_cfg=NaturalImageDriveConfig(
            visual_gain=stimulus.visual_gain,
            baseline_rate=stimulus.baseline_rate,
            periodic=periodic,
            projection_chunk_size=cfg.projection_chunk_size,
        ),
        l4_tunings=l4_tunings,
        l4_pref_dirs=l4_pref_dirs,
    )

    drive = NaturalImageL4Drive(
        dataset=dataset,
        preprocessor=preprocessor,
        projector=projector,
    )

    return drive, sampler


def _resolve_stimulus_config(
    stimulus_cfg: StimulusConfig | None,
    model_cfg: ModelConfig | StimulusConfig | None,
) -> StimulusConfig:
    """Resolves stimulus configuration, fallback to model config if necessary."""
    if stimulus_cfg is not None:
        return stimulus_cfg
    if model_cfg is None:
        raise ValueError("stimulus_cfg or model_cfg must be provided.")
    return getattr(model_cfg, "stimulus", model_cfg)


def _resolve_periodic(
    layers_cfg: LayersConfig | None,
    model_cfg: ModelConfig | StimulusConfig | None,
) -> bool:
    """Resolves layer periodicity (boundary conditions) from config hierarchies."""
    if layers_cfg is not None:
        return bool(layers_cfg.periodic)
    if model_cfg is not None:
        root_model = getattr(model_cfg, "model", None)
        if root_model is not None and hasattr(root_model, "layers"):
            return bool(root_model.layers.periodic)
        if hasattr(model_cfg, "layers"):
            return bool(model_cfg.layers.periodic)
        if hasattr(model_cfg, "periodic"):
            return bool(model_cfg.periodic)
    return True
