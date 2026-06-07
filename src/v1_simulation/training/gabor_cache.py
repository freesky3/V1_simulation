from __future__ import annotations

import hashlib
import json
import os
import numpy as np
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from v1_simulation.stimuli.natural import L4NaturalImageProjector, NaturalImagePreprocessor
    from v1_simulation.data.natural_images import NaturalImageSample

class GaborProjectionCache:
    """Manages the serialization cache for Gabor projections of natural image crops."""

    def __init__(self, cache_dir: str | Path = "data/.gabor_cache") -> None:
        self.cache_dir = Path(cache_dir)

    def _get_cache_paths(self, config_hash: str) -> tuple[Path, Path]:
        return (
            self.cache_dir / f"{config_hash}.npy",
            self.cache_dir / f"{config_hash}.json",
        )

    def compute_hash(
        self,
        projector: L4NaturalImageProjector,
        preprocessor: NaturalImagePreprocessor,
        samples: Sequence[NaturalImageSample],
    ) -> str:
        h = hashlib.sha256()

        # 1. L4 coords
        h.update(np.ascontiguousarray(projector.x_i).tobytes())
        h.update(np.ascontiguousarray(projector.y_i).tobytes())
        # 2. Gabor parameters
        h.update(str(projector.rf_bank.cfg).encode())
        # 3. Drive config
        h.update(str(projector.drive_cfg).encode())
        # 4. Preprocessor config
        h.update(str(preprocessor.cfg).encode())

        # 5. Unique samples sorted to be order-independent
        unique_samples = sorted(
            list(set(samples)),
            key=lambda s: (
                str(s.path),
                s.crop.top if s.crop else 0,
                s.crop.left if s.crop else 0,
                s.crop.height if s.crop else 0,
                s.crop.width if s.crop else 0,
            ),
        )
        for s in unique_samples:
            crop_str = (
                f"{s.crop.top},{s.crop.left},{s.crop.height},{s.crop.width}"
                if s.crop
                else "None"
            )
            s_str = f"{s.path}:{crop_str}\n"
            h.update(s_str.encode())

        return h.hexdigest()

    def load_or_build(
        self,
        projector: L4NaturalImageProjector,
        preprocessor: NaturalImagePreprocessor,
        dataset,
        samples: Sequence[NaturalImageSample],
    ) -> dict[NaturalImageSample, np.ndarray]:
        """Loads cached projections from disk if they exist; otherwise computes and caches them."""
        # Find all unique samples to project
        unique_samples = list(set(samples))
        
        # Sort them to guarantee deterministic ordering
        unique_samples_sorted = sorted(
            unique_samples,
            key=lambda s: (
                str(s.path),
                s.crop.top if s.crop else 0,
                s.crop.left if s.crop else 0,
                s.crop.height if s.crop else 0,
                s.crop.width if s.crop else 0,
            ),
        )

        config_hash = self.compute_hash(projector, preprocessor, unique_samples_sorted)
        npy_path, json_path = self._get_cache_paths(config_hash)

        if npy_path.exists() and json_path.exists():
            try:
                rates_array = np.load(npy_path)
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                
                # Reconstruct sample mapping using the active input objects
                sample_map = {s: rates_array[idx] for idx, s in enumerate(unique_samples_sorted)}
                return sample_map
            except Exception:
                # Fallback to building if loading fails
                pass

        # Build projections. Samples are sorted by image path, so reading the
        # current image once is enough without retaining the whole epoch in RAM.
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        chunk_size = _projection_chunk_size()
        rate_chunks = []
        frame_chunk = []
        current_path = None
        current_image = None

        for sample in unique_samples_sorted:
            if sample.path != current_path:
                current_path = sample.path
                current_image = dataset.read(sample.path)
            frame = preprocessor.transform(current_image, sample)
            frame_chunk.append(frame)
            if len(frame_chunk) >= chunk_size:
                rate_chunks.append(_project_frame_chunk(projector, frame_chunk))
                frame_chunk = []

        if frame_chunk:
            rate_chunks.append(_project_frame_chunk(projector, frame_chunk))

        rates_array = np.concatenate(rate_chunks, axis=0) if rate_chunks else np.empty((0, int(projector.l4.N)))
        
        # Save to disk
        np.save(npy_path, rates_array)
        
        metadata_samples = []
        for s in unique_samples_sorted:
            crop_info = (
                {
                    "top": s.crop.top,
                    "left": s.crop.left,
                    "height": s.crop.height,
                    "width": s.crop.width,
                }
                if s.crop
                else None
            )
            metadata_samples.append({"path": str(s.path), "crop": crop_info})
            
        metadata = {
            "config_hash": config_hash,
            "samples": metadata_samples,
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        sample_map = {s: rates_array[i] for i, s in enumerate(unique_samples_sorted)}
        return sample_map


def _projection_chunk_size() -> int:
    raw = os.environ.get("V1_GABOR_CACHE_CHUNK_SIZE", "64")
    try:
        value = int(raw)
    except ValueError:
        value = 64
    return max(1, value)


def _project_frame_chunk(projector: L4NaturalImageProjector, frames: Sequence[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.empty((0, int(projector.l4.N)))

    first = np.asarray(frames[0], dtype=float)
    if first.ndim != 2:
        raise ValueError("cached natural-image frames must be two-dimensional.")

    H, W = first.shape
    matrix = projector._get_projection_matrix(H, W)
    flattened = np.empty((len(frames), H * W), dtype=float)
    flattened[0] = first.ravel()
    for index, frame in enumerate(frames[1:], start=1):
        arr = np.asarray(frame, dtype=float)
        if arr.shape != (H, W):
            raise ValueError("all cached natural-image frames in a chunk must share the same shape.")
        flattened[index] = arr.ravel()

    integrals = flattened @ matrix.T
    return (
        np.maximum(0.0, float(projector.drive_cfg.baseline_rate) + integrals)
        * float(projector.drive_cfg.visual_gain)
    )
