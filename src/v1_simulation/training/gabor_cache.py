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

        # Build projections
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        rates_list = []
        image_cache = {}
        
        for sample in unique_samples_sorted:
            if sample.path not in image_cache:
                image_cache[sample.path] = dataset.read(sample.path)
            frame = preprocessor.transform(image_cache[sample.path], sample)
            rates_list.append(projector.project(frame))
            
        rates_array = np.stack(rates_list)
        
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
