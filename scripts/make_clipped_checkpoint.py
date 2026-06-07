#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from scipy import sparse

from v1_simulation.network.state import NetworkState, load_trained_network_state
from v1_simulation.training.checkpoints import save_checkpoint


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    source_path = run_dir if (run_dir / "layout.npz").exists() else run_dir / args.source_name
    w_max = float(args.w_max) if args.w_max is not None else _load_w_max(run_dir)

    source = load_trained_network_state(source_path).network
    clipped, metadata = make_clipped_network(source, w_max=w_max)
    metadata.update(
        {
            "mode": "clipped_only",
            "source_checkpoint": str(source_path),
            "output_name": str(args.output_name),
        }
    )

    output_root = source_path.parent if (run_dir / "layout.npz").exists() else run_dir
    output_path = save_checkpoint(output_root, args.output_name, clipped, metadata=metadata)
    print(f"Saved clipped-only checkpoint: {output_path}")
    print(
        "Clipped weights: "
        f"W_EE={metadata['W_EE_clipped_count']}/{metadata['W_EE_total']}, "
        f"W_IE={metadata['W_IE_clipped_count']}/{metadata['W_IE_total']}, "
        f"w_max={w_max}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a checkpoint with only excitatory-source BCM weights clipped to w_max.",
    )
    parser.add_argument("run_dir", type=Path, help="Training run directory or checkpoint directory.")
    parser.add_argument("--source-name", default="network_initial")
    parser.add_argument("--output-name", default="network_clipped_only")
    parser.add_argument("--w-max", type=float, default=None)
    args = parser.parse_args(argv)
    if args.w_max is not None and args.w_max <= 0.0:
        raise ValueError("--w-max must be positive.")
    return args


def make_clipped_network(network: NetworkState, *, w_max: float) -> tuple[NetworkState, dict[str, Any]]:
    if w_max <= 0.0 or not np.isfinite(float(w_max)):
        raise ValueError("w_max must be positive and finite.")

    weights = network.weights.toarray() if sparse.issparse(network.weights) else np.asarray(network.weights, dtype=float).copy()
    topology = network.connectivity.toarray().astype(bool)
    if weights.ndim != 2:
        raise ValueError("network weights must be a 2D matrix.")

    metadata: dict[str, Any] = {"w_max": float(w_max)}
    for name, targets, sources in (
        ("W_EE", network.idx_E, network.idx_E),
        ("W_IE", network.idx_I, network.idx_E),
    ):
        block = weights[np.ix_(targets, sources)].copy()
        mask = topology[np.ix_(targets, sources)]
        values_before = block[mask]
        max_before = float(np.max(values_before)) if values_before.size else float("nan")
        clipped_mask = mask & (block > float(w_max))
        clipped_count = int(np.sum(clipped_mask))
        if clipped_count:
            block[clipped_mask] = float(w_max)
            weights[np.ix_(targets, sources)] = block
        metadata[f"{name}_clipped_count"] = clipped_count
        metadata[f"{name}_total"] = int(np.sum(mask))
        metadata[f"{name}_max_before"] = max_before

    metadata["W_EE_max_after"] = _block_max(weights, topology, network.idx_E, network.idx_E)
    metadata["W_IE_max_after"] = _block_max(weights, topology, network.idx_I, network.idx_E)
    clipped_network = NetworkState(
        layout=network.layout,
        connectivity=network.connectivity,
        weights=sparse.csr_matrix(weights),
        source={**dict(network.source), "mode": "clipped_only", "w_max": float(w_max)},
    )
    return clipped_network, metadata


def _block_max(weights: np.ndarray, topology: np.ndarray, targets: np.ndarray, sources: np.ndarray) -> float:
    if len(targets) == 0 or len(sources) == 0:
        return float("nan")
    block = weights[np.ix_(targets, sources)]
    values = block[topology[np.ix_(targets, sources)]]
    return float(np.max(values)) if values.size else float("nan")


def _load_w_max(run_dir: Path) -> float:
    candidates = [run_dir / "run_config.json", run_dir.parent / "run_config.json"]
    for path in candidates:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        config = payload.get("config", payload)
        value = (
            config.get("training", {})
            .get("bcm", {})
            .get("w_max")
        )
        if value is not None:
            return float(value)
    return 30.0


if __name__ == "__main__":
    main()
