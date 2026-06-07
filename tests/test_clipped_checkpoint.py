import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

from v1_simulation.network.geometry import SheetGeometry
from v1_simulation.network.state import NetworkState, PopulationLayout


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_clipped_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("make_clipped_checkpoint", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
clip_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = clip_script
_SPEC.loader.exec_module(clip_script)

make_clipped_network = clip_script.make_clipped_network


def test_make_clipped_network_only_clips_excitatory_source_bcm_blocks() -> None:
    layout = PopulationLayout(
        l23=SheetGeometry(2, 2.0, 0.1),
        l4=SheetGeometry(1, 2.0, 0.0),
        l23_types=np.array(["E", "E", "I", "I"]),
        l4_tunings=np.array(["U"]),
        l4_pref_dirs=np.array([np.nan]),
    )
    weights = np.zeros(layout.shape, dtype=float)
    weights[0, 1] = 50.0  # W_EE: clip
    weights[1, 0] = 10.0  # W_EE: keep
    weights[2, 0] = 45.0  # W_IE: clip
    weights[0, 2] = 55.0  # W_EI: keep, not a BCM excitatory-source block
    weights[0, 4] = 60.0  # W_EX: keep, feedforward input is not clipped by this control
    network = NetworkState(
        layout=layout,
        connectivity=sparse.csr_matrix(weights != 0.0),
        weights=sparse.csr_matrix(weights),
    )

    clipped, metadata = make_clipped_network(network, w_max=30.0)
    out = clipped.weights.toarray()

    assert out[0, 1] == 30.0
    assert out[1, 0] == 10.0
    assert out[2, 0] == 30.0
    assert out[0, 2] == 55.0
    assert out[0, 4] == 60.0
    assert metadata["W_EE_clipped_count"] == 1
    assert metadata["W_IE_clipped_count"] == 1
    assert metadata["W_EE_max_before"] == 50.0
    assert metadata["W_IE_max_before"] == 45.0
