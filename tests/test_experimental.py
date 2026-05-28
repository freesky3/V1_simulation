import numpy as np
import pytest
from types import SimpleNamespace

from v1_simulation.data.experimental import ExperimentalData


def test_experimental_data_loading_and_derivation(tmp_path) -> None:
    # 1. Create a temporary npz file representing experimental data
    npz_path = tmp_path / "mock_experimental.npz"
    np.savez(
        npz_path,
        eta_I=0.2,       # I/E ratio
        eta_X=0.5,       # L4/L2_3 E ratio
        etaT_X=0.6,      # Tuned L4 ratio
    )

    # 2. Setup mock config
    # n_side = 8 -> N_X = 64
    # exact_N_E = 64 / 0.5 = 128
    # exact_N_I = 128 * 0.2 = 25.6
    # exact_N_E + exact_N_I = 153.6 -> sqrt(153.6) ~ 12.39 -> l2_3_n_side = 13
    # n_total = 169
    # N_E = int(169 / 1.2) = 140
    # N_I = 169 - 140 = 29
    # NT_X = int(0.6 * 64) = 38
    cfg_model = SimpleNamespace(
        layers=SimpleNamespace(l4=SimpleNamespace(n_side=8)),
        connectivity=SimpleNamespace(p_ee=0.25),
    )

    exp_data = ExperimentalData(cfg_model, npz_path)

    assert exp_data.eta_I == 0.2
    assert exp_data.eta_X == 0.5
    assert exp_data.N_X == 64
    assert exp_data.l2_3_n_side == 13
    assert exp_data.N_E == 140
    assert exp_data.N_I == 29
    assert exp_data.p_EE == 0.25
    assert exp_data.NT_X == 38
    assert exp_data.pT_X == pytest.approx(38 / 64)


def test_experimental_data_file_not_found() -> None:
    cfg_model = SimpleNamespace(
        layers=SimpleNamespace(l4=SimpleNamespace(n_side=8)),
        connectivity=SimpleNamespace(p_ee=0.25),
    )
    with pytest.raises(FileNotFoundError, match="Experimental data file not found"):
        ExperimentalData(cfg_model, "non_existent_file_path.npz")
