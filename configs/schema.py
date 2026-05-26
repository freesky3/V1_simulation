from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class LayerConfig:
    # use periodic boundary conditions
    periodic: bool = True

    # L4 parameters
    l4_n_side: int = 40
    l4_region_size: float = 2.0
    l4_z_pos: float = 0.0
    all_tuned: bool = True

    # L2/3 parameters
    l23_region_size: float = 2.0
    l23_z_pos: float = 0.1
    random_i: bool = False

@dataclass
class ConnectivityConfig:
    p_ee: float = 0.12
    j: float = 3.0
    g: float = 5.5
    j_ee_scale: float = 1.0
    j_ei_scale: float = 1.08
    j_ex_scale: float = 1.0
    j_ie_scale: float = 1.0
    j_ii_scale: float = 1.0
    j_ix_scale: float = 1.0
    sigma_narrow: float = 0.5
    sigma_broad: float = 1.0
    kappa: float = 0.45

@dataclass
class StimulusConfig:
    kind: str = "drifting_grating"
    stimulus_size: float = 2.0
    sigma: float = 0.085
    gamma: float = 1.3
    k: float = 14.137166941154069
    psi: float = 0.0
    r0: float = 0.0
    res: int = 300
    l0: float = 1.0
    epsilon: float = 1.0
    omega: float = 6.283185307179586
    visual_gain: float = 400.0
    n_theta: int = 8

@dataclass
class SolverConfig:
    backend: str = "scipy"
    method: str = "RK4"
    tau_e: float = 0.02
    tau_i: float = 0.01
    tau_rp: float = 2e-3
    theta: float = 20.0
    v_r: float = 10.0
    sigma_t: float = 10.0
    mu_tab_max: float = 100.0

@dataclass
class PathsConfig:
    data_dir: Path = Path("data")
    run_dir: Path = Path("runs")
    sample_data: Path = Path("data/sample_data.pkl")
    natural_image_dir: Path = Path("data/vanhateren_iml")

@dataclass
class RootConfig:
    seed: int = 42
    layers: LayerConfig = field(default_factory=LayerConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)