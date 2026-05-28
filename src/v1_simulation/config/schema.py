from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# ==========================================
# 1. Analysis Configuration
# ==========================================

@dataclass
class LouvainConfig:
    thr_prop: float = 0.2
    gamma: float = 1.0
    num_runs: int = 1000
    consensus_reps: int = 200

@dataclass
class AnalysisConfig:
    seed: Optional[int] = None
    num_surrogates: int = 10000
    center_side_fraction: float = 0.5
    osi_threshold: float = 0.4
    random_sample_fraction: float = 0.5
    louvain: LouvainConfig = field(default_factory=LouvainConfig)
    active_threshold: float = 1.0e-6
    simulation_id: Optional[str] = None

# ==========================================
# 2. Background Configuration
# ==========================================

@dataclass
class BackgroundConfig:
    enabled: bool = False
    seed: Optional[int] = None
    interpolation: str = "linear"
    tau_e: float = 0.05
    tau_i: float = 0.05
    mu_e: float = 0.0
    mu_i: float = 0.0
    sigma_e: float = 0.0
    sigma_i: float = 0.0

# ==========================================
# 3. Stimulus Configuration
# ==========================================

@dataclass
class GaborConfig:
    """Configuration for Gabor receptive fields and visual stimulus spatial features.

    The gamma parameter is applied linearly to y_prime**2 in RF kernels.
    """
    sigma: float = 0.085
    gamma: float = 1.0
    spatial_frequency: float = 14.137166941154069
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("stimulus.gabor.sigma must be positive.")
        if self.gamma <= 0:
            raise ValueError("stimulus.gabor.gamma must be positive.")


@dataclass
class StimulusConfig:
    kind: str = "drifting_grating"
    stimulus_size: float = 2.0
    baseline_rate: float = 0.0
    resolution: int = 300
    luminance: float = 1.0
    contrast: float = 1.0
    temporal_frequency: float = 6.283185307179586
    visual_gain: float = 400.0
    n_theta: int = 8
    gabor: GaborConfig = field(default_factory=GaborConfig)

    # Alias to support self.cfg.receptive_field references in stimuli
    @property
    def receptive_field(self) -> "StimulusConfig":
        """Alias to support self.cfg.receptive_field legacy references."""
        return self

    # Backward compatibility properties (old flat field names)
    @property
    def size(self) -> float:
        return self.stimulus_size

    @size.setter
    def size(self, val: float):
        self.stimulus_size = val

    @property
    def r0(self) -> float:
        return self.baseline_rate

    @r0.setter
    def r0(self, val: float):
        self.baseline_rate = val

    @property
    def res(self) -> int:
        return self.resolution

    @res.setter
    def res(self, val: int):
        self.resolution = val

    @property
    def l0(self) -> float:
        return self.luminance

    @l0.setter
    def l0(self, val: float):
        self.luminance = val

    @property
    def epsilon(self) -> float:
        return self.contrast

    @epsilon.setter
    def epsilon(self, val: float):
        self.contrast = val

    @property
    def omega(self) -> float:
        return self.temporal_frequency

    @omega.setter
    def omega(self, val: float):
        self.temporal_frequency = val

    # Properties delegating to the nested gabor config for backward compatibility
    @property
    def sigma(self) -> float:
        return self.gabor.sigma

    @sigma.setter
    def sigma(self, val: float):
        self.gabor.sigma = val

    @property
    def gamma(self) -> float:
        return self.gabor.gamma

    @gamma.setter
    def gamma(self, val: float):
        self.gabor.gamma = val

    @property
    def spatial_frequency(self) -> float:
        return self.gabor.spatial_frequency

    @spatial_frequency.setter
    def spatial_frequency(self, val: float):
        self.gabor.spatial_frequency = val

    @property
    def phase(self) -> float:
        return self.gabor.phase

    @phase.setter
    def phase(self, val: float):
        self.gabor.phase = val

    @property
    def k(self) -> float:
        return self.gabor.spatial_frequency

    @k.setter
    def k(self, val: float):
        self.gabor.spatial_frequency = val

    @property
    def psi(self) -> float:
        return self.gabor.phase

    @psi.setter
    def psi(self, val: float):
        self.gabor.phase = val

# ==========================================
# 4. Model Configuration
# ==========================================

@dataclass
class L4Config:
    n_side: int = 40
    region_size: float = 2.0
    z_pos: float = 0.0
    all_tuned: bool = True
    N_theta: int = 8

    @property
    def l4(self) -> "L4Config":
        """Self-reference to support self.cfg.l4.all_tuned syntax."""
        return self

@dataclass
class L23Config:
    n_side: Optional[int] = None
    region_size: float = 2.0
    z_pos: float = 0.1
    inhibitory_fraction: Optional[float] = None
    random_inhibitory: bool = False

    @property
    def random_I(self) -> bool:
        return self.random_inhibitory

    @random_I.setter
    def random_I(self, val: bool):
        self.random_inhibitory = val

@dataclass
class LayersConfig:
    periodic: bool = True
    l4: L4Config = field(default_factory=L4Config)
    l23: L23Config = field(default_factory=L23Config)
    N_theta: int = 8

    # Backward compatibility properties
    @property
    def l23_region_size(self) -> float:
        return self.l23.region_size

    @l23_region_size.setter
    def l23_region_size(self, val: float):
        self.l23.region_size = val

    @property
    def l23_z_pos(self) -> float:
        return self.l23.z_pos

    @l23_z_pos.setter
    def l23_z_pos(self, val: float):
        self.l23.z_pos = val

    @property
    def random_I(self) -> bool:
        return self.l23.random_inhibitory

    @random_I.setter
    def random_I(self, val: bool):
        self.l23.random_inhibitory = val

@dataclass
class ConnectivityScalesConfig:
    ee: float = 1.0
    ei: float = 1.08
    ex: float = 1.0
    ie: float = 1.0
    ii: float = 1.0
    ix: float = 1.0

@dataclass
class ConnectivityKernelConfig:
    sigma_narrow: float = 0.075
    sigma_broad: float = 0.225
    kappa: float = 0.45

@dataclass
class ConnectivityConfig:
    p_ee: float = 0.12
    j: float = 3.0
    g: float = 5.5
    equalize_indegree: bool = True
    scales: ConnectivityScalesConfig = field(default_factory=ConnectivityScalesConfig)
    kernel: ConnectivityKernelConfig = field(default_factory=ConnectivityKernelConfig)

    # Backward compatibility properties for old flat fields
    @property
    def j_ee_scale(self) -> float:
        return self.scales.ee

    @j_ee_scale.setter
    def j_ee_scale(self, val: float):
        self.scales.ee = val

    @property
    def j_ei_scale(self) -> float:
        return self.scales.ei

    @j_ei_scale.setter
    def j_ei_scale(self, val: float):
        self.scales.ei = val

    @property
    def j_ex_scale(self) -> float:
        return self.scales.ex

    @j_ex_scale.setter
    def j_ex_scale(self, val: float):
        self.scales.ex = val

    @property
    def j_ie_scale(self) -> float:
        return self.scales.ie

    @j_ie_scale.setter
    def j_ie_scale(self, val: float):
        self.scales.ie = val

    @property
    def j_ii_scale(self) -> float:
        return self.scales.ii

    @j_ii_scale.setter
    def j_ii_scale(self, val: float):
        self.scales.ii = val

    @property
    def j_ix_scale(self) -> float:
        return self.scales.ix

    @j_ix_scale.setter
    def j_ix_scale(self, val: float):
        self.scales.ix = val

    @property
    def sigma_narrow(self) -> float:
        return self.kernel.sigma_narrow

    @sigma_narrow.setter
    def sigma_narrow(self, val: float):
        self.kernel.sigma_narrow = val

    @property
    def sigma_broad(self) -> float:
        return self.kernel.sigma_broad

    @sigma_broad.setter
    def sigma_broad(self, val: float):
        self.kernel.sigma_broad = val

    @property
    def kappa(self) -> float:
        return self.kernel.kappa

    @kappa.setter
    def kappa(self, val: float):
        self.kernel.kappa = val

@dataclass
class ModelConfig:
    layers: LayersConfig = field(default_factory=LayersConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    trained_network_path: Optional[str] = None
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)

# ==========================================
# 5. Paths Configuration
# ==========================================

@dataclass
class PathsConfig:
    data_dir: Path = Path("data")
    sample_data_path: Path = Path("data/sample_data.pkl")
    natural_image_dir: Path = Path("data/vanhateren_iml")
    run_root: Path = Path("runs")

    @property
    def run_dir(self) -> Path:
        """Alias for run_root for backward compatibility."""
        return self.run_root

    @run_dir.setter
    def run_dir(self, val: Path):
        self.run_root = val

    @property
    def sample_data(self) -> Path:
        """Alias for sample_data_path for backward compatibility."""
        return self.sample_data_path

    @sample_data.setter
    def sample_data(self, val: Path):
        self.sample_data_path = val

# ==========================================
# 6. Transfer Configuration
# ==========================================

@dataclass
class TransferConfig:
    kind: str = "siegert"
    sigma_t: float = 10.0
    tau_e: float = 0.02
    tau_i: float = 0.01
    tau_rp: float = 0.002
    theta: float = 20.0
    v_r: float = 10.0
    mu_tab_max: float = 100.0

# ==========================================
# 7. Solver Configuration
# ==========================================

@dataclass
class JaxSolverConfig:
    prefer_sparse: bool = True
    dense_max_mb: float = 128.0

@dataclass
class DiffraxSolverConfig:
    solver: str = "tsit5"

@dataclass
class SolverConfig:
    backend: str = "scipy"
    method: str = "RK4"
    jax: Optional[JaxSolverConfig] = None
    diffrax: Optional[DiffraxSolverConfig] = None
    transfer: TransferConfig = field(default_factory=TransferConfig)

# ==========================================
# 8. Sweep Configuration
# ==========================================

@dataclass
class SweepConfig:
    max_workers: int = 2
    resume: bool = True
    output_csv: Optional[str] = None
    grid: Dict[str, Any] = field(default_factory=dict)

# ==========================================
# 9. Training Configuration
# ==========================================

@dataclass
class TrainingNaturalImageConfig:
    dir: Optional[str] = None
    limit: Optional[int] = None
    seed: Optional[int] = None
    crop_size: Optional[int] = 512
    patches_per_image: int = 4
    res: int = 128
    normalization: str = "log-zscore"
    clip_zscore: Optional[float] = 3.0
    projection_chunk_size: int = 64

@dataclass
class TrainingBCMConfig:
    epochs: int = 1
    batch_size: int = 16
    eta: float = 3.0e-5
    theta_beta: float = 0.01
    theta_eps: float = 1.0e-6
    theta_update_order: str = "pre"
    theta_init: Optional[float] = 1.0
    theta_floor: Optional[float] = 1.0e-3
    w_max: Optional[float] = 30.0
    row_sum_max_scale: Optional[float] = 1.05
    save_every: int = 100
    dynamic_steady_state: bool = True
    steady_state_abs_tol: float = 1.0e-3
    steady_state_rel_tol: float = 1.0e-5
    steady_state_window: int = 5
    steady_state_min_tau: float = 5.0

@dataclass
class TrainingConfig:
    enabled: bool = False
    natural_image: TrainingNaturalImageConfig = field(default_factory=TrainingNaturalImageConfig)
    bcm: TrainingBCMConfig = field(default_factory=TrainingBCMConfig)

# ==========================================
# 10. Root Configuration
# ==========================================

@dataclass
class RootConfig:
    seed: int = 42
    mode: str = "simulate"
    job_name: str = "v1"

    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
