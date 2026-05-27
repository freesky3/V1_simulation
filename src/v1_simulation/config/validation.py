import math

from v1_simulation.config.schema import RootConfig


def _require_bool(value: bool, path: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean, got {type(value).__name__}")


def _require_optional_non_negative_int(value: int | None, path: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer or null, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{path} must be non-negative, got {value}")


def _finite_float(value: float, path: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{path} must be finite, got {value}")
    return value

def validate_config(cfg: RootConfig) -> None:
    """Validate all fields in a RootConfig instance for physical correctness and constraints.
    
    Checks value ranges, dependencies, and typings for numerical settings of the model.
    
    Raises:
        ValueError: If any numeric values violate physical/mathematical bounds.
        TypeError: If any values have incorrect types.
    """
    # 1. Seed & Mode
    if cfg.seed < 0:
        raise ValueError(f"Global seed must be non-negative, got {cfg.seed}")
    if cfg.mode not in {"simulate", "train"}:
        raise ValueError(f"Global mode must be 'simulate' or 'train', got '{cfg.mode}'")

    # 2. Analysis Config
    analysis = cfg.analysis
    if analysis.num_surrogates <= 0:
        raise ValueError(f"analysis.num_surrogates must be positive, got {analysis.num_surrogates}")
    if not (0.0 <= analysis.center_side_fraction <= 1.0):
        raise ValueError(f"analysis.center_side_fraction must be in [0.0, 1.0], got {analysis.center_side_fraction}")
    if not (0.0 <= analysis.osi_threshold <= 1.0):
        raise ValueError(f"analysis.osi_threshold must be in [0.0, 1.0], got {analysis.osi_threshold}")
    if not (0.0 <= analysis.random_sample_fraction <= 1.0):
        raise ValueError(f"analysis.random_sample_fraction must be in [0.0, 1.0], got {analysis.random_sample_fraction}")
    if analysis.active_threshold <= 0.0:
        raise ValueError(f"analysis.active_threshold must be positive, got {analysis.active_threshold}")
    
    louvain = analysis.louvain
    if not (0.0 <= louvain.thr_prop <= 1.0):
        raise ValueError(f"analysis.louvain.thr_prop must be in [0.0, 1.0], got {louvain.thr_prop}")
    if louvain.gamma <= 0.0:
        raise ValueError(f"analysis.louvain.gamma must be positive, got {louvain.gamma}")
    if louvain.num_runs <= 0:
        raise ValueError(f"analysis.louvain.num_runs must be positive, got {louvain.num_runs}")
    if louvain.consensus_reps <= 0:
        raise ValueError(f"analysis.louvain.consensus_reps must be positive, got {louvain.consensus_reps}")

    # 3. Background Config
    bg = cfg.background
    _require_bool(bg.enabled, "background.enabled")
    _require_optional_non_negative_int(bg.seed, "background.seed")
    if bg.interpolation not in {"linear", "sample_hold"}:
        raise ValueError(
            "background.interpolation must be 'linear' or 'sample_hold', "
            f"got {bg.interpolation!r}"
        )
    tau_e = _finite_float(bg.tau_e, "background.tau_e")
    tau_i = _finite_float(bg.tau_i, "background.tau_i")
    sigma_e = _finite_float(bg.sigma_e, "background.sigma_e")
    sigma_i = _finite_float(bg.sigma_i, "background.sigma_i")
    _finite_float(bg.mu_e, "background.mu_e")
    _finite_float(bg.mu_i, "background.mu_i")
    if tau_e <= 0.0 or tau_i <= 0.0:
        raise ValueError(f"Background time constants (tau_e, tau_i) must be positive, got tau_e={bg.tau_e}, tau_i={bg.tau_i}")
    if sigma_e < 0.0 or sigma_i < 0.0:
        raise ValueError(f"Background noise standard deviations (sigma_e, sigma_i) must be non-negative, got sigma_e={bg.sigma_e}, sigma_i={bg.sigma_i}")

    # 4. Model Config
    model = cfg.model
    # Layers
    layers = model.layers
    if layers.l4.n_side <= 0:
        raise ValueError(f"model.layers.l4.n_side must be positive, got {layers.l4.n_side}")
    if layers.l4.region_size <= 0.0:
        raise ValueError(f"model.layers.l4.region_size must be positive, got {layers.l4.region_size}")
    if layers.l23.region_size <= 0.0:
        raise ValueError(f"model.layers.l23.region_size must be positive, got {layers.l23.region_size}")
    if layers.periodic:
        if abs(layers.l4.region_size - layers.l23.region_size) > 1e-7:
            raise ValueError(
                f"Periodic cross-layer simulations require matching region_size between L4 and L23, "
                f"got L4 region_size={layers.l4.region_size} and L23 region_size={layers.l23.region_size}"
            )
    
    # Connectivity
    conn = model.connectivity
    if not (0.0 <= conn.p_ee <= 1.0):
        raise ValueError(f"model.connectivity.p_ee must be in [0.0, 1.0], got {conn.p_ee}")
    if conn.j <= 0.0:
        raise ValueError(f"model.connectivity.j (coupling weight scale) must be positive, got {conn.j}")
    if conn.g <= 0.0:
        raise ValueError(f"model.connectivity.g (inhibition-to-excitation ratio) must be positive, got {conn.g}")
    
    # Scales
    for scale_name in ["ee", "ei", "ex", "ie", "ii", "ix"]:
        val = getattr(conn.scales, scale_name)
        if val < 0.0:
            raise ValueError(f"model.connectivity.scales.{scale_name} must be non-negative, got {val}")

    # Kernel
    if conn.kernel.sigma_narrow <= 0.0:
        raise ValueError(f"model.connectivity.kernel.sigma_narrow must be positive, got {conn.kernel.sigma_narrow}")
    if conn.kernel.sigma_broad <= 0.0:
        raise ValueError(f"model.connectivity.kernel.sigma_broad must be positive, got {conn.kernel.sigma_broad}")
    if conn.kernel.sigma_narrow >= conn.kernel.sigma_broad:
        raise ValueError(
            f"Connectivity kernel sigma_narrow must be smaller than sigma_broad, "
            f"got sigma_narrow={conn.kernel.sigma_narrow}, sigma_broad={conn.kernel.sigma_broad}"
        )
    if conn.kernel.kappa < 0.0:
        raise ValueError(f"model.connectivity.kernel.kappa must be non-negative, got {conn.kernel.kappa}")

    # 5. Transfer Config
    trans = cfg.transfer
    if trans.sigma_t <= 0.0:
        raise ValueError(f"transfer.sigma_t must be positive, got {trans.sigma_t}")
    if trans.tau_e <= 0.0 or trans.tau_i <= 0.0 or trans.tau_rp <= 0.0:
        raise ValueError(f"transfer time constants (tau_e, tau_i, tau_rp) must be positive")
    if trans.theta <= 0.0:
        raise ValueError(f"transfer threshold theta must be positive, got {trans.theta}")
    if trans.v_r <= 0.0:
        raise ValueError(f"transfer reset potential v_r must be positive, got {trans.v_r}")
    if trans.mu_tab_max <= 0.0:
        raise ValueError(f"transfer mu_tab_max must be positive, got {trans.mu_tab_max}")

    # 6. Solver Config
    sol = cfg.solver
    if sol.backend not in {"scipy", "jax-rk4", "diffrax"}:
        raise ValueError(f"solver.backend must be 'scipy', 'jax-rk4', or 'diffrax', got '{sol.backend}'")

    # 7. Stimulus Config
    stim = cfg.stimulus
    if stim.kind not in {"drifting_grating", "natural_image"}:
        raise ValueError(f"stimulus.kind must be 'drifting_grating' or 'natural_image', got '{stim.kind}'")
    if stim.stimulus_size <= 0.0:
        raise ValueError(f"stimulus.stimulus_size must be positive, got {stim.stimulus_size}")
    if stim.gabor.sigma <= 0.0:
        raise ValueError(f"stimulus.gabor.sigma must be positive, got {stim.gabor.sigma}")
    if stim.gabor.gamma <= 0.0:
        raise ValueError(f"stimulus.gabor.gamma must be positive, got {stim.gabor.gamma}")
    if stim.gabor.spatial_frequency <= 0.0:
        raise ValueError(f"stimulus.gabor.spatial_frequency must be positive, got {stim.gabor.spatial_frequency}")
    if stim.resolution <= 1:
        raise ValueError(f"stimulus.resolution must be greater than 1, got {stim.resolution}")
    if stim.luminance <= 0.0:
        raise ValueError(f"stimulus.luminance must be positive, got {stim.luminance}")
    if not (0.0 <= stim.contrast <= 1.0):
        raise ValueError(f"stimulus.contrast must be in [0.0, 1.0], got {stim.contrast}")
    if stim.temporal_frequency < 0.0:
        raise ValueError(f"stimulus.temporal_frequency must be non-negative, got {stim.temporal_frequency}")
    if stim.visual_gain <= 0.0:
        raise ValueError(f"stimulus.visual_gain must be positive, got {stim.visual_gain}")
    if stim.n_theta <= 0:
        raise ValueError(f"stimulus.n_theta must be positive, got {stim.n_theta}")

    # 8. Training Config
    train = cfg.training
    if train.enabled:
        img = train.natural_image
        if img.dir is None:
            raise ValueError("training.natural_image.dir must be set when training is enabled")
        if img.limit is not None and img.limit < 0:
            raise ValueError(f"training.natural_image.limit must be non-negative, got {img.limit}")
        if img.crop_size is not None and img.crop_size <= 0:
            raise ValueError(f"training.natural_image.crop_size must be positive, got {img.crop_size}")
        if img.patches_per_image <= 0:
            raise ValueError(f"training.natural_image.patches_per_image must be positive, got {img.patches_per_image}")
        if img.res <= 1:
            raise ValueError(f"training.natural_image.res must be greater than 1, got {img.res}")
        if img.normalization not in {"log-zscore", "zscore", "maxscale"}:
            raise ValueError(f"training.natural_image.normalization is unsupported: {img.normalization}")
        if img.clip_zscore is not None and img.clip_zscore <= 0:
            raise ValueError(f"training.natural_image.clip_zscore must be positive, got {img.clip_zscore}")
        if img.projection_chunk_size <= 0:
            raise ValueError(
                f"training.natural_image.projection_chunk_size must be positive, got {img.projection_chunk_size}"
            )
        
        bcm = train.bcm
        if bcm.epochs <= 0:
            raise ValueError(f"training.bcm.epochs must be positive, got {bcm.epochs}")
        if bcm.batch_size <= 0:
            raise ValueError(f"training.bcm.batch_size must be positive, got {bcm.batch_size}")
        if bcm.eta <= 0.0:
            raise ValueError(f"training.bcm.eta must be positive, got {bcm.eta}")
        if not (0.0 < bcm.theta_beta < 1.0):
            raise ValueError(f"training.bcm.theta_beta must be in (0.0, 1.0), got {bcm.theta_beta}")
        if bcm.theta_eps <= 0.0:
            raise ValueError(f"training.bcm.theta_eps must be positive, got {bcm.theta_eps}")
        if bcm.theta_init <= 0.0:
            raise ValueError(f"training.bcm.theta_init must be positive, got {bcm.theta_init}")
        if bcm.theta_floor <= 0.0:
            raise ValueError(f"training.bcm.theta_floor must be positive, got {bcm.theta_floor}")
        if bcm.w_max <= 0.0:
            raise ValueError(f"training.bcm.w_max must be positive, got {bcm.w_max}")
        if bcm.row_sum_max_scale <= 0.0:
            raise ValueError(f"training.bcm.row_sum_max_scale must be positive, got {bcm.row_sum_max_scale}")
        if bcm.save_every <= 0:
            raise ValueError(f"training.bcm.save_every must be positive, got {bcm.save_every}")
