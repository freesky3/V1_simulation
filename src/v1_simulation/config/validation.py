import math

from v1_simulation.config.schema import RootConfig, TransferConfig


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


def _optional_finite_float(value: float | None, path: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, path)


def _validate_transfer_config(trans: TransferConfig, path: str) -> None:
    if trans.kind != "siegert":
        raise ValueError(f"{path}.kind must be 'siegert', got {trans.kind!r}")
    if trans.sigma_t <= 0.0:
        raise ValueError(f"{path}.sigma_t must be positive, got {trans.sigma_t}")
    if trans.tau_e <= 0.0 or trans.tau_i <= 0.0 or trans.tau_rp <= 0.0:
        raise ValueError(f"{path} time constants (tau_e, tau_i, tau_rp) must be positive")
    if trans.theta <= 0.0:
        raise ValueError(f"{path} threshold theta must be positive, got {trans.theta}")
    if trans.v_r <= 0.0:
        raise ValueError(f"{path} reset potential v_r must be positive, got {trans.v_r}")
    if trans.mu_tab_max <= 0.0:
        raise ValueError(f"{path} mu_tab_max must be positive, got {trans.mu_tab_max}")
    rate_max = getattr(trans, 'rate_max', None)
    if rate_max is not None:
        rate_max = float(rate_max)
        if not math.isfinite(rate_max) or rate_max <= 0.0:
            raise ValueError(f"{path}.rate_max must be positive and finite when set, got {rate_max}")


def _validate_solver_method(backend: str, method: str) -> None:
    if backend == "scipy":
        if method not in {"RK4", "RK45", "DOP853", "BDF", "Radau", "LSODA"}:
            raise ValueError(
                "solver.method must be one of RK4, RK45, DOP853, BDF, Radau, or LSODA "
                "when solver.backend is 'scipy'."
            )
        return
    if backend == "jax-rk4":
        if method != "RK4":
            raise ValueError("solver.backend 'jax-rk4' requires solver.method 'RK4'.")
        return
    if backend == "diffrax":
        if method != "adaptive":
            raise ValueError("solver.backend 'diffrax' requires solver.method 'adaptive'.")
        return
    raise ValueError(f"solver.backend must be 'scipy', 'jax-rk4', or 'diffrax', got '{backend}'")


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
    _require_bool(analysis.save_plots, "analysis.save_plots")
    if analysis.num_surrogates <= 0:
        raise ValueError(f"analysis.num_surrogates must be positive, got {analysis.num_surrogates}")
    if not (0.0 < analysis.center_side_fraction <= 1.0):
        raise ValueError(f"analysis.center_side_fraction must be in (0.0, 1.0], got {analysis.center_side_fraction}")
    if not (0.0 <= analysis.osi_threshold <= 1.0):
        raise ValueError(f"analysis.osi_threshold must be in [0.0, 1.0], got {analysis.osi_threshold}")
    if not (0.0 < analysis.random_sample_fraction <= 1.0):
        raise ValueError(f"analysis.random_sample_fraction must be in (0.0, 1.0], got {analysis.random_sample_fraction}")
    if analysis.active_threshold <= 0.0:
        raise ValueError(f"analysis.active_threshold must be positive, got {analysis.active_threshold}")
    
    louvain = analysis.louvain
    if not (0.0 < louvain.thr_prop <= 1.0):
        raise ValueError(f"analysis.louvain.thr_prop must be in (0.0, 1.0], got {louvain.thr_prop}")
    if louvain.gamma <= 0.0:
        raise ValueError(f"analysis.louvain.gamma must be positive, got {louvain.gamma}")
    if louvain.num_runs <= 0:
        raise ValueError(f"analysis.louvain.num_runs must be positive, got {louvain.num_runs}")
    if not (0.0 <= louvain.consensus_tau <= 1.0):
        raise ValueError(f"analysis.louvain.consensus_tau must be in [0.0, 1.0], got {louvain.consensus_tau}")
    if louvain.consensus_reps <= 0:
        raise ValueError(f"analysis.louvain.consensus_reps must be positive, got {louvain.consensus_reps}")
    if louvain.min_module_degree < 0.0:
        raise ValueError(
            f"analysis.louvain.min_module_degree must be non-negative, got {louvain.min_module_degree}"
        )
    if louvain.min_cluster_size <= 0:
        raise ValueError(f"analysis.louvain.min_cluster_size must be positive, got {louvain.min_cluster_size}")

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
    if layers.N_theta <= 0:
        raise ValueError(f"model.layers.N_theta must be positive, got {layers.N_theta}")
    if layers.l4.N_theta <= 0:
        raise ValueError(f"model.layers.l4.N_theta must be positive, got {layers.l4.N_theta}")
    if layers.l4.n_side <= 0:
        raise ValueError(f"model.layers.l4.n_side must be positive, got {layers.l4.n_side}")
    if layers.l4.region_size <= 0.0:
        raise ValueError(f"model.layers.l4.region_size must be positive, got {layers.l4.region_size}")
    if layers.l23.n_side is not None and layers.l23.n_side <= 0:
        raise ValueError(f"model.layers.l23.n_side must be positive when set, got {layers.l23.n_side}")
    if layers.l23.region_size <= 0.0:
        raise ValueError(f"model.layers.l23.region_size must be positive, got {layers.l23.region_size}")
    if layers.l23.inhibitory_fraction is not None:
        if not (0.0 <= layers.l23.inhibitory_fraction <= 1.0):
            raise ValueError(
                "model.layers.l23.inhibitory_fraction must be in [0.0, 1.0] "
                f"when set, got {layers.l23.inhibitory_fraction}"
            )
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
    _require_bool(conn.equalize_indegree, "model.connectivity.equalize_indegree")
    
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
    if not (0.0 <= conn.kernel.kappa <= 1.0):
        raise ValueError(f"model.connectivity.kernel.kappa must be in [0.0, 1.0], got {conn.kernel.kappa}")

    # 5. Transfer Config
    _validate_transfer_config(cfg.transfer, "transfer")

    # 6. Solver Config
    sol = cfg.solver
    if sol.backend not in {"scipy", "jax-rk4", "diffrax"}:
        raise ValueError(f"solver.backend must be 'scipy', 'jax-rk4', or 'diffrax', got '{sol.backend}'")
    _validate_solver_method(sol.backend, sol.method)
    _validate_transfer_config(sol.transfer, "solver.transfer")
    if getattr(sol, 'early_stop', None) is not None:
        _require_bool(sol.early_stop.enabled, "solver.early_stop.enabled")
        _finite_float(sol.early_stop.min_time, "solver.early_stop.min_time")
        if sol.early_stop.min_time < 0.0:
            raise ValueError(f"solver.early_stop.min_time must be non-negative, got {sol.early_stop.min_time}")
        if sol.early_stop.min_steps <= 0:
            raise ValueError(f"solver.early_stop.min_steps must be positive, got {sol.early_stop.min_steps}")
        _finite_float(sol.early_stop.f_atol, "solver.early_stop.f_atol")
        if sol.early_stop.f_atol < 0.0:
            raise ValueError(f"solver.early_stop.f_atol must be non-negative, got {sol.early_stop.f_atol}")
        _finite_float(sol.early_stop.f_rtol, "solver.early_stop.f_rtol")
        if sol.early_stop.f_rtol < 0.0:
            raise ValueError(f"solver.early_stop.f_rtol must be non-negative, got {sol.early_stop.f_rtol}")
        if sol.early_stop.norm not in {"max", "l2"}:
            raise ValueError(f"solver.early_stop.norm must be 'max' or 'l2', got {sol.early_stop.norm!r}")
        if sol.early_stop.rk4_window <= 0:
            raise ValueError(f"solver.early_stop.rk4_window must be positive, got {sol.early_stop.rk4_window}")
        _require_bool(sol.early_stop.only_static_input, "solver.early_stop.only_static_input")
    if sol.jax is not None:
        if sol.jax.dense_max_mb <= 0.0:
            raise ValueError(f"solver.jax.dense_max_mb must be positive, got {sol.jax.dense_max_mb}")
        if sol.jax.dtype not in {"float32", "float64"}:
            raise ValueError(f"solver.jax.dtype must be 'float32' or 'float64', got {sol.jax.dtype!r}")
    if sol.diffrax is not None:
        if sol.diffrax.solver not in {"tsit5", "heun"}:
            raise ValueError(f"solver.diffrax.solver must be 'tsit5' or 'heun', got {sol.diffrax.solver!r}")
        if sol.diffrax.steady_state_tail_points <= 0:
            raise ValueError(
                f"solver.diffrax.steady_state_tail_points must be positive, got {sol.diffrax.steady_state_tail_points}"
            )

    # 7. Simulation Config
    sim = cfg.simulation
    t_start = _finite_float(sim.t_start, "simulation.t_start")
    if sim.t_stop is not None:
        t_stop = _finite_float(sim.t_stop, "simulation.t_stop")
        if t_stop <= t_start:
            raise ValueError("simulation.t_stop must be greater than simulation.t_start")
    if _finite_float(sim.duration_tau_e, "simulation.duration_tau_e") <= 0.0:
        raise ValueError(f"simulation.duration_tau_e must be positive, got {sim.duration_tau_e}")
    if sim.dt is not None and _finite_float(sim.dt, "simulation.dt") <= 0.0:
        raise ValueError(f"simulation.dt must be positive when set, got {sim.dt}")
    if _finite_float(sim.dt_tau_i_fraction, "simulation.dt_tau_i_fraction") <= 0.0:
        raise ValueError(
            "simulation.dt_tau_i_fraction must be positive, "
            f"got {sim.dt_tau_i_fraction}"
        )
    _require_bool(sim.store_trajectory, "simulation.store_trajectory")
    _require_bool(sim.save_network, "simulation.save_network")

    # 8. Stimulus Config
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

    # 9. Training Config
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
        if _finite_float(bcm.eta, "training.bcm.eta") < 0.0:
            raise ValueError(f"training.bcm.eta must be non-negative, got {bcm.eta}")
        if not (0.0 < _finite_float(bcm.theta_beta, "training.bcm.theta_beta") <= 1.0):
            raise ValueError(f"training.bcm.theta_beta must be in (0.0, 1.0], got {bcm.theta_beta}")
        if _finite_float(bcm.theta_eps, "training.bcm.theta_eps") <= 0.0:
            raise ValueError(f"training.bcm.theta_eps must be positive, got {bcm.theta_eps}")
        if bcm.theta_update_order not in {"pre", "post"}:
            raise ValueError(
                "training.bcm.theta_update_order must be 'pre' or 'post', "
                f"got {bcm.theta_update_order!r}"
            )
        theta_init = _optional_finite_float(bcm.theta_init, "training.bcm.theta_init")
        if theta_init is not None and theta_init <= 0.0:
            raise ValueError(f"training.bcm.theta_init must be positive, got {bcm.theta_init}")
        theta_floor = _optional_finite_float(bcm.theta_floor, "training.bcm.theta_floor")
        if theta_floor is not None and theta_floor <= 0.0:
            raise ValueError(f"training.bcm.theta_floor must be positive, got {bcm.theta_floor}")
        w_max = _optional_finite_float(bcm.w_max, "training.bcm.w_max")
        if w_max is not None and w_max <= 0.0:
            raise ValueError(f"training.bcm.w_max must be positive, got {bcm.w_max}")
        row_sum_scale = _optional_finite_float(bcm.row_sum_max_scale, "training.bcm.row_sum_max_scale")
        if row_sum_scale is not None and row_sum_scale < 0.0:
            raise ValueError(
                f"training.bcm.row_sum_max_scale must be non-negative, got {bcm.row_sum_max_scale}"
            )
        if bcm.save_every <= 0:
            raise ValueError(f"training.bcm.save_every must be positive, got {bcm.save_every}")
        if _finite_float(bcm.steady_state_abs_tol, "training.bcm.steady_state_abs_tol") <= 0.0:
            raise ValueError(
                f"training.bcm.steady_state_abs_tol must be positive, got {bcm.steady_state_abs_tol}"
            )
        if _finite_float(bcm.steady_state_rel_tol, "training.bcm.steady_state_rel_tol") <= 0.0:
            raise ValueError(
                f"training.bcm.steady_state_rel_tol must be positive, got {bcm.steady_state_rel_tol}"
            )
        if bcm.steady_state_window <= 0:
            raise ValueError(f"training.bcm.steady_state_window must be positive, got {bcm.steady_state_window}")
        if _finite_float(bcm.steady_state_min_tau, "training.bcm.steady_state_min_tau") < 0.0:
            raise ValueError(
                f"training.bcm.steady_state_min_tau must be non-negative, got {bcm.steady_state_min_tau}"
            )
        rate_exp = _optional_finite_float(bcm.rate_explosion_threshold, "training.bcm.rate_explosion_threshold")
        if rate_exp is not None and rate_exp <= 0.0:
            raise ValueError(
                f"training.bcm.rate_explosion_threshold must be positive when set, got {bcm.rate_explosion_threshold}"
            )
        sat_frac = _finite_float(bcm.saturation_fraction_threshold, "training.bcm.saturation_fraction_threshold")
        if sat_frac < 0.0 or sat_frac > 1.0:
            raise ValueError(
                f"training.bcm.saturation_fraction_threshold must be in [0.0, 1.0], got {bcm.saturation_fraction_threshold}"
            )
        if bcm.max_consecutive_bad_batches <= 0:
            raise ValueError(
                f"training.bcm.max_consecutive_bad_batches must be positive, got {bcm.max_consecutive_bad_batches}"
            )
        duration_tau_e = _finite_float(bcm.duration_tau_e, "training.bcm.duration_tau_e")
        if duration_tau_e <= 0.0:
            raise ValueError(f"training.bcm.duration_tau_e must be positive, got {bcm.duration_tau_e}")


    # 10. Sweep Config
    if getattr(cfg, 'sweep', None) is not None:
        _require_bool(cfg.sweep.resume, "sweep.resume")
        if cfg.sweep.max_workers <= 0:
            raise ValueError(f"sweep.max_workers must be positive, got {cfg.sweep.max_workers}")
        if not isinstance(cfg.sweep.grid, dict):
            raise TypeError(f"sweep.grid must be a dictionary/mapping, got {type(cfg.sweep.grid).__name__}")
