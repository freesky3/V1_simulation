import unittest

import numpy as np

from v1_simulation.config import load_config
from v1_simulation.config.schema import BackgroundConfig, RootConfig
from v1_simulation.config.validation import validate_config
from v1_simulation.solvers.base import validate_background_trace
from v1_simulation.stimuli.background import (
    BackgroundTrace,
    OUParams,
    generate_background_trace,
    generate_ou_background,
)


class BackgroundTests(unittest.TestCase):
    def test_generates_time_major_trace_from_yaml_schema(self) -> None:
        cfg = load_config(overrides=["background=ou"])
        time = np.linspace(0.0, 0.5, 51)

        trace = generate_background_trace(
            cfg.background,
            n_exc=4,
            n_inh=3,
            n_batch=2,
            time=time,
        )

        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.exc.shape, (51, 2, 4))
        self.assertEqual(trace.inh.shape, (51, 2, 3))
        self.assertTrue(np.allclose(trace.time, time))

    def test_disabled_background_returns_none(self) -> None:
        cfg = load_config(overrides=["background=none"])

        trace = generate_background_trace(
            cfg.background,
            n_exc=1,
            n_inh=1,
            n_batch=1,
            time=np.array([0.0, 0.1]),
        )

        self.assertIsNone(trace)

    def test_seed_reproducibility_and_independent_population_streams(self) -> None:
        time = np.linspace(0.0, 0.2, 11)
        kwargs = dict(
            n_inh=2,
            n_batch=3,
            time=time,
            exc=OUParams(mean=0.0, stationary_std=1.0, tau=0.05),
            inh=OUParams(mean=0.0, stationary_std=1.0, tau=0.05),
            seed=123,
        )

        trace_a = generate_ou_background(n_exc=4, **kwargs)
        trace_b = generate_ou_background(n_exc=4, **kwargs)
        trace_without_exc = generate_ou_background(n_exc=0, **kwargs)

        self.assertTrue(np.allclose(trace_a.exc, trace_b.exc))
        self.assertTrue(np.allclose(trace_a.inh, trace_b.inh))
        self.assertTrue(np.allclose(trace_a.inh, trace_without_exc.inh))

    def test_invalid_background_config_fails_fast(self) -> None:
        cfg = RootConfig(background=BackgroundConfig(enabled=True, tau_e=float("nan")))

        with self.assertRaisesRegex(ValueError, "background.tau_e must be finite"):
            validate_config(cfg)

        disabled_cfg = RootConfig(background=BackgroundConfig(enabled=False, sigma_i=-1.0))
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            validate_config(disabled_cfg)

    def test_generation_rejects_invalid_time_and_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            generate_ou_background(
                n_exc=1,
                n_inh=1,
                n_batch=1,
                time=np.array([0.0, np.inf]),
                exc=OUParams(),
                inh=OUParams(),
                seed=1,
            )

        with self.assertRaisesRegex(TypeError, "n_batch must be an integer"):
            generate_ou_background(
                n_exc=1,
                n_inh=1,
                n_batch=1.5,
                time=np.array([0.0, 0.1]),
                exc=OUParams(),
                inh=OUParams(),
                seed=1,
            )

    def test_trace_copies_and_freezes_arrays(self) -> None:
        time = np.array([0.0, 0.1])
        exc = np.zeros((2, 1, 1))
        inh = np.zeros((2, 1, 1))

        trace = BackgroundTrace(time=time, exc=exc, inh=inh)
        time[0] = -1.0
        exc[0, 0, 0] = 99.0

        self.assertEqual(trace.time[0], 0.0)
        self.assertEqual(trace.exc[0, 0, 0], 0.0)
        with self.assertRaises(ValueError):
            trace.exc[0, 0, 0] = 1.0

    def test_solver_validation_rejects_time_grid_mismatch(self) -> None:
        time = np.array([0.0, 0.1, 0.2])
        trace = BackgroundTrace(
            time=time,
            exc=np.zeros((3, 2, 4)),
            inh=np.zeros((3, 2, 3)),
        )

        with self.assertRaisesRegex(ValueError, "time grid"):
            validate_background_trace(
                trace,
                n_exc=4,
                n_inh=3,
                n_batch=2,
                time=np.array([0.0, 0.15, 0.2]),
            )


if __name__ == "__main__":
    unittest.main()
