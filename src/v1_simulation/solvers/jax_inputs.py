from __future__ import annotations

import numpy as np

from v1_simulation.solvers.base import (
    ExternalDrive,
    FloatArray,
    NetworkLayout,
    prepare_rk4_background,
    validate_external_drive_value,
)


def precompute_diffrax_inputs(
    external_drive: ExternalDrive,
    time: FloatArray,
    *,
    n_ext: int,
    n_batch: int,
) -> FloatArray:
    values = []
    for t in time:
        values.append(validate_external_drive_value(external_drive(float(t)), n_ext=n_ext, n_batch=n_batch))
    return np.stack(values)


def precompute_diffrax_background(
    trace,
    *,
    layout: NetworkLayout,
    n_batch: int,
    time: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    if trace is None:
        return (
            np.zeros((time.size, layout.n_exc, n_batch), dtype=np.float64),
            np.zeros((time.size, layout.n_inh, n_batch), dtype=np.float64),
        )

    return (
        np.transpose(trace.exc, (0, 2, 1)),
        np.transpose(trace.inh, (0, 2, 1)),
    )


def precompute_rk4_inputs(
    external_drive: ExternalDrive,
    time: FloatArray,
    *,
    n_ext: int,
    n_batch: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    left = []
    mid = []
    right = []
    for t0, t1 in zip(time[:-1], time[1:]):
        dt = float(t1 - t0)
        left.append(validate_external_drive_value(external_drive(float(t0)), n_ext=n_ext, n_batch=n_batch))
        mid.append(validate_external_drive_value(external_drive(float(t0 + 0.5 * dt)), n_ext=n_ext, n_batch=n_batch))
        right.append(validate_external_drive_value(external_drive(float(t1)), n_ext=n_ext, n_batch=n_batch))
    return np.stack(left), np.stack(mid), np.stack(right)


def precompute_rk4_background(
    trace,
    *,
    layout: NetworkLayout,
    n_batch: int,
    time: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    samples = prepare_rk4_background(
        trace,
        n_exc=layout.n_exc,
        n_inh=layout.n_inh,
        n_batch=n_batch,
        time=time,
    )
    if samples is None:
        n_steps = time.size - 1
        return (
            np.zeros((n_steps, layout.n_exc, n_batch), dtype=np.float64),
            np.zeros((n_steps, layout.n_exc, n_batch), dtype=np.float64),
            np.zeros((n_steps, layout.n_exc, n_batch), dtype=np.float64),
            np.zeros((n_steps, layout.n_inh, n_batch), dtype=np.float64),
            np.zeros((n_steps, layout.n_inh, n_batch), dtype=np.float64),
            np.zeros((n_steps, layout.n_inh, n_batch), dtype=np.float64),
        )

    return (
        np.transpose(samples.exc_left, (0, 2, 1)),
        np.transpose(samples.exc_mid, (0, 2, 1)),
        np.transpose(samples.exc_right, (0, 2, 1)),
        np.transpose(samples.inh_left, (0, 2, 1)),
        np.transpose(samples.inh_mid, (0, 2, 1)),
        np.transpose(samples.inh_right, (0, 2, 1)),
    )
