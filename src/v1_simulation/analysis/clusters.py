from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def labels_array(labels: ArrayLike, *, n_neurons: int | None = None) -> NDArray[np.int64]:
    """Validates and formats community labels into a 1D int64 numpy array.

    Args:
        labels: Array-like object containing community labels.
        n_neurons: Optional expected number of neurons.

    Returns:
        A validated copy of the labels as a 1D numpy array.

    Raises:
        ValueError: If shape is incorrect or if any label is negative.
    """
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if n_neurons is not None and arr.shape != (int(n_neurons),):
        raise ValueError(f"labels must have shape ({int(n_neurons)},), got {arr.shape}.")
    if np.any(arr < 0):
        raise ValueError("labels must be non-negative; 0 is reserved for unclassified neurons.")
    return arr.copy()


def cluster_ids(labels: ArrayLike) -> list[int]:
    """Returns a list of unique non-zero cluster IDs in the labels.

    Args:
        labels: Array-like object containing community labels.

    Returns:
        A sorted list of unique integer cluster IDs (excluding 0).
    """
    arr = labels_array(labels)
    return [int(c_id) for c_id in np.unique(arr) if c_id != 0]


def cluster_members(labels: ArrayLike) -> dict[int, NDArray[np.int64]]:
    """Groups neuron indices by their cluster labels.

    Args:
        labels: Array-like object containing community labels.

    Returns:
        A dictionary mapping each cluster ID to a numpy array of neuron indices.
    """
    arr = labels_array(labels)
    return {c_id: np.flatnonzero(arr == c_id) for c_id in cluster_ids(arr)}


def relabel_consecutive(labels: ArrayLike) -> NDArray[np.int64]:
    """Maps arbitrary cluster labels to consecutive integers starting from 1.

    Unclassified neurons (0 or NaN) remain labeled as 0.

    Args:
        labels: Array-like object containing community labels.

    Returns:
        A new 1D numpy array with consecutive cluster IDs.
    """
    arr = labels_array(np.nan_to_num(labels, nan=0.0).astype(np.int64, copy=False))
    out = np.zeros_like(arr, dtype=np.int64)
    for new_id, old_id in enumerate(cluster_ids(arr), start=1):
        out[arr == old_id] = new_id
    return out
