import numpy as np
import pytest

from v1_simulation.analysis.clusters import (
    cluster_ids,
    cluster_members,
    labels_array,
    relabel_consecutive,
)


def test_labels_array_validation() -> None:
    # Valid
    arr = labels_array([1, 2, 0, 3], n_neurons=4)
    assert np.array_equal(arr, np.array([1, 2, 0, 3], dtype=np.int64))

    # Length mismatch
    with pytest.raises(ValueError, match="labels must have shape"):
        labels_array([1, 2, 0], n_neurons=4)

    # Negative labels (excluding 0 which is allowed)
    with pytest.raises(ValueError, match="labels must be non-negative"):
        labels_array([1, -1, 0])


def test_cluster_ids() -> None:
    # 0 is excluded, other unique elements returned sorted
    labels = [3, 0, 1, 3, 5, 0]
    ids = cluster_ids(labels)
    assert ids == [1, 3, 5]


def test_cluster_members() -> None:
    labels = [3, 0, 1, 3, 5, 0]
    members = cluster_members(labels)

    assert set(members.keys()) == {1, 3, 5}
    assert np.array_equal(members[1], np.array([2]))
    assert np.array_equal(members[3], np.array([0, 3]))
    assert np.array_equal(members[5], np.array([4]))


def test_relabel_consecutive() -> None:
    # Unclassified remains 0, other active labels map to 1, 2, 3, ... based on their sorted order
    labels = [10, 0, 5, np.nan, 10, 20]
    relabeled = relabel_consecutive(labels)

    expected = np.array([2, 0, 1, 0, 2, 3], dtype=np.int64)
    assert np.array_equal(relabeled, expected)
