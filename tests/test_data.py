import pytest

from sumbench.data import sample_ids, sample_indices


def test_sampling_is_deterministic_and_sorted():
    a = sample_indices(11490, 500, 42)
    assert a == sample_indices(11490, 500, 42)
    assert a == sorted(a) and len(set(a)) == 500


def test_different_seed_differs():
    assert sample_indices(1000, 50, 1) != sample_indices(1000, 50, 2)


def test_sample_ids_maps_indices():
    ids = [f"id{i}" for i in range(100)]
    assert sample_ids(ids, 5, 7) == [ids[i] for i in sample_indices(100, 5, 7)]


def test_oversample_raises():
    with pytest.raises(ValueError):
        sample_indices(5, 6, 0)
