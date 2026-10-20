from __future__ import annotations

import numpy as np

from nsmp_metr_la.data import ClientSplit, FederatedTimeSeries
from nsmp_metr_la.scenarios import apply_training_scenario


def _dataset() -> FederatedTimeSeries:
    rng = np.random.default_rng(8)
    train = []
    val = []
    test = []
    for _ in range(4):
        train.append(ClientSplit(rng.normal(size=(100, 4)), rng.normal(size=100)))
        val.append(ClientSplit(rng.normal(size=(20, 4)), rng.normal(size=20)))
        test.append(ClientSplit(rng.normal(size=(25, 4)), rng.normal(size=25)))
    return FederatedTimeSeries(
        client_ids=["a", "b", "c", "d"],
        train=train,
        val=val,
        test=test,
        target_mean=np.array([30.0, 35.0, 40.0, 45.0]),
        target_std=np.array([5.0, 6.0, 7.0, 8.0]),
        similarity=np.ones((4, 4)) - np.eye(4),
        history=3,
        add_bias=True,
    )


def _raw_targets(dataset: FederatedTimeSeries, split: str) -> list[np.ndarray]:
    return [
        dataset.inverse_target(index, item.y)
        for index, item in enumerate(getattr(dataset, split))
    ]


def test_low_data_keeps_evaluation_targets_and_recomputes_scaling() -> None:
    original = _dataset()
    original_val = _raw_targets(original, "val")
    reduced, metadata = apply_training_scenario(
        original,
        {
            "name": "low",
            "seed": 3,
            "train_data_fraction": 0.2,
            "low_data_client_fraction": 1.0,
            "missing_pattern": "none",
            "missing_fraction": 0.0,
            "recompute_normalization": True,
        },
    )
    assert all(split.y.size == 20 for split in reduced.train)
    assert metadata["mean_retained_fraction"] == 0.2
    for expected, actual in zip(original_val, _raw_targets(reduced, "val"), strict=True):
        np.testing.assert_allclose(actual, expected)


def test_block_missing_only_changes_selected_client_count() -> None:
    reduced, metadata = apply_training_scenario(
        _dataset(),
        {
            "name": "block",
            "seed": 4,
            "train_data_fraction": 1.0,
            "low_data_client_fraction": 1.0,
            "missing_pattern": "block",
            "missing_fraction": 0.5,
            "missing_client_fraction": 0.5,
            "recompute_normalization": False,
        },
    )
    sizes = sorted(split.y.size for split in reduced.train)
    assert sizes == [50, 50, 100, 100]
    assert metadata["mean_retained_fraction"] == 0.75
