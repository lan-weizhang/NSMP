from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .data import ClientSplit, FederatedTimeSeries


def _select_clients(
    rng: np.random.Generator, num_clients: int, fraction: float
) -> set[int]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Client fraction must be in [0, 1].")
    count = int(round(num_clients * fraction))
    if fraction > 0.0:
        count = max(1, count)
    if count == 0:
        return set()
    return set(rng.choice(num_clients, size=count, replace=False).tolist())


def _uniform_subset(indices: np.ndarray, count: int) -> np.ndarray:
    if count >= indices.size:
        return indices.copy()
    positions = np.linspace(0, indices.size - 1, num=count)
    return indices[np.unique(np.rint(positions).astype(int))]


def _remove_missing(
    indices: np.ndarray,
    fraction: float,
    pattern: str,
    rng: np.random.Generator,
    minimum: int,
) -> np.ndarray:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("missing_fraction must be in [0, 1).")
    remove_count = min(int(round(indices.size * fraction)), indices.size - minimum)
    if remove_count <= 0:
        return indices
    if pattern == "random":
        removed = rng.choice(indices.size, size=remove_count, replace=False)
    elif pattern == "block":
        start = int(rng.integers(0, indices.size - remove_count + 1))
        removed = np.arange(start, start + remove_count)
    else:
        raise ValueError("missing_pattern must be 'none', 'random', or 'block'.")
    keep = np.ones(indices.size, dtype=bool)
    keep[removed] = False
    return indices[keep]


def _transform_split(
    split: ClientSplit,
    old_mean: float,
    old_std: float,
    new_mean: float,
    new_std: float,
    add_bias: bool,
) -> ClientSplit:
    feature_count = split.x.shape[1] - int(add_bias)
    raw_x = split.x[:, :feature_count] * old_std + old_mean
    x = (raw_x - new_mean) / new_std
    if add_bias:
        x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    raw_y = split.y * old_std + old_mean
    y = (raw_y - new_mean) / new_std
    return ClientSplit(x=x, y=y)


def apply_training_scenario(
    dataset: FederatedTimeSeries,
    config: dict[str, Any] | None,
) -> tuple[FederatedTimeSeries, dict[str, Any]]:
    """Reduce or remove training windows while leaving val/test targets intact.

    Low-data selection is applied first. Missingness then removes random windows or
    one contiguous block from the remaining training timeline. When requested,
    normalization is recomputed using only observations in retained windows.
    """
    settings = deepcopy(config or {})
    name = str(settings.get("name", "full_data"))
    seed = int(settings.get("seed", 2026))
    rng = np.random.default_rng(seed)
    minimum = int(settings.get("min_train_samples", dataset.dimension + 2))
    low_fraction = float(settings.get("train_data_fraction", 1.0))
    low_client_fraction = float(settings.get("low_data_client_fraction", 1.0))
    missing_fraction = float(settings.get("missing_fraction", 0.0))
    missing_client_fraction = float(settings.get("missing_client_fraction", 1.0))
    missing_pattern = str(settings.get("missing_pattern", "none"))
    recompute = bool(settings.get("recompute_normalization", True))

    if not 0.0 < low_fraction <= 1.0:
        raise ValueError("train_data_fraction must be in (0, 1].")
    if missing_pattern == "none" and missing_fraction != 0.0:
        raise ValueError("Use random/block missing_pattern when missing_fraction > 0.")

    low_clients = _select_clients(rng, dataset.num_clients, low_client_fraction)
    missing_clients = _select_clients(
        rng, dataset.num_clients, missing_client_fraction
    )
    selected_indices: list[np.ndarray] = []
    for client, split in enumerate(dataset.train):
        indices = np.arange(split.y.size)
        if client in low_clients and low_fraction < 1.0:
            target_count = max(minimum, int(round(indices.size * low_fraction)))
            indices = _uniform_subset(indices, min(target_count, indices.size))
        if (
            client in missing_clients
            and missing_fraction > 0.0
            and missing_pattern != "none"
        ):
            indices = _remove_missing(
                indices, missing_fraction, missing_pattern, rng, minimum
            )
        if indices.size < minimum:
            raise ValueError(
                f"Scenario leaves client {client} with {indices.size} samples; "
                f"minimum is {minimum}."
            )
        selected_indices.append(indices)

    target_mean = dataset.target_mean.copy()
    target_std = dataset.target_std.copy()
    train: list[ClientSplit] = []
    val: list[ClientSplit] = []
    test: list[ClientSplit] = []
    client_rows: list[dict[str, Any]] = []
    for client, indices in enumerate(selected_indices):
        old_train = dataset.train[client]
        retained = ClientSplit(x=old_train.x[indices], y=old_train.y[indices])
        old_mean = float(dataset.target_mean[client])
        old_std = float(dataset.target_std[client])
        if recompute:
            feature_count = retained.x.shape[1] - int(dataset.add_bias)
            raw_x = retained.x[:, :feature_count] * old_std + old_mean
            raw_y = retained.y * old_std + old_mean
            observations = np.concatenate([raw_x.ravel(), raw_y])
            new_mean = float(observations.mean())
            new_std = float(observations.std())
            if new_std < 1e-8:
                new_std = 1.0
        else:
            new_mean, new_std = old_mean, old_std
        target_mean[client] = new_mean
        target_std[client] = new_std
        train.append(
            _transform_split(
                retained,
                old_mean,
                old_std,
                new_mean,
                new_std,
                dataset.add_bias,
            )
        )
        val.append(
            _transform_split(
                dataset.val[client],
                old_mean,
                old_std,
                new_mean,
                new_std,
                dataset.add_bias,
            )
        )
        test.append(
            _transform_split(
                dataset.test[client],
                old_mean,
                old_std,
                new_mean,
                new_std,
                dataset.add_bias,
            )
        )
        client_rows.append(
            {
                "client_id": dataset.client_ids[client],
                "original_train_samples": int(old_train.y.size),
                "retained_train_samples": int(indices.size),
                "retained_fraction": float(indices.size / old_train.y.size),
                "low_data_client": client in low_clients,
                "missing_client": client in missing_clients,
            }
        )

    transformed = FederatedTimeSeries(
        client_ids=list(dataset.client_ids),
        train=train,
        val=val,
        test=test,
        target_mean=target_mean,
        target_std=target_std,
        similarity=dataset.similarity.copy(),
        history=dataset.history,
        add_bias=dataset.add_bias,
    )
    retained = np.array([row["retained_fraction"] for row in client_rows])
    metadata: dict[str, Any] = {
        "name": name,
        "seed": seed,
        "train_data_fraction": low_fraction,
        "low_data_client_fraction": low_client_fraction,
        "missing_pattern": missing_pattern,
        "missing_fraction": missing_fraction,
        "missing_client_fraction": missing_client_fraction,
        "recompute_normalization": recompute,
        "mean_retained_fraction": float(retained.mean()),
        "min_retained_fraction": float(retained.min()),
        "max_retained_fraction": float(retained.max()),
        "clients": client_rows,
    }
    return transformed, metadata
