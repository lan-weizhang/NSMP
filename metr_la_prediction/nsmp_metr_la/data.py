from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class ClientSplit:
    x: np.ndarray
    y: np.ndarray


@dataclass
class FederatedTimeSeries:
    client_ids: list[str]
    train: list[ClientSplit]
    val: list[ClientSplit]
    test: list[ClientSplit]
    target_mean: np.ndarray
    target_std: np.ndarray
    similarity: np.ndarray
    history: int
    add_bias: bool

    @property
    def num_clients(self) -> int:
        return len(self.client_ids)

    @property
    def dimension(self) -> int:
        return self.train[0].x.shape[1]

    def inverse_target(self, client_index: int, values: np.ndarray) -> np.ndarray:
        return (
            np.asarray(values) * self.target_std[client_index]
            + self.target_mean[client_index]
        )


def _read_speed_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf", ".hdf5"}:
        frame = pd.read_hdf(path)
        values = frame.to_numpy(dtype=np.float64)
        client_ids = [str(column) for column in frame.columns]
    elif suffix == ".csv":
        frame = pd.read_csv(path, index_col=0)
        values = frame.to_numpy(dtype=np.float64)
        client_ids = [str(column) for column in frame.columns]
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=True)
        key = "data" if "data" in archive.files else archive.files[0]
        values = np.asarray(archive[key], dtype=np.float64)
        if values.ndim == 3:
            values = values[..., 0]
        if values.ndim != 2:
            raise ValueError(f"Expected a 2-D speed matrix, got {values.shape}.")
        if "sensor_ids" in archive.files:
            client_ids = [str(item) for item in archive["sensor_ids"].tolist()]
        else:
            client_ids = [str(i) for i in range(values.shape[1])]
    else:
        raise ValueError(f"Unsupported speed file format: {path.suffix}")

    if values.ndim != 2:
        raise ValueError(f"Expected [time, sensor] data, got {values.shape}.")
    return values, client_ids


def _read_similarity(
    adjacency_path: Path | None,
    speed_client_ids: Sequence[str],
) -> tuple[np.ndarray | None, list[str] | None]:
    if adjacency_path is None:
        return None, None
    with adjacency_path.open("rb") as handle:
        try:
            payload = pickle.load(handle, encoding="latin1")
        except TypeError:
            payload = pickle.load(handle)

    if isinstance(payload, (tuple, list)) and len(payload) >= 3:
        sensor_ids = [str(item) for item in payload[0]]
        similarity = np.asarray(payload[2], dtype=np.float64)
    elif isinstance(payload, dict):
        raw_ids = payload.get("sensor_ids", speed_client_ids)
        raw_matrix = payload.get("adj_mx", payload.get("adjacency"))
        if raw_matrix is None:
            raise ValueError("Adjacency dictionary has no 'adj_mx' or 'adjacency'.")
        sensor_ids = [str(item) for item in raw_ids]
        similarity = np.asarray(raw_matrix, dtype=np.float64)
    else:
        similarity = np.asarray(payload, dtype=np.float64)
        sensor_ids = list(speed_client_ids)

    if similarity.shape != (len(sensor_ids), len(sensor_ids)):
        raise ValueError(
            "Adjacency shape does not agree with its sensor ID list: "
            f"{similarity.shape} versus {len(sensor_ids)} IDs."
        )
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    similarity = np.maximum(similarity, similarity.T)
    np.fill_diagonal(similarity, 0.0)
    return similarity, sensor_ids


def _align_data_and_similarity(
    values: np.ndarray,
    speed_ids: list[str],
    similarity: np.ndarray | None,
    graph_ids: list[str] | None,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    if similarity is None or graph_ids is None:
        return values, speed_ids, similarity

    speed_lookup = {sensor_id: index for index, sensor_id in enumerate(speed_ids)}
    if set(graph_ids).issubset(speed_lookup):
        order = [speed_lookup[sensor_id] for sensor_id in graph_ids]
        return values[:, order], graph_ids, similarity

    if values.shape[1] != similarity.shape[0]:
        raise ValueError(
            "Could not align speed columns with adjacency sensor IDs, and their "
            "dimensions differ."
        )
    return values, speed_ids, similarity


def _fill_missing(values: np.ndarray, zero_is_missing: bool) -> np.ndarray:
    frame = pd.DataFrame(values)
    if zero_is_missing:
        frame = frame.mask(frame == 0.0)
    frame = frame.interpolate(axis=0, limit_direction="both").ffill().bfill()
    if frame.isna().any().any():
        medians = frame.median(axis=0).fillna(0.0)
        frame = frame.fillna(medians)
    return frame.to_numpy(dtype=np.float64)


def _correlation_similarity(train_values: np.ndarray) -> np.ndarray:
    similarity = np.corrcoef(train_values.T)
    similarity = np.nan_to_num(np.abs(similarity), nan=0.0)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def _connected_order(similarity: np.ndarray, count: int) -> np.ndarray:
    n = similarity.shape[0]
    if count >= n:
        return np.arange(n)

    graph = similarity > 0
    degree = graph.sum(axis=1)
    start = int(np.argmax(degree))
    selected: list[int] = []
    queue = [start]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    while queue and len(selected) < count:
        node = queue.pop(0)
        selected.append(node)
        neighbors = np.flatnonzero(graph[node])
        neighbors = neighbors[np.argsort(similarity[node, neighbors])[::-1]]
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(int(neighbor))

    if len(selected) < count:
        remaining = np.flatnonzero(~visited)
        selected.extend(remaining[: count - len(selected)].tolist())
    return np.asarray(selected, dtype=int)


def _make_windows(
    standardized: np.ndarray,
    target_indices: np.ndarray,
    history: int,
    add_bias: bool,
) -> list[ClientSplit]:
    splits: list[ClientSplit] = []
    for client in range(standardized.shape[1]):
        x = np.stack(
            [standardized[index - history : index, client] for index in target_indices],
            axis=0,
        )
        if add_bias:
            x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        y = standardized[target_indices, client]
        splits.append(ClientSplit(x=x, y=y))
    return splits


def load_metr_la(
    speed_path: str | Path,
    adjacency_path: str | Path | None,
    *,
    history: int = 12,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
    num_clients: int | None = None,
    graph_source: str = "adjacency",
    add_bias: bool = True,
    zero_is_missing: bool = True,
) -> FederatedTimeSeries:
    """Load METR-LA as one autoregressive regression client per sensor."""
    speed_path = Path(speed_path)
    adjacency = Path(adjacency_path) if adjacency_path is not None else None
    values, speed_ids = _read_speed_matrix(speed_path)
    similarity, graph_ids = _read_similarity(adjacency, speed_ids)
    values, client_ids, similarity = _align_data_and_similarity(
        values, speed_ids, similarity, graph_ids
    )
    values = _fill_missing(values, zero_is_missing)

    train_end = int(values.shape[0] * train_fraction)
    val_end = int(values.shape[0] * (train_fraction + val_fraction))
    if train_end <= history or val_end <= train_end or val_end >= values.shape[0]:
        raise ValueError("Invalid chronological split for the available time series.")

    if graph_source == "correlation":
        similarity = _correlation_similarity(values[:train_end])
    elif graph_source == "adjacency":
        if similarity is None:
            raise ValueError(
                "graph_source='adjacency' requires an adjacency pickle file."
            )
    else:
        raise ValueError("graph_source must be 'adjacency' or 'correlation'.")

    requested = values.shape[1] if num_clients is None else int(num_clients)
    if requested <= 0 or requested > values.shape[1]:
        raise ValueError("num_clients must be between 1 and the sensor count.")
    chosen = _connected_order(similarity, requested)
    values = values[:, chosen]
    similarity = similarity[np.ix_(chosen, chosen)]
    client_ids = [client_ids[index] for index in chosen]

    target_mean = values[:train_end].mean(axis=0)
    target_std = values[:train_end].std(axis=0)
    target_std = np.where(target_std < 1e-8, 1.0, target_std)
    standardized = (values - target_mean) / target_std

    train_indices = np.arange(history, train_end)
    val_indices = np.arange(train_end, val_end)
    test_indices = np.arange(val_end, values.shape[0])
    return FederatedTimeSeries(
        client_ids=client_ids,
        train=_make_windows(standardized, train_indices, history, add_bias),
        val=_make_windows(standardized, val_indices, history, add_bias),
        test=_make_windows(standardized, test_indices, history, add_bias),
        target_mean=target_mean,
        target_std=target_std,
        similarity=similarity,
        history=history,
        add_bias=add_bias,
    )
