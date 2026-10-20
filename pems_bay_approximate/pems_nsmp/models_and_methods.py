"""Shared model, data processing and paper Local/FedAvg/FedRep dependencies."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
import tables
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ExperimentConfig:
    num_sensors: int = 30
    num_clients: int = 6
    lags: int = 12
    hidden_dim: int = 16
    train_windows: int = 800
    val_windows: int = 240
    test_windows: int = 480
    rounds: int = 20
    batch_size: int = 128
    learning_rate: float = 0.003
    weight_decay: float = 1e-5
    seed: int = 20260728


@dataclass
class ClientData:
    client_id: int
    sensor_positions: np.ndarray
    sensor_ids: list[str]
    adjacency: torch.Tensor
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


class GraphTemporalNet(nn.Module):
    """Small nonlinear temporal encoder followed by one graph-convolution step."""

    def __init__(self, lags: int, hidden_dim: int) -> None:
        super().__init__()
        self.temporal = nn.Linear(lags, hidden_dim)
        self.graph = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        adapter: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [batch, lags, nodes]
        hidden = torch.relu(self.temporal(x.transpose(1, 2)))
        hidden = torch.einsum("nm,bmh->bnh", adjacency, hidden)
        hidden = self.graph(hidden)
        if adapter is not None:
            hidden = hidden + adapter.view(1, 1, -1)
        return self.head(torch.relu(hidden)).squeeze(-1)




def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def interpolate_speeds(frame: pd.DataFrame) -> np.ndarray:
    clean = frame.astype(np.float32).replace(0.0, np.nan)
    clean = clean.interpolate(axis=0, limit_direction="both").ffill().bfill()
    if clean.isna().any().any():
        raise ValueError("Speed matrix still contains missing values after interpolation")
    return clean.to_numpy(dtype=np.float32)


def read_speed_frame(speed_path: Path) -> pd.DataFrame:
    """Read pandas HDF frames, including legacy pandas 0.15 fixed-format files."""

    try:
        return pd.read_hdf(speed_path)
    except TypeError as error:
        if "bytes-like object" not in str(error):
            raise
        with tables.open_file(speed_path, mode="r") as handle:
            if not hasattr(handle.root, "speed"):
                raise ValueError(f"Legacy HDF file has no /speed group: {speed_path}") from error
            group = handle.root.speed
            columns = [str(value) for value in group.axis0.read()]
            index = pd.to_datetime(group.axis1.read())
            values = group.block0_values.read()
        return pd.DataFrame(values, index=index, columns=columns)


def load_metr_la(speed_path: Path, adjacency_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    if not speed_path.exists() or not adjacency_path.exists():
        raise FileNotFoundError(f"Missing METR-LA inputs: {speed_path}, {adjacency_path}")
    frame = read_speed_frame(speed_path)
    with adjacency_path.open("rb") as handle:
        graph_payload = pickle.load(handle, encoding="latin1")
    sensor_ids = [str(value) for value in graph_payload[0]]
    adjacency = np.asarray(graph_payload[2], dtype=np.float32)
    frame.columns = [str(value) for value in frame.columns]
    if set(frame.columns) != set(sensor_ids):
        raise ValueError("Speed-frame columns and adjacency sensor IDs differ")
    frame = frame.loc[:, sensor_ids]
    speeds = interpolate_speeds(frame)
    if speeds.shape[1] != adjacency.shape[0]:
        raise ValueError("Speed and adjacency sensor counts differ")
    return speeds, sensor_ids, adjacency


def largest_component(adjacency: np.ndarray) -> np.ndarray:
    undirected = np.maximum(adjacency, adjacency.T)
    binary = (undirected > 0).astype(np.int8)
    component_count, labels = connected_components(binary, directed=False)
    sizes = np.bincount(labels, minlength=component_count)
    return np.flatnonzero(labels == int(np.argmax(sizes)))


def connected_bfs_order(adjacency: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    if count > len(candidates):
        raise ValueError(f"Requested {count} sensors from a component of size {len(candidates)}")
    graph = np.maximum(adjacency, adjacency.T)
    degree = graph[candidates][:, candidates].sum(axis=1)
    start = int(candidates[int(np.argmax(degree))])
    allowed = set(int(value) for value in candidates)
    queue = [start]
    visited: list[int] = []
    seen = {start}
    while queue and len(visited) < count:
        node = queue.pop(0)
        visited.append(node)
        neighbors = np.flatnonzero(graph[node] > 0)
        neighbors = sorted(
            (int(value) for value in neighbors if int(value) in allowed and int(value) not in seen),
            key=lambda value: float(graph[node, value]),
            reverse=True,
        )
        seen.update(neighbors)
        queue.extend(neighbors)
    if len(visited) != count:
        raise RuntimeError("BFS did not recover the requested connected sensor subset")
    return np.asarray(visited, dtype=np.int64)


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    graph = np.maximum(adjacency, adjacency.T).astype(np.float32)
    graph = graph + np.eye(graph.shape[0], dtype=np.float32)
    degree = graph.sum(axis=1)
    inv_sqrt = np.power(np.maximum(degree, 1e-8), -0.5)
    return (inv_sqrt[:, None] * graph) * inv_sqrt[None, :]


def sampled_targets(start: int, end: int, lags: int, limit: int) -> np.ndarray:
    first = max(start, lags)
    targets = np.arange(first, end, dtype=np.int64)
    if len(targets) <= limit:
        return targets
    positions = np.linspace(0, len(targets) - 1, num=limit, dtype=np.int64)
    return targets[positions]


def build_windows(
    normalized: np.ndarray,
    targets: np.ndarray,
    lags: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.stack([normalized[target - lags : target] for target in targets], axis=0)
    y = normalized[targets]
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


def region_tree(region_series: np.ndarray) -> list[tuple[int, int, int]]:
    correlation = np.corrcoef(region_series.T)
    correlation = np.nan_to_num(correlation, nan=0.0)
    similarity = np.clip((correlation + 1.0) / 2.0, 1e-6, 1.0)
    distance = 1.0 - similarity + 1e-6
    np.fill_diagonal(distance, 0.0)
    sparse_tree = minimum_spanning_tree(distance).toarray()
    undirected_edges: list[tuple[int, int]] = []
    for row, col in zip(*np.nonzero(sparse_tree)):
        undirected_edges.append((int(row), int(col)))
    if len(undirected_edges) != region_series.shape[1] - 1:
        raise RuntimeError("Client similarity graph did not yield a spanning tree")

    neighbors: dict[int, list[int]] = {idx: [] for idx in range(region_series.shape[1])}
    for left, right in undirected_edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    parent = {0: -1}
    queue = [0]
    order = [0]
    while queue:
        current = queue.pop(0)
        for neighbor in neighbors[current]:
            if neighbor in parent:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
            order.append(neighbor)
    depth = {0: 0}
    for node in order[1:]:
        depth[node] = depth[parent[node]] + 1
    oriented: list[tuple[int, int, int]] = []
    for edge_id, child in enumerate(sorted(order[1:], key=lambda node: (depth[node], node))):
        oriented.append((edge_id, child, parent[child]))
    return oriented


def prepare_clients(
    speeds: np.ndarray,
    sensor_ids: list[str],
    adjacency: np.ndarray,
    config: ExperimentConfig,
) -> tuple[list[ClientData], np.ndarray, list[tuple[int, int, int]]]:
    selected_global = connected_bfs_order(
        adjacency,
        largest_component(adjacency),
        config.num_sensors,
    )
    selected_speeds = speeds[:, selected_global]
    selected_ids = [sensor_ids[index] for index in selected_global]
    total_steps = selected_speeds.shape[0]
    train_end = int(total_steps * 0.70)
    val_end = int(total_steps * 0.80)

    train_mean = selected_speeds[:train_end].mean(axis=0)
    train_std = selected_speeds[:train_end].std(axis=0)
    train_std = np.maximum(train_std, 1e-3)
    normalized = (selected_speeds - train_mean) / train_std

    train_targets = sampled_targets(config.lags, train_end, config.lags, config.train_windows)
    val_targets = sampled_targets(train_end, val_end, config.lags, config.val_windows)
    test_targets = sampled_targets(val_end, total_steps, config.lags, config.test_windows)
    full_train_x, full_train_y = build_windows(normalized, train_targets, config.lags)
    full_val_x, full_val_y = build_windows(normalized, val_targets, config.lags)
    full_test_x, full_test_y = build_windows(normalized, test_targets, config.lags)

    groups = [np.asarray(group, dtype=np.int64) for group in np.array_split(np.arange(config.num_sensors), config.num_clients)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Each client must own at least one sensor")

    selected_graph = np.maximum(adjacency, adjacency.T)[np.ix_(selected_global, selected_global)]
    clients: list[ClientData] = []
    for client_id, positions in enumerate(groups):
        local_graph = normalize_adjacency(selected_graph[np.ix_(positions, positions)])
        clients.append(
            ClientData(
                client_id=client_id,
                sensor_positions=positions,
                sensor_ids=[selected_ids[index] for index in positions],
                adjacency=torch.from_numpy(local_graph),
                train_x=full_train_x[:, :, positions],
                train_y=full_train_y[:, positions],
                val_x=full_val_x[:, :, positions],
                val_y=full_val_y[:, positions],
                test_x=full_test_x[:, :, positions],
                test_y=full_test_y[:, positions],
                mean=torch.from_numpy(train_mean[positions].astype(np.float32)),
                std=torch.from_numpy(train_std[positions].astype(np.float32)),
            )
        )

    region_means = np.stack(
        [selected_speeds[:train_end, group].mean(axis=1) for group in groups],
        axis=1,
    )
    tree = region_tree(region_means)
    return clients, selected_global, tree


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def train_one_epoch(
    model: GraphTemporalNet,
    client: ClientData,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    seed: int,
    adapter_parameters: Iterable[torch.Tensor] | None = None,
) -> float:
    model.train()
    adjacency = client.adjacency.to(device)
    adapter_list = list(adapter_parameters or [])
    loader = make_loader(client.train_x, client.train_y, batch_size, True, seed)
    total_loss = 0.0
    total_count = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        adapter = None
        if adapter_list:
            adapter = torch.stack(adapter_list, dim=0).mean(dim=0)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch_x, adjacency, adapter)
        loss = torch.mean((prediction - batch_y) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(batch_x)
        total_count += len(batch_x)
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_client(
    model: GraphTemporalNet,
    client: ClientData,
    split: str,
    device: torch.device,
    batch_size: int,
    adapter_parameters: Iterable[torch.Tensor] | None = None,
) -> dict[str, float]:
    model.eval()
    x = getattr(client, f"{split}_x")
    y = getattr(client, f"{split}_y")
    loader = make_loader(x, y, batch_size, False, seed=0)
    adjacency = client.adjacency.to(device)
    mean = client.mean.to(device)
    std = client.std.to(device)
    adapter_list = list(adapter_parameters or [])
    adapter = torch.stack(adapter_list, dim=0).mean(dim=0) if adapter_list else None
    squared_error = 0.0
    absolute_error = 0.0
    absolute_percentage_error = 0.0
    count = 0
    mape_count = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        prediction = model(batch_x, adjacency, adapter)
        prediction = prediction * std + mean
        target = batch_y * std + mean
        error = prediction - target
        squared_error += float(torch.sum(error**2))
        absolute_error += float(torch.sum(torch.abs(error)))
        count += error.numel()
        valid = torch.abs(target) > 1e-3
        absolute_percentage_error += float(torch.sum(torch.abs(error[valid] / target[valid])))
        mape_count += int(torch.sum(valid))
    return {
        "sse": squared_error,
        "sae": absolute_error,
        "sape": absolute_percentage_error,
        "count": float(count),
        "mape_count": float(mape_count),
    }


def aggregate_metrics(parts: list[dict[str, float]]) -> dict[str, float]:
    totals = {key: sum(part[key] for part in parts) for key in parts[0]}
    return {
        "rmse": math.sqrt(totals["sse"] / totals["count"]),
        "mae": totals["sae"] / totals["count"],
        "mape": 100.0 * totals["sape"] / totals["mape_count"],
    }


def evaluate_models(
    models: list[GraphTemporalNet],
    clients: list[ClientData],
    split: str,
    device: torch.device,
    batch_size: int,
    adapters: list[dict[int, torch.Tensor]] | None = None,
) -> dict[str, float]:
    parts = []
    for index, (model, client) in enumerate(zip(models, clients)):
        adapter_values = adapters[index].values() if adapters is not None else None
        parts.append(evaluate_client(model, client, split, device, batch_size, adapter_values))
    return aggregate_metrics(parts)


def average_states(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    result: dict[str, torch.Tensor] = {}
    for key in states[0]:
        result[key] = sum(state[key] * (weight / total) for state, weight in zip(states, weights))
    return result


def parameter_count(model: nn.Module, prefixes: tuple[str, ...] | None = None) -> int:
    if prefixes is None:
        return sum(parameter.numel() for parameter in model.parameters())
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    )


def run_local(
    clients: list[ClientData],
    config: ExperimentConfig,
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    models = [GraphTemporalNet(config.lags, config.hidden_dim).to(device) for _ in clients]
    for model in models:
        load_state(model, base_state)
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        for model in models
    ]
    best_rmse = float("inf")
    best_states = [clone_state(model) for model in models]
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for round_index in range(1, config.rounds + 1):
        for client_index, (model, client, optimizer) in enumerate(zip(models, clients, optimizers)):
            train_one_epoch(
                model,
                client,
                optimizer,
                device,
                config.batch_size,
                config.seed + 10_000 * round_index + client_index,
            )
        metrics = evaluate_models(models, clients, "val", device, config.batch_size)
        history.append({"round": round_index, **metrics})
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_states = [clone_state(model) for model in models]
    for model, state in zip(models, best_states):
        load_state(model, state)
    metrics = evaluate_models(models, clients, "test", device, config.batch_size)
    metrics.update({"communication_bytes": 0, "runtime_seconds": time.perf_counter() - start})
    return metrics, history


def run_fedavg(
    clients: list[ClientData],
    config: ExperimentConfig,
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    global_state = copy.deepcopy(base_state)
    best_state = copy.deepcopy(global_state)
    best_rmse = float("inf")
    history: list[dict[str, float]] = []
    sample_weights = [float(len(client.train_x)) for client in clients]
    start = time.perf_counter()
    for round_index in range(1, config.rounds + 1):
        local_states = []
        for client_index, client in enumerate(clients):
            model = GraphTemporalNet(config.lags, config.hidden_dim).to(device)
            load_state(model, global_state)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
            train_one_epoch(
                model,
                client,
                optimizer,
                device,
                config.batch_size,
                config.seed + 20_000 * round_index + client_index,
            )
            local_states.append(clone_state(model))
        global_state = average_states(local_states, sample_weights)
        models = [GraphTemporalNet(config.lags, config.hidden_dim).to(device) for _ in clients]
        for model in models:
            load_state(model, global_state)
        metrics = evaluate_models(models, clients, "val", device, config.batch_size)
        history.append({"round": round_index, **metrics})
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_state = copy.deepcopy(global_state)
    models = [GraphTemporalNet(config.lags, config.hidden_dim).to(device) for _ in clients]
    for model in models:
        load_state(model, best_state)
    metrics = evaluate_models(models, clients, "test", device, config.batch_size)
    model_parameters = parameter_count(models[0])
    metrics.update(
        {
            "communication_bytes": 2 * len(clients) * model_parameters * 4 * config.rounds,
            "runtime_seconds": time.perf_counter() - start,
        }
    )
    return metrics, history


def run_fedrep(
    clients: list[ClientData],
    config: ExperimentConfig,
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    models = [GraphTemporalNet(config.lags, config.hidden_dim).to(device) for _ in clients]
    for model in models:
        load_state(model, base_state)
    shared_keys = [key for key in base_state if key.startswith(("temporal.", "graph."))]
    shared_state = {key: base_state[key].clone() for key in shared_keys}
    best_states = [clone_state(model) for model in models]
    best_rmse = float("inf")
    history: list[dict[str, float]] = []
    sample_weights = [float(len(client.train_x)) for client in clients]
    start = time.perf_counter()
    for round_index in range(1, config.rounds + 1):
        client_shared_states = []
        for client_index, (model, client) in enumerate(zip(models, clients)):
            state = model.state_dict()
            for key, value in shared_state.items():
                state[key].copy_(value.to(state[key].device))
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
            train_one_epoch(
                model,
                client,
                optimizer,
                device,
                config.batch_size,
                config.seed + 30_000 * round_index + client_index,
            )
            current = clone_state(model)
            client_shared_states.append({key: current[key] for key in shared_keys})
        shared_state = average_states(client_shared_states, sample_weights)
        for model in models:
            state = model.state_dict()
            for key, value in shared_state.items():
                state[key].copy_(value.to(state[key].device))
        metrics = evaluate_models(models, clients, "val", device, config.batch_size)
        history.append({"round": round_index, **metrics})
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_states = [clone_state(model) for model in models]
    for model, state in zip(models, best_states):
        load_state(model, state)
    metrics = evaluate_models(models, clients, "test", device, config.batch_size)
    shared_parameters = parameter_count(models[0], prefixes=("temporal.", "graph."))
    metrics.update(
        {
            "communication_bytes": 2 * len(clients) * shared_parameters * 4 * config.rounds,
            "runtime_seconds": time.perf_counter() - start,
        }
    )
    return metrics, history


def incident_edges(
    client_count: int,
    tree: list[tuple[int, int, int]],
) -> list[list[int]]:
    result = [[] for _ in range(client_count)]
    for edge_id, child, parent in tree:
        result[child].append(edge_id)
        result[parent].append(edge_id)
    return result














