"""Approximate NSMP for a non-convex graph-temporal model on PEMS-BAY.

Each tree edge owns a low-dimensional shared parameter.  A non-root client
samples its parent interface, approximately minimizes its local/subtree
objective over private model parameters and child interfaces, and trains an
constant, linear, quadratic, and MLP candidates to represent the resulting
reduced message.  Validation selects the smallest near-best representation.
The root then minimizes the assembled approximate objective and a downward pass
recovers all client models.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .models_and_methods import (
    ClientData,
    ExperimentConfig,
    GraphTemporalNet,
    aggregate_metrics,
    clone_state,
    load_metr_la,
    load_state,
    make_loader,
    prepare_clients,
    resolve_device,
    run_fedavg,
    run_fedrep,
    run_local,
    set_seed,
    train_one_epoch,
)


@dataclass(frozen=True)
class ApproxMessageConfig:
    interface_dim: int = 2
    interface_radius: float = 1.5
    message_hidden: int = 32
    message_samples: int = 48
    validation_samples: int = 16
    holdout_samples: int = 16
    inner_steps: int = 35
    recovery_steps: int = 120
    root_steps: int = 160
    multi_starts: int = 2
    message_epochs: int = 600
    inner_learning_rate: float = 0.01
    message_learning_rate: float = 0.01
    pretrain_epochs: int = 20
    private_regularization: float = 1e-5
    interface_regularization: float = 1e-3
    selection_tolerance_fraction: float = 0.10


class OverlapGraphTemporalNet(GraphTemporalNet):
    """Private graph-temporal model augmented by incident edge parameters."""

    def forward_with_edges(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        edge_values: dict[int, torch.Tensor],
        edge_bases: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        hidden = torch.relu(self.temporal(x.transpose(1, 2)))
        hidden = self.graph(torch.einsum("nm,bmh->bnh", adjacency, hidden))
        activated = torch.relu(hidden)
        prediction = self.head(activated).squeeze(-1)
        if edge_values:
            shared_terms = []
            for edge_id, value in edge_values.items():
                basis = edge_bases[edge_id]
                shared_features = torch.einsum("bnh,hd->bnd", activated, basis)
                shared_terms.append(torch.einsum("bnd,d->bn", shared_features, value))
            prediction = prediction + torch.stack(shared_terms).sum(dim=0) / math.sqrt(
                len(shared_terms)
            )
        return prediction


class MessageMLP(nn.Module):
    """Scalar reduced-message approximation with stored input/output scaling."""

    def __init__(self, input_dim: int, hidden_dim: int, radius: float) -> None:
        super().__init__()
        self.radius = float(radius)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("output_mean", torch.zeros(()))
        self.register_buffer("output_scale", torch.ones(()))

    def normalized_prediction(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z / self.radius).squeeze(-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.output_mean + self.output_scale * self.normalized_prediction(z)

    def set_output_scaling(self, targets: torch.Tensor) -> None:
        with torch.no_grad():
            self.output_mean.copy_(targets.mean())
            self.output_scale.copy_(targets.std(unbiased=False).clamp_min(1e-5))

    def payload_values(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()) + 2


def polynomial_features(z: torch.Tensor, degree: int, radius: float) -> torch.Tensor:
    """Constant/linear/quadratic features on a normalized interface domain."""

    normalized = z / radius
    features = [torch.ones((*normalized.shape[:-1], 1), device=z.device, dtype=z.dtype)]
    if degree >= 1:
        features.append(normalized)
    if degree >= 2:
        quadratic = []
        for left in range(normalized.shape[-1]):
            for right in range(left, normalized.shape[-1]):
                quadratic.append((normalized[..., left] * normalized[..., right]).unsqueeze(-1))
        features.append(torch.cat(quadratic, dim=-1))
    if degree not in (0, 1, 2):
        raise ValueError("Only constant, linear, and quadratic messages are supported")
    return torch.cat(features, dim=-1)


class PolynomialMessage(nn.Module):
    """Closed-form fitted constant, linear, or quadratic reduced message."""

    def __init__(
        self,
        input_dim: int,
        degree: int,
        radius: float,
        coefficients: torch.Tensor,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.degree = int(degree)
        self.radius = float(radius)
        self.register_buffer("coefficients", coefficients.detach().clone())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return polynomial_features(z, self.degree, self.radius) @ self.coefficients

    def payload_values(self) -> int:
        return self.coefficients.numel()

    @property
    def representation_name(self) -> str:
        return {0: "constant", 1: "linear", 2: "quadratic"}[self.degree]


@dataclass
class MinimizeResult:
    objective: float
    model_state: dict[str, torch.Tensor]
    child_interfaces: dict[int, torch.Tensor]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[1]
    shared_data = here / "data" / "pems_bay"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed-data", type=Path, default=shared_data / "pems-bay.h5")
    parser.add_argument("--adjacency", type=Path, default=shared_data / "adj_mx_bay.pkl")
    parser.add_argument("--dataset-name", type=str, default="PEMS-BAY")
    parser.add_argument("--output", type=Path, default=here / "results" / "latest" / "pems_bay_6clients_128_hidden4")
    parser.add_argument("--num-sensors", type=int, default=30)
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=140)
    parser.add_argument("--train-windows", type=int, default=128)
    parser.add_argument("--val-windows", type=int, default=240)
    parser.add_argument("--test-windows", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--interface-dim", type=int, default=2)
    parser.add_argument("--interface-radius", type=float, default=1.0)
    parser.add_argument("--message-hidden", type=int, default=32)
    parser.add_argument("--message-samples", type=int, default=48)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--holdout-samples", type=int, default=16)
    parser.add_argument("--inner-steps", type=int, default=35)
    parser.add_argument("--message-epochs", type=int, default=600)
    parser.add_argument("--multi-starts", type=int, default=2)
    parser.add_argument("--recovery-steps", type=int, default=120)
    parser.add_argument("--root-steps", type=int, default=120)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--inner-learning-rate", type=float, default=0.01)
    parser.add_argument("--message-learning-rate", type=float, default=0.01)
    parser.add_argument("--private-regularization", type=float, default=1e-5)
    parser.add_argument("--interface-regularization", type=float, default=0.01)
    parser.add_argument("--selection-tolerance-fraction", type=float, default=0.10)
    return parser.parse_args()


def tree_structure(
    client_count: int,
    tree: list[tuple[int, int, int]],
) -> tuple[dict[int, int], dict[int, int], dict[int, list[int]], dict[int, int]]:
    parent: dict[int, int] = {}
    parent_edge: dict[int, int] = {}
    children = {node: [] for node in range(client_count)}
    edge_for_child: dict[int, int] = {}
    for edge_id, child, parent_node in tree:
        parent[child] = parent_node
        parent_edge[child] = edge_id
        edge_for_child[child] = edge_id
        children[parent_node].append(child)
    if len(parent) != client_count - 1:
        raise ValueError("Tree must contain exactly one parent for each non-root node")
    return parent, parent_edge, children, edge_for_child


def tree_depths(children: dict[int, list[int]], root: int = 0) -> dict[int, int]:
    depths = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in children[node]:
            depths[child] = depths[node] + 1
            queue.append(child)
    if len(depths) != len(children):
        raise ValueError("Client tree is disconnected")
    return depths


def make_edge_bases(
    tree: list[tuple[int, int, int]],
    hidden_dim: int,
    interface_dim: int,
    seed: int,
    device: torch.device,
) -> list[dict[int, torch.Tensor]]:
    client_count = len(tree) + 1
    result: list[dict[int, torch.Tensor]] = [{} for _ in range(client_count)]
    generator = torch.Generator().manual_seed(seed + 811)
    for edge_id, child, parent in tree:
        raw = torch.randn(hidden_dim, interface_dim, generator=generator)
        basis = torch.linalg.qr(raw, mode="reduced").Q.to(device)
        result[child][edge_id] = basis
        result[parent][edge_id] = basis
    return result


def latin_hypercube(
    count: int,
    dimension: int,
    radius: float,
    seed: int,
) -> torch.Tensor:
    if count < 2:
        raise ValueError("At least two interface samples are required")
    rng = np.random.default_rng(seed)
    values = np.empty((count, dimension), dtype=np.float32)
    for column in range(dimension):
        bins = (np.arange(count, dtype=np.float32) + rng.random(count)) / count
        rng.shuffle(bins)
        values[:, column] = 2.0 * radius * bins - radius
    return torch.from_numpy(values)


def constrained_interface(raw: torch.Tensor, radius: float) -> torch.Tensor:
    return radius * torch.tanh(raw)


def model_objective(
    model: OverlapGraphTemporalNet,
    client: ClientData,
    edge_values: dict[int, torch.Tensor],
    edge_bases: dict[int, torch.Tensor],
    child_messages: dict[int, nn.Module],
    config: ApproxMessageConfig,
    device: torch.device,
) -> torch.Tensor:
    prediction = model.forward_with_edges(
        client.train_x.to(device),
        client.adjacency.to(device),
        edge_values,
        edge_bases,
    )
    data_loss = torch.mean((prediction - client.train_y.to(device)) ** 2)
    private_penalty = sum(torch.sum(parameter**2) for parameter in model.parameters())
    interface_penalty = sum(torch.sum(value**2) for value in edge_values.values())
    message_value = torch.zeros((), device=device)
    for child, message in child_messages.items():
        message_value = message_value + message(edge_values[child])
    return (
        data_loss
        + 0.5 * config.private_regularization * private_penalty
        + 0.5 * config.interface_regularization * interface_penalty
        + message_value
    )


def jitter_model(model: nn.Module, scale: float, generator: torch.Generator) -> None:
    if scale <= 0:
        return
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            parameter.add_(scale * noise.to(parameter.device))


def approximate_minimize(
    node: int,
    fixed_parent_interface: torch.Tensor | None,
    child_nodes: list[int],
    child_edge_ids: dict[int, int],
    child_messages: dict[int, nn.Module],
    clients: list[ClientData],
    edge_bases: list[dict[int, torch.Tensor]],
    initial_states: list[dict[str, torch.Tensor]],
    experiment: ExperimentConfig,
    message_config: ApproxMessageConfig,
    device: torch.device,
    steps: int,
    starts: int,
    seed: int,
    parent_edge_id: int | None,
) -> MinimizeResult:
    best: MinimizeResult | None = None
    for start_index in range(starts):
        model = OverlapGraphTemporalNet(experiment.lags, experiment.hidden_dim).to(device)
        load_state(model, initial_states[node])
        generator = torch.Generator().manual_seed(seed + 1009 * start_index)
        jitter_model(model, 0.015 * start_index, generator)
        raw_children = {
            child: nn.Parameter(
                0.25
                * torch.randn(
                    message_config.interface_dim,
                    generator=generator,
                    device="cpu",
                ).to(device)
            )
            for child in child_nodes
        }
        parameters = list(model.parameters()) + list(raw_children.values())
        optimizer = torch.optim.Adam(parameters, lr=message_config.inner_learning_rate)
        fixed = None if fixed_parent_interface is None else fixed_parent_interface.to(device)
        for _ in range(steps):
            edge_values: dict[int, torch.Tensor] = {}
            if parent_edge_id is not None:
                if fixed is None:
                    raise ValueError("A non-root node requires a fixed parent interface")
                edge_values[parent_edge_id] = fixed
            for child, raw in raw_children.items():
                edge_values[child_edge_ids[child]] = constrained_interface(
                    raw, message_config.interface_radius
                )
            optimizer.zero_grad(set_to_none=True)
            objective = model_objective(
                model,
                clients[node],
                edge_values,
                edge_bases[node],
                {
                    child_edge_ids[child]: child_messages[child]
                    for child in child_nodes
                },
                message_config,
                device,
            )
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            optimizer.step()

        with torch.no_grad():
            edge_values = {}
            if parent_edge_id is not None and fixed is not None:
                edge_values[parent_edge_id] = fixed
            recovered = {}
            for child, raw in raw_children.items():
                value = constrained_interface(raw, message_config.interface_radius)
                edge_values[child_edge_ids[child]] = value
                recovered[child] = value.detach().cpu().clone()
            final_objective = float(
                model_objective(
                    model,
                    clients[node],
                    edge_values,
                    edge_bases[node],
                    {
                        child_edge_ids[child]: child_messages[child]
                        for child in child_nodes
                    },
                    message_config,
                    device,
                ).detach()
            )
        candidate = MinimizeResult(final_objective, clone_state(model), recovered)
        if best is None or candidate.objective < best.objective:
            best = candidate
    if best is None:
        raise RuntimeError("Approximate minimization produced no candidate")
    return best


def fit_message(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    holdout_z: torch.Tensor,
    holdout_y: torch.Tensor,
    config: ApproxMessageConfig,
    device: torch.device,
    seed: int,
) -> tuple[MessageMLP, dict[str, float], list[dict[str, float]]]:
    set_seed(seed)
    message = MessageMLP(config.interface_dim, config.message_hidden, config.interface_radius).to(
        device
    )
    train_z = train_z.to(device)
    train_y = train_y.to(device)
    holdout_z = holdout_z.to(device)
    holdout_y = holdout_y.to(device)
    message.set_output_scaling(train_y)
    normalized_targets = (train_y - message.output_mean) / message.output_scale
    optimizer = torch.optim.Adam(message.parameters(), lr=config.message_learning_rate)
    best_state = copy.deepcopy(message.state_dict())
    best_holdout = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, config.message_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((message.normalized_prediction(train_z) - normalized_targets) ** 2)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == config.message_epochs:
            with torch.no_grad():
                train_rmse = float(torch.sqrt(torch.mean((message(train_z) - train_y) ** 2)))
                holdout_rmse = float(
                    torch.sqrt(torch.mean((message(holdout_z) - holdout_y) ** 2))
                )
            history.append(
                {"epoch": epoch, "train_rmse": train_rmse, "holdout_rmse": holdout_rmse}
            )
            if holdout_rmse < best_holdout:
                best_holdout = holdout_rmse
                best_state = copy.deepcopy(message.state_dict())
    message.load_state_dict(best_state)
    for parameter in message.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        train_prediction = message(train_z)
        holdout_prediction = message(holdout_z)
        label_std = float(train_y.std(unbiased=False))
        diagnostics = {
            "train_rmse": float(torch.sqrt(torch.mean((train_prediction - train_y) ** 2))),
            "holdout_rmse": float(torch.sqrt(torch.mean((holdout_prediction - holdout_y) ** 2))),
            "holdout_max_abs_error": float(torch.max(torch.abs(holdout_prediction - holdout_y))),
            "label_mean": float(train_y.mean()),
            "label_std": label_std,
            "normalized_holdout_rmse": best_holdout / max(label_std, 1e-8),
        }
    return message, diagnostics, history


def fit_polynomial_message(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    degree: int,
    config: ApproxMessageConfig,
    device: torch.device,
) -> PolynomialMessage:
    design = polynomial_features(
        train_z.to(torch.float64), degree, config.interface_radius
    )
    coefficients = torch.linalg.lstsq(design, train_y.to(torch.float64)).solution.to(
        torch.float32
    )
    return PolynomialMessage(
        config.interface_dim,
        degree,
        config.interface_radius,
        coefficients,
    ).to(device)


def representation_name(message: nn.Module) -> str:
    if isinstance(message, PolynomialMessage):
        return message.representation_name
    if isinstance(message, MessageMLP):
        return "mlp"
    raise TypeError(f"Unknown message representation: {type(message).__name__}")


def fit_adaptive_message(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    validation_z: torch.Tensor,
    validation_y: torch.Tensor,
    holdout_z: torch.Tensor,
    holdout_y: torch.Tensor,
    config: ApproxMessageConfig,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, dict[str, float | str], list[dict[str, float]]]:
    """Select the smallest message whose validation error is near the best."""

    candidates: dict[str, nn.Module] = {}
    for degree, name in ((0, "constant"), (1, "linear"), (2, "quadratic")):
        candidates[name] = fit_polynomial_message(train_z, train_y, degree, config, device)
    mlp, _, history = fit_message(
        train_z,
        train_y,
        validation_z,
        validation_y,
        config,
        device,
        seed,
    )
    candidates["mlp"] = mlp

    validation_z_device = validation_z.to(device)
    validation_y_device = validation_y.to(device)
    validation_errors = {}
    with torch.no_grad():
        for name, candidate in candidates.items():
            validation_errors[name] = float(
                torch.sqrt(
                    torch.mean((candidate(validation_z_device) - validation_y_device) ** 2)
                )
            )
    best_validation = min(validation_errors.values())
    label_std = float(train_y.std(unbiased=False))
    tolerance = config.selection_tolerance_fraction * max(label_std, 1e-5)
    eligible = [
        name
        for name, error in validation_errors.items()
        if error <= best_validation + tolerance
    ]
    selected_name = min(
        eligible,
        key=lambda name: (candidates[name].payload_values(), validation_errors[name]),
    )
    selected = candidates[selected_name]
    for parameter in selected.parameters():
        parameter.requires_grad_(False)

    train_z_device = train_z.to(device)
    train_y_device = train_y.to(device)
    holdout_z_device = holdout_z.to(device)
    holdout_y_device = holdout_y.to(device)
    with torch.no_grad():
        train_prediction = selected(train_z_device)
        holdout_prediction = selected(holdout_z_device)
        holdout_rmse = float(
            torch.sqrt(torch.mean((holdout_prediction - holdout_y_device) ** 2))
        )
        diagnostics: dict[str, float | str] = {
            "selected_model": selected_name,
            "selection_tolerance": tolerance,
            "best_validation_rmse": best_validation,
            "selected_validation_rmse": validation_errors[selected_name],
            "constant_validation_rmse": validation_errors["constant"],
            "linear_validation_rmse": validation_errors["linear"],
            "quadratic_validation_rmse": validation_errors["quadratic"],
            "mlp_validation_rmse": validation_errors["mlp"],
            "train_rmse": float(
                torch.sqrt(torch.mean((train_prediction - train_y_device) ** 2))
            ),
            "holdout_rmse": holdout_rmse,
            "holdout_max_abs_error": float(
                torch.max(torch.abs(holdout_prediction - holdout_y_device))
            ),
            "label_mean": float(train_y_device.mean()),
            "label_std": label_std,
            "normalized_holdout_rmse": holdout_rmse / max(label_std, 1e-8),
        }
    return selected, diagnostics, history


def pretrain_private_models(
    clients: list[ClientData],
    experiment: ExperimentConfig,
    base_state: dict[str, torch.Tensor],
    message_config: ApproxMessageConfig,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    states = []
    for client_index, client in enumerate(clients):
        model = GraphTemporalNet(experiment.lags, experiment.hidden_dim).to(device)
        load_state(model, base_state)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=experiment.learning_rate,
            weight_decay=experiment.weight_decay,
        )
        for epoch in range(message_config.pretrain_epochs):
            train_one_epoch(
                model,
                client,
                optimizer,
                device,
                experiment.batch_size,
                experiment.seed + 50_000 * (epoch + 1) + client_index,
            )
        states.append(clone_state(model))
    return states


@torch.no_grad()
def evaluate_overlap_models(
    models: list[OverlapGraphTemporalNet],
    clients: list[ClientData],
    edge_values: dict[int, torch.Tensor],
    edge_bases: list[dict[int, torch.Tensor]],
    split: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    parts = []
    for node, (model, client) in enumerate(zip(models, clients)):
        model.eval()
        x = getattr(client, f"{split}_x")
        y = getattr(client, f"{split}_y")
        loader = make_loader(x, y, batch_size, False, seed=0)
        adjacency = client.adjacency.to(device)
        mean = client.mean.to(device)
        std = client.std.to(device)
        local_edges = {
            edge_id: edge_values[edge_id].to(device) for edge_id in edge_bases[node]
        }
        sse = sae = sape = 0.0
        count = mape_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model.forward_with_edges(
                batch_x, adjacency, local_edges, edge_bases[node]
            )
            prediction = prediction * std + mean
            target = batch_y * std + mean
            error = prediction - target
            sse += float(torch.sum(error**2))
            sae += float(torch.sum(torch.abs(error)))
            count += error.numel()
            valid = torch.abs(target) > 1e-3
            sape += float(torch.sum(torch.abs(error[valid] / target[valid])))
            mape_count += int(torch.sum(valid))
        parts.append(
            {
                "sse": sse,
                "sae": sae,
                "sape": sape,
                "count": float(count),
                "mape_count": float(mape_count),
            }
        )
    return aggregate_metrics(parts)


def refit_private_models_with_fixed_interfaces(
    clients: list[ClientData],
    edge_values: dict[int, torch.Tensor],
    edge_bases: list[dict[int, torch.Tensor]],
    initial_states: list[dict[str, torch.Tensor]],
    experiment: ExperimentConfig,
    message_config: ApproxMessageConfig,
    device: torch.device,
) -> list[OverlapGraphTemporalNet]:
    """Recover private variables with fixed interfaces and validation selection."""

    recovered_models: list[OverlapGraphTemporalNet] = []
    for node, client in enumerate(clients):
        model = OverlapGraphTemporalNet(experiment.lags, experiment.hidden_dim).to(device)
        load_state(model, initial_states[node])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=experiment.learning_rate,
            weight_decay=experiment.weight_decay,
        )
        local_edges = {
            edge_id: edge_values[edge_id].to(device) for edge_id in edge_bases[node]
        }
        best_state = clone_state(model)
        best_validation_rmse = float("inf")
        for _ in range(message_config.recovery_steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_with_edges(
                client.train_x.to(device),
                client.adjacency.to(device),
                local_edges,
                edge_bases[node],
            )
            loss = torch.mean((prediction - client.train_y.to(device)) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            validation = evaluate_overlap_models(
                [model],
                [client],
                edge_values,
                [edge_bases[node]],
                "val",
                device,
                experiment.batch_size,
            )
            if validation["rmse"] < best_validation_rmse:
                best_validation_rmse = validation["rmse"]
                best_state = clone_state(model)
        load_state(model, best_state)
        recovered_models.append(model)
    return recovered_models


def run_approx_mlp_nsmp(
    clients: list[ClientData],
    tree: list[tuple[int, int, int]],
    experiment: ExperimentConfig,
    message_config: ApproxMessageConfig,
    base_state: dict[str, torch.Tensor],
    device: torch.device,
    output: Path,
) -> tuple[dict[str, float], list[dict[str, float | str]], dict[int, torch.Tensor]]:
    start_time = time.perf_counter()
    parent, parent_edge, children, child_edge = tree_structure(len(clients), tree)
    depths = tree_depths(children)
    edge_bases = make_edge_bases(
        tree,
        experiment.hidden_dim,
        message_config.interface_dim,
        experiment.seed,
        device,
    )
    print("  Pretraining private variables locally...")
    initial_states = pretrain_private_models(
        clients, experiment, base_state, message_config, device
    )
    messages: dict[int, nn.Module] = {}
    fit_rows: list[dict[str, float | str]] = []
    prediction_rows: list[dict[str, float]] = []
    message_history_rows: list[dict[str, float]] = []
    postorder = sorted(parent, key=lambda node: depths[node], reverse=True)
    for node in postorder:
        print(
            f"  Building message {node}->{parent[node]} "
            f"(children={len(children[node])})..."
        )
        total_samples = (
            message_config.message_samples
            + message_config.validation_samples
            + message_config.holdout_samples
        )
        samples = latin_hypercube(
            total_samples,
            message_config.interface_dim,
            message_config.interface_radius,
            experiment.seed + 100_000 + node,
        )
        targets = []
        for sample_index, interface in enumerate(samples):
            result = approximate_minimize(
                node=node,
                fixed_parent_interface=interface,
                child_nodes=children[node],
                child_edge_ids=child_edge,
                child_messages=messages,
                clients=clients,
                edge_bases=edge_bases,
                initial_states=initial_states,
                experiment=experiment,
                message_config=message_config,
                device=device,
                steps=message_config.inner_steps,
                starts=message_config.multi_starts,
                seed=experiment.seed + 1_000_000 * node + sample_index,
                parent_edge_id=parent_edge[node],
            )
            targets.append(result.objective)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        train_z = samples[: message_config.message_samples]
        train_y = targets_tensor[: message_config.message_samples]
        validation_end = message_config.message_samples + message_config.validation_samples
        validation_z = samples[message_config.message_samples : validation_end]
        validation_y = targets_tensor[message_config.message_samples : validation_end]
        holdout_z = samples[validation_end:]
        holdout_y = targets_tensor[validation_end:]
        message, diagnostics, fit_history = fit_adaptive_message(
            train_z,
            train_y,
            validation_z,
            validation_y,
            holdout_z,
            holdout_y,
            message_config,
            device,
            experiment.seed + 2_000_000 + node,
        )
        messages[node] = message
        fit_rows.append(
            {
                "node": node,
                "parent": parent[node],
                "edge_id": parent_edge[node],
                "train_samples": message_config.message_samples,
                "validation_samples": message_config.validation_samples,
                "holdout_samples": message_config.holdout_samples,
                "payload_values": message.payload_values(),
                **diagnostics,
            }
        )
        for split, z_values, y_values in (
            ("train", train_z, train_y),
            ("validation", validation_z, validation_y),
            ("holdout", holdout_z, holdout_y),
        ):
            with torch.no_grad():
                predictions = message(z_values.to(device)).cpu()
            for row_index, (z_value, target, prediction) in enumerate(
                zip(z_values, y_values, predictions)
            ):
                row = {
                    "node": node,
                    "split": split,
                    "sample": row_index,
                    "target": float(target),
                    "prediction": float(prediction),
                }
                row.update(
                    {f"z{dimension}": float(value) for dimension, value in enumerate(z_value)}
                )
                prediction_rows.append(row)
        for history_row in fit_history:
            message_history_rows.append({"node": node, **history_row})

    print("  Minimizing the approximate root objective...")
    root_result = approximate_minimize(
        node=0,
        fixed_parent_interface=None,
        child_nodes=children[0],
        child_edge_ids=child_edge,
        child_messages=messages,
        clients=clients,
        edge_bases=edge_bases,
        initial_states=initial_states,
        experiment=experiment,
        message_config=message_config,
        device=device,
        steps=message_config.root_steps,
        starts=max(message_config.multi_starts, 3),
        seed=experiment.seed + 3_000_000,
        parent_edge_id=None,
    )
    recovered_states: dict[int, dict[str, torch.Tensor]] = {0: root_result.model_state}
    recovered_edges: dict[int, torch.Tensor] = {
        child_edge[child]: value for child, value in root_result.child_interfaces.items()
    }
    preorder = sorted(parent, key=lambda node: depths[node])
    for node in preorder:
        result = approximate_minimize(
            node=node,
            fixed_parent_interface=recovered_edges[parent_edge[node]],
            child_nodes=children[node],
            child_edge_ids=child_edge,
            child_messages=messages,
            clients=clients,
            edge_bases=edge_bases,
            initial_states=initial_states,
            experiment=experiment,
            message_config=message_config,
            device=device,
            steps=message_config.recovery_steps,
            starts=max(message_config.multi_starts, 3),
            seed=experiment.seed + 4_000_000 + node,
            parent_edge_id=parent_edge[node],
        )
        recovered_states[node] = result.model_state
        for child, value in result.child_interfaces.items():
            recovered_edges[child_edge[child]] = value

    models = refit_private_models_with_fixed_interfaces(
        clients,
        recovered_edges,
        edge_bases,
        initial_states,
        experiment,
        message_config,
        device,
    )
    metrics = evaluate_overlap_models(
        models,
        clients,
        recovered_edges,
        edge_bases,
        "test",
        device,
        experiment.batch_size,
    )
    upward_values = sum(message.payload_values() for message in messages.values())
    downward_values = len(tree) * message_config.interface_dim
    metrics.update(
        {
            "communication_bytes": 4 * (upward_values + downward_values),
            "runtime_seconds": time.perf_counter() - start_time,
        }
    )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "message_fit_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_rows[0]))
        writer.writeheader()
        writer.writerows(fit_rows)
    prediction_fields = list(prediction_rows[0])
    with (output / "message_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (output / "message_training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(message_history_rows[0]))
        writer.writeheader()
        writer.writerows(message_history_rows)
    with (output / "recovered_interfaces.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["edge_id"] + [f"z{index}" for index in range(message_config.interface_dim)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for edge_id, value in sorted(recovered_edges.items()):
            row = {"edge_id": edge_id}
            row.update({f"z{index}": float(item) for index, item in enumerate(value)})
            writer.writerow(row)
    with (output / "per_client_test_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["client_id", "rmse", "mae", "mape", "sensor_count"]
        )
        writer.writeheader()
        for node, (model, client) in enumerate(zip(models, clients)):
            client_metrics = evaluate_overlap_models(
                [model],
                [client],
                recovered_edges,
                [edge_bases[node]],
                "test",
                device,
                experiment.batch_size,
            )
            writer.writerow(
                {
                    "client_id": node,
                    **client_metrics,
                    "sensor_count": len(client.sensor_positions),
                }
            )
    torch.save(
        {
            "types": {node: representation_name(message) for node, message in messages.items()},
            "states": {
                node: {
                    key: value.detach().cpu() for key, value in message.state_dict().items()
                }
                for node, message in messages.items()
            },
        },
        output / "message_models.pt",
    )
    return metrics, fit_rows, recovered_edges


def save_comparison(
    output: Path,
    experiment: ExperimentConfig,
    message_config: ApproxMessageConfig,
    device: torch.device,
    tree: list[tuple[int, int, int]],
    results: dict[str, dict[str, float]],
    histories: dict[str, list[dict[str, float]]],
    dataset_name: str,
    speed_data: Path,
    adjacency_data: Path,
) -> None:
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "rmse", "mae", "mape", "communication_bytes", "runtime_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, metrics in results.items():
            writer.writerow({"method": method, **metrics})
    with (output / "baseline_training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "round", "rmse", "mae", "mape"])
        writer.writeheader()
        for method, rows in histories.items():
            for row in rows:
                writer.writerow({"method": method, **row})
    metadata = {
        "experiment": asdict(experiment),
        "approximate_message": asdict(message_config),
        "device": str(device),
        "torch_version": torch.__version__,
        "tree_edges": len(tree),
        "dataset_name": dataset_name,
        "speed_data": Path(os.path.relpath(speed_data.resolve(), Path(__file__).resolve().parents[1])).as_posix(),
        "adjacency_data": Path(os.path.relpath(adjacency_data.resolve(), Path(__file__).resolve().parents[1])).as_posix(),
        "method_note": (
            "AdaptiveMsg-NSMP follows the paper procedure: sampled interfaces, "
            "multi-start gradient-based approximate partial minimization, node-specific "
            "message regression, validation-based selection among constant/linear/"
            "quadratic/MLP representations, root minimization, and downward recovery."
        ),
        "communication_note": (
            "Float32 payload only. AdaptiveMsg-NSMP counts one upward transmission of each "
            "selected message state and one downward interface vector per tree edge."
        ),
        "private_recovery_note": (
            "After the downward pass fixes all interfaces, each client's private model is "
            "re-optimized from the local-pretraining state for recovery_steps updates; the "
            "checkpoint is selected using only that client's validation data."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    methods = list(results)
    rmse = [results[method]["rmse"] for method in methods]
    communication = [results[method]["communication_bytes"] / 1e6 for method in methods]
    colors = ("#777777", "#4C78A8", "#F58518", "#54A24B")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(methods, rmse, color=colors)
    axes[0].set_ylim(0, 1.15 * max(rmse))
    axes[0].set_ylabel("Test RMSE")
    axes[0].set_title(f"{dataset_name} predictive performance")
    axes[1].bar(methods, communication, color=colors)
    axes[1].set_ylabel("Communication (MB, Float32 payload)")
    axes[1].set_title("Communicated numerical payload")
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    experiment = ExperimentConfig(
        num_sensors=args.num_sensors,
        num_clients=args.num_clients,
        hidden_dim=args.hidden_dim,
        train_windows=args.train_windows,
        val_windows=args.val_windows,
        test_windows=args.test_windows,
        rounds=args.rounds,
        seed=args.seed,
    )
    message_config = ApproxMessageConfig(
        interface_dim=args.interface_dim,
        interface_radius=args.interface_radius,
        message_hidden=args.message_hidden,
        message_samples=args.message_samples,
        validation_samples=args.validation_samples,
        holdout_samples=args.holdout_samples,
        inner_steps=args.inner_steps,
        message_epochs=args.message_epochs,
        multi_starts=args.multi_starts,
        recovery_steps=args.recovery_steps,
        root_steps=args.root_steps,
        pretrain_epochs=args.pretrain_epochs,
        inner_learning_rate=args.inner_learning_rate,
        message_learning_rate=args.message_learning_rate,
        private_regularization=args.private_regularization,
        interface_regularization=args.interface_regularization,
        selection_tolerance_fraction=args.selection_tolerance_fraction,
    )
    if experiment.num_sensors < experiment.num_clients:
        raise ValueError("num_sensors must be at least num_clients")
    set_seed(experiment.seed)
    device = resolve_device(args.device)
    print(f"Loading {args.dataset_name} on {device}...")
    speeds, sensor_ids, adjacency = load_metr_la(args.speed_data, args.adjacency)
    clients, _, tree = prepare_clients(speeds, sensor_ids, adjacency, experiment)
    base_model = GraphTemporalNet(experiment.lags, experiment.hidden_dim)
    base_state = clone_state(base_model)

    results: dict[str, dict[str, float]] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    baseline_runners = (
        ("Local", lambda: run_local(clients, experiment, base_state, device)),
        ("FedAvg", lambda: run_fedavg(clients, experiment, base_state, device)),
        ("FedRep", lambda: run_fedrep(clients, experiment, base_state, device)),
    )
    for method, runner in baseline_runners:
        set_seed(experiment.seed)
        print(f"Running {method}...")
        metrics, history = runner()
        results[method] = metrics
        histories[method] = history
        print(f"  RMSE={metrics['rmse']:.4f}, communication={metrics['communication_bytes']:,} B")

    set_seed(experiment.seed)
    print("Running AdaptiveMsg-NSMP...")
    metrics, _, _ = run_approx_mlp_nsmp(
        clients,
        tree,
        experiment,
        message_config,
        base_state,
        device,
        args.output,
    )
    results["AdaptiveMsg-NSMP"] = metrics
    print(f"  RMSE={metrics['rmse']:.4f}, communication={metrics['communication_bytes']:,} B")
    save_comparison(
        args.output,
        experiment,
        message_config,
        device,
        tree,
        results,
        histories,
        args.dataset_name,
        args.speed_data,
        args.adjacency,
    )
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
