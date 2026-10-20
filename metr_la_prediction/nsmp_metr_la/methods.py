from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .data import FederatedTimeSeries
from .graph import OrientedTree, TreeEdge, orient_tree
from .metrics import evaluate_parameters
from .quadratic import Quadratic, objective


@dataclass
class MethodResult:
    name: str
    parameters: np.ndarray
    runtime_seconds: float
    communication_rounds: int
    messages: int
    communication_bytes: int
    history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, float | int | str] = field(default_factory=dict)


def _stack_local_solutions(quadratics: list[Quadratic]) -> np.ndarray:
    return np.stack(
        [
            np.linalg.solve(quadratic.hessian, quadratic.linear)
            for quadratic in quadratics
        ]
    )


def fit_local(quadratics: list[Quadratic]) -> MethodResult:
    start = time.perf_counter()
    parameters = _stack_local_solutions(quadratics)
    runtime = time.perf_counter() - start
    return MethodResult(
        name="Local",
        parameters=parameters,
        runtime_seconds=runtime,
        communication_rounds=0,
        messages=0,
        communication_bytes=0,
    )


def _validation_rmse(
    dataset: FederatedTimeSeries, parameters: np.ndarray
) -> float:
    summary, _ = evaluate_parameters(dataset, parameters, "val")
    return summary.rmse


def fit_fedavg(
    dataset: FederatedTimeSeries,
    quadratics: list[Quadratic],
    *,
    rounds: int = 200,
    local_epochs: int = 5,
    learning_rate: float | None = None,
    learning_rate_factor: float = 0.8,
    patience: int = 30,
) -> MethodResult:
    """Full-batch FedAvg on the common linear autoregressive model."""
    start = time.perf_counter()
    num_clients = len(quadratics)
    dimension = quadratics[0].linear.size
    sample_counts = np.array([item.sample_count for item in quadratics], dtype=float)
    client_weights = sample_counts / sample_counts.sum()
    if learning_rate is None:
        lipschitz = max(
            float(np.linalg.eigvalsh(item.hessian).max()) for item in quadratics
        )
        learning_rate = learning_rate_factor / max(lipschitz, 1e-12)

    global_parameter = np.zeros(dimension, dtype=np.float64)
    best_parameter = global_parameter.copy()
    best_score = float("inf")
    best_round = 0
    history: list[dict[str, float]] = []
    rounds_run = 0
    for round_index in range(1, rounds + 1):
        local_parameters = []
        for quadratic in quadratics:
            local_parameter = global_parameter.copy()
            for _ in range(local_epochs):
                gradient = (
                    quadratic.hessian @ local_parameter - quadratic.linear
                )
                local_parameter -= learning_rate * gradient
            local_parameters.append(local_parameter)
        global_parameter = np.average(
            np.stack(local_parameters), axis=0, weights=client_weights
        )
        parameters = np.repeat(global_parameter[None, :], num_clients, axis=0)
        score = _validation_rmse(dataset, parameters)
        history.append({"round": float(round_index), "val_rmse": score})
        rounds_run = round_index
        if score < best_score - 1e-10:
            best_score = score
            best_parameter = global_parameter.copy()
            best_round = round_index
        elif round_index - best_round >= patience:
            break

    runtime = time.perf_counter() - start
    parameters = np.repeat(best_parameter[None, :], num_clients, axis=0)
    bytes_per_round = 2 * num_clients * dimension * 8
    return MethodResult(
        name="FedAvg",
        parameters=parameters,
        runtime_seconds=runtime,
        communication_rounds=rounds_run,
        messages=2 * num_clients * rounds_run,
        communication_bytes=bytes_per_round * rounds_run,
        history=history,
        metadata={
            "best_round": best_round,
            "learning_rate": float(learning_rate),
            "local_epochs": local_epochs,
        },
    )


def _fedrep_indices(
    dimension: int, shared_lags: int, add_bias: bool
) -> tuple[np.ndarray, np.ndarray]:
    lag_count = dimension - int(add_bias)
    if shared_lags <= 0 or shared_lags > lag_count:
        raise ValueError(
            f"shared_lags must be between 1 and {lag_count}, got {shared_lags}."
        )
    # Window columns are ordered from oldest to most recent. Share the most
    # recent lag coefficients and keep older lags (plus bias) client-private.
    shared = np.arange(lag_count - shared_lags, lag_count, dtype=int)
    private = np.setdiff1d(np.arange(dimension, dtype=int), shared)
    return shared, private


def _fedrep_private_solution(
    quadratic: Quadratic,
    shared_parameter: np.ndarray,
    shared: np.ndarray,
    private: np.ndarray,
) -> np.ndarray:
    h_pp = quadratic.hessian[np.ix_(private, private)]
    h_ps = quadratic.hessian[np.ix_(private, shared)]
    rhs = quadratic.linear[private] - h_ps @ shared_parameter
    return np.linalg.solve(h_pp, rhs)


def _assemble_fedrep_parameters(
    quadratics: list[Quadratic],
    shared_parameter: np.ndarray,
    shared: np.ndarray,
    private: np.ndarray,
) -> np.ndarray:
    parameters = np.zeros((len(quadratics), quadratics[0].linear.size))
    parameters[:, shared] = shared_parameter
    for index, quadratic in enumerate(quadratics):
        parameters[index, private] = _fedrep_private_solution(
            quadratic, shared_parameter, shared, private
        )
    return parameters


def solve_fedrep_closed_form(
    quadratics: list[Quadratic],
    *,
    shared_lags: int,
    add_bias: bool = True,
) -> np.ndarray:
    """Exact solution of the convex fixed shared/private linear objective."""
    dimension = quadratics[0].linear.size
    shared, private = _fedrep_indices(dimension, shared_lags, add_bias)
    sample_counts = np.array([item.sample_count for item in quadratics], dtype=float)
    weights = sample_counts / sample_counts.sum()
    reduced_hessian = np.zeros((shared.size, shared.size))
    reduced_linear = np.zeros(shared.size)
    for weight, quadratic in zip(weights, quadratics, strict=True):
        h_ss = quadratic.hessian[np.ix_(shared, shared)]
        h_sp = quadratic.hessian[np.ix_(shared, private)]
        h_pp = quadratic.hessian[np.ix_(private, private)]
        b_s = quadratic.linear[shared]
        b_p = quadratic.linear[private]
        h_pp_inv_h_ps = np.linalg.solve(h_pp, h_sp.T)
        h_pp_inv_b_p = np.linalg.solve(h_pp, b_p)
        reduced_hessian += weight * (h_ss - h_sp @ h_pp_inv_h_ps)
        reduced_linear += weight * (b_s - h_sp @ h_pp_inv_b_p)
    shared_parameter = np.linalg.solve(reduced_hessian, reduced_linear)
    return _assemble_fedrep_parameters(
        quadratics, shared_parameter, shared, private
    )


def fit_fedrep(
    dataset: FederatedTimeSeries,
    quadratics: list[Quadratic],
    *,
    shared_lags: int,
    rounds: int = 300,
    local_epochs: int = 5,
    learning_rate: float | None = None,
    learning_rate_factor: float = 0.8,
    patience: int = 40,
) -> MethodResult:
    """FedRep-style fixed split: recent lags shared, remaining terms private."""
    start = time.perf_counter()
    num_clients = len(quadratics)
    dimension = quadratics[0].linear.size
    shared, private = _fedrep_indices(
        dimension, shared_lags, dataset.add_bias
    )
    sample_counts = np.array([item.sample_count for item in quadratics], dtype=float)
    client_weights = sample_counts / sample_counts.sum()
    if learning_rate is None:
        lipschitz = max(
            float(
                np.linalg.eigvalsh(
                    item.hessian[np.ix_(shared, shared)]
                ).max()
            )
            for item in quadratics
        )
        learning_rate = learning_rate_factor / max(lipschitz, 1e-12)

    global_shared = np.zeros(shared.size, dtype=np.float64)
    best_parameters = _assemble_fedrep_parameters(
        quadratics, global_shared, shared, private
    )
    best_score = float("inf")
    best_round = 0
    history: list[dict[str, float]] = []
    rounds_run = 0
    for round_index in range(1, rounds + 1):
        local_shared_parameters = []
        for quadratic in quadratics:
            private_parameter = _fedrep_private_solution(
                quadratic, global_shared, shared, private
            )
            local_shared = global_shared.copy()
            h_ss = quadratic.hessian[np.ix_(shared, shared)]
            h_sp = quadratic.hessian[np.ix_(shared, private)]
            b_s = quadratic.linear[shared]
            for _ in range(local_epochs):
                gradient = h_ss @ local_shared + h_sp @ private_parameter - b_s
                local_shared -= learning_rate * gradient
            local_shared_parameters.append(local_shared)
        global_shared = np.average(
            np.stack(local_shared_parameters), axis=0, weights=client_weights
        )
        parameters = _assemble_fedrep_parameters(
            quadratics, global_shared, shared, private
        )
        score = _validation_rmse(dataset, parameters)
        history.append({"round": float(round_index), "val_rmse": score})
        rounds_run = round_index
        if score < best_score - 1e-10:
            best_score = score
            best_parameters = parameters.copy()
            best_round = round_index
        elif round_index - best_round >= patience:
            break

    exact_parameters = solve_fedrep_closed_form(
        quadratics,
        shared_lags=shared_lags,
        add_bias=dataset.add_bias,
    )
    relative_error_to_exact = float(
        np.linalg.norm(parameters - exact_parameters)
        / max(np.linalg.norm(exact_parameters), 1e-12)
    )
    runtime = time.perf_counter() - start
    bytes_per_round = 2 * num_clients * shared.size * 8
    return MethodResult(
        name="FedRep",
        parameters=best_parameters,
        runtime_seconds=runtime,
        communication_rounds=rounds_run,
        messages=2 * num_clients * rounds_run,
        communication_bytes=bytes_per_round * rounds_run,
        history=history,
        metadata={
            "best_round": best_round,
            "learning_rate": float(learning_rate),
            "local_epochs": local_epochs,
            "shared_lags": int(shared.size),
            "private_parameters": int(private.size),
            "final_relative_error_to_exact": relative_error_to_exact,
        },
    )


@dataclass
class _NSMPStoredMap:
    offset: np.ndarray
    parent_map: np.ndarray


def _solve_nsmp_messages(
    quadratics: list[Quadratic],
    tree: OrientedTree,
    coupling: float,
    *,
    normalize_edge_weights: bool,
) -> tuple[np.ndarray, float]:
    num_clients = len(quadratics)
    dimension = quadratics[0].linear.size
    identity = np.eye(dimension)
    positive_weights = tree.parent_weight[tree.parent_weight > 0]
    weight_scale = (
        float(np.mean(positive_weights))
        if normalize_edge_weights and positive_weights.size
        else 1.0
    )
    message_hessian: list[np.ndarray | None] = [None] * num_clients
    message_linear: list[np.ndarray | None] = [None] * num_clients
    message_constant = np.zeros(num_clients)
    stored_maps: list[_NSMPStoredMap | None] = [None] * num_clients

    root_parameter: np.ndarray | None = None
    root_value = float("nan")
    for node in tree.postorder:
        quadratic = quadratics[node]
        aggregate_hessian = quadratic.hessian.copy()
        aggregate_linear = quadratic.linear.copy()
        aggregate_constant = quadratic.constant
        for child in tree.children[node]:
            child_hessian = message_hessian[child]
            child_linear = message_linear[child]
            if child_hessian is None or child_linear is None:
                raise RuntimeError("A child NSMP message was not computed.")
            aggregate_hessian += child_hessian
            aggregate_linear += child_linear
            aggregate_constant += message_constant[child]

        if node == tree.root:
            root_parameter = np.linalg.solve(
                aggregate_hessian, aggregate_linear
            )
            root_value = float(
                aggregate_constant
                - 0.5 * aggregate_linear @ root_parameter
            )
            continue

        edge_weight = tree.parent_weight[node] / weight_scale
        strength = coupling * edge_weight
        if strength <= 0:
            raise ValueError("All NSMP tree-edge coupling strengths must be positive.")
        eliminated_hessian = aggregate_hessian + strength * identity
        offset = np.linalg.solve(eliminated_hessian, aggregate_linear)
        parent_map = np.linalg.solve(
            eliminated_hessian, strength * identity
        )
        reduced_hessian = strength * identity - strength * parent_map
        reduced_hessian = 0.5 * (reduced_hessian + reduced_hessian.T)
        reduced_linear = strength * offset
        reduced_constant = float(
            aggregate_constant - 0.5 * aggregate_linear @ offset
        )
        message_hessian[node] = reduced_hessian
        message_linear[node] = reduced_linear
        message_constant[node] = reduced_constant
        stored_maps[node] = _NSMPStoredMap(offset, parent_map)

    if root_parameter is None:
        raise RuntimeError("NSMP root was not solved.")
    parameters = np.zeros((num_clients, dimension))
    parameters[tree.root] = root_parameter
    for node in tree.preorder:
        if node == tree.root:
            continue
        parent = tree.parent[node]
        stored = stored_maps[node]
        if stored is None:
            raise RuntimeError("Missing NSMP minimizer map.")
        parameters[node] = stored.offset + stored.parent_map @ parameters[parent]
    return parameters, root_value


def solve_nsmp_centralized(
    quadratics: list[Quadratic],
    edges: list[TreeEdge],
    coupling: float,
    *,
    normalize_edge_weights: bool = True,
) -> np.ndarray:
    num_clients = len(quadratics)
    dimension = quadratics[0].linear.size
    block_hessian = sparse.block_diag(
        [item.hessian for item in quadratics], format="csr"
    )
    laplacian = sparse.lil_matrix((num_clients, num_clients))
    mean_weight = (
        np.mean([edge.weight for edge in edges])
        if normalize_edge_weights and edges
        else 1.0
    )
    for edge in edges:
        weight = edge.weight / mean_weight
        laplacian[edge.first, edge.first] += weight
        laplacian[edge.second, edge.second] += weight
        laplacian[edge.first, edge.second] -= weight
        laplacian[edge.second, edge.first] -= weight
    full_hessian = block_hessian + coupling * sparse.kron(
        laplacian.tocsr(), sparse.eye(dimension), format="csr"
    )
    full_linear = np.concatenate([item.linear for item in quadratics])
    solution = spsolve(full_hessian, full_linear)
    return np.asarray(solution).reshape(num_clients, dimension)


def fit_nsmp(
    quadratics: list[Quadratic],
    edges: list[TreeEdge],
    *,
    coupling: float,
    root: int | str = "center",
    normalize_edge_weights: bool = True,
    verify_centralized: bool = True,
) -> MethodResult:
    if coupling <= 0:
        raise ValueError("NSMP coupling must be positive.")
    start = time.perf_counter()
    tree = orient_tree(len(quadratics), edges, root=root)
    parameters, root_value = _solve_nsmp_messages(
        quadratics,
        tree,
        coupling,
        normalize_edge_weights=normalize_edge_weights,
    )
    runtime = time.perf_counter() - start
    dimension = quadratics[0].linear.size
    symmetric_entries = dimension * (dimension + 1) // 2
    upward_scalars = symmetric_entries + dimension + 1
    downward_scalars = dimension
    edge_count = len(edges)
    metadata: dict[str, float | int | str] = {
        "coupling": coupling,
        "root": tree.root,
        "tree_depth": max(
            _node_depth(node, tree.parent) for node in range(len(quadratics))
        ),
        "root_objective": root_value,
    }
    if verify_centralized:
        centralized = solve_nsmp_centralized(
            quadratics,
            edges,
            coupling,
            normalize_edge_weights=normalize_edge_weights,
        )
        relative_error = float(
            np.linalg.norm(parameters - centralized)
            / max(np.linalg.norm(centralized), 1e-12)
        )
        edge_tuples = [(e.first, e.second, e.weight) for e in edges]
        nsmp_objective = objective(
            parameters,
            quadratics,
            edges=edge_tuples,
            coupling=coupling / (
                np.mean([edge.weight for edge in edges])
                if normalize_edge_weights and edges
                else 1.0
            ),
        )
        centralized_objective = objective(
            centralized,
            quadratics,
            edges=edge_tuples,
            coupling=coupling / (
                np.mean([edge.weight for edge in edges])
                if normalize_edge_weights and edges
                else 1.0
            ),
        )
        metadata.update(
            {
                "relative_error_to_centralized": relative_error,
                "objective_gap": float(nsmp_objective - centralized_objective),
            }
        )
    return MethodResult(
        name="NSMP",
        parameters=parameters,
        runtime_seconds=runtime,
        communication_rounds=2,
        messages=2 * edge_count,
        communication_bytes=8
        * edge_count
        * (upward_scalars + downward_scalars),
        metadata=metadata,
    )


def _node_depth(node: int, parent: np.ndarray) -> int:
    depth = 0
    while parent[node] >= 0:
        node = int(parent[node])
        depth += 1
    return depth
