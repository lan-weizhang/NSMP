from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .quadratic import QuadraticFactor, combine_factors, objective_value


@dataclass
class SolverResult:
    method: str
    solution: np.ndarray
    objective: float
    rounds: int
    messages: int
    communicated_scalars: int
    history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class _Recovery:
    eliminated: tuple[int, ...]
    separator: tuple[int, ...]
    intercept: np.ndarray
    slope: np.ndarray


def _dimension(factors: Sequence[QuadraticFactor]) -> int:
    coordinates = [coordinate for factor in factors for coordinate in factor.scope]
    if not coordinates:
        return 0
    return max(coordinates) + 1


def _stable_solve(matrix: np.ndarray, rhs: np.ndarray, jitter: float) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    if matrix.size == 0:
        return np.zeros_like(rhs, dtype=float)
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.solve(matrix + jitter * scale * np.eye(matrix.shape[0]), rhs)


def solve_centralized(
    factors: Sequence[QuadraticFactor], *, jitter: float = 1e-10
) -> SolverResult:
    combined = combine_factors(factors, name="centralized")
    local_solution = _stable_solve(combined.hessian, combined.linear, jitter)
    solution = np.full(_dimension(factors), np.nan, dtype=float)
    solution[np.asarray(combined.scope, dtype=int)] = local_solution
    return SolverResult(
        method="Centralized WLS",
        solution=solution,
        objective=objective_value(factors, solution),
        rounds=0,
        messages=0,
        communicated_scalars=0,
    )


def _tree_orientation(
    num_nodes: int, edges: Sequence[tuple[int, int]], root: int
) -> tuple[list[int], list[list[int]], list[int], int]:
    adjacency = [[] for _ in range(num_nodes)]
    for first, second in edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    if len(edges) != num_nodes - 1:
        raise ValueError("NSMP routing graph must be a tree with M-1 edges.")
    parent = [-2] * num_nodes
    parent[root] = -1
    order: list[int] = []
    queue: deque[int] = deque([root])
    depth = [0] * num_nodes
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adjacency[node]:
            if parent[neighbor] != -2:
                continue
            parent[neighbor] = node
            depth[neighbor] = depth[node] + 1
            queue.append(neighbor)
    if len(order) != num_nodes:
        raise ValueError("NSMP routing tree is disconnected.")
    children = [[] for _ in range(num_nodes)]
    for node, ancestor in enumerate(parent):
        if ancestor >= 0:
            children[ancestor].append(node)
    return parent, children, order, max(depth)


def _partial_minimize(
    factor: QuadraticFactor,
    separator: tuple[int, ...],
    *,
    jitter: float,
    name: str,
) -> tuple[QuadraticFactor, _Recovery]:
    separator = tuple(sorted(separator))
    separator_set = set(separator)
    eliminated = tuple(item for item in factor.scope if item not in separator_set)
    scope_position = {coordinate: index for index, coordinate in enumerate(factor.scope)}
    u_index = np.asarray([scope_position[item] for item in eliminated], dtype=int)
    z_index = np.asarray([scope_position[item] for item in separator], dtype=int)
    if not eliminated:
        return (
            QuadraticFactor(
                separator,
                factor.hessian[np.ix_(z_index, z_index)],
                factor.linear[z_index],
                factor.constant,
                name,
            ),
            _Recovery((), separator, np.zeros(0), np.zeros((0, len(separator)))),
        )
    h_uu = factor.hessian[np.ix_(u_index, u_index)]
    h_uz = factor.hessian[np.ix_(u_index, z_index)]
    h_zz = factor.hessian[np.ix_(z_index, z_index)]
    b_u = factor.linear[u_index]
    b_z = factor.linear[z_index]
    inverse_b = _stable_solve(h_uu, b_u, jitter)
    inverse_huz = _stable_solve(h_uu, h_uz, jitter)
    message = QuadraticFactor(
        separator,
        h_zz - h_uz.T @ inverse_huz,
        b_z - h_uz.T @ inverse_b,
        factor.constant - 0.5 * float(b_u @ inverse_b),
        name,
    )
    recovery = _Recovery(eliminated, separator, inverse_b, -inverse_huz)
    return message, recovery


def solve_nsmp(
    factors: Sequence[QuadraticFactor],
    tree_edges: Sequence[tuple[int, int]],
    *,
    root: int,
    jitter: float = 1e-10,
) -> SolverResult:
    """Exact quadratic NSMP with complete subtree separators.

    The routing tree selects an elimination order.  Separator scopes are the
    full coordinate intersections across each subtree cut, so non-tree overlap
    dependencies are retained rather than dropped.
    """

    num_nodes = len(factors)
    parent, children, order, height = _tree_orientation(num_nodes, tree_edges, root)
    all_clients = set(range(num_nodes))
    local_scopes = [set(factor.scope) for factor in factors]
    subtrees: list[set[int]] = [set() for _ in range(num_nodes)]
    separators: list[tuple[int, ...]] = [tuple() for _ in range(num_nodes)]
    for node in reversed(order):
        subtree = {node}
        for child in children[node]:
            subtree.update(subtrees[child])
        subtrees[node] = subtree
        if node == root:
            separators[node] = ()
            continue
        inside = set().union(*(local_scopes[index] for index in subtree))
        outside_clients = all_clients - subtree
        outside = set().union(*(local_scopes[index] for index in outside_clients))
        separators[node] = tuple(sorted(inside & outside))

    messages: list[QuadraticFactor | None] = [None] * num_nodes
    recoveries: list[_Recovery | None] = [None] * num_nodes
    communicated_scalars = 0
    for node in reversed(order):
        incoming = [messages[child] for child in children[node]]
        combined = combine_factors(
            [factors[node], *[item for item in incoming if item is not None]],
            name=f"node_{node}_combined",
        )
        message, recovery = _partial_minimize(
            combined,
            separators[node],
            jitter=jitter,
            name=f"message_{node}",
        )
        messages[node] = message
        recoveries[node] = recovery
        if node != root:
            separator_size = len(separators[node])
            communicated_scalars += (
                separator_size * (separator_size + 1) // 2
                + separator_size
                + 1
            )

    solution = np.full(_dimension(factors), np.nan, dtype=float)
    for node in order:
        recovery = recoveries[node]
        assert recovery is not None
        if recovery.separator:
            separator_values = solution[np.asarray(recovery.separator, dtype=int)]
            if np.any(~np.isfinite(separator_values)):
                raise RuntimeError("NSMP downward pass encountered an unset separator.")
        else:
            separator_values = np.zeros(0)
        eliminated_values = recovery.intercept + recovery.slope @ separator_values
        if recovery.eliminated:
            solution[np.asarray(recovery.eliminated, dtype=int)] = eliminated_values
        if node != root:
            communicated_scalars += len(recovery.separator)
    if np.any(~np.isfinite(solution)):
        raise RuntimeError("NSMP did not reconstruct every global coordinate.")
    return SolverResult(
        method="Exact NSMP",
        solution=solution,
        objective=objective_value(factors, solution),
        rounds=2 * height,
        messages=2 * (num_nodes - 1),
        communicated_scalars=communicated_scalars,
        metadata={"root": root, "tree_height": height},
    )


def _coordinate_holders(factors: Sequence[QuadraticFactor]) -> dict[int, list[int]]:
    holders: dict[int, list[int]] = {}
    for client, factor in enumerate(factors):
        for coordinate in factor.scope:
            holders.setdefault(coordinate, []).append(client)
    return holders


def solve_consensus_admm(
    factors: Sequence[QuadraticFactor],
    *,
    rounds: int = 300,
    rho: float = 0.0,
    oracle_objective: float | None = None,
    oracle_solution: np.ndarray | None = None,
    initial_solution: np.ndarray | None = None,
    jitter: float = 1e-10,
) -> SolverResult:
    """Coordinate-consensus ADMM over the clients that own each coordinate."""

    if rho <= 0.0:
        positive_curvatures: list[float] = []
        for factor in factors:
            eigenvalues = np.linalg.eigvalsh(factor.hessian)
            threshold = max(1.0, float(eigenvalues[-1])) * 1e-12
            positive = eigenvalues[eigenvalues > threshold]
            if positive.size:
                positive_curvatures.append(float(positive[0]))
        rho = len(factors) * min(positive_curvatures)
        rho = max(rho, 1e-12)
    oracle_hessian = (
        combine_factors(factors, name="oracle_gap").hessian
        if oracle_solution is not None
        else None
    )

    holders = _coordinate_holders(factors)
    shared = {coordinate for coordinate, clients in holders.items() if len(clients) > 1}
    local_position = [
        {coordinate: index for index, coordinate in enumerate(factor.scope)}
        for factor in factors
    ]
    dimension = _dimension(factors)
    initial = (
        np.zeros(dimension, dtype=float)
        if initial_solution is None
        else np.asarray(initial_solution, dtype=float).reshape(-1).copy()
    )
    if initial.size != dimension:
        raise ValueError("initial_solution has the wrong dimension")
    local_x = [initial[np.asarray(factor.scope, dtype=int)].copy() for factor in factors]
    local_u = [np.zeros(len(factor.scope), dtype=float) for factor in factors]
    global_z = initial.copy()
    history: list[dict[str, float]] = []
    per_round_scalars = 2 * sum(
        1 for factor in factors for coordinate in factor.scope if coordinate in shared
    )
    for iteration in range(1, rounds + 1):
        previous_z = global_z.copy()
        for client, factor in enumerate(factors):
            mask = np.asarray([coordinate in shared for coordinate in factor.scope], dtype=float)
            rhs = factor.linear + rho * mask * (
                global_z[np.asarray(factor.scope, dtype=int)] - local_u[client]
            )
            matrix = factor.hessian + rho * np.diag(mask)
            local_x[client] = _stable_solve(matrix, rhs, jitter)
        for coordinate in shared:
            values = []
            for client in holders[coordinate]:
                position = local_position[client][coordinate]
                values.append(local_x[client][position] + local_u[client][position])
            global_z[coordinate] = float(np.mean(values))
        for coordinate, clients in holders.items():
            if len(clients) == 1:
                client = clients[0]
                global_z[coordinate] = local_x[client][local_position[client][coordinate]]
        primal_squared = 0.0
        for client, factor in enumerate(factors):
            for coordinate in shared:
                position = local_position[client].get(coordinate)
                if position is None:
                    continue
                residual = local_x[client][position] - global_z[coordinate]
                local_u[client][position] += residual
                primal_squared += residual * residual
        dual_squared = 0.0
        for coordinate in shared:
            difference = global_z[coordinate] - previous_z[coordinate]
            dual_squared += len(holders[coordinate]) * (rho * difference) ** 2
        current_objective = objective_value(factors, global_z)
        record = {
            "round": float(iteration),
            "objective": current_objective,
            "primal_residual": float(np.sqrt(primal_squared)),
            "dual_residual": float(np.sqrt(dual_squared)),
            "communicated_scalars": float(iteration * per_round_scalars),
        }
        if oracle_objective is not None:
            if oracle_solution is not None and oracle_hessian is not None:
                difference = global_z - oracle_solution
                gap = 0.5 * float(difference @ oracle_hessian @ difference)
            else:
                gap = max(0.0, current_objective - oracle_objective)
            record["relative_objective_gap"] = gap / max(1.0, abs(oracle_objective))
        history.append(record)
    return SolverResult(
        method="Distributed consensus ADMM",
        solution=global_z,
        objective=objective_value(factors, global_z),
        rounds=rounds,
        messages=2 * len(factors) * rounds,
        communicated_scalars=per_round_scalars * rounds,
        history=history,
        metadata={"rho": rho},
    )


def _metropolis_matrices(
    factors: Sequence[QuadraticFactor],
    communication_edges: Sequence[tuple[int, int]],
) -> dict[int, tuple[list[int], np.ndarray]]:
    holders = _coordinate_holders(factors)
    matrices: dict[int, tuple[list[int], np.ndarray]] = {}
    edge_set = {tuple(sorted(edge)) for edge in communication_edges}
    for coordinate, clients in holders.items():
        if len(clients) < 2:
            continue
        client_position = {client: index for index, client in enumerate(clients)}
        local_edges = [
            edge for edge in edge_set if edge[0] in client_position and edge[1] in client_position
        ]
        adjacency = {client: set() for client in clients}
        for first, second in local_edges:
            adjacency[first].add(second)
            adjacency[second].add(first)
        visited = {clients[0]}
        queue = deque([clients[0]])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(visited) != len(clients):
            # Route the shared coordinate along a deterministic holder chain.
            local_edges = list(zip(clients[:-1], clients[1:]))
            adjacency = {client: set() for client in clients}
            for first, second in local_edges:
                adjacency[first].add(second)
                adjacency[second].add(first)
        matrix = np.zeros((len(clients), len(clients)), dtype=float)
        for first, second in local_edges:
            weight = 1.0 / (1.0 + max(len(adjacency[first]), len(adjacency[second])))
            i = client_position[first]
            j = client_position[second]
            matrix[i, j] = weight
            matrix[j, i] = weight
        for client, index in client_position.items():
            matrix[index, index] = 1.0 - float(np.sum(matrix[index]))
        matrices[coordinate] = (clients, matrix)
    return matrices


def solve_decentralized_gossip(
    factors: Sequence[QuadraticFactor],
    communication_edges: Sequence[tuple[int, int]],
    *,
    rounds: int = 300,
    learning_rate_factor: float = 0.05,
    oracle_objective: float | None = None,
    oracle_solution: np.ndarray | None = None,
    initial_solution: np.ndarray | None = None,
) -> SolverResult:
    """Gradient tracking with coordinate-wise Metropolis gossip.

    A common positive rescaling of every local objective is used internally to
    remove PMU precision units from the stable step-size calculation.  This
    changes neither the optimizer nor the reported objective.
    """

    holders = _coordinate_holders(factors)
    local_position = [
        {coordinate: index for index, coordinate in enumerate(factor.scope)}
        for factor in factors
    ]
    dimension = _dimension(factors)
    initial = (
        np.zeros(dimension, dtype=float)
        if initial_solution is None
        else np.asarray(initial_solution, dtype=float).reshape(-1).copy()
    )
    if initial.size != dimension:
        raise ValueError("initial_solution has the wrong dimension")
    local_x = [initial[np.asarray(factor.scope, dtype=int)].copy() for factor in factors]
    matrices = _metropolis_matrices(factors, communication_edges)
    lipschitz = max(
        max(1e-12, float(np.linalg.eigvalsh(factor.hessian).max()))
        for factor in factors
    )
    scaled_hessians = [factor.hessian / lipschitz for factor in factors]
    scaled_linears = [factor.linear / lipschitz for factor in factors]
    gradients = [-linear.copy() for linear in scaled_linears]
    trackers = [gradient.copy() for gradient in gradients]
    step = learning_rate_factor
    # Both the decision vector and the gradient tracker are exchanged.
    per_round_scalars = 4 * sum(
        len(set(factors[first].scope) & set(factors[second].scope))
        for first, second in communication_edges
    )
    history: list[dict[str, float]] = []
    oracle_hessian = (
        combine_factors(factors, name="oracle_gap").hessian
        if oracle_solution is not None
        else None
    )
    global_solution = initial.copy()
    for iteration in range(1, rounds + 1):
        mixed_x = [values.copy() for values in local_x]
        mixed_trackers = [values.copy() for values in trackers]
        for coordinate, (clients, matrix) in matrices.items():
            positions = [local_position[client][coordinate] for client in clients]
            values = np.asarray(
                [local_x[client][position] for client, position in zip(clients, positions)]
            )
            tracker_values = np.asarray(
                [trackers[client][position] for client, position in zip(clients, positions)]
            )
            for client, position, value, tracker_value in zip(
                clients, positions, matrix @ values, matrix @ tracker_values
            ):
                mixed_x[client][position] = value
                mixed_trackers[client][position] = tracker_value
        new_x = [mixed - step * tracker for mixed, tracker in zip(mixed_x, trackers)]
        new_gradients = [
            hessian @ values - linear
            for hessian, values, linear in zip(scaled_hessians, new_x, scaled_linears)
        ]
        trackers = [
            mixed + new_gradient - old_gradient
            for mixed, new_gradient, old_gradient in zip(
                mixed_trackers, new_gradients, gradients
            )
        ]
        local_x = new_x
        gradients = new_gradients
        disagreement_squared = 0.0
        for coordinate, clients in holders.items():
            values = np.asarray(
                [local_x[client][local_position[client][coordinate]] for client in clients]
            )
            global_solution[coordinate] = float(np.mean(values))
            disagreement_squared += float(np.sum((values - np.mean(values)) ** 2))
        current_objective = objective_value(factors, global_solution)
        record = {
            "round": float(iteration),
            "objective": current_objective,
            "consensus_residual": float(np.sqrt(disagreement_squared)),
            "learning_rate": float(step),
            "communicated_scalars": float(iteration * per_round_scalars),
        }
        if oracle_objective is not None:
            if oracle_solution is not None and oracle_hessian is not None:
                difference = global_solution - oracle_solution
                gap = 0.5 * float(difference @ oracle_hessian @ difference)
            else:
                gap = max(0.0, current_objective - oracle_objective)
            record["relative_objective_gap"] = gap / max(1.0, abs(oracle_objective))
        history.append(record)
    return SolverResult(
        method="Decentralized gradient/gossip",
        solution=global_solution,
        objective=objective_value(factors, global_solution),
        rounds=rounds,
        messages=4 * len(communication_edges) * rounds,
        communicated_scalars=per_round_scalars * rounds,
        history=history,
        metadata={"learning_rate": step, "objective_scale": lipschitz},
    )
