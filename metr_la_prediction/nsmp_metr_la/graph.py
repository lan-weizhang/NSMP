from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree


@dataclass(frozen=True)
class TreeEdge:
    first: int
    second: int
    weight: float


@dataclass
class OrientedTree:
    root: int
    parent: np.ndarray
    parent_weight: np.ndarray
    children: list[list[int]]
    preorder: list[int]
    postorder: list[int]


def maximum_similarity_spanning_tree(similarity: np.ndarray) -> list[TreeEdge]:
    """Return an MST using inverse similarity as edge cost."""
    similarity = np.asarray(similarity, dtype=np.float64)
    similarity = np.maximum(similarity, similarity.T)
    np.fill_diagonal(similarity, 0.0)
    graph = csr_matrix(similarity > 0)
    components, _ = connected_components(graph, directed=False)
    if components != 1:
        raise ValueError(
            f"The selected graph has {components} connected components. "
            "Use graph_source='correlation' or select a connected client subset."
        )

    rows, cols = np.nonzero(np.triu(similarity, 1))
    costs = np.zeros_like(similarity)
    costs[rows, cols] = 1.0 / np.maximum(similarity[rows, cols], 1e-12)
    costs[cols, rows] = costs[rows, cols]
    tree = minimum_spanning_tree(csr_matrix(costs))
    row, col = tree.nonzero()
    edges = [
        TreeEdge(int(i), int(j), float(similarity[i, j]))
        for i, j in zip(row, col, strict=True)
    ]
    if len(edges) != similarity.shape[0] - 1:
        raise RuntimeError("Spanning-tree construction did not return M-1 edges.")
    return edges


def _tree_adjacency(
    num_nodes: int, edges: list[TreeEdge]
) -> list[list[tuple[int, float]]]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for edge in edges:
        adjacency[edge.first].append((edge.second, edge.weight))
        adjacency[edge.second].append((edge.first, edge.weight))
    return adjacency


def tree_center(num_nodes: int, edges: list[TreeEdge]) -> int:
    """Choose a center that minimizes maximum unweighted tree depth."""
    if num_nodes == 1:
        return 0
    adjacency = _tree_adjacency(num_nodes, edges)
    degree = np.array([len(items) for items in adjacency])
    leaves = deque(np.flatnonzero(degree <= 1).tolist())
    remaining = num_nodes
    while remaining > 2:
        layer_size = len(leaves)
        remaining -= layer_size
        for _ in range(layer_size):
            leaf = leaves.popleft()
            for neighbor, _ in adjacency[leaf]:
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    leaves.append(neighbor)
    return int(min(leaves))


def orient_tree(
    num_nodes: int,
    edges: list[TreeEdge],
    root: int | str = "center",
) -> OrientedTree:
    if num_nodes == 1:
        return OrientedTree(
            root=0,
            parent=np.array([-1]),
            parent_weight=np.array([0.0]),
            children=[[]],
            preorder=[0],
            postorder=[0],
        )
    root_index = tree_center(num_nodes, edges) if root == "center" else int(root)
    if not 0 <= root_index < num_nodes:
        raise ValueError("Root index is outside the client range.")

    adjacency = _tree_adjacency(num_nodes, edges)
    parent = np.full(num_nodes, -2, dtype=int)
    parent_weight = np.zeros(num_nodes, dtype=np.float64)
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    parent[root_index] = -1
    preorder: list[int] = []
    queue = deque([root_index])
    while queue:
        node = queue.popleft()
        preorder.append(node)
        for neighbor, weight in adjacency[node]:
            if parent[neighbor] != -2:
                continue
            parent[neighbor] = node
            parent_weight[neighbor] = weight
            children[node].append(neighbor)
            queue.append(neighbor)
    if np.any(parent == -2):
        raise ValueError("The supplied edges do not form a connected tree.")
    return OrientedTree(
        root=root_index,
        parent=parent,
        parent_weight=parent_weight,
        children=children,
        preorder=preorder,
        postorder=preorder[::-1],
    )
