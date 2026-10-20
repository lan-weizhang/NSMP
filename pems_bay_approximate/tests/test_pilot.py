from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from pems_nsmp.models_and_methods import (  # noqa: E402
    GraphTemporalNet,
    connected_bfs_order,
    incident_edges,
    normalize_adjacency,
    region_tree,
)


def test_graph_model_output_shape() -> None:
    model = GraphTemporalNet(lags=12, hidden_dim=8)
    x = torch.randn(7, 12, 5)
    adjacency = torch.eye(5)
    adapter = torch.zeros(8)
    assert model(x, adjacency, adapter).shape == (7, 5)


def test_connected_selection_and_normalization() -> None:
    adjacency = np.zeros((8, 8), dtype=np.float32)
    for index in range(7):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    selected = connected_bfs_order(adjacency, np.arange(8), 6)
    assert len(selected) == 6
    assert len(set(selected.tolist())) == 6
    normalized = normalize_adjacency(adjacency[np.ix_(selected, selected)])
    assert normalized.shape == (6, 6)
    assert np.all(np.isfinite(normalized))


def test_client_tree_has_n_minus_one_edges() -> None:
    time = np.linspace(0, 4 * np.pi, 100)
    region_series = np.stack(
        [np.sin(time + shift) for shift in np.linspace(0, 0.8, 6)], axis=1
    )
    tree = region_tree(region_series)
    assert len(tree) == 5
    edges = incident_edges(6, tree)
    assert sum(len(values) for values in edges) == 10
