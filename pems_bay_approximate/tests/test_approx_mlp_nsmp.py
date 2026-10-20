from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from pems_nsmp.messages import (  # noqa: E402
    ApproxMessageConfig,
    MessageMLP,
    OverlapGraphTemporalNet,
    PolynomialMessage,
    fit_adaptive_message,
    latin_hypercube,
    polynomial_features,
    tree_depths,
    tree_structure,
)


def test_message_mlp_shape_and_payload() -> None:
    message = MessageMLP(input_dim=2, hidden_dim=8, radius=1.5)
    z = torch.randn(11, 2)
    output = message(z)
    assert output.shape == (11,)
    expected_parameters = (2 * 8 + 8) + (8 * 8 + 8) + (8 + 1)
    assert message.payload_values() == expected_parameters + 2


def test_latin_hypercube_stays_in_domain_and_is_reproducible() -> None:
    first = latin_hypercube(16, 2, 1.5, 123)
    second = latin_hypercube(16, 2, 1.5, 123)
    assert torch.equal(first, second)
    assert torch.all(first >= -1.5)
    assert torch.all(first <= 1.5)
    assert torch.all(torch.std(first, dim=0) > 0.5)


def test_polynomial_features_and_payload() -> None:
    z = torch.tensor([[1.5, -1.5], [0.0, 0.0]])
    features = polynomial_features(z, degree=2, radius=1.5)
    assert features.shape == (2, 6)
    message = PolynomialMessage(2, 2, 1.5, torch.arange(6, dtype=torch.float32))
    assert message(z).shape == (2,)
    assert message.payload_values() == 6


def test_adaptive_message_selects_quadratic_for_quadratic_targets() -> None:
    generator = torch.Generator().manual_seed(17)
    train_z = 3.0 * torch.rand(64, 2, generator=generator) - 1.5
    validation_z = 3.0 * torch.rand(24, 2, generator=generator) - 1.5
    holdout_z = 3.0 * torch.rand(24, 2, generator=generator) - 1.5

    def target(z: torch.Tensor) -> torch.Tensor:
        return 0.7 + 0.2 * z[:, 0] - 0.1 * z[:, 1] + 0.3 * z[:, 0] ** 2

    config = ApproxMessageConfig(message_epochs=50, selection_tolerance_fraction=0.01)
    message, diagnostics, _ = fit_adaptive_message(
        train_z,
        target(train_z),
        validation_z,
        target(validation_z),
        holdout_z,
        target(holdout_z),
        config,
        torch.device("cpu"),
        seed=19,
    )
    assert diagnostics["selected_model"] == "quadratic"
    assert message.payload_values() == 6
    assert float(diagnostics["holdout_rmse"]) < 1e-5


def test_tree_structure_and_depths() -> None:
    tree = [(0, 1, 0), (1, 2, 1), (2, 3, 1)]
    parent, parent_edge, children, child_edge = tree_structure(4, tree)
    assert parent == {1: 0, 2: 1, 3: 1}
    assert parent_edge == child_edge == {1: 0, 2: 1, 3: 2}
    assert children == {0: [1], 1: [2, 3], 2: [], 3: []}
    assert tree_depths(children) == {0: 0, 1: 1, 2: 2, 3: 2}


def test_overlap_model_is_differentiable_in_shared_interface() -> None:
    model = OverlapGraphTemporalNet(lags=3, hidden_dim=4)
    x = torch.randn(5, 3, 2)
    adjacency = torch.eye(2)
    z = torch.randn(2, requires_grad=True)
    basis = torch.linalg.qr(torch.randn(4, 2), mode="reduced").Q
    prediction = model.forward_with_edges(x, adjacency, {0: z}, {0: basis})
    loss = prediction.square().mean()
    loss.backward()
    assert prediction.shape == (5, 2)
    assert z.grad is not None
    assert np.isfinite(z.grad.detach().numpy()).all()
