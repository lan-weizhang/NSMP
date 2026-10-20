from __future__ import annotations

import numpy as np

from nsmp_metr_la.graph import maximum_similarity_spanning_tree
from nsmp_metr_la.methods import (
    fit_nsmp,
    solve_fedrep_closed_form,
    solve_nsmp_centralized,
)
from nsmp_metr_la.quadratic import Quadratic


def _quadratics(seed: int = 7) -> list[Quadratic]:
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(7):
        x = rng.normal(size=(80, 5))
        y = rng.normal(size=80)
        hessian = x.T @ x / x.shape[0] + 0.05 * np.eye(x.shape[1])
        linear = x.T @ y / x.shape[0]
        items.append(
            Quadratic(
                hessian=hessian,
                linear=linear,
                constant=float(0.5 * y @ y / y.size),
                sample_count=y.size,
            )
        )
    return items


def test_nsmp_matches_centralized_sparse_solution() -> None:
    quadratics = _quadratics()
    similarity = np.array(
        [
            [0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0, 0.8, 0.7, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0, 0.0, 0.6, 0.0, 0.0],
            [0.0, 0.7, 0.0, 0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.6, 0.0, 0, 0.0, 0.4],
            [0.0, 0.0, 0.0, 0.5, 0.0, 0, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0],
        ],
        dtype=float,
    )
    edges = maximum_similarity_spanning_tree(similarity)
    result = fit_nsmp(quadratics, edges, coupling=0.7)
    centralized = solve_nsmp_centralized(quadratics, edges, coupling=0.7)
    np.testing.assert_allclose(result.parameters, centralized, rtol=1e-9, atol=1e-10)
    assert result.metadata["relative_error_to_centralized"] < 1e-9
    assert abs(result.metadata["objective_gap"]) < 1e-9


def test_fedrep_closed_form_satisfies_fixed_split_stationarity() -> None:
    quadratics = _quadratics()
    shared_lags = 2
    personalized = solve_fedrep_closed_form(
        quadratics, shared_lags=shared_lags, add_bias=True
    )
    shared = np.array([2, 3])
    private = np.array([0, 1, 4])
    sample_counts = np.array([item.sample_count for item in quadratics], dtype=float)
    weights = sample_counts / sample_counts.sum()
    np.testing.assert_allclose(
        personalized[:, shared],
        np.repeat(personalized[0, shared][None, :], len(quadratics), axis=0),
    )
    shared_gradient = np.zeros(shared.size)
    for item, parameter in zip(quadratics, personalized, strict=True):
        gradient = item.hessian @ parameter - item.linear
        assert np.linalg.norm(gradient[private]) < 1e-9
    for weight, item, parameter in zip(
        weights, quadratics, personalized, strict=True
    ):
        shared_gradient += weight * (
            item.hessian @ parameter - item.linear
        )[shared]
    assert np.linalg.norm(shared_gradient) < 1e-9
