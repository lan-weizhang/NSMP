from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import ClientSplit


@dataclass(frozen=True)
class Quadratic:
    """f(w) = 0.5 w^T H w - b^T w + c."""

    hessian: np.ndarray
    linear: np.ndarray
    constant: float
    sample_count: int


def make_quadratics(
    clients: list[ClientSplit],
    ridge: float,
    *,
    regularize_bias: bool = False,
) -> list[Quadratic]:
    quadratics: list[Quadratic] = []
    for split in clients:
        n = split.x.shape[0]
        dimension = split.x.shape[1]
        penalty = np.ones(dimension)
        if not regularize_bias:
            penalty[-1] = 0.0
        hessian = split.x.T @ split.x / n + ridge * np.diag(penalty)
        linear = split.x.T @ split.y / n
        constant = float(0.5 * split.y @ split.y / n)
        quadratics.append(
            Quadratic(
                hessian=0.5 * (hessian + hessian.T),
                linear=linear,
                constant=constant,
                sample_count=n,
            )
        )
    return quadratics


def objective(
    parameters: np.ndarray,
    quadratics: list[Quadratic],
    *,
    edges: list[tuple[int, int, float]] | None = None,
    coupling: float = 0.0,
) -> float:
    total = 0.0
    for index, quadratic in enumerate(quadratics):
        value = parameters[index]
        total += (
            0.5 * value @ quadratic.hessian @ value
            - quadratic.linear @ value
            + quadratic.constant
        )
    if edges:
        for first, second, weight in edges:
            difference = parameters[first] - parameters[second]
            total += 0.5 * coupling * weight * float(difference @ difference)
    return float(total)
