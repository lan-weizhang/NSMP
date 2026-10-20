from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class QuadraticFactor:
    """A coordinate-indexed quadratic 0.5*x'Hx - b'x + c."""

    scope: tuple[int, ...]
    hessian: np.ndarray
    linear: np.ndarray
    constant: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        scope = tuple(int(item) for item in self.scope)
        if len(scope) != len(set(scope)):
            raise ValueError("Quadratic factor scope contains duplicate coordinates.")
        hessian = np.asarray(self.hessian, dtype=float)
        linear = np.asarray(self.linear, dtype=float).reshape(-1)
        if hessian.shape != (len(scope), len(scope)):
            raise ValueError(
                f"Hessian shape {hessian.shape} does not match scope size {len(scope)}."
            )
        if linear.shape != (len(scope),):
            raise ValueError(
                f"Linear term shape {linear.shape} does not match scope size {len(scope)}."
            )
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "hessian", 0.5 * (hessian + hessian.T))
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "constant", float(self.constant))

    @classmethod
    def from_least_squares(
        cls,
        scope: Sequence[int],
        design: np.ndarray,
        observation: np.ndarray,
        precision: np.ndarray | Sequence[float] | float,
        *,
        name: str = "",
    ) -> "QuadraticFactor":
        design = np.asarray(design, dtype=float)
        observation = np.asarray(observation, dtype=float).reshape(-1)
        if design.shape != (observation.size, len(scope)):
            raise ValueError(
                "Design matrix must have one row per observation and one column "
                "per scoped coordinate."
            )
        precision_array = np.asarray(precision, dtype=float)
        if precision_array.ndim == 0:
            weighted_design = float(precision_array) * design
            weighted_observation = float(precision_array) * observation
        elif precision_array.ndim == 1:
            if precision_array.size != observation.size:
                raise ValueError("Precision vector has the wrong length.")
            weighted_design = precision_array[:, None] * design
            weighted_observation = precision_array * observation
        elif precision_array.shape == (observation.size, observation.size):
            weighted_design = precision_array @ design
            weighted_observation = precision_array @ observation
        else:
            raise ValueError("Precision must be scalar, diagonal vector, or square matrix.")
        return cls(
            tuple(scope),
            design.T @ weighted_design,
            design.T @ weighted_observation,
            0.5 * float(observation @ weighted_observation),
            name,
        )

    def evaluate_local(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != len(self.scope):
            raise ValueError("Local value vector has the wrong size.")
        return float(
            0.5 * values @ self.hessian @ values
            - self.linear @ values
            + self.constant
        )


def combine_factors(
    factors: Iterable[QuadraticFactor], *, name: str = "combined"
) -> QuadraticFactor:
    factors = list(factors)
    if not factors:
        return QuadraticFactor((), np.zeros((0, 0)), np.zeros(0), 0.0, name)
    scope = tuple(sorted({coordinate for factor in factors for coordinate in factor.scope}))
    positions = {coordinate: index for index, coordinate in enumerate(scope)}
    hessian = np.zeros((len(scope), len(scope)), dtype=float)
    linear = np.zeros(len(scope), dtype=float)
    constant = 0.0
    for factor in factors:
        local = np.array([positions[coordinate] for coordinate in factor.scope], dtype=int)
        hessian[np.ix_(local, local)] += factor.hessian
        linear[local] += factor.linear
        constant += factor.constant
    return QuadraticFactor(scope, hessian, linear, constant, name)


def objective_value(factors: Sequence[QuadraticFactor], solution: np.ndarray) -> float:
    solution = np.asarray(solution, dtype=float).reshape(-1)
    value = 0.0
    for factor in factors:
        local = solution[np.asarray(factor.scope, dtype=int)]
        value += factor.evaluate_local(local)
    return float(value)


def embed_solution(factor: QuadraticFactor, local_solution: np.ndarray, dimension: int) -> np.ndarray:
    solution = np.full(dimension, np.nan, dtype=float)
    solution[np.asarray(factor.scope, dtype=int)] = np.asarray(local_solution, dtype=float)
    return solution

