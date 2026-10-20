from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import FederatedTimeSeries


@dataclass(frozen=True)
class MetricSummary:
    mae: float
    rmse: float
    mape: float


def evaluate_parameters(
    dataset: FederatedTimeSeries,
    parameters: np.ndarray,
    split: str,
    *,
    mape_min_speed: float = 1.0,
) -> tuple[MetricSummary, list[dict[str, float | str]]]:
    client_splits = getattr(dataset, split)
    absolute_errors: list[np.ndarray] = []
    squared_errors: list[np.ndarray] = []
    percentage_errors: list[np.ndarray] = []
    rows: list[dict[str, float | str]] = []
    for index, client_split in enumerate(client_splits):
        prediction_standardized = client_split.x @ parameters[index]
        prediction = dataset.inverse_target(index, prediction_standardized)
        target = dataset.inverse_target(index, client_split.y)
        error = prediction - target
        absolute = np.abs(error)
        squared = error**2
        valid_mape = np.abs(target) >= mape_min_speed
        percentage = (
            absolute[valid_mape] / np.abs(target[valid_mape]) * 100.0
            if np.any(valid_mape)
            else np.array([], dtype=np.float64)
        )
        absolute_errors.append(absolute)
        squared_errors.append(squared)
        percentage_errors.append(percentage)
        rows.append(
            {
                "client_id": dataset.client_ids[index],
                "mae": float(absolute.mean()),
                "rmse": float(np.sqrt(squared.mean())),
                "mape": float(percentage.mean()) if percentage.size else float("nan"),
                "samples": int(target.size),
            }
        )

    all_absolute = np.concatenate(absolute_errors)
    all_squared = np.concatenate(squared_errors)
    nonempty_percentage = [item for item in percentage_errors if item.size]
    all_percentage = (
        np.concatenate(nonempty_percentage)
        if nonempty_percentage
        else np.array([np.nan])
    )
    return (
        MetricSummary(
            mae=float(all_absolute.mean()),
            rmse=float(np.sqrt(all_squared.mean())),
            mape=float(np.nanmean(all_percentage)),
        ),
        rows,
    )
