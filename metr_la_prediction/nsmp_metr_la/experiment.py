from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import FederatedTimeSeries, load_metr_la
from .graph import TreeEdge, maximum_similarity_spanning_tree
from .methods import (
    MethodResult,
    fit_fedavg,
    fit_fedrep,
    fit_local,
    fit_nsmp,
)
from .metrics import evaluate_parameters
from .quadratic import make_quadratics
from .scenarios import apply_training_scenario


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}.")


def _choose_by_validation(
    dataset: FederatedTimeSeries,
    candidates: list[tuple[float, MethodResult]],
    metric: str = "mape",
) -> tuple[float, MethodResult, list[dict[str, float]]]:
    metric_name = metric.lower()
    if metric_name not in {"mae", "rmse", "mape"}:
        raise ValueError("validation metric must be one of mae, rmse, mape.")
    trials: list[dict[str, float]] = []
    best_value = candidates[0][0]
    best_result = candidates[0][1]
    best_score = float("inf")
    for value, result in candidates:
        metrics, _ = evaluate_parameters(dataset, result.parameters, "val")
        score = float(getattr(metrics, metric_name))
        trials.append(
            {
                "value": float(value),
                "val_mae": metrics.mae,
                "val_rmse": metrics.rmse,
                "val_mape": metrics.mape,
            }
        )
        if score < best_score:
            best_score = score
            best_value = value
            best_result = result
    return best_value, best_result, trials


def _plot_metric(summary_frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for axis, metric, label in zip(
        axes,
        ["test_mae", "test_rmse", "test_mape"],
        ["MAE (mph)", "RMSE (mph)", "MAPE (%)"],
        strict=True,
    ):
        axis.bar(summary_frame["method"], summary_frame[metric], color="#4C78A8")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_histories(results: list[MethodResult], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    plotted = False
    for result in results:
        if not result.history:
            continue
        rounds = [item["round"] for item in result.history]
        scores = [item["val_rmse"] for item in result.history]
        axis.plot(rounds, scores, label=result.name)
        plotted = True
    if plotted:
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Validation RMSE (mph)")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _evaluate_and_save(
    dataset: FederatedTimeSeries,
    results: list[MethodResult],
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_client_rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for result in results:
        validation, _ = evaluate_parameters(dataset, result.parameters, "val")
        test, client_rows = evaluate_parameters(dataset, result.parameters, "test")
        rows.append(
            {
                "method": result.name,
                "val_mae": validation.mae,
                "val_rmse": validation.rmse,
                "val_mape": validation.mape,
                "test_mae": test.mae,
                "test_rmse": test.rmse,
                "test_mape": test.mape,
                "runtime_seconds": result.runtime_seconds,
                "communication_rounds": result.communication_rounds,
                "messages": result.messages,
                "communication_bytes": result.communication_bytes,
                **{
                    f"meta_{key}": value
                    for key, value in result.metadata.items()
                },
            }
        )
        for client_row in client_rows:
            all_client_rows.append({"method": result.name, **client_row})
        histories[result.name] = result.history
        metadata[result.name] = result.metadata
        np.save(output_dir / f"parameters_{result.name.lower()}.npy", result.parameters)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(all_client_rows).to_csv(
        output_dir / "per_client_metrics.csv", index=False
    )
    (output_dir / "histories.json").write_text(
        json.dumps(histories, indent=2, default=_json_default), encoding="utf-8"
    )
    (output_dir / "method_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )
    _plot_metric(summary, output_dir / "test_metrics.png")
    _plot_histories(results, output_dir / "validation_convergence.png")
    return summary


def run_experiment(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_config = config["data"]
    dataset = load_metr_la(
        speed_path=data_config["speed_path"],
        adjacency_path=data_config.get("adjacency_path"),
        history=int(data_config.get("history", 12)),
        train_fraction=float(data_config.get("train_fraction", 0.7)),
        val_fraction=float(data_config.get("val_fraction", 0.1)),
        num_clients=data_config.get("num_clients"),
        graph_source=data_config.get("graph_source", "adjacency"),
        add_bias=bool(data_config.get("add_bias", True)),
        zero_is_missing=bool(data_config.get("zero_is_missing", True)),
    )
    dataset, scenario_metadata = apply_training_scenario(
        dataset, data_config.get("scenario")
    )
    pd.DataFrame(scenario_metadata["clients"]).to_csv(
        output_dir / "client_training_samples.csv", index=False
    )
    method_config = config["methods"]
    ridge = float(method_config.get("ridge", 1e-3))
    validation_metric = str(method_config.get("validation_metric", "mape"))
    quadratics = make_quadratics(
        dataset.train,
        ridge,
        regularize_bias=bool(method_config.get("regularize_bias", False)),
    )
    edges = maximum_similarity_spanning_tree(dataset.similarity)

    results: list[MethodResult] = [fit_local(quadratics)]
    fedavg_config = method_config.get("fedavg", {})
    results.append(
        fit_fedavg(
            dataset,
            quadratics,
            rounds=int(fedavg_config.get("rounds", 300)),
            local_epochs=int(fedavg_config.get("local_epochs", 5)),
            learning_rate=fedavg_config.get("learning_rate"),
            learning_rate_factor=float(
                fedavg_config.get("learning_rate_factor", 0.8)
            ),
            patience=int(fedavg_config.get("patience", 40)),
        )
    )

    fedrep_config = method_config.get("fedrep", {})
    fedrep_values = [
        int(value)
        for value in fedrep_config.get("shared_lags_grid", [3, 6, 9, 12])
    ]
    fedrep_candidates = [
        (
            value,
            fit_fedrep(
                dataset,
                quadratics,
                shared_lags=value,
                rounds=int(fedrep_config.get("rounds", 300)),
                local_epochs=int(fedrep_config.get("local_epochs", 1)),
                learning_rate=fedrep_config.get("learning_rate"),
                learning_rate_factor=float(
                    fedrep_config.get("learning_rate_factor", 0.8)
                ),
                patience=int(fedrep_config.get("patience", 40)),
            ),
        )
        for value in fedrep_values
    ]
    selected_fedrep, fedrep_result, fedrep_trials = _choose_by_validation(
        dataset, fedrep_candidates, metric=validation_metric
    )
    fedrep_result.metadata["selected_shared_lags"] = int(selected_fedrep)
    fedrep_result.metadata["validation_metric"] = validation_metric
    fedrep_result.metadata["validation_trials"] = json.dumps(fedrep_trials)
    results.append(fedrep_result)

    nsmp_config = method_config.get("nsmp", {})
    nsmp_values = [
        float(value)
        for value in nsmp_config.get(
            "coupling_grid", [0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 10.0]
        )
    ]
    root: int | str = nsmp_config.get("root", "center")
    nsmp_candidates = [
        (
            value,
            fit_nsmp(
                quadratics,
                edges,
                coupling=value,
                root=root,
                normalize_edge_weights=bool(
                    nsmp_config.get("normalize_edge_weights", True)
                ),
                verify_centralized=bool(
                    nsmp_config.get("verify_centralized", True)
                ),
            ),
        )
        for value in nsmp_values
    ]
    selected_nsmp, nsmp_result, nsmp_trials = _choose_by_validation(
        dataset, nsmp_candidates, metric=validation_metric
    )
    nsmp_result.metadata["selected_coupling"] = selected_nsmp
    nsmp_result.metadata["validation_trials"] = json.dumps(nsmp_trials)
    results.append(nsmp_result)

    tree_rows = [
        {
            "sensor_i": dataset.client_ids[edge.first],
            "sensor_j": dataset.client_ids[edge.second],
            "similarity": edge.weight,
        }
        for edge in edges
    ]
    pd.DataFrame(tree_rows).to_csv(output_dir / "mst_edges.csv", index=False)
    resolved_config = {
        **config,
        "resolved": {
            "num_clients": dataset.num_clients,
            "dimension": dataset.dimension,
            "client_ids": dataset.client_ids,
            "selected_fedrep_shared_lags": int(selected_fedrep),
            "selected_nsmp_coupling": selected_nsmp,
            "scenario": {
                key: value
                for key, value in scenario_metadata.items()
                if key != "clients"
            },
        },
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, default=_json_default),
        encoding="utf-8",
    )
    summary = _evaluate_and_save(dataset, results, output_dir)
    summary.insert(0, "scenario", scenario_metadata["name"])
    summary.insert(
        1, "mean_retained_fraction", scenario_metadata["mean_retained_fraction"]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary
