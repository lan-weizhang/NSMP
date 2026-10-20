from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nsmp_metr_la.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run METR-LA low-data and missing-training-window scenarios."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "paper.json",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        help="Optional scenario names. By default all configured scenarios run.",
    )
    parser.add_argument("--num-clients", type=int)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override output_root from the configuration file.",
    )
    return parser.parse_args()


def _plot_comparison(summary: pd.DataFrame, output: Path) -> None:
    metrics = ["test_mae", "test_rmse", "test_mape"]
    labels = ["Test MAE (mph)", "Test RMSE (mph)", "Test MAPE (%)"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for axis, metric, label in zip(axes, metrics, labels, strict=True):
        pivot = summary.pivot(index="scenario", columns="method", values=metric)
        pivot.plot(kind="bar", ax=axis)
        axis.set_ylabel(label)
        axis.set_xlabel("")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
        axis.legend().remove()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root or Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.scenarios or [])
    configured_names = {item["name"] for item in config["scenarios"]}
    unknown = selected - configured_names
    if unknown:
        raise SystemExit(f"Unknown scenarios: {sorted(unknown)}")

    frames: list[pd.DataFrame] = []
    for scenario in config["scenarios"]:
        if selected and scenario["name"] not in selected:
            continue
        experiment = deepcopy(config["experiment"])
        experiment["data"]["scenario"] = scenario
        if args.num_clients is not None:
            experiment["data"]["num_clients"] = args.num_clients
        experiment["output_dir"] = str(output_root / scenario["name"])
        print(f"\n=== Scenario: {scenario['name']} ===")
        frames.append(run_experiment(experiment))

    if not frames:
        raise SystemExit("No scenarios selected.")
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(output_root / "all_scenarios_summary.csv", index=False)
    _plot_comparison(summary, output_root / "scenario_comparison.png")
    columns = [
        "scenario",
        "mean_retained_fraction",
        "method",
        "test_mae",
        "test_rmse",
        "test_mape",
        "communication_bytes",
    ]
    print("\n=== Combined summary ===")
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
