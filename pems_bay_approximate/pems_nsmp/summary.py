"""Save full and compact method-comparison tables from PEMS-BAY runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

SEEDS = (20260728, 20260729, 20260730)
METHODS = ("Local", "FedAvg", "FedRep", "AdaptiveMsg-NSMP")


def run_directory_name(seed: int) -> str:
    base = "pems_bay_6clients_128_hidden4"
    return base if seed == SEEDS[0] else f"{base}_seed{seed}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(results_root: Path, output_root: Path | None = None,
                      seeds: tuple[int, ...] | None = None) -> None:
    source = Path(results_root)
    destination = Path(output_root) if output_root is not None else source
    metadata: dict = {}
    if (source / "per_run_results.csv").is_file():
        raw_rows = read_rows(source / "per_run_results.csv")
        if (source / "run_metadata.json").is_file():
            metadata = json.loads((source / "run_metadata.json").read_text(encoding="utf-8"))
    else:
        raw_rows = []
        runs = []
        for seed in (seeds if seeds is not None else SEEDS):
            folder = source / run_directory_name(seed)
            rows = read_rows(folder / "summary.csv")
            raw_rows.extend({"seed": seed, **row} for row in rows)
            runs.append({"seed": seed, "configuration": json.loads(
                (folder / "metadata.json").read_text(encoding="utf-8"))})
        metadata["runs"] = runs

    if not raw_rows:
        raise ValueError("No experiment results to summarize.")
    for row in raw_rows:
        row["seed"] = int(row["seed"])
        for field in ("rmse", "mae", "mape", "runtime_seconds"):
            row[field] = float(row[field])
        row["communication_bytes"] = int(row["communication_bytes"])
    actual_seeds = sorted({row["seed"] for row in raw_rows})
    for seed in actual_seeds:
        rows = [row for row in raw_rows if row["seed"] == seed]
        if len(rows) != len(METHODS) or {row["method"] for row in rows} != set(METHODS):
            raise ValueError(f"Expected all four methods exactly once for seed {seed}.")

    full_rows = []
    compact_rows = []
    for method in METHODS:
        rows = [row for row in raw_rows if row["method"] == method]
        result = {"method": method, "runs": len(rows)}
        for field in ("rmse", "mae", "mape", "communication_bytes", "runtime_seconds"):
            values = [row[field] for row in rows]
            result[f"{field}_mean"] = mean(values)
            result[f"{field}_sample_std"] = stdev(values) if len(values) > 1 else None
        full_rows.append(result)
        rmse = f"{result['rmse_mean']:.3f}"
        if len(rows) > 1:
            rmse += f" ± {result['rmse_sample_std']:.3f}"
        payload = float(result["communication_bytes_mean"])
        compact_rows.append({"Method": method, "Test RMSE": rmse,
                             "Bytes": int(payload) if payload.is_integer() else payload})

    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "summary.csv", full_rows)
    write_rows(destination / "summary_compact.csv", compact_rows)
    write_rows(destination / "per_run_results.csv", raw_rows)
    metadata.update({"dataset": "PEMS-BAY", "seeds": actual_seeds,
                     "methods": list(METHODS), "standard_deviation": "sample (ddof=1)",
                     "communication": "Float32 numerical payload, 4 bytes per value"})
    (destination / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'Method':<20} {'Test RMSE':<20} {'Bytes':>12}")
    for row in compact_rows:
        print(f"{row['Method']:<20} {row['Test RMSE']:<20} {row['Bytes']:>12}")
