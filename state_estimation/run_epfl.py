#!/usr/bin/env python3
"""Run distributed WLS state-estimation experiments.

The real-data path expects EPFL-format PMU text files plus a per-unit network
admittance matrix supplied by the researcher. Only the paper real-data experiment is exposed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent / "epfl_pmu_state_estimation"
sys.path.insert(0, str(PROJECT_DIR))

from epfl_wls import (
    combine_factors,
    solve_centralized,
    solve_consensus_admm,
    solve_decentralized_gossip,
    solve_nsmp,
)
from epfl_wls.data import load_epfl_snapshots, load_column_config
from epfl_wls.model import (
    NetworkSpec,
    build_local_factors,
    complex_voltage_to_state,
    load_ybus,
    voltage_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data/epfl_wayback_2014_08_04_19h", help="directory containing EPFL PMU TXT/CSV files")
    parser.add_argument("--network-config", type=Path, default=PROJECT_DIR / "configs/epfl_wayback_2014_08_04_19h/epfl_network.json", help="network/PMU JSON configuration for real data")
    parser.add_argument("--column-map", type=Path, default=PROJECT_DIR / "configs/epfl_wayback_2014_08_04_19h/epfl_columns.json", help="raw-file column JSON configuration for real data")
    parser.add_argument("--ybus", type=Path, default=PROJECT_DIR / "configs/epfl_wayback_2014_08_04_19h/epfl_ybus.npz", help="complex per-unit Ybus in .npy or .npz format")
    parser.add_argument("--ybus-key", default="ybus", help="array key when --ybus is an .npz file")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results/latest")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--stride", type=int, default=10000)
    parser.add_argument("--admm-rounds", type=int, default=1000)
    parser.add_argument(
        "--admm-rho", type=float, default=0.0,
        help="ADMM penalty; non-positive selects a curvature-scaled value",
    )
    parser.add_argument("--gossip-rounds", type=int, default=3000)
    parser.add_argument("--gossip-learning-factor", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def timed(call: Callable[[], object]) -> tuple[object, float]:
    start = time.perf_counter()
    value = call()
    return value, time.perf_counter() - start




def reference_flat_start(snapshot, spec: NetworkSpec) -> np.ndarray:
    """Repeat the configured reference-bus phasor at every network bus."""

    reference_bus = min(spec.reference_voltage_buses)
    pmu_id = next(
        key for key, bus in spec.pmu_bus_map.items() if bus == reference_bus
    )
    reference = np.asarray(snapshot.voltage[pmu_id], dtype=complex)
    if not snapshot.per_unit:
        reference = (
            reference * spec.voltage_scale_to_volts / spec.phase_voltage_base_v
        )
    return complex_voltage_to_state(np.tile(reference, (spec.num_buses, 1)))




def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.stride <= 0:
        raise ValueError("--max-samples and --stride must be positive")

    missing = [
        flag
        for flag, value in (
            ("--network-config", args.network_config),
            ("--column-map", args.column_map),
            ("--ybus", args.ybus),
        )
        if value is None
    ]
    if missing:
        raise ValueError("real-data mode also requires " + ", ".join(missing))
    spec = NetworkSpec.from_json(args.network_config)
    ybus = load_ybus(args.ybus, key=args.ybus_key)
    column_config = load_column_config(args.column_map)
    snapshots = load_epfl_snapshots(args.data_dir, spec, column_config)
    true_states = None
    mode_name = "epfl_real_data"

    selected_indices = list(range(0, len(snapshots), args.stride))[: args.max_samples]
    if not selected_indices:
        raise RuntimeError("no synchronized PMU snapshots were loaded")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for sample_number, snapshot_index in enumerate(selected_indices):
        snapshot = snapshots[snapshot_index]
        factors = build_local_factors(snapshot, spec, ybus)
        flat_start = reference_flat_start(snapshot, spec)

        centralized, central_time = timed(lambda: solve_centralized(factors))
        oracle_objective = 0.0
        oracle_hessian = combine_factors(factors, name="stable_gap").hessian
        nsmp, nsmp_time = timed(
            lambda: solve_nsmp(
                factors,
                spec.communication_edges,
                root=spec.nsmp_root,
            )
        )
        admm, admm_time = timed(
            lambda: solve_consensus_admm(
                factors,
                rho=args.admm_rho,
                rounds=args.admm_rounds,
                oracle_objective=oracle_objective,
                oracle_solution=centralized.solution,
                initial_solution=flat_start,
            )
        )
        gossip, gossip_time = timed(
            lambda: solve_decentralized_gossip(
                factors,
                spec.communication_edges,
                rounds=args.gossip_rounds,
                learning_rate_factor=args.gossip_learning_factor,
                oracle_objective=oracle_objective,
                oracle_solution=centralized.solution,
                initial_solution=flat_start,
            )
        )

        initialization_messages = len(spec.communication_edges)
        initialization_scalars = (
            2 * spec.phases * len(spec.communication_edges)
        )
        for iterative_result in (admm, gossip):
            iterative_result.messages += initialization_messages
            iterative_result.communicated_scalars += initialization_scalars
            for item in iterative_result.history:
                item["communicated_scalars"] += initialization_scalars

        methods = (
            ("centralized_wls", centralized, central_time),
            ("nsmp", nsmp, nsmp_time),
            ("consensus_admm", admm, admm_time),
            ("decentralized_gossip", gossip, gossip_time),
        )
        for method, result, runtime in methods:
            difference = result.solution - centralized.solution
            stable_gap = max(
                0.0, 0.5 * float(difference @ oracle_hessian @ difference)
            )
            metrics = voltage_metrics(result.solution, snapshot, spec, exclude_reference=True)
            row: dict[str, object] = {
                "mode": mode_name,
                "sample": sample_number,
                "timestamp": str(snapshot.timestamp),
                "method": method,
                "objective": stable_gap,
                "relative_objective_gap": stable_gap
                / max(1.0, abs(oracle_objective)),
                "runtime_seconds": runtime,
                "rounds": int(result.rounds),
                "messages": int(result.messages),
                "communicated_scalars": int(result.communicated_scalars),
                "rho": result.metadata.get("rho", np.nan),
                "learning_rate": result.metadata.get("learning_rate", np.nan),
                **metrics,
            }
            if true_states is not None:
                truth = true_states[snapshot_index]
                row["state_rmse_pu"] = float(
                    np.sqrt(np.mean((result.solution - truth) ** 2))
                )
            records.append(row)

    details = pd.DataFrame(records)
    summary = (
        details.groupby("method", sort=False)
        .mean(numeric_only=True)
        .reset_index()
        .drop(columns=["sample"], errors="ignore")
    )
    details.to_csv(output_dir / "per_snapshot_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    # Match the columns and displayed precision of the paper table.
    paper_rows = summary.set_index("method").loc[
        ["nsmp", "consensus_admm", "decentralized_gossip"]
    ]
    paper_summary = pd.DataFrame({
        "Dataset": ["EPFL"] * 3,
        "Method": ["NSMP", "ADMM", "Gossip"],
        "Error metric": paper_rows["relative_objective_gap"].to_numpy(),
        "Hop rounds/iterations": paper_rows["rounds"].to_numpy().astype(int),
        "Bytes": (paper_rows["communicated_scalars"].to_numpy() * 8).round().astype(int),
    })
    paper_summary.to_csv(output_dir / "summary_compact.csv", index=False, float_format="%.2e")

    run_metadata = {
        "mode": mode_name,
        "samples": len(selected_indices),
        "stride": args.stride,
        "admm_rounds": args.admm_rounds,
        "admm_rho": args.admm_rho,
        "gossip_rounds": args.gossip_rounds,
        "gossip_learning_factor": args.gossip_learning_factor,
        "seed": args.seed,
        "objective_reporting": (
            "shifted quadratic gap 0.5*(x-x_centralized)'H(x-x_centralized); "
            "centralized reference is zero to avoid constant-term cancellation"
        ),
        "network_config": Path(os.path.relpath(args.network_config.resolve(), PROJECT_DIR)).as_posix() if args.network_config else None,
        "column_map": Path(os.path.relpath(args.column_map.resolve(), PROJECT_DIR)).as_posix() if args.column_map else None,
        "ybus": Path(os.path.relpath(args.ybus.resolve(), PROJECT_DIR)).as_posix() if args.ybus else None,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
