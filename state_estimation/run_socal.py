from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent / "socal28_state_estimation"
EPFL_PROJECT = PROJECT_DIR.parent / "epfl_pmu_state_estimation"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(EPFL_PROJECT))

LOCAL_DEPENDENCIES = Path(__file__).resolve().parent.parent / ".runtime/socal_dependencies"
if LOCAL_DEPENDENCIES.is_dir():
    sys.path.insert(0, str(LOCAL_DEPENDENCIES))

from epfl_wls.quadratic import combine_factors
from epfl_wls.solvers import (
    solve_centralized,
    solve_consensus_admm,
    solve_decentralized_gossip,
    solve_nsmp,
)
from socal_wls import build_factors, load_socal_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoCal 28-Bus real-phasor WLS comparison")
    parser.add_argument("--start", default="2024-11-14T07:00:00")
    parser.add_argument("--end", default="2024-11-14T07:05:00")
    parser.add_argument("--snapshots", type=int, default=3)
    parser.add_argument("--admm-rounds", type=int, default=2000)
    parser.add_argument("--admm-rho", type=float, default=0.0)
    parser.add_argument("--gossip-rounds", type=int, default=5000)
    parser.add_argument("--gossip-learning-factor", type=float, default=0.05)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "results" / "latest"
    )
    return parser.parse_args()


def timed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def normalized_voltage_rmse(solution, case, snapshot) -> float:
    errors = []
    for node, measured in snapshot.voltage_by_node.items():
        local = solution[6 * node : 6 * node + 6]
        estimate = local[:3] + 1j * local[3:]
        target = measured / case.voltage_bases[node]
        errors.append(np.abs(estimate - target) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(errors))))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case = load_socal_case(
        PROJECT_DIR / "official_repo",
        start=args.start,
        end=args.end,
        max_snapshots=args.snapshots,
    )

    records = []
    for snapshot_index, snapshot in enumerate(case.snapshots):
        factors = build_factors(case, snapshot)
        combined = combine_factors(factors, name="gap_hessian")
        centralized, centralized_seconds = timed(lambda: solve_centralized(factors))
        nsmp, nsmp_seconds = timed(
            lambda: solve_nsmp(factors, case.tree_edges, root=0)
        )
        admm, admm_seconds = timed(
            lambda: solve_consensus_admm(
                factors,
                rounds=args.admm_rounds,
                rho=args.admm_rho,
                oracle_objective=centralized.objective,
                oracle_solution=centralized.solution,
            )
        )
        gossip, gossip_seconds = timed(
            lambda: solve_decentralized_gossip(
                factors,
                case.tree_edges,
                rounds=args.gossip_rounds,
                learning_rate_factor=args.gossip_learning_factor,
                oracle_objective=centralized.objective,
                oracle_solution=centralized.solution,
            )
        )

        results = (
            ("centralized_wls", centralized, centralized_seconds),
            ("nsmp", nsmp, nsmp_seconds),
            ("consensus_admm", admm, admm_seconds),
            ("decentralized_gossip", gossip, gossip_seconds),
        )
        for method, result, runtime in results:
            difference = result.solution - centralized.solution
            shifted_gap = 0.5 * float(
                difference @ combined.hessian @ difference
            )
            records.append(
                {
                    "snapshot": snapshot_index,
                    "timestamp": snapshot.timestamp,
                    "method": method,
                    "runtime_s": runtime,
                    "rounds": result.rounds,
                    "messages": result.messages,
                    "communicated_scalars": result.communicated_scalars,
                    "state_rmse_vs_centralized_pu": float(
                        np.sqrt(np.mean(difference**2))
                    ),
                    "shifted_quadratic_gap": shifted_gap,
                    "metered_voltage_complex_rmse_pu": normalized_voltage_rmse(
                        result.solution, case, snapshot
                    ),
                }
            )
    per_snapshot = pd.DataFrame(records)
    summary = (
        per_snapshot.groupby("method", sort=False)
        .agg(
            snapshots=("snapshot", "count"),
            runtime_s_mean=("runtime_s", "mean"),
            rounds=("rounds", "max"),
            messages=("messages", "max"),
            communicated_scalars=("communicated_scalars", "max"),
            state_rmse_vs_centralized_pu_mean=(
                "state_rmse_vs_centralized_pu",
                "mean",
            ),
            shifted_quadratic_gap_mean=("shifted_quadratic_gap", "mean"),
            metered_voltage_complex_rmse_pu_mean=(
                "metered_voltage_complex_rmse_pu",
                "mean",
            ),
        )
        .reset_index()
    )
    per_snapshot.to_csv(args.output_dir / "per_snapshot_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    # Match the columns and displayed precision of the paper table.
    paper_rows = summary.set_index("method").loc[
        ["nsmp", "consensus_admm", "decentralized_gossip"]
    ]
    paper_summary = pd.DataFrame({
        "Dataset": ["SoCal"] * 3,
        "Method": ["NSMP", "ADMM", "Gossip"],
        "Error metric": paper_rows["state_rmse_vs_centralized_pu_mean"].to_numpy(),
        "Hop rounds/iterations": paper_rows["rounds"].to_numpy().astype(int),
        "Bytes": (paper_rows["communicated_scalars"].to_numpy() * 8).round().astype(int),
    })
    paper_summary.to_csv(args.output_dir / "summary_compact.csv", index=False, float_format="%.2e")
    metadata = {
        "dataset": "SoCal 28-Bus public real synchrophasor sample",
        "source": "https://github.com/caltech-netlab/digital-twin-dataset",
        "share": "https://caltech.box.com/s/5baxy2ogbalqohpidh1lyxgnnxmv5tuc",
        "time_window": [args.start, args.end],
        "snapshots_used": len(case.snapshots),
        "official_subproblem": "A-side bus-injection V-I state estimation",
        "electrical_nodes_after_official_aggregation_and_pruning": case.num_nodes,
        "three_phase_complex_state_dimension": case.state_dimension,
        "meter_sites": len(case.measured_nodes),
        "zero_injection_nodes": case.num_nodes - len(case.measured_nodes),
        "routing_edges": len(case.tree_edges),
        "admm_rounds": args.admm_rounds,
        "admm_rho_requested": args.admm_rho,
        "gossip_rounds": args.gossip_rounds,
        "gossip_learning_factor": args.gossip_learning_factor,
        "important_scope_note": (
            "SoCal 28-Bus is the dataset name. The official public phasor notebook "
            "solves one A-side connected component, which reduces to 12 electrical "
            "nodes (72 real voltage coordinates), not a literal 28-node estimator."
        ),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
