#!/usr/bin/env python3
"""Prepare archived EPFL PMU data for the distributed WLS experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from epfl_wls.epfl_cfg import build_ybus_from_cfg, parse_epfl_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-power-va", type=float, default=1_000_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topology = parse_epfl_cfg(args.data_dir / "CFG.dat")
    ybus = build_ybus_from_cfg(topology, base_power_va=args.base_power_va)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "epfl_ybus.npz", ybus=ybus)

    physical_edges = [list(edge) for edge in topology.line_endpoints]
    adjacency = [[] for _ in topology.node_numbers]
    for first, second in topology.line_endpoints:
        adjacency[first].append(second)
        adjacency[second].append(first)
    eccentricity = []
    for start in range(len(adjacency)):
        distances = [-1] * len(adjacency)
        distances[start] = 0
        queue = [start]
        for node in queue:
            for neighbor in adjacency[node]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        eccentricity.append(max(distances))
    root = min(range(len(eccentricity)), key=lambda index: eccentricity[index])
    reference_buses = [
        index for index, mode in enumerate(topology.se_inputs) if "V" in mode.upper()
    ]
    network = {
        "num_buses": len(topology.node_numbers),
        "phases": 3,
        "pmu_bus_map": {
            pmu_id: index for index, pmu_id in enumerate(topology.pmu_ids)
        },
        "reference_voltage_buses": reference_buses[:1],
        "communication_edges": physical_edges,
        "nsmp_root": root,
        "voltage_sigma_pu": 0.005,
        "current_sigma_pu": 0.1,
        "base_line_line_voltage_v": float(
            np.sqrt(3.0) * topology.base_phase_voltage_v
        ),
        "base_power_va": args.base_power_va,
        "voltage_scale_to_volts": 1.0,
        "current_scale_to_amps": 1.0,
        "current_sign_by_pmu": {
            pmu_id: (1.0 if index == 0 else -1.0)
            for index, pmu_id in enumerate(topology.pmu_ids)
        },
        "node_names": list(topology.node_names),
        "source_cfg": str((args.data_dir / "CFG.dat").resolve()),
    }
    columns = {
        "timestamp": "timestamp",
        "timestamp_unit": "s",
        "angle_unit": "radian",
        "voltage_magnitude": [
            f"phase{phase}.voltage.magnitude" for phase in "ABC"
        ],
        "voltage_angle": [f"phase{phase}.voltage.angle" for phase in "ABC"],
        "current_magnitude": [
            f"phase{phase}.current.magnitude" for phase in "ABC"
        ],
        "current_angle": [f"phase{phase}.current.angle" for phase in "ABC"],
    }
    (args.output_dir / "epfl_network.json").write_text(
        json.dumps(network, indent=2), encoding="utf-8"
    )
    (args.output_dir / "epfl_columns.json").write_text(
        json.dumps(columns, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(topology.pmu_ids)} PMUs and {len(topology.line_names)} lines")
    print(f"Physical communication edges: {physical_edges}; NSMP root: {root}")
    print(f"Assets written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
