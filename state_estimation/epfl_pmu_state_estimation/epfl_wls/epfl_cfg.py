from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EPFLTopology:
    base_phase_voltage_v: float
    node_names: tuple[str, ...]
    node_numbers: tuple[int, ...]
    pmu_ids: tuple[str, ...]
    se_inputs: tuple[str, ...]
    line_names: tuple[str, ...]
    line_endpoints: tuple[tuple[int, int], ...]
    line_lengths_km: tuple[float, ...]
    resistance_ohm_per_km: np.ndarray
    reactance_ohm_per_km: np.ndarray
    susceptance_us_per_km: np.ndarray


def _matrix(text: str) -> np.ndarray:
    return np.asarray(
        [[float(value) for value in row.split(",")] for row in text.split(";")],
        dtype=float,
    )


def parse_epfl_cfg(path: str | Path) -> EPFLTopology:
    """Parse the first complete topology block in EPFL's repeated CFG.dat."""

    wanted = {
        "base_voltage",
        "node_names",
        "node_numbers",
        "PMU_ID",
        "SE_input",
        "incidence_matrix 1ph",
        "line_names",
        "line_lengths-km",
        "line R-ohm/km",
        "line X-ohm/km",
        "line B-uS/km",
    }
    values: dict[str, list[str]] = {}
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in raw_line.split("\t") if part.strip()]
        if not parts or not parts[0].startswith("[") or not parts[0].endswith("]"):
            continue
        key = parts[0][1:-1]
        if key in wanted and key not in values:
            values[key] = parts[1:]
        if wanted.issubset(values):
            break
    missing = sorted(wanted - values.keys())
    if missing:
        raise ValueError(f"CFG is missing fields: {missing}")

    node_numbers = tuple(int(item) for item in values["node_numbers"])
    incidence = values["incidence_matrix 1ph"]
    endpoints: list[tuple[int, int]] = []
    for encoded in incidence:
        positions = tuple(index for index, marker in enumerate(encoded) if marker == "1")
        if len(positions) != 2:
            raise ValueError(f"Invalid one-phase incidence row {encoded!r}")
        endpoints.append((positions[0], positions[1]))
    line_names = tuple(values["line_names"])
    lengths = tuple(float(item) for item in values["line_lengths-km"])
    if not (len(line_names) == len(lengths) == len(endpoints)):
        raise ValueError("CFG line names, lengths, and incidence rows disagree")

    return EPFLTopology(
        base_phase_voltage_v=float(values["base_voltage"][0]),
        node_names=tuple(values["node_names"]),
        node_numbers=node_numbers,
        pmu_ids=tuple(values["PMU_ID"]),
        se_inputs=tuple(values["SE_input"]),
        line_names=line_names,
        line_endpoints=tuple(endpoints),
        line_lengths_km=lengths,
        resistance_ohm_per_km=_matrix(values["line R-ohm/km"][0]),
        reactance_ohm_per_km=_matrix(values["line X-ohm/km"][0]),
        susceptance_us_per_km=_matrix(values["line B-uS/km"][0]),
    )


def build_ybus_from_cfg(
    topology: EPFLTopology, *, base_power_va: float = 1_000_000.0
) -> np.ndarray:
    """Construct a three-phase per-unit Ybus using a nominal-pi cable model."""

    phases = topology.resistance_ohm_per_km.shape[0]
    num_buses = len(topology.node_numbers)
    shape = (num_buses * phases, num_buses * phases)
    ybus_si = np.zeros(shape, dtype=complex)
    z_per_km = topology.resistance_ohm_per_km + 1j * topology.reactance_ohm_per_km
    b_per_km = topology.susceptance_us_per_km * 1e-6
    for (first, second), length in zip(
        topology.line_endpoints, topology.line_lengths_km
    ):
        series = np.linalg.inv(z_per_km * length)
        shunt = 0.5j * b_per_km * length
        first_slice = slice(first * phases, (first + 1) * phases)
        second_slice = slice(second * phases, (second + 1) * phases)
        ybus_si[first_slice, first_slice] += series + shunt
        ybus_si[second_slice, second_slice] += series + shunt
        ybus_si[first_slice, second_slice] -= series
        ybus_si[second_slice, first_slice] -= series
    base_line_line_voltage_v = np.sqrt(3.0) * topology.base_phase_voltage_v
    impedance_base_ohm = base_line_line_voltage_v**2 / base_power_va
    return ybus_si * impedance_base_ohm
