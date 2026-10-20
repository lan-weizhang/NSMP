from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from .quadratic import QuadraticFactor


PHASES = ("a", "b", "c")


@dataclass
class MeasurementSnapshot:
    timestamp: str
    voltage: dict[str, np.ndarray]
    current: dict[str, np.ndarray]
    per_unit: bool = False


@dataclass
class NetworkSpec:
    num_buses: int
    phases: int
    pmu_bus_map: dict[str, int]
    reference_voltage_buses: set[int]
    communication_edges: list[tuple[int, int]]
    nsmp_root: int
    voltage_sigma_pu: float
    current_sigma_pu: float
    base_line_line_voltage_v: float
    base_power_va: float
    voltage_scale_to_volts: float = 1.0
    current_scale_to_amps: float = 1.0
    current_sign_by_pmu: dict[str, float] = field(default_factory=dict)

    @property
    def pmu_ids(self) -> list[str]:
        return list(self.pmu_bus_map)

    @property
    def state_dimension(self) -> int:
        return self.num_buses * 2 * self.phases

    @property
    def phase_voltage_base_v(self) -> float:
        return self.base_line_line_voltage_v / np.sqrt(3.0)

    @property
    def current_base_a(self) -> float:
        return self.base_power_va / (
            np.sqrt(3.0) * self.base_line_line_voltage_v
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "NetworkSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            num_buses=int(payload["num_buses"]),
            phases=int(payload.get("phases", 3)),
            pmu_bus_map={str(key): int(value) for key, value in payload["pmu_bus_map"].items()},
            reference_voltage_buses={int(item) for item in payload["reference_voltage_buses"]},
            communication_edges=[tuple(map(int, edge)) for edge in payload["communication_edges"]],
            nsmp_root=int(payload["nsmp_root"]),
            voltage_sigma_pu=float(payload.get("voltage_sigma_pu", 1e-3)),
            current_sigma_pu=float(payload.get("current_sigma_pu", 5e-3)),
            base_line_line_voltage_v=float(payload.get("base_line_line_voltage_v", 20_000.0)),
            base_power_va=float(payload.get("base_power_va", 1_000_000.0)),
            voltage_scale_to_volts=float(payload.get("voltage_scale_to_volts", 1.0)),
            current_scale_to_amps=float(payload.get("current_scale_to_amps", 1.0)),
            current_sign_by_pmu={
                str(key): float(value)
                for key, value in payload.get("current_sign_by_pmu", {}).items()
            },
        )


def load_ybus(path: str | Path, *, key: str = "ybus") -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            if key not in archive:
                raise KeyError(f"{path} has no array named {key!r}.")
            ybus = archive[key]
    elif path.suffix.lower() == ".npy":
        ybus = np.load(path)
    else:
        raise ValueError("Ybus must be supplied as a complex .npy or .npz array.")
    ybus = np.asarray(ybus, dtype=complex)
    if ybus.ndim != 2 or ybus.shape[0] != ybus.shape[1]:
        raise ValueError("Ybus must be a square complex matrix.")
    return ybus


def bus_coordinates(bus: int, phases: int = 3) -> tuple[int, ...]:
    start = bus * 2 * phases
    return tuple(range(start, start + 2 * phases))


def state_to_complex_voltage(solution: np.ndarray, num_buses: int, phases: int) -> np.ndarray:
    solution = np.asarray(solution, dtype=float).reshape(-1)
    voltage = np.zeros((num_buses, phases), dtype=complex)
    for bus in range(num_buses):
        start = bus * 2 * phases
        voltage[bus] = (
            solution[start : start + phases]
            + 1j * solution[start + phases : start + 2 * phases]
        )
    return voltage


def complex_voltage_to_state(voltage: np.ndarray) -> np.ndarray:
    voltage = np.asarray(voltage, dtype=complex)
    rows: list[np.ndarray] = []
    for bus in range(voltage.shape[0]):
        rows.append(np.concatenate([voltage[bus].real, voltage[bus].imag]))
    return np.concatenate(rows)


def _complex_operator_to_real(
    operator: np.ndarray, *, num_buses: int, phases: int
) -> np.ndarray:
    operator = np.asarray(operator, dtype=complex)
    design = np.zeros((2 * operator.shape[0], num_buses * 2 * phases), dtype=float)
    for output in range(operator.shape[0]):
        for complex_column in range(operator.shape[1]):
            bus = complex_column // phases
            phase = complex_column % phases
            coefficient = operator[output, complex_column]
            real_column = bus * 2 * phases + phase
            imag_column = bus * 2 * phases + phases + phase
            design[output, real_column] = coefficient.real
            design[output, imag_column] = -coefficient.imag
            design[operator.shape[0] + output, real_column] = coefficient.imag
            design[operator.shape[0] + output, imag_column] = coefficient.real
    return design


def _voltage_selection(bus: int, num_buses: int, phases: int) -> np.ndarray:
    selection = np.zeros((phases, num_buses * phases), dtype=complex)
    selection[:, bus * phases : (bus + 1) * phases] = np.eye(phases)
    return selection


def _to_per_unit(
    snapshot: MeasurementSnapshot, spec: NetworkSpec
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if snapshot.per_unit:
        voltage = {
            key: np.asarray(value, dtype=complex) for key, value in snapshot.voltage.items()
        }
        current = {
            key: np.asarray(value, dtype=complex) for key, value in snapshot.current.items()
        }
    else:
        voltage = {
            key: np.asarray(value, dtype=complex)
            * spec.voltage_scale_to_volts
            / spec.phase_voltage_base_v
            for key, value in snapshot.voltage.items()
        }
        current = {
            key: np.asarray(value, dtype=complex)
            * spec.current_scale_to_amps
            / spec.current_base_a
            for key, value in snapshot.current.items()
        }
    current = {
        key: spec.current_sign_by_pmu.get(key, 1.0) * value
        for key, value in current.items()
    }
    return voltage, current


def build_local_factors(
    snapshot: MeasurementSnapshot,
    spec: NetworkSpec,
    ybus: np.ndarray,
    *,
    support_tolerance: float = 1e-12,
) -> list[QuadraticFactor]:
    expected_ybus_shape = (spec.num_buses * spec.phases,) * 2
    if ybus.shape != expected_ybus_shape:
        raise ValueError(
            f"Ybus shape {ybus.shape} does not match {expected_ybus_shape}."
        )
    voltage, current = _to_per_unit(snapshot, spec)
    factors: list[QuadraticFactor] = []
    for pmu_id, bus in spec.pmu_bus_map.items():
        if pmu_id not in current or pmu_id not in voltage:
            raise KeyError(f"Snapshot has no synchronized data for PMU {pmu_id}.")
        current_operator = ybus[
            bus * spec.phases : (bus + 1) * spec.phases, :
        ]
        support_buses = []
        for candidate in range(spec.num_buses):
            block = current_operator[
                :, candidate * spec.phases : (candidate + 1) * spec.phases
            ]
            if np.any(np.abs(block) > support_tolerance):
                support_buses.append(candidate)
        if bus not in support_buses:
            support_buses.append(bus)
        scope = tuple(
            coordinate
            for support_bus in sorted(support_buses)
            for coordinate in bus_coordinates(support_bus, spec.phases)
        )
        current_design_global = _complex_operator_to_real(
            current_operator, num_buses=spec.num_buses, phases=spec.phases
        )
        rows = [current_design_global[:, np.asarray(scope, dtype=int)]]
        current_value = np.asarray(current[pmu_id], dtype=complex)
        observations = [np.concatenate([current_value.real, current_value.imag])]
        precisions = [
            np.full(2 * spec.phases, 1.0 / spec.current_sigma_pu**2)
        ]
        if bus in spec.reference_voltage_buses:
            voltage_operator = _voltage_selection(
                bus, spec.num_buses, spec.phases
            )
            voltage_design_global = _complex_operator_to_real(
                voltage_operator, num_buses=spec.num_buses, phases=spec.phases
            )
            rows.append(voltage_design_global[:, np.asarray(scope, dtype=int)])
            voltage_value = np.asarray(voltage[pmu_id], dtype=complex)
            observations.append(
                np.concatenate([voltage_value.real, voltage_value.imag])
            )
            precisions.append(
                np.full(2 * spec.phases, 1.0 / spec.voltage_sigma_pu**2)
            )
        factors.append(
            QuadraticFactor.from_least_squares(
                scope,
                np.vstack(rows),
                np.concatenate(observations),
                np.concatenate(precisions),
                name=f"PMU_{pmu_id}_bus_{bus}",
            )
        )
    return factors


def voltage_metrics(
    solution: np.ndarray,
    snapshot: MeasurementSnapshot,
    spec: NetworkSpec,
    *,
    exclude_reference: bool = True,
) -> dict[str, float]:
    measured_voltage, _ = _to_per_unit(snapshot, spec)
    estimated = state_to_complex_voltage(
        solution, spec.num_buses, spec.phases
    )
    magnitude_errors: list[np.ndarray] = []
    angle_errors: list[np.ndarray] = []
    for pmu_id, bus in spec.pmu_bus_map.items():
        if exclude_reference and bus in spec.reference_voltage_buses:
            continue
        measured = np.asarray(measured_voltage[pmu_id], dtype=complex)
        magnitude_errors.append(np.abs(estimated[bus]) - np.abs(measured))
        angle_difference = np.angle(estimated[bus]) - np.angle(measured)
        wrapped = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))
        angle_errors.append(np.rad2deg(wrapped))
    if not magnitude_errors:
        return {"voltage_magnitude_rmse_pu": np.nan, "voltage_angle_mae_deg": np.nan}
    magnitude = np.concatenate(magnitude_errors)
    angle = np.concatenate(angle_errors)
    return {
        "voltage_magnitude_rmse_pu": float(np.sqrt(np.mean(magnitude**2))),
        "voltage_angle_mae_deg": float(np.mean(np.abs(angle))),
    }


def synthetic_epfl_problem(
    *, num_snapshots: int = 8, seed: int = 20260725
) -> tuple[NetworkSpec, np.ndarray, list[MeasurementSnapshot], list[np.ndarray]]:
    """Generate a six-bus, three-phase self-test; this is not real-data evidence."""

    rng = np.random.default_rng(seed)
    num_buses = 6
    phases = 3
    ybus = np.zeros((num_buses * phases, num_buses * phases), dtype=complex)
    series = 5.0 - 12.0j
    for first in range(num_buses - 1):
        second = first + 1
        for phase in range(phases):
            i = first * phases + phase
            j = second * phases + phase
            ybus[i, i] += series
            ybus[j, j] += series
            ybus[i, j] -= series
            ybus[j, i] -= series
    ybus += np.eye(num_buses * phases) * (1e-3 + 2e-3j)
    spec = NetworkSpec(
        num_buses=num_buses,
        phases=phases,
        pmu_bus_map={str(index + 1): index for index in range(5)},
        reference_voltage_buses={0},
        communication_edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
        nsmp_root=2,
        voltage_sigma_pu=2e-4,
        current_sigma_pu=5e-4,
        base_line_line_voltage_v=20_000.0,
        base_power_va=1_000_000.0,
    )
    snapshots: list[MeasurementSnapshot] = []
    states: list[np.ndarray] = []
    phase_offsets = np.deg2rad(np.array([0.0, -120.0, 120.0]))
    for time_index in range(num_snapshots):
        voltage = np.zeros((num_buses, phases), dtype=complex)
        for bus in range(num_buses):
            magnitude = 1.0 - 0.004 * bus + 0.002 * np.sin(0.4 * time_index + bus)
            angle = phase_offsets - 0.006 * bus + 0.002 * np.cos(0.3 * time_index + bus)
            voltage[bus] = magnitude * np.exp(1j * angle)
        current = (ybus @ voltage.reshape(-1)).reshape(num_buses, phases)
        measured_voltage: dict[str, np.ndarray] = {}
        measured_current: dict[str, np.ndarray] = {}
        for pmu_id, bus in spec.pmu_bus_map.items():
            measured_voltage[pmu_id] = voltage[bus] + spec.voltage_sigma_pu * (
                rng.standard_normal(phases) + 1j * rng.standard_normal(phases)
            )
            measured_current[pmu_id] = current[bus] + spec.current_sigma_pu * (
                rng.standard_normal(phases) + 1j * rng.standard_normal(phases)
            )
        snapshots.append(
            MeasurementSnapshot(
                timestamp=f"synthetic-{time_index:04d}",
                voltage=measured_voltage,
                current=measured_current,
                per_unit=True,
            )
        )
        states.append(complex_voltage_to_state(voltage))
    return spec, ybus, snapshots, states
