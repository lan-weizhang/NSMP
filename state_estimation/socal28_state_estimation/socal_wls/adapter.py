from __future__ import annotations

import copy
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


PHASES = ("a", "b", "c")
PRUNE_NODE_MEMBER = "bus_1130"

# This is the measured-injection map used by Caltech's official BIM notebook.
# Each tuple is: representative voltage bus, measured branch, current sign, meter.
MEASURED_SITES = (
    ("bus_1038", "line_381", +1.0, "egauge_21"),
    ("bus_1119", "cb_146", -1.0, "egauge_7"),
    ("bus_1083", "cb_137", -1.0, "egauge_9"),
    ("bus_1098", "cb_140", -1.0, "egauge_13"),
    ("bus_1107", "cb_143", -1.0, "egauge_3"),
    ("bus_1070", "cb_134", -1.0, "egauge_11"),
)


@dataclass(frozen=True)
class SoCalSnapshot:
    timestamp: str
    voltage_by_node: dict[int, np.ndarray]
    current_by_node: dict[int, np.ndarray]


@dataclass(frozen=True)
class SoCalCase:
    node_names: tuple[str, ...]
    edge_names: tuple[str, ...]
    tree_edges: tuple[tuple[int, int], ...]
    ybus: np.ndarray
    voltage_bases: np.ndarray
    current_bases: np.ndarray
    measured_nodes: tuple[int, ...]
    snapshots: tuple[SoCalSnapshot, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def state_dimension(self) -> int:
        return self.num_nodes * 6


def _import_official_utils(official_repo: Path):
    """Import only the official topology/phasor helpers.

    The topology module imports cvxpy and ntplib at module import time even
    though neither package is used by the preprocessing below. Lightweight
    stubs keep this adapter independent of optional solver/network packages.
    """

    utils_dir = str(official_repo / "utils")
    if not (official_repo / "utils" / "topology.py").is_file():
        raise FileNotFoundError(
            "SoCal upstream utilities are missing. Follow the source and data "
            "preparation steps in state_estimation/README.md."
        )
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    if "cvxpy" not in sys.modules:
        sys.modules["cvxpy"] = types.ModuleType("cvxpy")
    if "ntplib" not in sys.modules:
        ntplib = types.ModuleType("ntplib")
        ntplib.NTPClient = object
        sys.modules["ntplib"] = ntplib
    return (
        importlib.import_module("topology"),
        importlib.import_module("utils"),
        importlib.import_module("phasor_utils"),
        importlib.import_module("global_param"),
    )


def _contains_bus(group_name: str, bus_name: str) -> bool:
    return bus_name in group_name.split("|")


def _load_csv_window(path: Path, start: np.datetime64, end: np.datetime64) -> dict[str, np.ndarray]:
    frame = pd.read_csv(
        path,
        usecols=["t", "frequency", "rms", "phase_angle_harmonic_0"],
    )
    timestamp = pd.to_datetime(frame.pop("t"), errors="raise").to_numpy(dtype="datetime64[us]")
    # Keep a small margin because each meter timestamp is offset from the
    # shared trigger timestamp and is aligned to it by the official routine.
    margin = np.timedelta64(2, "s")
    mask = (timestamp >= start - margin) & (timestamp < end + margin)
    output = {"t": timestamp[mask]}
    for column in frame.columns:
        output[column] = frame[column].to_numpy(dtype=float)[mask]
    # The official aligner accepts preloaded dictionaries, but in that branch
    # expects the convenience alias that its file-loading branch normally
    # creates internally.
    output["phase"] = output["phase_angle_harmonic_0"].copy()
    return output


def _complex_to_real_operator(operator: np.ndarray, num_nodes: int) -> np.ndarray:
    operator = np.asarray(operator, dtype=complex)
    rows = operator.shape[0]
    design = np.zeros((2 * rows, num_nodes * 6), dtype=float)
    for output in range(rows):
        for complex_column in range(operator.shape[1]):
            node = complex_column // 3
            phase = complex_column % 3
            coefficient = operator[output, complex_column]
            real_column = node * 6 + phase
            imag_column = node * 6 + 3 + phase
            design[output, real_column] = coefficient.real
            design[output, imag_column] = -coefficient.imag
            design[rows + output, real_column] = coefficient.imag
            design[rows + output, imag_column] = coefficient.real
    return design


def load_socal_case(
    official_repo: str | Path,
    *,
    start: str = "2024-11-14T07:00:00",
    end: str = "2024-11-14T07:05:00",
    max_snapshots: int | None = None,
    base_power_va: float = 1_000_000.0,
) -> SoCalCase:
    official_repo = Path(official_repo).resolve()
    topology, official_utils, phasor_utils, global_param = _import_official_utils(official_repo)
    start_np = np.datetime64(start, "us")
    end_np = np.datetime64(end, "us")
    if end_np <= start_np:
        raise ValueError("end must be later than start")

    network_dir = official_repo / "sample_dataset" / "topology" / "network_files" / "circuit3"
    net_data = topology.load_net_files([str(network_dir)], t=start)
    parents = official_utils.dict_parents(
        official_utils.load_json(global_param.FILE_PATHS["element_inheritance"], raise_err=True)
    )
    nodes, edges, decorators = topology.net2graph(
        net_data,
        parents,
        fix_implicit=False,
        connected_edges=True,
        print_warning=False,
    )
    electrical_nodes, electrical_edges, _ = topology.graph2electrical_graph(
        copy.deepcopy(nodes), copy.deepcopy(edges), {}, net_data, parents
    )

    root = next(name for name in electrical_nodes if _contains_bus(name, "bus_1038"))
    graph = nx.Graph(topology.to_edge_list(electrical_edges))
    keep = nx.node_connected_component(graph, root)
    electrical_nodes = {name: value for name, value in electrical_nodes.items() if name in keep}
    electrical_edges = {
        name: value
        for name, value in electrical_edges.items()
        if value["source"] in keep and value["target"] in keep
    }
    prune = next(
        (name for name in electrical_nodes if _contains_bus(name, PRUNE_NODE_MEMBER)), None
    )
    if prune is not None:
        del electrical_nodes[prune]
        electrical_edges = {
            name: value
            for name, value in electrical_edges.items()
            if prune not in (value["source"], value["target"])
        }

    node_names = tuple(electrical_nodes)
    node_lookup = {name: index for index, name in enumerate(node_names)}
    tree_edges = tuple(
        (node_lookup[value["source"]], node_lookup[value["target"]])
        for value in electrical_edges.values()
    )
    if len(node_names) != 12 or len(tree_edges) != 11 or not nx.is_tree(nx.Graph(tree_edges)):
        raise RuntimeError(
            f"Unexpected A-side electrical graph: {len(node_names)} nodes, "
            f"{len(tree_edges)} edges."
        )

    ybus = topology.network_Y_matrix(electrical_edges, list(node_names), net_data)
    voltage_bases = np.empty(len(node_names), dtype=float)
    for index, group in enumerate(node_names):
        values = {
            float(net_data["Bus"][bus]["nominal_voltage"])
            for bus in group.split("|")
            if bus in net_data["Bus"] and "nominal_voltage" in net_data["Bus"][bus]
        }
        if len(values) != 1:
            raise RuntimeError(f"Ambiguous nominal voltage for {group}: {values}")
        voltage_bases[index] = values.pop()
    current_bases = base_power_va / (3.0 * voltage_bases)

    phasor_dir = official_repo / "sample_dataset" / "phasors"
    selected_meters = {site[3] for site in MEASURED_SITES}
    input_data: dict[str, dict[str, np.ndarray]] = {}
    for meter_name, decorator in decorators.items():
        if meter_name not in selected_meters:
            continue
        meter = topology.get_element(decorator, net_data)
        for register in meter["registers"]:
            if (
                register.get("unit") not in ("V", "A")
                or register.get("value") not in ("L1", "L2", "L3", "S1", "S2", "S3")
                or "element" not in register
            ):
                continue
            path = phasor_dir / f"{meter_name}-{register['value']}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing selected phasor file: {path}")
            input_data[register["element"]] = _load_csv_window(path, start_np, end_np)

    shared_time = pd.to_datetime(
        pd.read_csv(phasor_dir / "t.csv", usecols=["t"])["t"], errors="raise"
    ).to_numpy(dtype="datetime64[us]")
    shared_time = shared_time[(shared_time >= start_np) & (shared_time < end_np)]
    aligned_time, aligned = phasor_utils.align_phasors(
        input_data,
        time_column_file=shared_time,
        columns=["phase", "rms"],
        ref="bus_1038.ag",
        delta_t_threshold=1.0,
    )
    if max_snapshots is not None:
        aligned_time = aligned_time[:max_snapshots]
    if not len(aligned_time):
        raise RuntimeError("No fully aligned phasor snapshots in the requested interval.")

    measured_nodes: list[int] = []
    site_info: list[tuple[int, str, str, float]] = []
    for voltage_bus, current_element, sign, meter_name in MEASURED_SITES:
        node = next(index for index, group in enumerate(node_names) if _contains_bus(group, voltage_bus))
        measured_nodes.append(node)
        site_info.append((node, voltage_bus, current_element, sign))

    snapshots: list[SoCalSnapshot] = []
    for time_index, timestamp in enumerate(aligned_time):
        voltage_by_node: dict[int, np.ndarray] = {}
        current_by_node: dict[int, np.ndarray] = {}
        for node, voltage_bus, current_element, sign in site_info:
            voltage_by_node[node] = np.asarray(
                [
                    aligned[f"{voltage_bus}.{phase}g"]["rms"][time_index]
                    * np.exp(
                        1j
                        * np.deg2rad(
                            aligned[f"{voltage_bus}.{phase}g"]["phase"][time_index]
                        )
                    )
                    for phase in PHASES
                ],
                dtype=complex,
            )
            current_by_node[node] = sign * np.asarray(
                [
                    aligned[f"{current_element}.{phase}"]["rms"][time_index]
                    * np.exp(
                        1j
                        * np.deg2rad(
                            aligned[f"{current_element}.{phase}"]["phase"][time_index]
                        )
                    )
                    for phase in PHASES
                ],
                dtype=complex,
            )
        snapshots.append(
            SoCalSnapshot(
                timestamp=str(timestamp),
                voltage_by_node=voltage_by_node,
                current_by_node=current_by_node,
            )
        )

    return SoCalCase(
        node_names=node_names,
        edge_names=tuple(electrical_edges),
        tree_edges=tree_edges,
        ybus=np.asarray(ybus, dtype=complex),
        voltage_bases=voltage_bases,
        current_bases=current_bases,
        measured_nodes=tuple(measured_nodes),
        snapshots=tuple(snapshots),
    )


def build_factors(
    case: SoCalCase,
    snapshot: SoCalSnapshot,
    *,
    current_sigma_pu: float = 5e-3,
    voltage_sigma_pu: float = 5e-3,
    zero_injection_sigma_pu: float = 1e-3,
):
    # Imported here so this adapter remains usable for topology/data inspection
    # without imposing the location of the generic solver package.
    from epfl_wls.quadratic import QuadraticFactor

    num_nodes = case.num_nodes
    voltage_scaling = np.repeat(case.voltage_bases, 3)
    normalized_current_operator = case.ybus @ np.diag(voltage_scaling)
    current_real_global = _complex_to_real_operator(normalized_current_operator, num_nodes)
    factors = []
    for node in range(num_nodes):
        row_slice = slice(3 * node, 3 * node + 3)
        complex_rows = normalized_current_operator[row_slice]
        support_nodes = [
            candidate
            for candidate in range(num_nodes)
            if np.any(np.abs(complex_rows[:, 3 * candidate : 3 * candidate + 3]) > 1e-12)
        ]
        if node not in support_nodes:
            support_nodes.append(node)
        scope = tuple(
            coordinate
            for support_node in sorted(support_nodes)
            for coordinate in range(6 * support_node, 6 * support_node + 6)
        )

        current_rows = np.r_[np.arange(3 * node, 3 * node + 3), np.arange(3 * num_nodes + 3 * node, 3 * num_nodes + 3 * node + 3)]
        current_design = current_real_global[np.ix_(current_rows, np.asarray(scope))]
        current_design = current_design / case.current_bases[node]
        rows = [current_design]
        if node in snapshot.current_by_node:
            current = snapshot.current_by_node[node] / case.current_bases[node]
            observations = [np.concatenate([current.real, current.imag])]
            precisions = [np.full(6, 1.0 / current_sigma_pu**2)]
        else:
            observations = [np.zeros(6)]
            precisions = [np.full(6, 1.0 / zero_injection_sigma_pu**2)]

        if node in snapshot.voltage_by_node:
            selection = np.zeros((3, num_nodes * 3), dtype=complex)
            selection[:, 3 * node : 3 * node + 3] = np.eye(3)
            voltage_design_global = _complex_to_real_operator(selection, num_nodes)
            rows.append(voltage_design_global[:, np.asarray(scope)])
            voltage = snapshot.voltage_by_node[node] / case.voltage_bases[node]
            observations.append(np.concatenate([voltage.real, voltage.imag]))
            precisions.append(np.full(6, 1.0 / voltage_sigma_pu**2))

        factors.append(
            QuadraticFactor.from_least_squares(
                scope,
                np.vstack(rows),
                np.concatenate(observations),
                np.concatenate(precisions),
                name=f"bus_{node:02d}",
            )
        )
    return factors
