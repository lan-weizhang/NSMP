from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model import MeasurementSnapshot, NetworkSpec, PHASES


@dataclass
class ParsedPMU:
    pmu_id: str
    frame: pd.DataFrame


def _extract_column(frame: pd.DataFrame, selector: str | int) -> pd.Series:
    if isinstance(selector, int):
        return frame.iloc[:, selector]
    if selector in frame.columns:
        return frame[selector]
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): column
        for column in frame.columns
    }
    key = re.sub(r"[^a-z0-9]", "", selector.lower())
    if key not in normalized:
        raise KeyError(
            f"Could not find column {selector!r}. Available columns: {list(frame.columns)}"
        )
    return frame[normalized[key]]


def _read_one_file(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    if path.suffix.lower() == ".asc":
        return _read_clear_signed_epfl(path)
    delimiter = config.get("delimiter", r"[;,\s]+")
    if delimiter == "auto":
        delimiter = None
    header = 0 if config.get("has_header", True) else None
    return pd.read_csv(
        path,
        sep=delimiter,
        engine="python",
        comment=config.get("comment_prefix", config.get("comment", "#")),
        header=header,
        skiprows=int(config.get("skip_rows", 0)),
        on_bad_lines="skip",
    )


def _read_clear_signed_epfl(path: Path) -> pd.DataFrame:
    """Read EPFL's PGP clear-signed ``[data]`` tabular format."""

    columns: list[str] | None = None
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            if line.startswith("[column_data_names]"):
                columns = [item.strip() for item in line.split("\t")[1:] if item.strip()]
            elif line.startswith("[data]"):
                if columns is None:
                    raise ValueError(f"{path} contains data before its column header")
                values = [item.strip() for item in line.split("\t")[1:]]
                if values and values[-1] == "":
                    values.pop()
                if len(values) == len(columns):
                    rows.append(values)
    if columns is None or not rows:
        raise ValueError(f"{path} contains no EPFL clear-signed measurement table")
    return pd.DataFrame(rows, columns=columns)


def load_column_config(path: str | Path) -> dict[str, Any]:
    """Load a column map and accept either grouped lists or phase-wise keys."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for quantity in ("voltage", "current"):
        for field in ("magnitude", "angle"):
            grouped_key = f"{quantity}_{field}"
            if grouped_key not in config:
                continue
            selectors = config[grouped_key]
            if len(selectors) != len(PHASES):
                raise ValueError(f"{grouped_key} must contain one selector per phase")
            for phase, selector in zip(PHASES, selectors):
                config[f"{quantity}_{field}_{phase}"] = selector
    return config


def parse_pmu_files(
    paths: list[Path], pmu_id: str, column_config: dict[str, Any]
) -> ParsedPMU:
    if not paths:
        raise FileNotFoundError(f"No files found for PMU {pmu_id}.")
    frames = [_read_one_file(path, column_config) for path in sorted(paths)]
    raw = pd.concat(frames, ignore_index=True)
    timestamp_raw = _extract_column(raw, column_config["timestamp"])
    timestamp_unit = column_config.get("timestamp_unit")
    if timestamp_unit:
        timestamp = pd.to_datetime(
            pd.to_numeric(timestamp_raw, errors="coerce"),
            unit=timestamp_unit,
            utc=True,
            errors="coerce",
        )
    else:
        timestamp = pd.to_datetime(
            timestamp_raw,
            format=column_config.get("timestamp_format"),
            utc=True,
            errors="coerce",
        )
    canonical = pd.DataFrame({"timestamp": timestamp})
    angle_unit = str(column_config.get("angle_unit", "degree")).lower()
    for quantity in ("voltage", "current"):
        for phase in PHASES:
            magnitude = pd.to_numeric(
                _extract_column(raw, column_config[f"{quantity}_magnitude_{phase}"]),
                errors="coerce",
            )
            angle = pd.to_numeric(
                _extract_column(raw, column_config[f"{quantity}_angle_{phase}"]),
                errors="coerce",
            )
            if angle_unit.startswith("deg"):
                angle = np.deg2rad(angle)
            canonical[f"{quantity}_{phase}_real"] = magnitude * np.cos(angle)
            canonical[f"{quantity}_{phase}_imag"] = magnitude * np.sin(angle)
    canonical = canonical.dropna().sort_values("timestamp")
    canonical = canonical.drop_duplicates("timestamp", keep="last")
    if canonical.empty:
        raise ValueError(f"PMU {pmu_id} files contained no valid rows.")
    return ParsedPMU(pmu_id, canonical)


def load_aligned_snapshots(
    data_dir: str | Path,
    pmu_ids: list[str],
    column_map_path: str | Path,
    *,
    sampling_period_ms: int = 20,
    stride: int = 1,
    max_samples: int | None = None,
) -> list[MeasurementSnapshot]:
    column_config = load_column_config(column_map_path)
    return _load_aligned_snapshots(
        data_dir,
        pmu_ids,
        column_config,
        sampling_period_ms=sampling_period_ms,
        stride=stride,
        max_samples=max_samples,
    )


def load_epfl_snapshots(
    data_dir: str | Path,
    spec: NetworkSpec,
    column_config: dict[str, Any],
    *,
    sampling_period_ms: int = 20,
) -> list[MeasurementSnapshot]:
    """Load and align every PMU listed in ``spec``."""

    normalized = dict(column_config)
    for quantity in ("voltage", "current"):
        for field in ("magnitude", "angle"):
            grouped_key = f"{quantity}_{field}"
            if grouped_key in normalized:
                for phase, selector in zip(PHASES, normalized[grouped_key]):
                    normalized[f"{quantity}_{field}_{phase}"] = selector
    return _load_aligned_snapshots(
        data_dir,
        spec.pmu_ids,
        normalized,
        sampling_period_ms=sampling_period_ms,
    )


def _load_aligned_snapshots(
    data_dir: str | Path,
    pmu_ids: list[str],
    column_config: dict[str, Any],
    *,
    sampling_period_ms: int,
    stride: int = 1,
    max_samples: int | None = None,
) -> list[MeasurementSnapshot]:
    data_dir = Path(data_dir)
    parsed: list[ParsedPMU] = []
    all_files = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".asc"}
    ]
    for pmu_id in pmu_ids:
        pattern = re.compile(
            rf"pmu[_-]?id0*{re.escape(str(pmu_id))}(?:\D|$)", re.IGNORECASE
        )
        matches = [path for path in all_files if pattern.search(path.name)]
        parsed.append(parse_pmu_files(matches, str(pmu_id), column_config))
    rounded_frames: dict[str, pd.DataFrame] = {}
    common_index: pd.DatetimeIndex | None = None
    rounding = f"{sampling_period_ms}ms"
    for item in parsed:
        frame = item.frame.copy()
        frame["timestamp"] = frame["timestamp"].dt.round(rounding)
        frame = frame.groupby("timestamp", as_index=True).first()
        rounded_frames[item.pmu_id] = frame
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)
    if common_index is None or common_index.empty:
        raise ValueError("The PMU files have no timestamps in common after alignment.")
    selected = common_index.sort_values()[:: max(1, stride)]
    if max_samples is not None:
        selected = selected[:max_samples]
    snapshots: list[MeasurementSnapshot] = []
    for timestamp in selected:
        voltage: dict[str, np.ndarray] = {}
        current: dict[str, np.ndarray] = {}
        for pmu_id, frame in rounded_frames.items():
            row = frame.loc[timestamp]
            voltage[pmu_id] = np.asarray(
                [
                    row[f"voltage_{phase}_real"] + 1j * row[f"voltage_{phase}_imag"]
                    for phase in PHASES
                ]
            )
            current[pmu_id] = np.asarray(
                [
                    row[f"current_{phase}_real"] + 1j * row[f"current_{phase}_imag"]
                    for phase in PHASES
                ]
            )
        snapshots.append(
            MeasurementSnapshot(
                timestamp=timestamp.isoformat(),
                voltage=voltage,
                current=current,
                per_unit=False,
            )
        )
    return snapshots
