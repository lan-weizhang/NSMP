from pathlib import Path

import numpy as np
import pandas as pd

from epfl_wls.data import load_epfl_snapshots, parse_pmu_files
from epfl_wls.model import NetworkSpec


def _write_pmu(path: Path, offset: float) -> None:
    frame = pd.DataFrame(
        {
            "time": ["2024-01-01T00:00:00.001", "2024-01-01T00:00:00.021"],
            "va": [11547.0 + offset, 11548.0 + offset],
            "vb": [11547.0, 11548.0],
            "vc": [11547.0, 11548.0],
            "vpa": [0.0, 0.1],
            "vpb": [-120.0, -119.9],
            "vpc": [120.0, 120.1],
            "ia": [10.0, 10.1],
            "ib": [10.0, 10.1],
            "ic": [10.0, 10.1],
            "ipa": [0.0, 0.0],
            "ipb": [-120.0, -120.0],
            "ipc": [120.0, 120.0],
        }
    )
    frame.to_csv(path, index=False)


def test_configurable_parser_synchronizes_raw_phasors(tmp_path):
    _write_pmu(tmp_path / "sample_PMUID1.csv", 0.0)
    _write_pmu(tmp_path / "sample_PMUID2.csv", 1.0)
    spec = NetworkSpec(
        num_buses=3,
        phases=3,
        pmu_bus_map={"1": 0, "2": 1},
        reference_voltage_buses={0},
        communication_edges=[(0, 1)],
        nsmp_root=0,
        voltage_sigma_pu=1e-4,
        current_sigma_pu=1e-3,
        base_line_line_voltage_v=20000.0,
        base_power_va=1e6,
    )
    config = {
        "delimiter": ",",
        "timestamp": "time",
        "angle_unit": "degree",
        "voltage_magnitude": ["va", "vb", "vc"],
        "voltage_angle": ["vpa", "vpb", "vpc"],
        "current_magnitude": ["ia", "ib", "ic"],
        "current_angle": ["ipa", "ipb", "ipc"],
    }
    snapshots = load_epfl_snapshots(tmp_path, spec, config)
    assert len(snapshots) == 2
    assert set(snapshots[0].voltage) == {"1", "2"}
    assert not snapshots[0].per_unit
    assert np.isclose(abs(snapshots[0].voltage["1"][0]), 11547.0)


def test_clear_signed_epfl_parser(tmp_path):
    path = tmp_path / "PMU_ID2_sample.asc"
    path.write_text(
        "\n".join(
            [
                "-----BEGIN PGP SIGNED MESSAGE-----",
                "Hash: SHA1",
                "[column_data_names]\ttimestamp\tVa\tAa\tIa\tAIa\tVb\tAb\tIb\tAIb\tVc\tAc\tIc\tAIc",
                "[data]\t1.00\t10\t0\t1\t0\t10\t-2\t1\t-2\t10\t2\t1\t2",
                "-----BEGIN PGP SIGNATURE-----",
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "timestamp": "timestamp",
        "timestamp_unit": "s",
        "angle_unit": "radian",
        "voltage_magnitude_a": "Va",
        "voltage_angle_a": "Aa",
        "current_magnitude_a": "Ia",
        "current_angle_a": "AIa",
        "voltage_magnitude_b": "Vb",
        "voltage_angle_b": "Ab",
        "current_magnitude_b": "Ib",
        "current_angle_b": "AIb",
        "voltage_magnitude_c": "Vc",
        "voltage_angle_c": "Ac",
        "current_magnitude_c": "Ic",
        "current_angle_c": "AIc",
    }
    parsed = parse_pmu_files([path], "2", config)
    assert len(parsed.frame) == 1
    assert np.isclose(parsed.frame.iloc[0]["voltage_a_real"], 10.0)
