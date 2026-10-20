from pathlib import Path

import numpy as np

from epfl_wls.epfl_cfg import build_ybus_from_cfg, parse_epfl_cfg


def test_cfg_parser_and_ybus(tmp_path: Path):
    cfg = tmp_path / "CFG.dat"
    cfg.write_text(
        "\n".join(
            [
                "[base_voltage]\t11547.0054",
                "[node_names]\tA\tB",
                "[node_numbers]\t1\t2",
                "[PMU_ID]\t2\t3",
                "[SE_input]\tVI\tI",
                "[incidence_matrix 1ph]\t11",
                "[line_names]\tline12",
                "[line_lengths-km]\t1.0",
                "[line R-ohm/km]\t1,0,0;0,1,0;0,0,1",
                "[line X-ohm/km]\t1,0,0;0,1,0;0,0,1",
                "[line B-uS/km]\t0,0,0;0,0,0;0,0,0",
            ]
        ),
        encoding="utf-8",
    )
    topology = parse_epfl_cfg(cfg)
    ybus = build_ybus_from_cfg(topology, base_power_va=1e6)
    assert topology.line_endpoints == ((0, 1),)
    assert ybus.shape == (6, 6)
    assert np.allclose(ybus[:3, :3], -ybus[:3, 3:])
    assert np.allclose(ybus, ybus.T)
