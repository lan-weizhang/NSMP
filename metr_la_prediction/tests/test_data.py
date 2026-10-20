from __future__ import annotations

import numpy as np

from nsmp_metr_la.data import load_metr_la


def test_npz_correlation_loader(tmp_path) -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(size=150)
    values = np.stack(
        [
            base + 0.1 * rng.normal(size=150),
            0.8 * base + 0.2 * rng.normal(size=150),
            rng.normal(size=150),
            0.5 * base + 0.3 * rng.normal(size=150),
        ],
        axis=1,
    )
    path = tmp_path / "speeds.npz"
    np.savez(path, data=values, sensor_ids=np.array(["a", "b", "c", "d"]))
    dataset = load_metr_la(
        path,
        None,
        history=6,
        num_clients=3,
        graph_source="correlation",
        zero_is_missing=False,
    )
    assert dataset.num_clients == 3
    assert dataset.dimension == 7
    assert dataset.train[0].x.shape[0] == int(0.7 * 150) - 6
    np.testing.assert_allclose(dataset.train[0].y.mean(), 0.0, atol=0.15)
