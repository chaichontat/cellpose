from __future__ import annotations

import numpy as np

from cellpose import core


def test_run_net_skips_empty_tiles_and_scatter_fills_zeros(monkeypatch) -> None:
    class Net:
        nout = 3

    calls: list[np.ndarray] = []

    def fake_forward(_net, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append(batch.copy())
        return (
            np.ones((len(batch), 3, batch.shape[2], batch.shape[3]), dtype=np.float32),
            np.ones((len(batch), 256), dtype=np.float32),
        )

    monkeypatch.setattr(core, "_forward", fake_forward)
    images = np.zeros((2, 32, 32, 3), dtype=np.float32)
    images[1] = 1

    output, _styles = core.run_net(
        Net(),
        images,
        batch_size=1,
        bsize=32,
        single_tile_if_fit=True,
        skip_empty_tiles=True,
    )

    assert len(calls) == 2
    assert all(np.any(batch) for batch in calls)
    np.testing.assert_array_equal(output[0], 0)
    np.testing.assert_allclose(output[1], 1)
