import numpy as np

from cellpose.contrib.packed_infer import (
    compute_stripe_layout,
    pack_planes_to_stripes,
    unpack_stripes_to_planes,
)


def test_compute_layout_three_stripes():
    Ly = 68
    layout = compute_stripe_layout(Ly, bsize=256, pack_k=3, guard=16)
    assert layout is not None
    assert layout.K == 3
    assert layout.guard == 16
    assert layout.starts == (0, 84, 168)  # 68 + 16 = 84 step
    assert layout.stripe_height == 68
    assert layout.bsize == 256


def test_pack_unpack_identity():
    # Random 7 planes, small width, 3 channels
    Lz, Ly, Lx, C = 7, 68, 113, 3
    x = np.random.RandomState(0).randn(Lz, Ly, Lx, C).astype(np.float32)
    layout = compute_stripe_layout(Ly, bsize=256, pack_k=3, guard=16)
    assert layout is not None

    packed, mapping = pack_planes_to_stripes(x, layout)
    # Simulate a network head with 3 outputs (dy, dx, prob) using an injective map of x
    # so we can verify exact reconstruction through unpack.
    y_packed = np.concatenate([
        packed[..., :1].copy(),           # dy from channel 0
        2 * packed[..., 1:2].copy(),      # dx from channel 1
        3 * packed[..., 2:3].copy(),      # prob from channel 2
    ], axis=-1)

    y = unpack_stripes_to_planes(y_packed, mapping, Lz=Lz, Ly=Ly)
    # Recover the source signal and check equality
    x0 = np.concatenate([x[..., 0:1], 2 * x[..., 1:2], 3 * x[..., 2:3]], axis=-1)
    np.testing.assert_allclose(y, x0, rtol=0, atol=0)


def test_pack_unpack_remainder_two():
    # Lz not divisible by K (e.g., 5 with K=3)
    Lz, Ly, Lx, C = 5, 68, 77, 1
    x = np.random.RandomState(1).randn(Lz, Ly, Lx, C).astype(np.float32)
    layout = compute_stripe_layout(Ly, bsize=256, pack_k=3, guard=16)
    assert layout is not None and layout.K == 3
    packed, mapping = pack_planes_to_stripes(x, layout)
    # Mimic 3-head output by tiling
    y_packed = np.tile(packed, (1, 1, 1, 3))
    y = unpack_stripes_to_planes(y_packed, mapping, Lz=Lz, Ly=Ly)
    x0 = np.tile(x, (1, 1, 1, 3))
    np.testing.assert_allclose(y, x0, rtol=0, atol=0)


def test_pack_layout_insufficient_height():
    # If stripes cannot fit at least K=2, layout should be None
    Ly = 200
    layout = compute_stripe_layout(Ly, bsize=256, pack_k=3, guard=60)
    assert layout is None


def test_pack_unpack_exact_multiple():
    # Lz divisible by K (6 with K=3)
    Lz, Ly, Lx, C = 6, 68, 41, 2
    x = np.random.RandomState(2).randn(Lz, Ly, Lx, C).astype(np.float32)
    layout = compute_stripe_layout(Ly, bsize=256, pack_k=3, guard=16)
    assert layout is not None and layout.K == 3
    packed, mapping = pack_planes_to_stripes(x, layout)
    y_packed = packed[..., :1].repeat(3, axis=-1)
    y = unpack_stripes_to_planes(y_packed, mapping, Lz=Lz, Ly=Ly)
    expected = x[..., :1].repeat(3, axis=-1)
    np.testing.assert_allclose(y, expected, rtol=0, atol=0)
