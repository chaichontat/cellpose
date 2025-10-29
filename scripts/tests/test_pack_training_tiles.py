import math
import numpy as np
from unittest.mock import patch

from cellpose.train import (
    _PACK_STRIPE_BORDER,
    _pack_training_tiles,
    _resolve_pack_stripe_height,
    _prepad_height,
    _effective_height,
)

PACK_BORDER = _PACK_STRIPE_BORDER


def _fake_pack_planes_to_stripes(x, layout):
    groups = math.ceil(x.shape[0] / layout.K)
    packed = np.zeros((groups, layout.bsize, x.shape[2], x.shape[3]), dtype=x.dtype)
    mapping = []
    for i in range(x.shape[0]):
        group = min(i // layout.K, groups - 1)
        slot_start = layout.starts[i % layout.K]
        data_start = slot_start + layout.border
        data_end = data_start + layout.stripe_height
        packed[group, data_start:data_end, :, :] = x[i]
        mapping.append((group, data_start, data_end))
    return packed, mapping


def test_pack_training_tiles_auto_height():
    imgs = np.zeros((5, 3, 256, 128), dtype=np.float32)
    lbls = np.zeros((5, 4, 256, 128), dtype=np.float32)

    recorded_heights = []

    with patch(
        "cellpose.train.pack_planes_to_stripes",
        side_effect=lambda x, layout: recorded_heights.append(layout.stripe_height)
        or _fake_pack_planes_to_stripes(x, layout),
    ):
        _pack_training_tiles(
            imgs,
            lbls,
            pack_k=3,
            guard=16,
            bsize=256,
            stripe_height=None,
            border=PACK_BORDER,
        )

    # Called twice (images + labels) with the same resolved height
    assert recorded_heights == [64, 64]


def test_mixed_batch_planning_sizes_do_not_exceed_batch():
    # Simulate the revised mixing logic to ensure total tiles stay within batch size
    pack_k = 3
    batch_size = 16
    S = 16  # available items
    ratio = 0.5

    desired_packed_tiles = int(round(ratio * batch_size))
    desired_std_tiles = max(0, batch_size - desired_packed_tiles)
    std_tiles = min(S, desired_std_tiles)
    remaining_for_packed = max(0, batch_size - std_tiles)
    max_stripes_allowed = min(S - std_tiles, remaining_for_packed * pack_k)
    stripes_target = max(0, max_stripes_allowed)
    packed_tiles = (stripes_target + pack_k - 1) // pack_k

    assert packed_tiles + std_tiles <= batch_size
    # And we actually have both modes represented for these parameters
    assert packed_tiles > 0 and std_tiles > 0


def test_pack_training_tiles_respects_explicit_height():
    imgs = np.zeros((4, 3, 256, 64), dtype=np.float32)
    lbls = np.zeros((4, 5, 256, 64), dtype=np.float32)
    recorded_heights = []

    with patch(
        "cellpose.train.pack_planes_to_stripes",
        side_effect=lambda x, layout: recorded_heights.append(layout.stripe_height)
        or _fake_pack_planes_to_stripes(x, layout),
    ):
        _pack_training_tiles(
            imgs,
            lbls,
            pack_k=3,
            guard=16,
            bsize=256,
            stripe_height=68,
            border=PACK_BORDER,
        )

    assert recorded_heights == [68, 68]


def test_pack_training_tiles_clamps_invalid_height():
    imgs = np.zeros((3, 3, 256, 80), dtype=np.float32)
    lbls = np.zeros((3, 4, 256, 80), dtype=np.float32)
    recorded_heights = []

    # Request a stripe height that is too tall; expect fallback to the max admissible (with border)
    with patch(
        "cellpose.train.pack_planes_to_stripes",
        side_effect=lambda x, layout: recorded_heights.append(layout.stripe_height)
        or _fake_pack_planes_to_stripes(x, layout),
    ):
        _pack_training_tiles(
            imgs,
            lbls,
            pack_k=3,
            guard=16,
            bsize=256,
            stripe_height=240,
            border=PACK_BORDER,
        )

    assert recorded_heights == [64, 64]


def test_resolve_pack_height_auto_selection():
    height, auto = _resolve_pack_stripe_height(None, pack_k=3, guard=16, bsize=256, border=PACK_BORDER)
    assert auto is True
    assert height == 64


def test_resolve_pack_height_explicit_honored():
    height, auto = _resolve_pack_stripe_height(68, pack_k=3, guard=16, bsize=256, border=PACK_BORDER)
    assert auto is False
    assert height == 68


def test_prepad_height_channel_first():
    arr = np.zeros((3, 68, 1968), dtype=np.float32)
    assert _prepad_height(arr) == 68


def test_prepad_height_channel_last():
    arr = np.zeros((68, 1968, 3), dtype=np.float32)
    assert _prepad_height(arr) == 68


def test_prepad_height_2d_input():
    arr = np.zeros((512, 256), dtype=np.float32)
    assert _prepad_height(arr) == 512


def test_effective_height_respects_rescale():
    arr = np.zeros((3, 136, 1968), dtype=np.float32)
    assert _effective_height(arr, 2.0) == 68
    assert _effective_height(arr, 1.0) == 136
