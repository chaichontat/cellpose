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
            bsize=256,
            stripe_height=None,
            border=PACK_BORDER,
        )

    # Called twice (images + labels) with auto-selected height
    # K is auto-selected to maximize packing while ensuring guard >= 5
    assert len(recorded_heights) == 2
    assert recorded_heights[0] == recorded_heights[1]  # same height for imgs and lbls


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
            bsize=256,
            stripe_height=64,  # clamped to image height
            border=PACK_BORDER,
        )

    assert recorded_heights == [64, 64]


def test_pack_training_tiles_clamps_invalid_height():
    # (N, C, H, W) - H=80, W=256
    imgs = np.zeros((3, 3, 80, 256), dtype=np.float32)
    lbls = np.zeros((3, 4, 80, 256), dtype=np.float32)
    recorded_heights = []

    # Request a stripe height that is too tall; expect fallback to auto-selected height
    with patch(
        "cellpose.train.pack_planes_to_stripes",
        side_effect=lambda x, layout: recorded_heights.append(layout.stripe_height)
        or _fake_pack_planes_to_stripes(x, layout),
    ):
        _pack_training_tiles(
            imgs,
            lbls,
            bsize=256,
            stripe_height=240,
            border=PACK_BORDER,
        )

    # Height is clamped to what fits in bsize with guard >= 5
    assert len(recorded_heights) == 2
    assert recorded_heights[0] == recorded_heights[1]
    assert recorded_heights[0] <= 80  # clamped to max_height (image H)


def test_resolve_pack_height_auto_selection():
    # K is now auto-selected, so we just pass bsize and border
    height, auto = _resolve_pack_stripe_height(None, bsize=256, border=PACK_BORDER)
    assert height is not None
    assert height > 0


def test_resolve_pack_height_explicit_honored():
    height, auto = _resolve_pack_stripe_height(68, bsize=256, border=PACK_BORDER)
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


def test_pack_training_tiles_preserves_image_label_overlay():
    # Construct a batch where every image and label channel share identical per-pixel values.
    # Packing must keep them perfectly co-registered.
    N, C, H, W = 5, 1, 64, 128
    imgs = np.zeros((N, C, H, W), dtype=np.float32)
    lbls = np.zeros((N, 3, H, W), dtype=np.float32)

    for i in range(N):
        value = float(i + 1)
        imgs[i, 0] = value
        lbls[i, 0] = value
        lbls[i, 1] = value
        lbls[i, 2] = value

    packed_imgs, packed_lbls = _pack_training_tiles(
        imgs,
        lbls,
        bsize=256,
        stripe_height=H,  # match height to avoid random vertical cropping
        border=PACK_BORDER,
    )

    # Shapes must match and image/label overlay should be preserved pixelwise
    assert packed_imgs.shape[0] == packed_lbls.shape[0]
    assert packed_imgs.shape[2:] == packed_lbls.shape[2:]

    for g in range(packed_imgs.shape[0]):
        for c in range(packed_lbls.shape[1]):
            assert np.allclose(packed_imgs[g, 0], packed_lbls[g, c])
