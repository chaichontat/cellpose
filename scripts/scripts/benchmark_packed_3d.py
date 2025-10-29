#!/usr/bin/env python3
"""Quick 3D ortho packing benchmark.

Given a single input image, extract a center crop of shape (Y=68, X=984),
form a 2-channel plane, and stack it Z=30 times → [30, 68, 984, 2].

Run segmentation in 3D mode with:
  - Baseline: standard CellposeModel (no packing)
  - Packed:   PackedCellposeModel (K=3 stripes, guard=16)

Report wall time and the number of low-level model forwards (instrumented
via cellpose.core._forward) to highlight the reduction in runs.

Expected: YX orthogonal pass has Lz=30 and Ly=68. With K=3 stripes,
we form G=ceil(30/3)=10 packed groups — “10 runs” vs “30 runs” baseline,
at the YX pass level. Other passes may also pack if their Y' is small.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path
from typing import Tuple

import numpy as np


def _load_image(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
    else:
        try:
            from tifffile import imread
        except Exception as e:  # pragma: no cover
            raise RuntimeError("tifffile not available; supply .npy or install tifffile") from e
        arr = imread(str(p))
    return np.asarray(arr)


def _center_crop_2d(img: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    Ly, Lx = target
    if img.ndim == 2:
        H, W = img.shape
        pad_y = max(0, Ly - H)
        pad_x = max(0, Lx - W)
        if pad_y or pad_x:
            img = np.pad(img, ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)))
        H, W = img.shape
        y0 = max(0, (H - Ly) // 2)
        x0 = max(0, (W - Lx) // 2)
        return img[y0:y0 + Ly, x0:x0 + Lx]
    elif img.ndim >= 3:
        H, W = img.shape[-2:]
        pad_y = max(0, Ly - H)
        pad_x = max(0, Lx - W)
        if pad_y or pad_x:
            pad_spec = [(0, 0)] * (img.ndim - 2) + [(pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)]
            img = np.pad(img, pad_spec)
            H, W = img.shape[-2:]
        y0 = max(0, (H - Ly) // 2)
        x0 = max(0, (W - Lx) // 2)
        slc = [slice(None)] * (img.ndim - 2) + [slice(y0, y0 + Ly), slice(x0, x0 + Lx)]
        return img[tuple(slc)]
    else:
        raise ValueError(f"Unsupported image ndim={img.ndim}")


def _extract_2d_plane(img: np.ndarray, target_xy: Tuple[int, int]) -> np.ndarray:
    """Select a single 2D plane (Y,X) from an arbitrary-ND array by taking
    the two largest dimensions as spatial, and the first index along others.
    Then center-crop/pad to the requested target size.
    """
    if img.ndim == 2:
        base = img
    else:
        shape = list(img.shape)
        # pick two largest axes as Y,X
        axes_sorted = np.argsort(shape)[::-1]
        ydim, xdim = int(axes_sorted[0]), int(axes_sorted[1])
        # Build permutation that moves ydim, xdim to the last two positions
        others = [i for i in range(img.ndim) if i not in (ydim, xdim)]
        perm = others + [ydim, xdim]
        arr = np.transpose(img, perm)
        # Take the first slice across non-spatial dims
        if arr.ndim > 2:
            arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))[0]
        base = arr
    return _center_crop_2d(base, target_xy)


@contextlib.contextmanager
def count_core_forwards():
    from cellpose import core as _core
    orig_forward = _core._forward
    counter = {"n": 0}

    def wrapped(net, x):
        counter["n"] += 1
        return orig_forward(net, x)

    _core._forward = wrapped  # type: ignore[assignment]
    try:
        yield counter
    finally:
        _core._forward = orig_forward  # type: ignore[assignment]


def _build_stack(arr: np.ndarray, z: int = 30) -> np.ndarray:
    # Construct a 2-channel (fake) plane from the 2D crop: stack/duplicate if needed
    if arr.ndim == 2:
        plane2c = np.stack([arr, arr], axis=-1)
    elif arr.ndim >= 3:
        # Use first two channels if available, otherwise duplicate first
        if arr.shape[-1] >= 2:
            plane2c = arr[..., :2]
        else:
            plane2c = np.concatenate([arr, arr], axis=-1)
    else:
        raise ValueError(f"Unexpected array ndim {arr.ndim}")
    return np.repeat(plane2c[np.newaxis, ...], z, axis=0)  # [Z, Y, X, 2]


def main():
    ap = argparse.ArgumentParser(description="3D ortho packing benchmark (baseline vs packed)")
    ap.add_argument("--input", required=True, help="Path to source image (.npy or .tif)")
    ap.add_argument("--pretrained", default="cpsam", help="Pretrained model path/name (PyTorch)")
    ap.add_argument("--engine", default=None, help="TensorRT engine .plan; if set, uses TRT models")
    ap.add_argument("--bsize", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pack-k", type=int, default=3)
    ap.add_argument("--pack-guard", type=int, default=16)
    args = ap.parse_args()

    img = _load_image(args.input)
    crop2d = _extract_2d_plane(img, (68, 984))
    stack = _build_stack(crop2d, z=30).astype(np.float32)

    print(f"Stack: Z={stack.shape[0]}, Y={stack.shape[1]}, X={stack.shape[2]}, C={stack.shape[3]}")
    # Expected group count for YX pass
    K = args.pack_k
    groups = (stack.shape[0] + K - 1) // K
    print(f"Expected YX effective runs: baseline={stack.shape[0]} vs packed={groups} (K={K})")

    # Lazy import heavy deps only when actually running benchmark
    from cellpose import models
    from cellpose.contrib.packed_infer import PackedCellposeModel
    use_trt = args.engine is not None
    if use_trt:
        from cellpose.contrib.cellposetrt import CellposeModelTRT
        from cellpose.contrib.packed_infer import PackedCellposeModelTRT as PackedRT
        base = CellposeModelTRT(gpu=True, pretrained_model=args.engine)
        packed_model = PackedRT(gpu=True, pretrained_model=args.engine, pack_z_stripes=True, pack_k=args.pack_k, pack_guard=args.pack_guard)
    else:
        base = models.CellposeModel(gpu=True, pretrained_model=args.pretrained)
        packed_model = PackedCellposeModel(gpu=True, pretrained_model=args.pretrained, pack_z_stripes=True, pack_k=args.pack_k, pack_guard=args.pack_guard)
    times = []
    forwards = 0
    for i in range(args.warmup + args.repeat):
        with count_core_forwards() as cnt:
            t0 = time.time()
            _ = base.eval(stack, do_3D=True, channel_axis=-1, z_axis=0, bsize=args.bsize, batch_size=8, augment=False, tile_overlap=0.1, compute_masks=False)
            dt = (time.time() - t0) * 1000
        if i >= args.warmup:
            times.append(dt)
            forwards += cnt["n"]
    print(f"Baseline: {np.mean(times):.1f} ms (avg of {args.repeat}), net forwards≈{forwards//args.repeat} per run")

    # Packed
    packed = packed_model
    times = []
    forwards = 0
    for i in range(args.warmup + args.repeat):
        with count_core_forwards() as cnt:
            t0 = time.time()
            _ = packed.eval(stack, do_3D=True, channel_axis=-1, z_axis=0, bsize=args.bsize, batch_size=8, augment=False, tile_overlap=0.1, compute_masks=False)
            dt = (time.time() - t0) * 1000
        if i >= args.warmup:
            times.append(dt)
            forwards += cnt["n"]
    print(f"Packed:   {np.mean(times):.1f} ms (avg of {args.repeat}), net forwards≈{forwards//args.repeat} per run")


if __name__ == "__main__":  # pragma: no cover
    main()
