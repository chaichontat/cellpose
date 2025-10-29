#!/usr/bin/env python3
"""Quick 2D packing benchmark using direct run_net interception.

Build a Z-stack [30, 68, 984, C] from a single image, then compare:
  - Baseline: run_net(net, stack, ...)
  - Packed:   forward_packed_2d(net, stack, ...)

Works with PyTorch or TensorRT-backed net. For TRT engines compiled with
N∈[1..Nmax], pass --batch-size ≤ Nmax (e.g., 4).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np

# Reference ZCYX stack for regression/benchmarking.
DEFAULT_INPUT = Path("/working/20251001_JaxA3_Coro11/registered--3r+pi/reg-0072.tif")


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


def _extract_2d_plane(img: np.ndarray, target_xy: Tuple[int, int]) -> np.ndarray:
    Ly, Lx = target_xy
    if img.ndim == 2:
        base = img
    else:
        # Take two largest dims as spatial
        shape = list(img.shape)
        axes_sorted = np.argsort(shape)[::-1]
        ydim, xdim = int(axes_sorted[0]), int(axes_sorted[1])
        others = [i for i in range(img.ndim) if i not in (ydim, xdim)]
        perm = others + [ydim, xdim]
        arr = np.transpose(img, perm)
        if arr.ndim > 2:
            arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))[0]
        base = arr
    # Center-crop/pad to target
    H, W = base.shape[-2:]
    pad_y = max(0, Ly - H)
    pad_x = max(0, Lx - W)
    if pad_y or pad_x:
        base = np.pad(base, ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)))
    H, W = base.shape
    y0 = max(0, (H - Ly) // 2)
    x0 = max(0, (W - Lx) // 2)
    return base[y0 : y0 + Ly, x0 : x0 + Lx]


def _build_stack(arr: np.ndarray, z: int = 30, c: int = 3) -> np.ndarray:
    if c == 1:
        plane = arr[..., np.newaxis]
    else:
        plane = np.stack([arr] * c, axis=-1)
    return np.repeat(plane[np.newaxis, ...], z, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="2D packing benchmark (baseline vs packed)")
    ap.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to source image (.npy or .tif). Default is reg-0072.tif (ZCYX).",
    )
    ap.add_argument("--engine", default=None, help="TensorRT engine .plan; if set, uses TRT models")
    ap.add_argument("--pretrained", default="cpsam", help="Pretrained model path/name (PyTorch)")
    ap.add_argument("--bsize", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4, help="Mini-batch for tiling (≤ TRT Nmax)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pack-k", type=int, default=3)
    ap.add_argument("--pack-guard", type=int, default=16)
    ap.add_argument("--target-height", type=int, default=68)
    ap.add_argument("--target-width", type=int, default=984)
    args = ap.parse_args()

    img = _load_image(args.input)
    plane = _extract_2d_plane(img, (args.target_height, args.target_width))
    stack = _build_stack(plane, z=30, c=3)
    print(f"Stack: Z={stack.shape[0]}, Y={stack.shape[1]}, X={stack.shape[2]}, C={stack.shape[3]}")
    nx = int(np.ceil((1 + 2 * 0.1) * stack.shape[2] / args.bsize))
    packed_groups = (stack.shape[0] + args.pack_k - 1) // args.pack_k
    print(f"Expected effective runs: baseline planes={stack.shape[0]} × nx={nx}; packed stripes={packed_groups} × nx={nx}")

    # Model/net
    use_trt = args.engine is not None
    if use_trt:
        from cellpose.contrib.cellposetrt import CellposeModelTRT
        model = CellposeModelTRT(gpu=True, pretrained_model=args.engine)
    else:
        from cellpose import models
        model = models.CellposeModel(gpu=True, pretrained_model=args.pretrained)
    net = model.net

    from cellpose.contrib.pack_utils import forward_packed_2d, run_net_with_stats

    # Baseline
    times = []
    batches = []
    tiles = []
    for i in range(args.warmup + args.repeat):
        t0 = time.time()
        _, _, stats = run_net_with_stats(
            net,
            stack,
            bsize=args.bsize,
            batch_size=args.batch_size,
            augment=False,
            tile_overlap=0.1,
            return_stats=True,
        )
        dt = (time.time() - t0) * 1000
        if i >= args.warmup:
            times.append(dt)
            batches.append(stats["batches"])
            tiles.append(stats["tiles"])
    print(
        f"Baseline: {np.mean(times):.1f} ms (avg of {args.repeat}), batches/run≈{np.mean(batches):.1f}, tiles/run≈{np.mean(tiles):.1f}"
    )

    # Packed
    times = []
    batches = []
    tiles = []
    for i in range(args.warmup + args.repeat):
        t0 = time.time()
        _, _, stats = forward_packed_2d(
            net,
            stack,
            bsize=args.bsize,
            batch_size=args.batch_size,
            augment=False,
            tile_overlap=0.1,
            pack_k=args.pack_k,
            guard=args.pack_guard,
            return_stats=True,
        )
        dt = (time.time() - t0) * 1000
        if i >= args.warmup:
            times.append(dt)
            batches.append(stats["batches"])
            tiles.append(stats["tiles"])
    print(
        f"Packed:   {np.mean(times):.1f} ms (avg of {args.repeat}), batches/run≈{np.mean(batches):.1f}, tiles/run≈{np.mean(tiles):.1f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
