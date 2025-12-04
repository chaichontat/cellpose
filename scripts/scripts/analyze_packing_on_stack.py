#!/usr/bin/env python3
"""Analyze packing feasibility and tile counts on a ZCYX stack.

Loads a .tif stack (ZCYX as provided), reorders to ZYXC for Cellpose core,
then compares baseline tiling to packed tiling in 3D and 2D paths using a
dummy network (so the counts reflect tiling/packing only, not model cost).

Usage examples:
  CUDA_VISIBLE_DEVICES=1 conda run -n cp4 python scripts/scripts/analyze_packing_on_stack.py \
      --input /working/20251001_JaxA3_Coro11/registered--3r+pi/reg-0072.tif

Notes:
  - Requires tifffile in the environment (cp4 has it).
  - Does not download weights; uses a zero-op torch Module to count forward calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import tifffile as tiff
except Exception as e:  # pragma: no cover
    raise RuntimeError("tifffile is required for this analysis") from e

import torch

from cellpose.core import run_3D, run_net
from cellpose.contrib.packed_infer import _run_3d_with_packing
from cellpose import transforms
from cellpose.contrib.pack_utils import (
    _forward_counter,
    compute_stripe_layout,
    forward_packed_2d,
)


class DummyNet(torch.nn.Module):
    def __init__(self, nout: int = 4):
        super().__init__()
        self.nout = nout
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        y = torch.zeros(b, self.nout, h, w, device=x.device, dtype=x.dtype)
        style = torch.zeros(b, 256, device=x.device, dtype=x.dtype)
        return y, style


def load_zcyx(path: Path) -> np.ndarray:
    arr = tiff.imread(str(path))
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D ZCYX stack; got shape {arr.shape}")
    # Z,C,Y,X -> Z,Y,X,C
    z, c, y, x = arr.shape
    return np.transpose(arr, (0, 2, 3, 1)).astype(np.float32), (z, y, x, c)


def main():
    ap = argparse.ArgumentParser(description="Packing analysis for ZCYX stack")
    ap.add_argument(
        "--input",
        default="/working/20251001_JaxA3_Coro11/registered--3r+pi/reg-0072.tif",
        help="Path to ZCYX tif stack",
    )
    ap.add_argument("--bsize", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pack-k", type=int, default=3)
    ap.add_argument("--guard", type=int, default=16)
    ap.add_argument("--border", type=int, default=5)
    ap.add_argument(
        "--anisotropy",
        type=float,
        default=1.0,
        help="If !=1, emulate Cellpose 3D anisotropy resize (packing now allowed).",
    )
    ap.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="If set, apply the same XY scaling factor Cellpose uses (30/diameter).",
    )
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    stack, shape = load_zcyx(path)
    z, y, x, c = shape
    print(f"Loaded: Z={z} Y={y} X={x} C={c} from {path}")

    if args.diameter is not None:
        scale = 30.0 / float(args.diameter)
        y_scaled = max(1, int(round(y * scale)))
        x_scaled = max(1, int(round(x * scale)))
        stack = transforms.resize_image(stack, Ly=y_scaled, Lx=x_scaled)
        y, x = y_scaled, x_scaled
        print(
            f"Diameter={args.diameter:g} ⇒ scale={scale:.4f}, resized XY to ({y}, {x})"
        )
        # models adjusts anisotropy by the same factor after diameter scaling
        if args.anisotropy is not None:
            args.anisotropy = float(args.anisotropy) * scale
            print(f"Effective anisotropy after diameter scaling: {args.anisotropy:.4f}")

    # Emulate models._run_net anisotropy behavior: if anisotropy != 1.0,
    # transpose to (Y, Z, X, C), resize along the first axis, then transpose back.
    if args.anisotropy is not None and float(args.anisotropy) != 1.0:
        anis = float(args.anisotropy)
        tmp = stack.transpose(1, 0, 2, 3)
        tmp = transforms.resize_image(tmp, Ly=int(z * anis), Lx=x)
        stack = tmp.transpose(1, 0, 2, 3)
        z, y, x, c = stack.shape
        print(
            f"Anisotropy={anis:g} → resized stack to Z={z} Y={y} X={x}; packing enabled."
        )

    net = DummyNet(nout=4)
    params = dict(
        batch_size=args.batch_size, augment=False, tile_overlap=0.1, bsize=args.bsize
    )

    # Baseline 3D counts
    with _forward_counter() as stats_base3d:
        run_3D(net, stack, **params)
    # Packed 3D counts (only if packing is enabled)
    with _forward_counter() as stats_pack3d:
        _run_3d_with_packing(
            net,
            stack,
            batch_size=args.batch_size,
            augment=False,
            tile_overlap=0.1,
            bsize=args.bsize,
            pack_border=args.border,
            plane_weights=None,
        )
    print(
        f"3D tiles: baseline={stats_base3d['tiles']}, packed={stats_pack3d['tiles']}, "
        f"speedup={stats_base3d['tiles'] / max(1, stats_pack3d['tiles']):.2f}x"
    )

    # Orientation detail
    sstr = ["YX", "ZY", "ZX"]
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    for p, name in enumerate(sstr):
        xsl = stack.transpose(pm[p])
        Lzp, Lyp, Lxp = xsl.shape[:3]
        use_pack = name != "YX"
        layout = None
        if use_pack:
            # K and guard are auto-selected
            layout = compute_stripe_layout(
                Lyp,
                bsize=args.bsize,
                border=args.border,
            )
        with _forward_counter() as sb:
            run_net(net, xsl, **params)
        if use_pack and layout is not None:
            with _forward_counter() as sp:
                # route packed path through forward_packed_2d for this orientation
                from cellpose.contrib.pack_utils import pack_planes_to_stripes

                packed, _ = pack_planes_to_stripes(xsl, layout)
                run_net(net, packed, **{**params, "single_tile_if_fit": True})
            packed_tiles = sp["tiles"]
            print(
                f"  {name}: Lyp={Lyp}, K={layout.K}, guard={layout.guard}, "
                f"baseline_tiles={sb['tiles']}, packed_tiles={packed_tiles}"
            )
        else:
            print(
                f"  {name}: Lyp={Lyp}, layout=None, baseline_tiles={sb['tiles']} (no layout)"
            )

    # 2D path across Z planes
    net2d = DummyNet(nout=3)
    with _forward_counter() as stats_base2d:
        run_net(net2d, stack, **params)
    with _forward_counter() as stats_pack2d:
        forward_packed_2d(
            net2d,
            stack,
            bsize=args.bsize,
            batch_size=args.batch_size,
            augment=False,
            tile_overlap=0.1,
            pack_border=args.border,
            return_stats=False,
        )
    print(
        f"2D tiles: baseline={stats_base2d['tiles']}, packed={stats_pack2d['tiles']}, "
        f"speedup={stats_base2d['tiles'] / max(1, stats_pack2d['tiles']):.2f}x"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
