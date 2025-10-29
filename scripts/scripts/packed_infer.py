#!/usr/bin/env python3
"""Example runner for PackedCellposeModel.

This script demonstrates how to use the packing subclass without
modifying the original code. It loads an image stack from a .npy or .tif
file and runs 2D segmentation plane-wise with Z→Y packing enabled.

Usage
  python -m scripts.packed_infer --input path/to/stack.npy --pretrained cpsam \
      --pack-k 3 --pack-guard 16 --bsize 256 --do-3d
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    from tifffile import imread
except Exception:  # pragma: no cover
    imread = None

from cellpose.contrib.packed_infer import PackedCellposeModel


def _load(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".npy"}:
        x = np.load(p)
    elif p.suffix.lower() in {".tif", ".tiff"}:
        if imread is None:
            raise RuntimeError("tifffile not available; install it to read TIFF stacks")
        x = imread(p)
    else:
        raise ValueError(f"Unsupported input suffix: {p.suffix}")
    return x


def main():
    ap = argparse.ArgumentParser(description="Packed Cellpose inference example (2D plane-wise)")
    ap.add_argument("--input", required=True, help="Path to 3D stack (.npy or .tif)")
    ap.add_argument("--pretrained", default="cpsam", help="Pretrained model name/path")
    ap.add_argument("--bsize", type=int, default=256, help="Tile size (H=W)")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for tiling")
    ap.add_argument("--pack-k", type=int, default=3, help="Target stripes per packed frame")
    ap.add_argument("--pack-guard", type=int, default=16, help="Guard rows between stripes")
    ap.add_argument("--no-pack", action="store_true", help="Disable packing (use baseline)")
    ap.add_argument("--do-3d", action="store_true", help="Run 3D ortho mode (packing applies here)")
    args = ap.parse_args()

    x = _load(args.input)
    # Expect [Z, Y, X] or [Z, Y, X, C]; model will handle conversion in eval
    model = PackedCellposeModel(
        gpu=True,
        pretrained_model=args.pretrained,
        pack_z_stripes=(not args.no_pack),
        pack_k=args.pack_k,
        pack_guard=args.pack_guard,
    )
    masks, flows, styles = model.eval(
        x,
        do_3D=args.do_3d,
        bsize=args.bsize,
        batch_size=args.batch_size,
        augment=False,
        tile_overlap=0.1,
        compute_masks=True,
    )

    out_dir = Path("./packed_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "masks.npy", masks)
    np.save(out_dir / "flows.npy", np.array(flows, dtype=object))
    np.save(out_dir / "styles.npy", styles)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
