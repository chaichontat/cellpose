#!/usr/bin/env python3
"""Verify training data contains thin packed tiles after diameter scaling.

This script scans a dataset directory for TIFF files that have paired
``*_seg.npy`` masks, measures their pre-pad heights, applies the same
rescaling used during training (diameter-based), and asserts that at
least one sample qualifies as "thin" (< thin_threshold pixels after
rescaling). It also checks that label files are readable to confirm
paired loading works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import numpy as np
from cellpose.io import imread

try:
    from cellpose.train import _prepad_height, _effective_height
except ImportError as exc:  # pragma: no cover - guard for standalone use
    raise SystemExit(f"Cannot import cellpose helpers: {exc}")


def _iter_pairs(root: Path) -> Iterator[Tuple[Path, Path]]:
    for seg_path in sorted(root.rglob("*_seg.npy")):
        stem = seg_path.stem
        if not stem.endswith("_seg"):
            continue
        tif_name = stem[:-4] + ".tif"
        tif_path = seg_path.with_name(tif_name)
        if tif_path.exists():
            yield tif_path, seg_path


def _summarize_pairs(
    pairs: Iterable[Tuple[Path, Path]],
    *,
    thin_threshold: float,
    rescale_factor: float,
) -> tuple[int, int, int]:
    total = 0
    thin = 0
    square_like = 0
    for tif_path, seg_path in pairs:
        img = imread(tif_path.as_posix())
        # Ensure paired mask loads to catch corrupted entries
        _ = np.load(seg_path, allow_pickle=True)
        eff_height = _effective_height(img, rescale_factor)
        if eff_height < thin_threshold:
            thin += 1
        else:
            square_like += 1
        total += 1
    return total, thin, square_like


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("/working/cellpose-training/20251015_JaxA2_Sag7"),
        help="Dataset directory to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--thin-threshold",
        type=float,
        default=100.0,
        help="Pixels after rescaling to qualify as thin (default: %(default)s)",
    )
    parser.add_argument(
        "--rescale-factor",
        type=float,
        default=2.0,
        help="Diameter-based rescale factor applied before measuring height",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of pairs to inspect (0 = all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root
    if not root.exists():
        raise SystemExit(f"Dataset root {root} does not exist")

    all_pairs = list(_iter_pairs(root))
    if not all_pairs:
        raise SystemExit(f"No paired TIFF/_seg.npy files found under {root}")

    if args.limit > 0:
        sample_pairs = all_pairs[: args.limit]
    else:
        sample_pairs = all_pairs

    total, thin, square_like = _summarize_pairs(
        sample_pairs,
        thin_threshold=args.thin_threshold,
        rescale_factor=args.rescale_factor,
    )
    print(
        f"Scanned {total} paired samples (limit={args.limit or 'all'}) — "
        f"thin={thin}, square_or_large={square_like}"
    )
    if thin == 0:
        raise SystemExit(
            "No thin tiles detected after diameter scaling; verify data or thresholds."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
