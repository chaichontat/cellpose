#!/usr/bin/env python3
"""Compare baseline vs packed Cellpose masks on a ribbon tile (IoU)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tifffile import imread

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from cellpose import io, models
from cellpose.contrib.pack_utils import compute_max_guard, compute_stripe_layout, pack_planes_to_stripes
from cellpose.train import _PACK_STRIPE_BORDER, _ScaledTileSaver
from scripts.save_packed_masks import PackedCellposeModel, run_model

TARGET_HW = (68, 256)
DOWNSAMPLE = 2
BSIZE = 256
PACK_BORDER = _PACK_STRIPE_BORDER


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    bin_a = np.asarray(mask_a, dtype=np.int32) > 0
    bin_b = np.asarray(mask_b, dtype=np.int32) > 0
    intersection = np.logical_and(bin_a, bin_b).sum(dtype=np.float64)
    union = np.logical_or(bin_a, bin_b).sum(dtype=np.float64)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def prepare_square_stripes(path: Path, count: int, target_hw: tuple[int, int] = TARGET_HW) -> list[np.ndarray]:
    arr = imread(path.as_posix())
    if arr.ndim == 4:
        plane = arr[0]
    elif arr.ndim == 3:
        plane = arr
    else:
        raise ValueError(f"Unsupported ndim {arr.ndim} for {path}")
    plane = plane[:, ::DOWNSAMPLE, ::DOWNSAMPLE].astype(np.float32, copy=False)
    C, H, W = plane.shape
    if C < 3:
        pad = np.zeros((3 - C, H, W), dtype=plane.dtype)
        plane = np.concatenate([plane, pad], axis=0)
    elif C > 3:
        plane = plane[:3]

    Ly, Lx = target_hw
    stripes: list[np.ndarray] = []
    y_max = max(0, H - Ly)
    if y_max == 0:
        positions = [0] * max(1, count)
    else:
        step = max(1, y_max // max(1, count - 1))
        positions = [min(y_max, i * step) for i in range(count)]
    x0 = max(0, (W - Lx) // 2)
    for y0 in positions:
        stripe = plane[:, y0 : y0 + Ly, x0 : x0 + Lx]
        stripes.append(stripe)
    return stripes


def _load_ribbon_plane(path: Path) -> np.ndarray:
    arr = imread(path.as_posix())
    if arr.ndim == 4:
        plane = arr[5, :, ::DOWNSAMPLE, ::DOWNSAMPLE]
    elif arr.ndim == 3:
        plane = arr[:, ::DOWNSAMPLE, ::DOWNSAMPLE]
    else:
        raise ValueError(f"Unexpected ndim {arr.ndim} for {path}")
    return plane.astype(np.float32, copy=False)


def prepare_ribbon_stripes(
    path: Path,
    count: int,
    target_hw: tuple[int, int] = TARGET_HW,
) -> list[np.ndarray]:
    base = _load_ribbon_plane(path)
    Ly, Lx = target_hw
    C, H, W = base.shape
    pad_y = max(0, Ly - H)
    pad_x = max(0, Lx - W)
    if pad_y or pad_x:
        base = np.pad(
            base,
            (
                (0, 0),
                (pad_y // 2, pad_y - pad_y // 2),
                (pad_x // 2, pad_x - pad_x // 2),
            ),
        )
        _, H, W = base.shape
    y0 = max(0, (H - Ly) // 2)
    max_x_offset = max(0, W - Lx)
    if count <= 1 or max_x_offset == 0:
        x_offsets = [max(0, (W - Lx) // 2)]
    else:
        x_offsets = np.linspace(0, max_x_offset, num=count, dtype=int).tolist()
    stripes: list[np.ndarray] = []
    for x0 in x_offsets:
        stripe = base[:, y0 : y0 + Ly, x0 : x0 + Lx]
        stripes.append(stripe)
    return stripes


def _normalize_channels(tile: np.ndarray, target_channels: int = 3) -> np.ndarray:
    arr = np.asarray(tile, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected stripe ndim 3, got {arr.ndim}")
    channels, height, width = arr.shape
    if channels < target_channels:
        pad = np.zeros((target_channels - channels, height, width), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    elif channels > target_channels:
        arr = arr[:target_channels]
    return arr


def _save_scaled_tiles(
    stripes: list[np.ndarray],
    *,
    debug_save_scaled_dir: Path | None,
    debug_save_scaled_limit: int,
    debug_save_packed_dir: Path | None,
    pack_k: int,
    bsize: int,
    pack_border: int,
    flow_model: models.CellposeModel | None,
    flow_output_dir: Path | None,
) -> tuple[tuple[Path, int] | None, Path | None, Path | None]:
    if debug_save_scaled_dir is None:
        scaled_stats = None
        processed = [_normalize_channels(tile) for tile in stripes]
    else:
        saver = _ScaledTileSaver(debug_save_scaled_dir, debug_save_scaled_limit)
        processed = [_normalize_channels(tile) for tile in stripes]
        if processed:
            tiles = np.stack(processed, axis=0)
            saver.save_batch(tiles, phase="val", epoch=0, batch_index=0)
        else:
            tiles = np.zeros((0, 3, TARGET_HW[0], TARGET_HW[1]), dtype=np.float32)
            saver.save_batch(tiles, phase="val", epoch=0, batch_index=0)
        scaled_stats = (saver.root, saver.saved)

    packed_path, packed_flow_path = _save_packed_patch(
        processed,
        output_dir=debug_save_packed_dir,
        pack_k=pack_k,
        bsize=bsize,
        pack_border=pack_border,
        flow_model=flow_model,
        flow_output_dir=flow_output_dir,
    )
    return scaled_stats, packed_path, packed_flow_path


def _save_packed_patch(
    stripes_cf: list[np.ndarray],
    *,
    output_dir: Path | None,
    pack_k: int,
    bsize: int,
    pack_border: int,
    flow_model: models.CellposeModel | None,
    flow_output_dir: Path | None,
) -> tuple[Path | None, Path | None]:
    if output_dir is None or not stripes_cf:
        return None, None
    stack = np.stack([np.transpose(tile, (1, 2, 0)) for tile in stripes_cf], axis=0)
    Ly = stack.shape[1]
    guard_eff = compute_max_guard(
        int(Ly),
        bsize=bsize,
        pack_k=pack_k,
        border=pack_border,
    )
    layout = compute_stripe_layout(
        Ly,
        bsize=bsize,
        pack_k=pack_k,
        guard=guard_eff,
        border=pack_border,
    )
    if layout is None:
        return None, None
    packed, _ = pack_planes_to_stripes(stack, layout)
    if packed.size == 0:
        return None, None
    patch = packed[0]  # (bsize, width, channels)
    patch_cf = np.transpose(patch[:, :, :3], (2, 0, 1))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / "packed_patch.png"
    arr = _ScaledTileSaver._to_uint8(patch_cf)
    io.imsave(filename.as_posix(), arr)
    flow_path: Path | None = None
    if flow_model is not None and flow_output_dir is not None:
        masks, flows, _ = run_model(flow_model, [patch_cf])
        if flows and flows[0]:
            rgb = np.asarray(flows[0][0], dtype=np.float32)
            cf = np.transpose(rgb[:, :, :3], (2, 0, 1))
            flow_arr = _ScaledTileSaver._to_uint8(cf)
            flow_output_dir = Path(flow_output_dir)
            flow_output_dir.mkdir(parents=True, exist_ok=True)
            flow_path = flow_output_dir / "packed_patch_flow.png"
            io.imsave(flow_path.as_posix(), flow_arr)
    return filename, flow_path


def _save_flow_rgbs(
    flows: list[list[np.ndarray]],
    *,
    output_dir: Path | None,
    prefix: str,
    limit: int,
) -> list[Path]:
    if output_dir is None or not flows:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    max_items = max(0, int(limit)) if limit is not None else len(flows)
    for idx, flow_triplet in enumerate(flows):
        if max_items and len(saved) >= max_items:
            break
        if not flow_triplet:
            continue
        rgb = np.asarray(flow_triplet[0], dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            continue
        cf = np.transpose(rgb[:, :, :3], (2, 0, 1))
        arr = _ScaledTileSaver._to_uint8(cf)
        path = output_dir / f"{prefix}_stripe{idx:02d}.png"
        io.imsave(path.as_posix(), arr)
        saved.append(path)
    return saved


def evaluate(
    image_path: Path,
    model_path: str,
    gpu: bool,
    pack_guard: int,
    pack_k: int,
    ribbon_count: int,
    extra_square: Path | None,
    square_count: int,
    debug_save_scaled_dir: Path | None,
    debug_save_scaled_limit: int,
    debug_save_packed_dir: Path | None,
    debug_save_flow_dir: Path | None,
    debug_save_flow_limit: int,
) -> tuple[float, tuple[Path, int] | None, Path | None, dict[str, list[Path]]]:
    stripes = prepare_ribbon_stripes(image_path, ribbon_count)
    if extra_square is not None:
        stripes.extend(prepare_square_stripes(extra_square, square_count))

    baseline = models.CellposeModel(gpu=gpu, pretrained_model=model_path)
    packed = PackedCellposeModel(
        gpu=gpu,
        pretrained_model=model_path,
        pack_guard=pack_guard,
        pack_k=pack_k,
    )

    scaled_stats, packed_path, packed_flow_path = _save_scaled_tiles(
        stripes,
        debug_save_scaled_dir=debug_save_scaled_dir,
        debug_save_scaled_limit=debug_save_scaled_limit,
        debug_save_packed_dir=debug_save_packed_dir,
        pack_k=pack_k,
        bsize=BSIZE,
        pack_border=PACK_BORDER,
        flow_model=packed,
        flow_output_dir=debug_save_flow_dir,
    )

    masks_base, flows_base, _ = run_model(baseline, stripes)
    masks_pack, flows_pack, _ = run_model(packed, stripes)

    flow_debug = {
        "baseline": _save_flow_rgbs(
            flows_base,
            output_dir=debug_save_flow_dir,
            prefix="baseline_flow",
            limit=debug_save_flow_limit,
        ),
        "packed": _save_flow_rgbs(
            flows_pack,
            output_dir=debug_save_flow_dir,
            prefix="packed_flow",
            limit=debug_save_flow_limit,
        ),
    }

    ious = [mask_iou(mb, mp) for mb, mp in zip(masks_base, masks_pack)]
    return float(np.mean(ious)), scaled_stats, packed_path, packed_flow_path, flow_debug


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/working/cellpose-training/20251015_JaxA2_Sag7/2/2--reg-0240_orthozx-1244.tif"),
        help="Ribbon TIFF to test",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/working/cellpose-training/models/embryonicsam",
        help="Pretrained model path",
    )
    parser.add_argument("--gpu", action="store_true", help="Run inference on GPU")
    parser.add_argument("--pack-guard", type=int, default=20, help="Guard rows used for packing")
    parser.add_argument("--pack-k", type=int, default=3, help="Number of stripes packed together (default: %(default)s)")
    parser.add_argument(
        "--ribbon-count",
        type=int,
        default=1,
        help="Number of stripes to sample along the ribbon tile before packing (default: %(default)s)",
    )
    parser.add_argument("--extra-square", type=Path, default=None, help="Optional large square TIFF to crop into stripes")
    parser.add_argument("--square-count", type=int, default=4, help="Number of stripes to sample from the extra square tile")
    parser.add_argument("--min-iou", type=float, default=0.90, help="Minimum acceptable IoU")
    parser.add_argument(
        "--debug-save-scaled-dir",
        type=Path,
        default=None,
        help="Directory to dump scaled stripes as PNGs (falls back to CELLPOSE_SAVE_SCALED_DIR)",
    )
    parser.add_argument(
        "--debug-save-scaled-limit",
        type=int,
        default=32,
        help="Maximum number of scaled tiles to save when debug output is enabled",
    )
    parser.add_argument(
        "--debug-save-packed-dir",
        type=Path,
        default=None,
        help="Directory to save the 256x256 packed patch (defaults to --debug-save-scaled-dir)",
    )
    parser.add_argument(
        "--debug-save-flow-dir",
        type=Path,
        default=None,
        help="Directory to emit colorized network flow outputs (defaults to --debug-save-scaled-dir)",
    )
    parser.add_argument(
        "--debug-save-flow-limit",
        type=int,
        default=8,
        help="Maximum number of flow visualizations per mode to save",
    )
    args = parser.parse_args()

    debug_dir = args.debug_save_scaled_dir
    env_debug_dir = os.environ.get("CELLPOSE_SAVE_SCALED_DIR")
    if debug_dir is None and env_debug_dir:
        debug_dir = Path(env_debug_dir)

    packed_dir = args.debug_save_packed_dir
    if packed_dir is None:
        packed_dir = debug_dir

    flow_dir = args.debug_save_flow_dir
    if flow_dir is None:
        flow_dir = debug_dir

    debug_limit = args.debug_save_scaled_limit
    env_limit = os.environ.get("CELLPOSE_SAVE_SCALED_LIMIT")
    if env_limit is not None:
        try:
            debug_limit = int(env_limit)
        except ValueError:
            pass

    iou, scaled_stats, packed_path, packed_flow_path, flow_debug = evaluate(
        args.image,
        args.model,
        args.gpu,
        args.pack_guard,
        args.pack_k,
        args.ribbon_count,
        args.extra_square,
        args.square_count,
        debug_dir,
        debug_limit,
        packed_dir,
        flow_dir,
        args.debug_save_flow_limit,
    )
    print(f"Mask IoU (baseline vs packed): {iou:.6f}")
    if scaled_stats is not None:
        saved_dir, saved_count = scaled_stats
        print(f"Saved {saved_count} scaled tile PNGs to {saved_dir}")
    if packed_path is not None:
        print(f"Packed 256x256 patch saved to {packed_path}")
    if packed_flow_path is not None:
        print(f"Packed patch flow visualization saved to {packed_flow_path}")
    if flow_debug["baseline"]:
        print(f"Baseline flow visualizations: {', '.join(map(str, flow_debug['baseline']))}")
    if flow_debug["packed"]:
        print(f"Packed flow visualizations: {', '.join(map(str, flow_debug['packed']))}")
    if iou < args.min_iou:
        raise SystemExit(
            f"Mask IoU {iou:.3f} below threshold {args.min_iou:.3f}."
        )


if __name__ == "__main__":
    main()
