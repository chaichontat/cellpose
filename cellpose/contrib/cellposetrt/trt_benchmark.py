# %%
"""Cellpose vs TensorRT benchmarking

Compare full pipeline (masks/flows/styles) between Torch and TRT
using the same inputs, report IoU and percent error (sMAPE), and time both
implementations.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
import torch

from cellpose import models
from cellpose.contrib.cellposetrt import CellposeModelTRT

# ---- CONFIG ----
image_path = Path(
    os.environ.get(
        "CP_IMAGE",
        "/working/20251001_JaxA3_Coro11/analysis/deconv/registered--3r+pi/reg-0076.tif",
    )
)
# Slices to be applied to loaded image
TILE_SLICE = (5, slice(None), slice(0, 512), slice(0, 512))

# Pretrained bf16 Cellpose SAM weights
pretrained = os.environ.get(
    "CP_MODEL",
    "/working/cellpose-training/models/embryonicsam",
)

# TensorRT engine path processed from trt_build.py; must match above!
engine_path = Path(
    os.environ.get(
        "CP_ENGINE",
        "/home/chaichontat/cellpose/scripts/builds/cpsam_b4_sm120_bf16.plan",
    )
)

BATCH = 1

# Folder path for TEST E
DIR_ENV = os.environ.get("CP_ISO_DIR")
# Number of images from DIR_ENV to test IoU on
N_SAMPLES = int(os.environ.get("CP_ISO_DIR_N", "20"))


def print_smape(name: str, ref, tst) -> None:
    r = torch.as_tensor(ref).float().flatten()
    t = torch.as_tensor(tst).float().flatten()
    diff = (t - r).abs()
    mae = float(diff.mean())
    smape = float((2.0 * diff / (r.abs() + t.abs() + 1e-12)).mean() * 100.0)
    print(
        f"{name}: shape={tuple(torch.as_tensor(ref).shape)} | sMAPE={smape:.3f}%  MAE={mae:.6g}"
    )


def time_op(
    name: str,
    fn: Callable,
    *,
    warmup: int = 1,
    iters: int = 5,
) -> float:
    # Warmup
    for _ in range(warmup):
        _ = fn()

    # Run
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()

    dt = (time.perf_counter() - t0) / iters
    ms = dt * 1000.0
    print(f"{name}: {ms:.3f} ms/iter (avg over {iters}, warmup={warmup})")
    return ms


def time_op_cuda(
    name: str,
    fn,
    *,
    warmup: int = 10,
    iters: int = 100,
) -> float:
    """GPU kernel timing using CUDA events (net-only).

    Records elapsed time on the current CUDA stream across `iters` calls. Does
    not include Python/host sync beyond the final event synchronize.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    # Warmup to stabilize autotuning/caches
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"{name}: {ms:.3f} ms/iter (CUDA events, iters={iters}, warmup={warmup})")
    return ms


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / max(1, float(union))


tile = tifffile.imread(image_path)[TILE_SLICE]
print("Loaded tile:", tile.shape, tile.dtype)
print(f"Engine path: {engine_path}")

# ---- Build models ----
device = torch.device("cuda:0")
print(f"Using CUDA device: {device} | {torch.cuda.get_device_name(device)}")

base = models.CellposeModel(
    gpu=True,
    device=device,
    pretrained_model=pretrained,
)
trt_model = CellposeModelTRT(
    gpu=True, device=device, pretrained_model=pretrained, engine_path=str(engine_path)
)

with torch.inference_mode():
    base_out = base.eval(tile, compute_masks=True, batch_size=BATCH)
    trt_out = trt_model.eval(tile, compute_masks=True, batch_size=BATCH)

print("\n[TEST C] Full pipeline parity")
masks_pt, masks_trt = base_out[0], trt_out[0]
print(
    f"  masks: torch={masks_pt.shape} trt={masks_trt.shape}  IoU={iou_binary(masks_pt != 0, masks_trt != 0):.4f}"
)

flows_pt = base_out[1]
flows_trt = trt_out[1]
for k, (fpt, ftrt) in enumerate(zip(flows_pt, flows_trt)):
    print_smape(f"  flow[{k}]", fpt, ftrt)

# Timing (full pipeline, compute_masks=True)
with torch.inference_mode():
    print("\n[TIMING C] Full pipeline eval(tile3)")
    ms_base = time_op("  Torch eval", lambda: base.eval(tile, batch_size=BATCH))
    ms_trt = time_op(
        "  TRT eval",
        lambda: models.CellposeModel.eval(trt_model, tile, batch_size=BATCH),
    )

spd = ms_base / ms_trt
print(f"  Speedup vs Torch: x{spd:.2f}")

# Net-only timing on representative 4x3x256x256 batch (CUDA events)
with torch.inference_mode():
    print(f"\n[TIMING D] Net-only forward ({BATCH}x3x256x256)")
    Xb = torch.randn(BATCH, 3, 256, 256, device=device, dtype=torch.bfloat16)
    ms_torch_net = time_op_cuda("  Torch net", lambda: base.net(Xb))
    ms_trt_net = time_op_cuda("  TRT net  ", lambda: trt_model.net(Xb))
    if ms_trt_net > 0:
        print(f"  Speedup (net-only): x{ms_torch_net / ms_trt_net:.2f}")

# ---- TEST E: Folder IoU on first N images (Torch vs TRT masks) ----
if not DIR_ENV:
    DIR_ENV = str(Path(image_path).parent)

folder = Path(DIR_ENV)
# Collect first N image files (exclude *_masks and flows)
files = [p for p in sorted(folder.glob("*.tif"))]
sub = files[:N_SAMPLES]

print(f"\n[TEST E] IoU parity on first {len(sub)} images from: {folder}")
ious: list[float] = []
for idx, f in enumerate(sub):
    try:
        arr = tifffile.imread(f)[TILE_SLICE]
        with torch.inference_mode():
            out_t = base.eval(arr, compute_masks=True, batch_size=BATCH)
            out_r = trt_model.eval(arr, compute_masks=True, batch_size=BATCH)
            if not np.any(out_t[0]) or not np.any(out_r[0]):
                print(
                    f"  [warn] skipping {f.name}: empty masks at least one of the models"
                )
                continue

            m_t = out_t[0]
            m_r = out_r[0]
        iou = iou_binary(m_t != 0, m_r != 0)
        ious.append(iou)
        print(f"  {idx + 1}/{len(sub)} processed... IoU={iou:.4f}")
    except Exception as e:
        print(f"  [warn] skipping {f.name}: {e}")

a = np.array(ious, dtype=float)
print(
    f"  IoU range: min={a.min():.4f}  median={np.median(a):.4f}  max={a.max():.4f}  (N={len(a)})"
)
