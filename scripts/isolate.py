#%%
"""Cellpose vs TensorRT isolation harness — lean core.

Purpose: compare full pipeline (masks/flows/styles) between Torch and TRT
using the same inputs, report IoU and percent error (sMAPE), and time both
implementations. Assumes a BF16 engine built for batch=4.
"""

from __future__ import annotations

import os

# Enforce GPU 1 usage for this script. With this mask, the process will see a
# single logical device at index 0 that maps to physical GPU 1.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from cellpose import io as cp_io
from cellpose import models
from cellpose.trt import CellposeModelTRT, TRTEngineModule

# ---- CONFIG ----
image_path = Path(os.environ.get(
    "CP_ISO_IMAGE",
    "/working/20251001_JaxA3_Coro11/analysis/deconv/registered--3r+pi/reg-0076.tif",
))
engine_path = Path(os.environ.get(
    "CP_ISO_ENGINE",
    "/home/chaichontat/cellpose/scripts/builds/cpsam_b4_sm120_bf16.plan",
))  # must match checkpoint!
pretrained = os.environ.get(
    "CP_ISO_CKPT",
    "/working/cellpose-training/models/embryonicsam",
)  # SAM checkpoint
# Enforce batch_size == 4 as requested (used for both Torch and TRT paths)
BATCH = 4
TILE_SLICE = (5, slice(None), slice(0, 256), slice(0, 256))  # z=5, :, 256x256
# -------------

def to_numpy(obj) -> np.ndarray:
    if isinstance(obj, np.ndarray): return obj
    if hasattr(obj, "detach"): return obj.detach().cpu().numpy()
    if hasattr(obj, "cpu"):    return np.asarray(obj.cpu())
    return np.asarray(obj)

def print_smape(name: str, ref, tst) -> None:
    r = torch.as_tensor(ref).float().flatten()
    t = torch.as_tensor(tst).float().flatten()
    diff = (t - r).abs()
    smape = float((2.0 * diff / (r.abs() + t.abs() + 1e-12)).mean() * 100.0)
    print(f"{name}: shape={tuple(torch.as_tensor(ref).shape)} | sMAPE={smape:.3f}%")

def time_op(name: str, fn, *, warmup: int = 3, iters: int = 30, device: Optional[torch.device] = None) -> float:
    import time
    # warmup
    for _ in range(max(0, warmup)):
        _ = fn()
    # No manual synchronize: measure enqueue + any implicit syncs inside fn
    t0 = time.perf_counter()
    for _ in range(max(1, iters)):
        _ = fn()
    # Avoid explicit synchronize here as requested
    dt = (time.perf_counter() - t0) / max(1, iters)
    ms = dt * 1000.0
    print(f"{name}: {ms:.3f} ms/iter (avg over {iters}, warmup={warmup})")
    return ms

def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / max(1, float(union))

# ---- Load image and adapt to 3C for pipeline tests ----
img = cp_io.imread(str(image_path))
tile = img[TILE_SLICE]
if tile.ndim == 3 and tile.shape[0] <= 4:  # (C,H,W) -> (H,W,C)
    tile = np.moveaxis(tile, 0, -1)
print("Loaded tile:", tile.shape, tile.dtype)
print(f"Engine path: {engine_path}")

def adapt_to_3c(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:     return np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3:
        C = arr.shape[-1]
        if C == 3: return arr
        if C == 1: return np.repeat(arr, 3, axis=-1)
        if C == 2: return np.concatenate([arr, arr[..., -1:]], axis=-1)  # repeat last
        if C > 3:  return arr[..., :3]
    raise ValueError(f"cannot adapt {arr.shape} to 3C")

tile3 = adapt_to_3c(tile)

# ---- Build models ----
device = torch.device("cuda:0")  # logical 0 within CUDA_VISIBLE_DEVICES mask
print(f"Using CUDA device: {device} | {torch.cuda.get_device_name(device)}")

base = models.CellposeModel(
    gpu=True,
    device=device,
    pretrained_model=pretrained,
    use_bfloat16=True,
)

# Construct TRT-backed model without allocating a GPU PyTorch net in super().__init__
trt_model = CellposeModelTRT(
    gpu=False,
    device=None,
    pretrained_model=pretrained,
    engine_path=str(engine_path),
    fp16=False,
)
# Replace backend with a CUDA TRTEngineModule explicitly
trt_backend = TRTEngineModule(str(engine_path), device=device, fp16=False)
trt_model.net = trt_backend.eval()
# Advertise GPU path so base-class eval() takes the GPU branch
trt_model.gpu = True
trt_model.device = device

# ---- Full-pipeline parity (flows/styles/masks) ----
with torch.inference_mode():
    base_out = base.eval(tile3, compute_masks=True, batch_size=BATCH)
    # Use base-class eval to avoid any subclass overrides that force batch_size=1
    trt_out  = models.CellposeModel.eval(trt_model, tile3, compute_masks=True, batch_size=BATCH)

print("\n[TEST C] Full pipeline parity")
masks_pt = to_numpy(base_out[0]); masks_trt = to_numpy(trt_out[0])
print(f"  masks: torch={masks_pt.shape} trt={masks_trt.shape}  IoU={iou_binary(masks_pt!=0, masks_trt!=0):.4f}")

flows_pt = base_out[1]
flows_trt = trt_out[1]
for k, (fpt, ftrt) in enumerate(zip(flows_pt, flows_trt)):
    print_smape(f"  flow[{k}]", fpt, ftrt)

print_smape("  styles (pipeln)", base_out[2], trt_out[2])

# Timing (full pipeline, compute_masks=True)
with torch.inference_mode():
    print("\n[TIMING C] Full pipeline eval(tile3)")
    ms_base = time_op("  Torch eval", lambda: base.eval(tile3, compute_masks=True, batch_size=BATCH), device=device)
    ms_trt  = time_op("  TRT eval",   lambda: models.CellposeModel.eval(trt_model, tile3, compute_masks=True, batch_size=BATCH), device=device)
    spd = ms_base / ms_trt if ms_trt > 0 else float('inf')
    print(f"  Speedup vs Torch: x{spd:.2f}")
