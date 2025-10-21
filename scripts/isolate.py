#%%
"""Cellpose vs TensorRT isolation harness (SAM, 3C) — fixed .device/.dtype and 3C tiles."""

from __future__ import annotations

import os

# Enforce GPU 1 usage for this script. With this mask, the process will see a
# single logical device at index 0 that maps to physical GPU 1.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn

from cellpose import io as cp_io
from cellpose import models
from cellpose.trt import CellposeModelTRT

# ---- CONFIG ----
image_path = Path(os.environ.get("CP_ISO_IMAGE",
                      "/working/20251001_JaxA3_Coro11/analysis/deconv/registered--3r+pi/reg-0076.tif"))
engine_path = Path(os.environ.get("CP_ISO_ENGINE",
                      "/home/chaichontat/cellpose/scripts/builds/cyto2_bf16.plan"))  # must match checkpoint!
onnx_path   = Path(os.environ.get("CP_ISO_ONNX",
                      "/home/chaichontat/cellpose/scripts/builds/cyto2.onnx"))
pretrained  = os.environ.get("CP_ISO_CKPT",
                      "/working/cellpose-training/models/embryonicsam")              # SAM checkpoint
HW: Tuple[int,int] = (256, 256)   # engine profile
# Enforce batch_size == 4 as requested (used for both Torch and TRT paths)
BATCH = 4
DTYPE = os.environ.get("CP_ISO_DTYPE", "bf16")   # "bf16" | "fp16" | "fp32"
TILE_SLICE = (5, slice(None), slice(0,256), slice(0,256))  # z=5, :, 256x256

# execution knobs for constrained environments
GPU_ID = int(os.environ.get("CP_ISO_GPU", 0))
TRT_ONLY = os.environ.get("CP_ISO_TRT_ONLY", "0").lower() in {"1","true","yes"}
# -------------

def _torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]

def to_numpy(obj) -> np.ndarray:
    if isinstance(obj, np.ndarray): return obj
    if hasattr(obj, "detach"): return obj.detach().cpu().numpy()
    if hasattr(obj, "cpu"):    return np.asarray(obj.cpu())
    return np.asarray(obj)

def summarize(name: str, ref, tst) -> None:
    r = torch.as_tensor(ref).float().flatten()
    t = torch.as_tensor(tst).float().flatten()
    diff = (t - r).abs()
    mae  = float(diff.mean()); mx = float(diff.max())
    # symmetric mean absolute percentage error (sMAPE), robust near zero
    smape = float((2.0 * diff / (r.abs() + t.abs() + 1e-12)).mean() * 100.0)
    rel  = float((t - r).norm() / (r.norm() + 1e-12))
    print(f"{name}: shape={tuple(torch.as_tensor(ref).shape)} | %Err={smape:.3f}%  MAE={mae:.6g}  Max={mx:.6g}  relL2={rel:.6g}")

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

# ---- Capture wrapper with real .device/.dtype attributes and a safe .to() ----
class CaptureNet(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        # materialize attributes for Cellpose core
        p = next(self.net.parameters(), None)
        self.device = p.device if p is not None else torch.device("cuda")
        self.dtype  = p.dtype  if p is not None else torch.bfloat16
        self.calls: List[Dict[str, Any]] = []

    def to(self, *args, **kwargs):
        self.net.to(*args, **kwargs)
        # update materialized attrs
        p = next(self.net.parameters(), None)
        if p is not None:
            self.device = p.device
            self.dtype  = p.dtype
        return self

    def forward(self, x):
        out = self.net(x)
        outs = out if isinstance(out, (tuple, list)) else (out,)
        self.calls.append({
            "input":  x.detach().clone(),
            "outputs": tuple(o.detach().clone() if torch.is_tensor(o) else o for o in outs),
        })
        return out

# ---- Minimal TRT runner (queries shapes, uses current stream) ----
class TRTRunner:
    def __init__(self, plan: Path, device: torch.device):
        self.device = device
        logger = trt.Logger(trt.Logger.ERROR)
        with open(plan, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.has_tensors_api = hasattr(self.engine, "num_io_tensors") and hasattr(self.ctx, "set_input_shape")
        # simple caches to avoid per-iteration shape/programming overhead
        self._last_shape: Optional[Tuple[int,int,int,int]] = None
        self._y: Optional[torch.Tensor] = None
        self._s: Optional[torch.Tensor] = None

        def _dt(dt):
            if dt == trt.DataType.BF16:  return torch.bfloat16
            if dt == trt.DataType.HALF:  return torch.float16
            if dt == trt.DataType.FLOAT: return torch.float32
            raise ValueError(dt)

        if self.has_tensors_api:
            for n in ("input","y","style"): _ = self.engine.get_tensor_dtype(n)
            self.dt_in = _dt(self.engine.get_tensor_dtype("input"))
            self.dt_y  = _dt(self.engine.get_tensor_dtype("y"))
            self.dt_s  = _dt(self.engine.get_tensor_dtype("style"))
        else:
            self.idx_in = self.engine.get_binding_index("input")
            self.idx_y  = self.engine.get_binding_index("y")
            self.idx_s  = self.engine.get_binding_index("style")
            self.dt_in = _dt(self.engine.get_binding_dtype(self.idx_in))
            self.dt_y  = _dt(self.engine.get_binding_dtype(self.idx_y))
            self.dt_s  = _dt(self.engine.get_binding_dtype(self.idx_s))

    def query_output_shapes(self, N, C, H, W):
        if self.has_tensors_api:
            self.ctx.set_input_shape("input", (N, C, H, W))
            y_shape = tuple(self.ctx.get_tensor_shape("y"))
            s_shape = tuple(self.ctx.get_tensor_shape("style"))
        else:
            self.ctx.set_binding_shape(self.idx_in, (N, C, H, W))
            y_shape = tuple(self.ctx.get_binding_shape(self.idx_y))
            s_shape = tuple(self.ctx.get_binding_shape(self.idx_s))
        return y_shape, s_shape

    def __call__(self, X: torch.Tensor):
        assert X.is_cuda
        if X.dtype != self.dt_in: X = X.to(self.dt_in)
        X = X.contiguous()
        N, C, H, W = X.shape
        stream = torch.cuda.current_stream(self.device)
        sh = int(stream.cuda_stream)
        if self.has_tensors_api:
            # Set shape only if it changed
            if self._last_shape != (N, C, H, W):
                self.ctx.set_input_shape("input", (N, C, H, W))
                self._last_shape = (N, C, H, W)
            y_shape = tuple(self.ctx.get_tensor_shape("y"))
            s_shape = tuple(self.ctx.get_tensor_shape("style"))
            # Reuse output buffers if possible
            if self._y is None or tuple(self._y.shape) != y_shape or self._y.dtype != self.dt_y or self._y.device != X.device:
                self._y = torch.empty(y_shape, device=X.device, dtype=self.dt_y)
            if self._s is None or tuple(self._s.shape) != s_shape or self._s.dtype != self.dt_s or self._s.device != X.device:
                self._s = torch.empty(s_shape, device=X.device, dtype=self.dt_s)
            y, s = self._y, self._s
            self.ctx.set_tensor_address("input", X.data_ptr())
            self.ctx.set_tensor_address("y", y.data_ptr())
            self.ctx.set_tensor_address("style", s.data_ptr())
            ok = self.ctx.execute_async_v3(sh)
            if not ok: raise RuntimeError("execute_async_v3 failed")
            return y, s
        else:
            self.ctx.set_binding_shape(self.idx_in, (N, C, H, W))
            y_shape = tuple(self.ctx.get_binding_shape(self.idx_y))
            s_shape = tuple(self.ctx.get_binding_shape(self.idx_s))
            if self._y is None or tuple(self._y.shape) != y_shape or self._y.dtype != self.dt_y or self._y.device != X.device:
                self._y = torch.empty(y_shape, device=X.device, dtype=self.dt_y)
            if self._s is None or tuple(self._s.shape) != s_shape or self._s.dtype != self.dt_s or self._s.device != X.device:
                self._s = torch.empty(s_shape, device=X.device, dtype=self.dt_s)
            y, s = self._y, self._s
            bindings = [None]*self.engine.num_bindings
            bindings[self.idx_in] = X.data_ptr()
            bindings[self.idx_y]  = y.data_ptr()
            bindings[self.idx_s]  = s.data_ptr()
            ok = self.ctx.execute_async_v2(bindings, sh)
            if not ok: raise RuntimeError("execute_async_v2 failed")
            return y, s

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

 # ---- Build models, wrap nets for capture ----
device = torch.device(f"cuda:{GPU_ID}")
dtype = _torch_dtype(DTYPE)

print(f"Using CUDA device: {device}")

base = None
if not TRT_ONLY:
    # Put heavy PyTorch model on the selected CUDA device
    base = models.CellposeModel(gpu=True, device=device, pretrained_model=pretrained,
                                use_bfloat16=(DTYPE=="bf16"))
    base.net = CaptureNet(base.net.eval().to(device))

from cellpose.trt import TRTEngineModule


def _ensure_engine_for_device(plan_path: Path, onnx_path: Path, device: torch.device) -> Path:
    """Load a TRT engine; if incompatible with current GPU arch, rebuild from ONNX locally."""
    try:
        _ = TRTRunner(plan_path, device)
        return plan_path
    except Exception as e:
        msg = str(e)
        if "incompatible device" in msg or "deserializeCudaEngine" in msg or isinstance(e, AttributeError):
            maj, minr = torch.cuda.get_device_capability(device)
            tag = f"sm{maj}{minr}"
            new_plan = plan_path.with_name(plan_path.stem.split("_")[0] + f"_{tag}_bf16.plan")
            print(f"[TRT] Engine incompatible with {device} (capability {maj}.{minr}); rebuilding to {new_plan} ...")
            cmd = [sys.executable, str(Path(__file__).with_name("convert-to-rt.py")),
                   "--onnx", str(onnx_path), "--plan", str(new_plan), "--bf16", "--batch", str(BATCH)]
            # Preserve CUDA_VISIBLE_DEVICES mask (forced to "1"); use logical GPU 0 within the mask
            env = os.environ.copy(); env["CP_ISO_GPU"] = "0"
            subprocess.run(cmd, check=True, env=env)
            # Validate
            _ = TRTRunner(new_plan, device)
            return new_plan
        raise

# Ensure engine compatibility, then initialize runners
engine_path = _ensure_engine_for_device(engine_path, onnx_path, device)
trt_direct = TRTRunner(engine_path, device)

# Construct TRT-backed model without allocating a GPU PyTorch net in super().__init__
trt_model = CellposeModelTRT(gpu=False, device=None, pretrained_model=pretrained,
                             engine_path=str(engine_path), fp16=(DTYPE=="fp16"))
# Replace backend with a CUDA TRTEngineModule explicitly
trt_backend = TRTEngineModule(str(engine_path), device=device, fp16=(DTYPE=="fp16"))
trt_model.net = CaptureNet(trt_backend.eval())
# Advertise GPU path to the outer CellposeModel so eval() takes the GPU branch
trt_model.gpu = True
trt_model.device = device

# ---- Sanity: shapes/dtypes ----
N, C = BATCH, 3
H, W = HW
print(f"Engine query for input (N,C,H,W)=({N},{C},{H},{W}) ...")
y_shape_e, s_shape_e = trt_direct.query_output_shapes(N, C, H, W)
print("  TRT out 'y' shape :", y_shape_e)
print("  TRT out 'style'   :", s_shape_e)
print("  TRT dtypes        : in=", trt_direct.dt_in, " y=", trt_direct.dt_y, " s=", trt_direct.dt_s)

if not TRT_ONLY:
    with torch.inference_mode():
        X_probe = torch.randn(N, C, H, W, device=device, dtype=dtype)
        y_pt, s_pt = base.net.net(X_probe)[:2]
    print("  Torch out 'y' shape:", tuple(y_pt.shape))
    print("  Torch out 'style'  :", tuple(s_pt.shape))

if not TRT_ONLY:
    # ---- TEST A: net-only parity on random input ----
    torch.manual_seed(0)
    with torch.inference_mode():
        X = torch.randn(N, C, H, W, device=device, dtype=dtype)
        y_t, s_t = base.net.net(X)[:2]
        y_r, s_r = trt_direct(X)

    print("\n[TEST A] Random-input net parity (Torch vs TRT-engine)")
    summarize("y (random)", y_t, y_r)
    summarize("style (rand)", s_t, s_r)
    if torch.tensor(y_t).isnan().any() or torch.tensor(y_r).isnan().any():
        print("NaNs detected in y — check normalization / Resize attrs.")

    # Timing (net-only)
    with torch.inference_mode():
        _torch_fn = lambda: base.net.net(X)
        _trt_fn_d = lambda: trt_direct(X)
        _trt_fn_w = lambda: trt_model.net.net(X)
        print("\n[TIMING A] Net-only forward on random input")
        ms_torch = time_op("  Torch net", _torch_fn, device=device)
        ms_trt_d = time_op("  TRT direct", _trt_fn_d, device=device)
        ms_trt_w = time_op("  TRT wrapped", _trt_fn_w, device=device)
        spd_d = ms_torch / ms_trt_d if ms_trt_d > 0 else float('inf')
        spd_w = ms_torch / ms_trt_w if ms_trt_w > 0 else float('inf')
        print(f"  Speedup vs Torch: direct x{spd_d:.2f}, wrapped x{spd_w:.2f}")

if not TRT_ONLY:
    # ---- TEST B: captured-input parity from eval() (compute_masks=False) ----
    with torch.inference_mode():
        # use the same batch size for parity
        _ = base.eval(tile3, compute_masks=False, batch_size=BATCH)   # *** use 3C tile ***
        # Monkey-patch eval to base class to avoid forced batch_size=1 in CellposeModelTRT
        trt_model.eval = models.CellposeModel.eval.__get__(trt_model, models.CellposeModel)
        _ = trt_model.eval(tile3, compute_masks=False, batch_size=BATCH)

    calls_pt  = base.net.calls
    calls_trt = trt_model.net.calls

    print(f"\n[TEST B] Captured .net calls: torch={len(calls_pt)} trt={len(calls_trt)}")
    num_calls = min(len(calls_pt), len(calls_trt))
    for i in range(num_calls):
        Xpt  = calls_pt[i]["input"]
        Xtrt = calls_trt[i]["input"]
        print(f"  Call #{i}: input parity")
        if tuple(Xpt.shape) != tuple(Xtrt.shape):
            print(f"    X shapes differ: torch={tuple(Xpt.shape)} trt={tuple(Xtrt.shape)}; skipping compare.")
            continue
        summarize("    X", Xpt, Xtrt)

        y_pt_i = calls_pt[i]["outputs"][0]
        s_pt_i = calls_pt[i]["outputs"][1]
        y_tr_i = calls_trt[i]["outputs"][0]
        s_tr_i = calls_trt[i]["outputs"][1]
        if tuple(y_pt_i.shape) == tuple(y_tr_i.shape):
            summarize("    y (net)", y_pt_i, y_tr_i)
        else:
            print(f"    y shapes differ: torch={tuple(y_pt_i.shape)} trt={tuple(y_tr_i.shape)}; skipping.")
        if tuple(s_pt_i.shape) == tuple(s_tr_i.shape):
            summarize("    s (net)", s_pt_i, s_tr_i)
        else:
            print(f"    s shapes differ: torch={tuple(s_pt_i.shape)} trt={tuple(s_tr_i.shape)}; skipping.")

        with torch.inference_mode():
            y_trt_dir, s_trt_dir = trt_direct(Xpt.to(device))
        if tuple(y_pt_i.shape) == tuple(y_trt_dir.shape):
            summarize("    y (TRT-direct vs torch)", y_pt_i, y_trt_dir)
        else:
            print(f"    y (direct) shapes differ: torch={tuple(y_pt_i.shape)} trt={tuple(y_trt_dir.shape)}; skipping.")
        if tuple(s_pt_i.shape) == tuple(s_trt_dir.shape):
            summarize("    s (TRT-direct vs torch)", s_pt_i, s_trt_dir)
        else:
            print(f"    s (direct) shapes differ: torch={tuple(s_pt_i.shape)} trt={tuple(s_trt_dir.shape)}; skipping.")

if not TRT_ONLY:
    # ---- TEST C: full-pipeline parity (flows/styles/masks) ----
    with torch.inference_mode():
        base_out = base.eval(tile3, compute_masks=True, batch_size=BATCH)
        # ensure eval uses intended batch size on TRT
        trt_model.eval = models.CellposeModel.eval.__get__(trt_model, models.CellposeModel)
        trt_out  = trt_model.eval(tile3, compute_masks=True, batch_size=BATCH)

    print("\n[TEST C] Full pipeline parity")
    masks_pt = to_numpy(base_out[0]); masks_trt = to_numpy(trt_out[0])
    print(f"  masks: torch={masks_pt.shape} trt={masks_trt.shape}  IoU={iou_binary(masks_pt!=0, masks_trt!=0):.4f}")

    flows_pt = base_out[1]; flows_trt = trt_out[1]
    for k, (fpt, ftrt) in enumerate(zip(flows_pt, flows_trt)):
        summarize(f"  flow[{k}]", fpt, ftrt)

    summarize("  styles (pipeln)", base_out[2], trt_out[2])

    # Timing (full pipeline, compute_masks=True)
    with torch.inference_mode():
        print("\n[TIMING C] Full pipeline eval(tile3)")
        ms_base = time_op("  Torch eval", lambda: base.eval(tile3, compute_masks=True), device=device)
        ms_trt  = time_op("  TRT eval",   lambda: trt_model.eval(tile3, compute_masks=True), device=device)
        spd = ms_base / ms_trt if ms_trt > 0 else float('inf')
        print(f"  Speedup vs Torch: x{spd:.2f}")
else:
    # TRT-only smoke test
    with torch.inference_mode():
        _ = trt_model.eval(tile3, compute_masks=False)
    print("\n[TEST] TRT-only path executed (compute_masks=False).")
