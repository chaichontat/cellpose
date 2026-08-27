# cellpose/contrib/cellposetrt/fp8_build_vit.py
"""
PTQ to FP8 (E4M3) for CellposeSAM (ViT-based) and build a TensorRT engine.

Pipeline:
  Baseline ONNX (BF16) -> ModelOpt ONNX PTQ (explicit Q/DQ) -> TensorRT engine

Usage:
  python -m cellpose.contrib.cellposetrt.fp8_build_vit \
    --pretrained cpsam \
    --output builds/cpsam_vit_fp8_b4.plan \
    --calib-folder /data/registered --calib-samples 512 \
    --bsize 256 --batch-size 4 --skip-mask-head
"""

# cellpose/contrib/cellposetrt/quant_policies_vit.py
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import modelopt.onnx.quantization as moq
from modelopt.onnx.quantization import fp8 as mo_fp8
import modelopt.torch.quantization as mtq
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader


def _ensure_cudnn_path():
    """Best effort to expose cuDNN/TensorRT libs to ORT's TensorRT EP."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    candidates = [
        Path(conda_prefix) / "lib",
        Path(conda_prefix) / "lib" / "stubs",
        Path(conda_prefix) / "lib" / "python3.13" / "site-packages" / "nvidia" / "cudnn" / "lib",
        Path(conda_prefix) / "lib" / "python3.13" / "site-packages" / "tensorrt_libs",
    ]
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(":") if p] if ld else []
    updated = False
    for path in candidates:
        if not path.exists():
            continue
        path_str = str(path)
        if path_str not in parts:
            parts.insert(0, path_str)
            updated = True
    if updated:
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


_ensure_cudnn_path()
import numpy as np
import tensorrt as trt
import tifffile
import torch
import torch.nn as nn

from cellpose.contrib.cellposetrt import trt_build as _trt_build

# Use ModelOpt's built-in heuristics to skip GEMV/small MatMuls for TensorRT performance.
try:
    from modelopt.onnx.quantization import fp8 as _mo_fp8  # noqa: F401
    from modelopt.onnx.quantization import graph_utils as _mo_graph_utils  # noqa: F401
except Exception:
    pass


def fp8_vit_cfg(*, skip_mask_head: bool = True):
    """
    FP8 PTQ for Vision Transformers:
      - Quantize nn.Linear throughout attention (QKV/proj) and MLP (fc1, fc2).
      - Optionally quantize patch-embed (Conv/Linear) if it exists.
      - Keep LayerNorm/Softmax/SDPA in BF16 (no quantizers added).
    """
    cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)  # FP8 E4M3, per-channel weights, per-tensor activations
    qcfg = cfg["quant_cfg"]

    # Be explicit about modules we do NOT want quantized
    # (ModelOpt generally quantizes Linear/Conv; these entries keep norms & softmax high precision.)
    qcfg["nn.LayerNorm"] = {"*": {"enable": False}}
    qcfg["nn.Softmax"]   = {"*": {"enable": False}}

    # Ensure we use per-channel on weights (axis=0) and per-tensor on activations
    qcfg["*weight_quantizer"] = {"axis": 0, "enable": True}
    qcfg["*input_quantizer"]  = {"axis": None, "enable": True}

    # Optional: keep final mask head high precision
    if skip_mask_head:
        for pat in ("*mask*", "*logit*", "*output_upscaling*", "*iou*"):
            qcfg[pat] = {"*": {"enable": False}}

    # Use simple max amax calibration by default (robust, fast).
    # You can switch to SmoothQuant/AWQ by changing cfg["algorithm"].
    cfg["algorithm"] = "max"
    return cfg



def _device_banner():
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(dev)
    cc = torch.cuda.get_device_capability(dev)
    print(f"CUDA device: {dev} | {name} | SM{cc[0]}{cc[1]}")

def _calib_stream(folder: Path, n: int, *, bsize: int, device: torch.device):
    files = sorted([p for p in folder.glob("*.tif")])
    seen = 0
    for f in files:
        if seen >= n: break
        try:
            arr = tifffile.imread(f)
            # use same tiling as runtime; default slice matches your scripts
            tile = arr[np.s_[5, :, :bsize, :bsize]]
            tile = np.concatenate([tile, np.zeros_like(tile[[0]])], axis=0)
            x = torch.from_numpy(tile).to(device, dtype=torch.bfloat16)
            if x.ndim == 3:
                x = x.unsqueeze(0)  # NCHW
            yield x
            seen += 1
        except Exception as e:
            print(f"[warn] skipping {f.name}: {e}")

@torch.no_grad()
def _forward_step(model, xb):
    # Forward through CellposeSAM's net (ViT encoder + decoder path); returns (y, style)-compatible tuple
    return model(xb)[:2]

def _export_qdq_onnx(model: nn.Module, onnx_out: str, *, batch_size: int, bsize: int):
    device = torch.device("cuda")
    dummy = torch.randn(batch_size, 3, bsize, bsize, device=device, dtype=torch.bfloat16)
    # ModelOpt quantized modules export to ONNX with Q/DQ recognized by TensorRT.
    torch.onnx.export(
        model, dummy, onnx_out,
        input_names=["input"], output_names=["y","style"],
        opset_version=20, dynamo=True  # dynamo=True per NVIDIA/TE & PyTorch exporter guidance
    )
    print(f"[export] quantized ONNX written to {onnx_out}")


class _OrtFolderDataReader(CalibrationDataReader):
    """Calibration data reader for ONNX FP8 quantization from a folder of TIFs."""
    def __init__(self, input_name: str, folder: Path, *, bsize: int, max_samples: int):
        self.input_name = input_name
        self.paths = sorted([p for p in Path(folder).glob('*.tif')])
        self.bsize = int(bsize)
        self.max_samples = int(max_samples)
        self._idx = 0

    def get_next(self):
        import numpy as np
        import tifffile
        while self._idx < len(self.paths) and self._idx < self.max_samples:
            p = self.paths[self._idx]
            self._idx += 1
            try:
                arr = tifffile.imread(p)
                tile = arr[np.s_[5, :, :self.bsize, :self.bsize]]
                tile = np.concatenate([tile, np.zeros_like(tile[[0]])], axis=0)
                if tile.shape[0] > 3:
                    tile = tile[:3]
                x = tile.astype('float32', copy=False)[None, ...]
                return {self.input_name: x}
            except Exception as e:
                print(f"[warn] calibration skip {p.name}: {e}")
                continue
        return None

    def rewind(self):
        self._idx = 0

    # Some ModelOpt helper paths use a non-standard 'get_first' API; implement it.
    def get_first(self):
        self.rewind()
        item = self.get_next()
        # Reset for subsequent full pass
        self.rewind()
        return item

def _build_trt(onnx_path: str, plan_path: str, *, bsize: int, batch_size: int, vram_mb: int, fp8: bool = False):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse quantized ONNX")

    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, vram_mb * (1 << 20))
    # Do not set BF16/FP16 flags on a strongly-typed explicit Q/DQ network; types come from ONNX.
    # IMPORTANT: no FP8/INT8 flags for explicit Q/DQ networks. TRT honors Q/DQ directly.
    # When building from a plain ONNX fallback, allow FP8 kernels via BuilderFlag.FP8.
    if fp8:
        try:
            config.set_flag(trt.BuilderFlag.FP8)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        except Exception:
            pass

    # Dynamic N [1..batch_size], fixed H=W=bsize
    inp = network.get_input(0)
    _, C, _, _ = tuple(inp.shape)
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (1, C, bsize, bsize), (batch_size, C, bsize, bsize), (batch_size, C, bsize, bsize))
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TRT build failed")
    Path(plan_path).parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "wb") as f:
        f.write(bytes(engine))
    print(f"[build] TensorRT engine saved: {plan_path}")


def main():
    ap = argparse.ArgumentParser(description="FP8 PTQ + TensorRT build for CellposeSAM (ViT)")
    ap.add_argument("--pretrained", type=str, required=True)
    ap.add_argument("--output", "-o", type=str, required=True)
    ap.add_argument("--calib-folder", type=Path, required=True)
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--calibration-npy", type=Path, default=None, help="Path to numpy calibration data (e.g., calib.npy)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--bsize", type=int, default=256)
    ap.add_argument("--vram", type=int, default=12000)
    ap.add_argument("--skip-mask-head", action="store_true", default=False)
    ap.add_argument("--force", action="store_true", help="Rebuild even if cached artifacts exist")
    ap.add_argument("--calibrate-per-node", action="store_true", help="Enable per-node calibration (lower memory, slower)")
    ap.add_argument("--hp-dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                    help="High-precision dtype for non-quantized ops (bf16 preferred; fp16 fallback if toolchain limits).")
    args = ap.parse_args()

    _device_banner()

    out_plan = Path(args.output)
    onnx_fp32 = Path(args.output).with_suffix('.fp32.onnx')
    onnx_qdq = Path(args.output).with_suffix('.onnx')

    # Fast path: if plan already exists and not forcing, exit immediately
    if out_plan.exists() and not args.force:
        print(f"[cache] Existing engine found: {out_plan}. Use --force to rebuild.")
        return

    # Baseline FP32 ONNX export (returns y, style). Reuse if present unless --force.
    if not onnx_fp32.exists() or args.force:
        print("[export] exporting baseline FP32 ONNX ...")
        _trt_build.export_onnx(
            args.pretrained,
            str(onnx_fp32),
            batch_size=args.batch_size,
            bsize=args.bsize,
            opset=21,
            dtype=torch.float32,
            dynamo=True,
            dynamic_batch=False,
        )
    else:
        print(f"[cache] Reusing baseline ONNX: {onnx_fp32}")

    # ONNX‑level FP8 PTQ with explicit Q/DQ
    print("[PTQ] quantizing ONNX to FP8 (explicit Q/DQ) ...")
    # Quantized ONNX reuse if present unless --force
    if not onnx_qdq.exists() or args.force:
        input_name = 'input'
        # Prefer user-provided calibration numpy as per NVIDIA example
        calib_np = None
        if args.calibration_npy and Path(args.calibration_npy).is_file():
            try:
                import numpy as _np
                calib_np = _np.load(args.calibration_npy, allow_pickle=True)
                print(f"[calib] Loaded calibration numpy: {args.calibration_npy} shape={getattr(calib_np, 'shape', None)}")
            except Exception as e:
                print(f"[warn] Failed to load calibration numpy: {e}. Falling back to random calibration.")
                calib_np = None
        # Build shapes string only if no calibration numpy provided
        calib_shapes = f"{input_name}:1x3x{args.bsize}x{args.bsize}"

        # Prefer GPU-assisted calibration (TensorRT EP) with CPU fallback.
        # GPU-only calibration: use TRT EP with a data reader (avoids symbolic-shape path)
        calibration_eps = ['trt']
        calib_reader = _OrtFolderDataReader(input_name, args.calib_folder, bsize=args.bsize, max_samples=args.calib_samples)

        # Directly call fp8.quantize with a data reader, no CPU fallback
        qmodel = mo_fp8.quantize(
            onnx_path=str(onnx_fp32),
            calibration_method='max',
            calibration_data_reader=calib_reader,
            calibration_shapes=None,
            calibration_eps=calibration_eps,
            op_types_to_quantize=["MatMul", "Gemm"],
            op_types_to_exclude=["Conv"],
            nodes_to_exclude=[],
            high_precision_dtype=args.hp_dtype,
            mha_accumulation_dtype='fp32',
            direct_io_types=True,
            calibrate_per_node=args.calibrate_per_node,
            passes=['concat_elimination'],
            use_external_data_format=False,
        )
        onnx.save(qmodel, str(onnx_qdq))
    else:
        print(f"[cache] Reusing quantized ONNX: {onnx_qdq}")

    # Build TensorRT engine from the explicit Q/DQ ONNX
    _build_trt(str(onnx_qdq), args.output, bsize=args.bsize, batch_size=args.batch_size, vram_mb=args.vram)

if __name__ == "__main__":
    main()
