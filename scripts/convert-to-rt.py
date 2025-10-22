#!/usr/bin/env python3
# scripts/convert-to-rt.py
# Minimal TensorRT builder for fixed-shape (min=opt=max) engines.

import argparse
import os
from typing import Tuple

import tensorrt as trt
import torch

DEFAULT_HW: tuple[int, int] = (256, 256)  # Hardcoded per user request

def _to_bytes(obj) -> bytes:
    if obj is None:
        return b""
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    try:
        return memoryview(obj).tobytes()
    except Exception:
        pass
    if hasattr(obj, "tobytes"):
        try:
            return obj.tobytes()  # type: ignore[call-arg]
        except Exception:
            pass
    if hasattr(obj, "data"):
        try:
            return memoryview(obj.data).tobytes()  # type: ignore[attr-defined]
        except Exception:
            pass
    raise TypeError(f"Unsupported engine blob type: {type(obj)}")


def build_engine(onnx_path: str, plan_path: str, hw: Tuple[int, int], workspace_mb: int, batch: int):
    # Pin CUDA device explicitly to avoid building on a busy/incompatible GPU
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    # When a CUDA_VISIBLE_DEVICES mask is set (e.g., "1"), the process only
    # sees one logical device at index 0. Build on that logical device.
    gpu_id = int(os.environ.get("CP_ISO_GPU", 0 if vis else 0))
    try:
        torch.cuda.set_device(gpu_id)
    except Exception:
        pass
    # Also set device via CUDA runtime to steer TensorRT
    try:
        import ctypes
        libcudart = ctypes.CDLL('libcudart.so')
        libcudart.cudaSetDevice(ctypes.c_int(gpu_id))
    except Exception:
        pass
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse ONNX file {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    config.set_flag(trt.BuilderFlag.BF16)
    config.builder_optimization_level = 3
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # Conservative tactics: restrict to cuBLAS/cuBLAS_LT/cuDNN (no Cask)
    mask = 0
    for src in (trt.TacticSource.CUBLAS, trt.TacticSource.CUBLAS_LT, trt.TacticSource.CUDNN):
        mask |= int(getattr(src, "value", src))
    config.set_tactic_sources(mask)

    # Input must be NCHW with static C.
    inp = network.get_input(0)
    if inp.shape is None or len(inp.shape) != 4:
        raise ValueError(f"Expected NCHW input, got {inp.shape}")
    _, C_dim, _, _ = tuple(inp.shape)

    if not isinstance(C_dim, int) or C_dim <= 0:
        raise ValueError(f"Channel dimension must be static/int in ONNX, got {C_dim}")
    C = C_dim

    H, W = int(hw[0]), int(hw[1])
    N = int(batch)
    fixed = (N, C, H, W)
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, fixed, fixed, fixed)
    config.add_optimization_profile(profile)

    engine_blob = builder.build_serialized_network(network, config)
    data = _to_bytes(engine_blob)
    if not data:
        raise RuntimeError("TensorRT engine build failed or returned empty blob.")

    with open(plan_path, "wb") as f:
        f.write(data)
    print(f"Saved TensorRT engine: {plan_path} (shape {N}x{C}x{H}x{W}, dtype=bf16)")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--workspace_mb", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=4, help="Fixed batch dimension N")
    return ap.parse_args()


def main():
    args = parse_args()
    hw = DEFAULT_HW
    build_engine(args.onnx, args.plan, hw, args.workspace_mb, args.batch)


if __name__ == "__main__":
    main()
