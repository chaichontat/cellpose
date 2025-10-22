"""Builds a TensorRT engine (.plan) from a Cellpose model.

Speed up of 1.7x (batch size 4) - 2.2x (batch size 1) observed for
CellposeSAM net on RTX 5090 with BF16 engine
compared to the native PyTorch bfloat16 inference.

Requirements
- NVIDIA GPU with BF16 support (SM80+, e.g., Ampere or newer).
- TensorRT >= 10 (Python bindings with the tensors API).
- PyTorch with ONNX exporter.
- CellposeSAM bfloat16 weights (pretrained_model path).

Dependencies can be installed via pip:
  `pip install tensorrt-cu12 nvidia-cuda-runtime-cu12`
Ensure that the requested CUDA version matches your environment's CUDA version.

Behavior
- Exports ONNX that returns exactly (y, style), matching Cellpose segmentation.
- Fixed optimization profile: N=batch-size, C=3, H=W=bsize.
- BF16 engine with builder_optimization_level=3 and OBEY_PRECISION_CONSTRAINTS.

Gotchas
- Plan files are not portable: they are specific to GPU arch (SM) and
  TensorRT/CUDA/driver. Rebuild on each host/GPU family;
- Tensor size is fixed, bsize and batch_size must be specified and the same at runtime.

Usage
  python trt_build.py PRETRAINED -o OUTPUT.plan [--batch-size N] [--bsize 256] [--vram 12000] [--opset 18]

Runs in ~2 minutes on a Threadripper 7990X with RTX 5090
Tested on tensorrt-cu12 10.13.3.9, torch 2.9.0+cu128, Python 3.13.9
NVIDIA open driver 570.195.03, Ubuntu 24.04.2

Example output:
    Loaded tile: (2, 512, 512) uint16
    Engine path: /home/chaichontat/cellpose/scripts/builds/cpsam_b4_sm120_bf16.plan
    Using CUDA device: cuda:0 | NVIDIA GeForce RTX 5090

    [TEST C] Full pipeline parity
    masks: torch=(512, 512) trt=(512, 512)  IoU=0.9986
    flow[0]: shape=(512, 512, 3) | sMAPE=2.257%  MAE=0.176858
    flow[1]: shape=(2, 512, 512) | sMAPE=27.623%  MAE=0.0060048
    flow[2]: shape=(512, 512) | sMAPE=0.816%  MAE=0.0170394

    [TIMING C] Full pipeline eval(tile3)
    Torch eval: 222.155 ms/iter (avg over 5, warmup=1)
    TRT eval: 138.330 ms/iter (avg over 5, warmup=1)
    Speedup vs Torch: x1.61

    [TIMING D] Net-only forward (1x3x256x256)
    Torch net: 15.930 ms/iter (CUDA events, iters=100, warmup=10)
    TRT net  : 7.110 ms/iter (CUDA events, iters=100, warmup=10)
    Speedup (net-only): x2.24

    [TEST E] IoU parity on first 20 images from: /working/20251001_JaxA3_Coro11/analysis/deconv/registered--3r+pi
    1/20 processed... IoU=0.9994
    ...
    20/20 processed... IoU=0.9991
    IoU range: min=0.9816  median=0.9973  max=0.9996  (N=20)
"""

import argparse
import os
from pathlib import Path

import tensorrt as trt
import torch

from cellpose import models


class _CPNetWrapper(torch.nn.Module):
    """Wrap Cellpose net to expose exactly (y, style) for ONNX export.

    Contract matches the main segmentation workflow and the TensorRT engine.
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        y, style = self.net(x)[:2]
        return y, style


def export_onnx(pretrained_model: str, onnx_out: str, *, batch_size: int, bsize: int, opset: int = 18):
    device = torch.device("cuda")
    model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model, use_bfloat16=True)
    net = model.net.to(device).eval()

    # Ensure weights are BF16 as expected
    param_dtypes = {p.dtype for p in net.parameters()}
    if torch.float32 in param_dtypes:
        raise RuntimeError(f"Loaded model contains FP32 parameters: {param_dtypes}. Expected BF16 only.")
    wrapper = _CPNetWrapper(net)

    dummy = torch.randn(batch_size, 3, bsize, bsize, device=device, dtype=torch.bfloat16)
    Path(os.path.dirname(onnx_out) or ".").mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_out,
            opset_version=opset,
            dynamo=False,
            input_names=["input"],
            output_names=["y", "style"],
        )
    print(f"Exported ONNX to {onnx_out}.")


def build_engine(onnx_path: str, plan_path: str, *, bsize: int, vram: int, batch_size: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build a TensorRT engine.")
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse ONNX file {onnx_path}")

    config = builder.create_builder_config()
    config.builder_optimization_level = 3  # 4 doesn't help, 5 is too slow
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, vram * (1 << 20))
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    # Sanity check: input must be NCHW with static C
    inp = network.get_input(0)
    if inp.shape is None or len(inp.shape) != 4:
        raise ValueError(f"Expected NCHW input, got {inp.shape}")

    _, C_dim, _, _ = tuple(inp.shape)
    if not isinstance(C_dim, int) or C_dim <= 0:
        raise ValueError(f"Channel dimension must be static/int in ONNX, got {C_dim}")

    C = int(C_dim)
    N = int(batch_size)

    fixed = (N, C, bsize, bsize)
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, fixed, fixed, fixed)
    config.add_optimization_profile(profile)

    engine_blob = builder.build_serialized_network(network, config)
    if engine_blob is None:
        raise RuntimeError("TensorRT engine build failed or returned empty blob.")
    data = bytes(engine_blob)

    out_dir = Path(plan_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "wb") as f:
        f.write(data)
    print(f"Saved TensorRT engine: {plan_path} (shape {fixed}, dtype=bf16)")


def main():
    ap = argparse.ArgumentParser(description="Export Cellpose net to ONNX and build TensorRT engine")
    ap.add_argument("pretrained_model", type=str, help="Path/name of pretrained model (e.g., cpsam)")
    ap.add_argument("-o", "--output", type=str, required=True, help="TensorRT engine output path (.plan)")
    ap.add_argument("--vram", type=int, default=12000, help="Amount of GPU memory available (in MB) for TensorRT to optimize for")
    ap.add_argument("--batch-size", type=int, default=1, help="Fixed batch dimension N")
    ap.add_argument("--bsize", type=int, default=256, help="Tile size (256x256 by default)")
    ap.add_argument("--opset", type=int, default=20, help="ONNX opset version to use for export")
    args = ap.parse_args()

    plan_path = args.output
    p = Path(plan_path)
    onnx_out = str(p.with_suffix(".onnx"))
    export_onnx(args.pretrained_model, onnx_out, batch_size=args.batch_size, bsize=args.bsize, opset=args.opset)
    build_engine(onnx_out, plan_path, batch_size=args.batch_size, bsize=args.bsize, vram=args.vram)


if __name__ == "__main__":
    main()
