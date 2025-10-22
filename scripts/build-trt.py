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
    dynamic_axes = {
        "input": {0: "N", 2: "H", 3: "W"},
        "y": {0: "N", 2: "H", 3: "W"},
        "style": {0: "N"},
    }
    Path(os.path.dirname(onnx_out) or ".").mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_out,
            opset_version=opset,
            dynamo=True,
            input_names=["input"],
            output_names=["y", "style"],
            dynamic_axes=dynamic_axes,
        )
    print(f"Exported ONNX to {onnx_out}.")


def _to_bytes(blob) -> bytes:
    if blob is None:
        return b""
    return bytes(blob)


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
    config.builder_optimization_level = 4
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
    data = _to_bytes(engine_blob)
    if not data:
        raise RuntimeError("TensorRT engine build failed or returned empty blob.")
    Path(plan_path).mkdir(parents=True, exist_ok=True)
    with open(plan_path, "wb") as f:
        f.write(data)
    print(f"Saved TensorRT engine: {plan_path} (shape {fixed}, dtype=bf16)")


def main():
    ap = argparse.ArgumentParser(description="Export Cellpose net to ONNX and build TensorRT engine")
    ap.add_argument("pretrained_model", type=str, help="Path/name of pretrained model (e.g., cpsam)")
    ap.add_argument("-o", "--output", type=str, required=True, help="TensorRT engine output path (.plan)")
    ap.add_argument("--vram", type=int, default=12000, help="Amount of GPU memory available (in MB) for TensorRT to optimize for")
    ap.add_argument("--batch-size", type=int, default=4, help="Fixed batch dimension N")
    ap.add_argument("--bsize", type=int, default=256, help="Tile size (256x256 by default)")
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset version to use for export")
    args = ap.parse_args()

    plan_path = args.output
    p = Path(plan_path)
    onnx_out = str(p.with_suffix(".onnx"))
    export_onnx(args.pretrained_model, onnx_out, batch_size=args.batch_size, bsize=args.bsize, opset=args.opset)
    build_engine(onnx_out, plan_path, batch_size=args.batch_size, bsize=args.bsize, vram=args.vram)


if __name__ == "__main__":
    main()
