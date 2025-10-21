# scripts/export_cellpose_to_onnx.py
import argparse
import os

import torch

from cellpose import models


class _CPNetWrapper(torch.nn.Module):
    """Wrap Cellpose net so ONNX outputs are exactly (y, style)."""
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        y, style = self.net(x)[:2]
        return y, style

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained_model', type=str, default='cyto2',
                   help="Builtin name (e.g. 'cyto2','nuclei','cpsam') or path to .pth")
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--opset', type=int, default=18)
    p.add_argument('--onnx', type=str, required=True)
    # Only use opt_hw (export shape); min/max removed
    p.add_argument('--opt_hw', type=int, nargs=2, default=[256,256], help='Export resolution H W (used for dummy input)')
    p.add_argument('--batch', type=int, default=1)
    # Precision policy: model MUST be BF16. Raise if FP32 is requested or detected.
    p.add_argument('--fp32', action='store_true', help='(Forbidden) Export in float32 — will raise')
    args = p.parse_args()

    print(args.pretrained_model)
    device = torch.device('cuda' if args.gpu else 'cpu')
    if args.fp32:
        raise RuntimeError("This export enforces BF16. Do not pass --fp32.")
    model = models.CellposeModel(gpu=args.gpu, pretrained_model=args.pretrained_model,
                                 use_bfloat16=True)
    net = model.net.to(device)
    # Enforce BF16 weights strictly; fail fast if FP32 detected
    param_dtypes = {p.dtype for p in net.parameters()}
    if torch.float32 in param_dtypes:
        raise RuntimeError(f"Loaded model contains FP32 parameters: {param_dtypes}. Expected BF16 only.")
    target_dtype = torch.bfloat16
    net = net.eval()

    # detect channels from the network; default to 3 if SAM-like
    nchan = 3

    wrapper = _CPNetWrapper(net)

    # use opt shape for export; keep dynamic axes
    dummy = torch.randn(args.batch, nchan, args.opt_hw[0], args.opt_hw[1],
                        device=device, dtype=target_dtype)

    dynamic_axes = {
        'input':  {0:'N', 2:'H', 3:'W'},
        'y':      {0:'N', 2:'H', 3:'W'},
        'style':  {0:'N'}  # second dim (style) is static
    }

    os.makedirs(os.path.dirname(args.onnx) or '.', exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.onnx, opset_version=args.opset,
            dynamo=False,
            input_names=['input'], output_names=['y','style'],
            dynamic_axes=dynamic_axes
        )
    print(f"Exported ONNX to {args.onnx} (dtype={'bf16' if target_dtype==torch.bfloat16 else 'fp32'})")
    print(f"Assumed nchan={nchan}, dynamic N,H,W; batch={args.batch}")

if __name__ == '__main__':
    main()
