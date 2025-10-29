from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

from cellpose import models
from cellpose.contrib.cellposetrt import CellposeModelTRT

TILE_SLICE = np.s_[5, :, :512, :512]

def iou_binary(a,b):
    a,b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a,b).sum(); union = np.logical_or(a,b).sum()
    return float(inter)/max(1.0, float(union))

def smape(ref,tst):
    r = torch.as_tensor(ref).float().flatten()
    t = torch.as_tensor(tst).float().flatten()
    d = (t-r).abs()
    return float((2*d/(r.abs()+t.abs()+1e-12)).mean()*100.0), float(d.mean())

def time_op(name, fn, warmup=1, iters=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter()-t0)/iters*1000
    print(f"{name}: {dt:.3f} ms/iter (avg over {iters}, warmup={warmup})")
    return dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--engine-bf16", type=Path, required=True)
    ap.add_argument("--engine-fp8", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--folder", type=Path, default=None)
    ap.add_argument("--n-samples", type=int, default=20)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    # Baseline BF16 (TRT) engine and the FP8 engine
    trt_bf16 = CellposeModelTRT(gpu=True, device=dev, pretrained_model=str(args.engine_bf16))
    trt_fp8  = CellposeModelTRT(gpu=True, device=dev, pretrained_model=str(args.engine_fp8))

    eval_kwargs = dict(batch_size=args.batch_size, flow_threshold=0, compute_masks=True)

    tile = tifffile.imread(args.image)[TILE_SLICE]
    with torch.inference_mode():
        out_bf16 = trt_bf16.eval(tile, **eval_kwargs)
        out_fp8  = trt_fp8.eval(tile, **eval_kwargs)
    print("[PARITY] BF16 TRT vs FP8 TRT")
    print(f"  masks IoU  : {iou_binary(out_bf16[0]!=0, out_fp8[0]!=0):.4f}")
    for k,(r,t) in enumerate(zip(out_bf16[1], out_fp8[1])):
        s,mae = smape(r,t); print(f"  flow[{k}] sMAPE={s:.3f}% MAE={mae:.5g}")

    with torch.inference_mode():
        print("\n[TIMING] Full pipeline")
        _ = time_op("  BF16 TRT", lambda: trt_bf16.eval(tile, **eval_kwargs))
        _ = time_op("   FP8 TRT", lambda: trt_fp8.eval(tile, **eval_kwargs))

    folder = args.folder or args.image.parent
    files = sorted([p for p in folder.glob("*.tif")])[:args.n_samples]
    ious=[]
    with torch.inference_mode():
        for i,f in enumerate(files,1):
            arr = tifffile.imread(f)[TILE_SLICE]
            mb = trt_bf16.eval(arr, **eval_kwargs)[0]
            mf = trt_fp8.eval(arr, **eval_kwargs)[0]
            if np.any(mb) and np.any(mf):
                iou = iou_binary(mb!=0, mf!=0)
                ious.append(iou); print(f"  {i}/{len(files)} IoU={iou:.4f}")
    if ious:
        a = np.array(ious); print(f"IoU range: min={a.min():.4f} median={np.median(a):.4f} max={a.max():.4f} (N={len(a)})")

if __name__ == "__main__":
    main()
