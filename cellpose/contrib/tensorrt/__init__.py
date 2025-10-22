"""TensorRT-backed Cellpose model module."""

import ctypes

import torch

import tensorrt as trt
from cellpose import models


def _torch_ptr(t: torch.Tensor):
    return ctypes.c_void_p(t.data_ptr()).value


class TRTEngineModule(torch.nn.Module):
    """TensorRT-backed CellposeSAM model.

    It is not intended for auxiliary training/export variants that add extra outputs
    (e.g., BioImage.IO downsampled tensors, denoise/perceptual losses).

    Notes
    - Requires TensorRT >= 10.
    - Engines are compiled for fixed profiles batch size and tile size.
    """
    def __init__(self, engine_path: str, device=torch.device("cuda")):
        super().__init__()

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                f"TensorRT backend requires a CUDA device, got '{self.device.type}'. CPUs/MLX are unsupported."
            )

        ver = getattr(trt, "__version__", None)
        if not ver:
            raise RuntimeError("TensorRT >= 10 required (version unknown).")
        try:
            major = int(str(ver).split(".")[0])
        except Exception:
            raise RuntimeError(f"TensorRT >= 10 required (found {ver}).")
        if major < 10:
            raise RuntimeError(f"TensorRT >= 10 required (found {ver}).")

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._ctx = self._engine.create_execution_context()

        # Names exported by our ONNX: 'input' -> y/style
        self._name_in = "input"
        self._name_y = "y"
        self._name_s = "style"

        def _to_torch_dtype(dt):
            if dt == trt.DataType.BF16:
                return torch.bfloat16
            if dt == trt.DataType.HALF:
                return torch.float16
            if dt == trt.DataType.FLOAT:
                return torch.float32
            raise ValueError(f"Unsupported TensorRT dtype: {dt}")

        # Sanity: make sure the names exist and modes are right
        for name in (self._name_in, self._name_y, self._name_s):
            self._engine.get_tensor_dtype(name)

        # Capture per-tensor dtypes from engine
        self._dtype_in = _to_torch_dtype(self._engine.get_tensor_dtype(self._name_in))
        self._dtype_y = _to_torch_dtype(self._engine.get_tensor_dtype(self._name_y))
        self._dtype_s = _to_torch_dtype(self._engine.get_tensor_dtype(self._name_s))

        self.dtype = self._dtype_in

    def forward(self, X: torch.Tensor):
        assert X.is_cuda, "Input must be a CUDA tensor"
        if X.dtype != self._dtype_in:
            X = X.to(self._dtype_in)
        X = X.contiguous()
        N, C, H, W = X.shape

        # 1) Set input shape by name
        self._ctx.set_input_shape(self._name_in, (N, C, H, W))

        # 2) Allocate outputs; query shapes from engine (may have -1 -> allocate by heuristics if needed)
        #    Cellpose heads are [N,3,H,W] and [N,S], so we can shape from input.
        # Read S from engine if available; otherwise default to 256 (Cellpose style vec size) and adjust if needed.
        try:
            # If engine carries concrete dims (profile-dependent), use them
            s_dims = tuple(self._engine.get_tensor_shape(self._name_s))
            if any(d < 0 for d in s_dims):
                S = s_dims[-1] if s_dims[-1] > 0 else 256
            else:
                S = s_dims[-1]
        except Exception:
            S = 256

        y = torch.empty((N, 3, H, W), device=X.device, dtype=self._dtype_y)
        s = torch.empty((N, S), device=X.device, dtype=self._dtype_s)

        stream = torch.cuda.current_stream(self.device)
        stream_handle = int(stream.cuda_stream)

        self._ctx.set_tensor_address(self._name_in, _torch_ptr(X))
        self._ctx.set_tensor_address(self._name_y, _torch_ptr(y))
        self._ctx.set_tensor_address(self._name_s, _torch_ptr(s))

        ok = self._ctx.execute_async_v3(stream_handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        return y, s


class CellposeModelTRT(models.CellposeModel):
    """Drop-in replacement for CellposeModel (eval) using TensorRT.

    Preparation
    - Build an engine for your model first with scripts/trt_build.py, for example:
      python scripts/trt_build.py PRETRAINED -o OUTPUT.plan --batch-size 4 --bsize 256
      Then pass engine_path=OUTPUT.plan to this class.

    Contract
    - Uses a TensorRT engine whose forward returns exactly (y, style) as defined
      in TRTEngineModule; aligns with the main segmentation pipeline.
    - Not intended for denoise/perceptual-loss training utilities or BioImage.IO
      export paths that expect additional tensors beyond (y, style).
    """

    def __init__(
        self,
        gpu=False,
        pretrained_model="cyto2",
        model_type=None,
        diam_mean=None,
        device=None,
        nchan=None,
        use_bfloat16=True,
        engine_path=None,
    ):
        super().__init__(
            gpu=gpu,
            pretrained_model=pretrained_model,
            model_type=model_type,
            diam_mean=diam_mean,
            device=device,
            nchan=nchan,
            use_bfloat16=True,
        )
        dev = torch.device("cuda" if device is None else device)
        if not use_bfloat16:
            raise ValueError("CellposeModelTRT only supports use_bfloat16=True")
        if engine_path is None:
            raise ValueError("engine_path must be provided for CellposeModelTRT")
        self.net = TRTEngineModule(engine_path, device=dev)

    def eval(self, x, **kwargs):
        if kwargs.get("bsize", 256) != 256:
            raise ValueError("CellposeModelTRT only supports bsize=256")
        return super().eval(x, **kwargs)
