import ctypes

import tensorrt as trt
import torch

from cellpose import models


def _torch_ptr(t: torch.Tensor) -> int:
    return ctypes.c_void_p(t.data_ptr()).value


class TRTEngineModule(torch.nn.Module):
    def __init__(self, engine_path: str, device=torch.device("cuda")):
        super().__init__()

        if getattr(torch.version, "hip", None):
            raise RuntimeError("TensorRT backend is incompatible with ROCm/PyTorch HIP builds.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TensorRT backend requires an NVIDIA CUDA GPU. No CUDA device is available."
            )
        # Normalize device
        dev = torch.device(device)
        if dev.type != "cuda":
            raise RuntimeError(
                f"TensorRT backend requires a CUDA device, got '{dev.type}'. CPUs/MLX are unsupported."
            )
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Invalid CUDA device index {dev.index}; available count={torch.cuda.device_count()}"
            )
        self.device = dev

        # Enforce TensorRT >= 10 (tensors API only)
        try:
            ver = getattr(trt, "__version__", "0.0.0")
            major = int(str(ver).split(".")[0])
        except Exception:
            major = 0
            ver = "unknown"
        if major < 10:
            raise RuntimeError(
                f"TensorRT >= 10 required (found {ver}). Legacy bindings API is unsupported."
            )

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._ctx = self._engine.create_execution_context()
        # Validate tensors API presence
        if not (hasattr(self._engine, "num_io_tensors") and hasattr(self._ctx, "set_input_shape")):
            raise RuntimeError(
                "TensorRT tensors API not available. Please rebuild with TensorRT >= 10 and re-export the engine."
            )

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

        # -------- Tensor API (TRT 10+) --------
        # 1) Set input shape by name
        self._ctx.set_input_shape(self._name_in, (N, C, H, W))

        # 2) Allocate outputs; query shapes from engine (may have -1 -> allocate by heuristics if needed)
        #    Cellpose heads are [N,3,H,W] and [N,S], so we can shape from input.
        # Read S from engine if available; otherwise default to 256 (Cellpose style vec size) and adjust if needed.
        try:
            # If engine carries concrete dims (profile-dependent), use them
            s_dims = tuple(self._engine.get_tensor_shape(self._name_s))
            # Replace -1 with concrete batch N if needed
            if any(d < 0 for d in s_dims):
                S = s_dims[-1] if s_dims[-1] > 0 else 256
            else:
                S = s_dims[-1]
        except Exception:
            S = 256

        y = torch.empty((N, 3, H, W), device=X.device, dtype=self._dtype_y)
        s = torch.empty((N, S), device=X.device, dtype=self._dtype_s)

        # Use the current PyTorch stream to avoid hazards
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
    """
    Drop-in subclass that preserves the constructor and eval(...) interface.
    Args (extra):
      backend: 'trt' | 'onnxrt_trt'
      engine_path: path to .plan (for backend='trt')
      onnx_path: path to .onnx (for backend='onnxrt_trt' or validation)
      fp16: bool, default True
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
