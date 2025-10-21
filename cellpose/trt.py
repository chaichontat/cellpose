import ctypes

import tensorrt as trt
import torch

from cellpose import models


def _torch_ptr(t: torch.Tensor) -> int:
    return ctypes.c_void_p(t.data_ptr()).value


class TRTEngineModule(torch.nn.Module):
    def __init__(self, engine_path: str, device=torch.device("cuda"), fp16=True):
        super().__init__()
        self.device = device
        # dtype will be inferred from the engine per tensor; keep a legacy
        # attribute for callers that check .dtype (use input dtype).
        self.dtype = torch.float16 if fp16 else torch.float32

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._ctx = self._engine.create_execution_context()

        # Detect API surface
        self._has_tensors_api = hasattr(self._engine, "num_io_tensors") and hasattr(
            self._ctx, "set_input_shape"
        )
        self._has_bindings_api = hasattr(self._engine, "get_binding_index") and hasattr(
            self._ctx, "execute_v2"
        )

        # Names exported by our ONNX: 'input' -> y/style
        self._name_in = "input"
        self._name_y = "y"
        self._name_s = "style"

        # Map TRT dtypes to torch dtypes
        def _to_torch_dtype(dt):
            if dt == trt.DataType.BF16:
                return torch.bfloat16
            if dt == trt.DataType.HALF:
                return torch.float16
            if dt == trt.DataType.FLOAT:
                return torch.float32
            raise ValueError(f"Unsupported TensorRT dtype: {dt}")

        if self._has_tensors_api:
            # Sanity: make sure the names exist and modes are right
            # (TRT 10 API: get_tensor_mode / INPUT | OUTPUT)
            for name in (self._name_in, self._name_y, self._name_s):
                # Will raise if name is unknown
                self._engine.get_tensor_dtype(name)
            from tensorrt import TensorIOMode

            assert self._engine.get_tensor_mode(self._name_in) == TensorIOMode.INPUT, (
                "input must be INPUT tensor"
            )
            assert self._engine.get_tensor_mode(self._name_y) == TensorIOMode.OUTPUT, (
                "y must be OUTPUT tensor"
            )
            assert self._engine.get_tensor_mode(self._name_s) == TensorIOMode.OUTPUT, (
                "style must be OUTPUT tensor"
            )
            # Capture per-tensor dtypes from engine
            self._dtype_in = _to_torch_dtype(self._engine.get_tensor_dtype(self._name_in))
            self._dtype_y = _to_torch_dtype(self._engine.get_tensor_dtype(self._name_y))
            self._dtype_s = _to_torch_dtype(self._engine.get_tensor_dtype(self._name_s))
        elif self._has_bindings_api:
            # Legacy path: cache binding indices
            self._idx_in = self._engine.get_binding_index(self._name_in)
            self._idx_y = self._engine.get_binding_index(self._name_y)
            self._idx_s = self._engine.get_binding_index(self._name_s)
            self._dtype_in = _to_torch_dtype(self._engine.get_binding_dtype(self._idx_in))
            self._dtype_y = _to_torch_dtype(self._engine.get_binding_dtype(self._idx_y))
            self._dtype_s = _to_torch_dtype(self._engine.get_binding_dtype(self._idx_s))
        else:
            raise RuntimeError(
                "Unsupported TensorRT Python bindings: neither tensors API nor bindings API detected"
            )
        # For back-compat, reflect the input dtype
        self.dtype = self._dtype_in

    def forward(self, X: torch.Tensor):
        assert X.is_cuda, "Input must be a CUDA tensor"
        if X.dtype != self._dtype_in:
            X = X.to(self._dtype_in)
        X = X.contiguous()
        N, C, H, W = X.shape

        if self._has_tensors_api:
            # -------- Tensor API (TRT 10+) --------
            # 1) Set input shape by name
            self._ctx.set_input_shape(self._name_in, (N, C, H, W))

            # 2) Allocate outputs; query shapes from engine (may have -1 -> allocate by heuristics if needed)
            #    For fully dynamic outputs, we can either:
            #       a) query get_max_output_size & use IOutputAllocator, or
            #       b) set addresses to preallocated buffers of expected size and sync.
            #    Cellpose heads are [N,3,H,W] and [N,S], so we can shape from input.
            # y: [N, 3, H, W]; style: [N, S] where S is static per model (read from ONNX/trt if desired)
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

        else:
            # -------- Legacy bindings API (TRT 8/9) --------
            # Dynamic shapes via set_binding_shape
            self._ctx.set_binding_shape(self._idx_in, (N, C, H, W))
            y_shape = tuple(self._ctx.get_binding_shape(self._idx_y))
            s_shape = tuple(self._ctx.get_binding_shape(self._idx_s))
            y = torch.empty(y_shape, device=X.device, dtype=self._dtype_y)
            s = torch.empty(s_shape, device=X.device, dtype=self._dtype_s)

            bindings = [None] * self._engine.num_bindings
            bindings[self._idx_in] = _torch_ptr(X)
            bindings[self._idx_y] = _torch_ptr(y)
            bindings[self._idx_s] = _torch_ptr(s)

            ok = self._ctx.execute_v2(bindings)
            if not ok:
                raise RuntimeError("TensorRT execute_v2 failed")
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
        backend="trt",
        engine_path=None,
        fp16=True,
    ):
        super().__init__(
            gpu=gpu,
            pretrained_model=pretrained_model,
            model_type=model_type,
            diam_mean=diam_mean,
            device=device,
            nchan=nchan,
            use_bfloat16=use_bfloat16,
        )
        # Replace .net with a backend wrapper that exposes the same callable signature
        dev = torch.device("cuda" if (device is None and gpu) else (device or "cpu"))
        if backend == "trt":
            assert engine_path is not None, "Provide engine_path for backend='trt'"
            self.net = TRTEngineModule(engine_path, device=dev, fp16=fp16)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def eval(self, *args, **kwargs):
        kwargs |= {"batch_size": 1}  # TRT engine built for batch size 1
        return super().eval(*args, **kwargs)
