"""
Transformer Engine (TE) integration helpers.

This module provides opt-in patching of Linear and LayerNorm layers inside the
SAM image encoder with NVIDIA Transformer Engine equivalents, plus utilities
to gate FP8 safely. If TE is not installed or the device is not CUDA, it
silently disables itself.

We intentionally avoid changing convolutions or custom attention math; only
Linear and LayerNorm are swapped to reduce risk while capturing most of the
speedup in MLP and projection layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

try:
    import transformer_engine.pytorch as te  # type: ignore
    _TE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    te = None  # type: ignore
    _TE_AVAILABLE = False


@dataclass
class TEStatus:
    enabled: bool
    fp8_enabled: bool
    reason: str = ""


def _replace_linear(parent: nn.Module, name: str, linear: nn.Linear, params_dtype: torch.dtype) -> None:
    assert _TE_AVAILABLE and te is not None
    bias = linear.bias is not None
    new_linear = te.Linear(
        linear.in_features,
        linear.out_features,
        bias=bias,
        params_dtype=params_dtype,
    )
    # copy weights/bias
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight.detach().to(new_linear.weight.dtype))
        if bias:
            assert new_linear.bias is not None
            new_linear.bias.copy_(linear.bias.detach().to(new_linear.bias.dtype))
    setattr(parent, name, new_linear)


# Note: TE's LayerNorm API does not expose elementwise_affine the same way as torch.nn.LayerNorm.
# To avoid semantic drift, we currently keep LayerNorm modules as-is and only swap Linear layers.


def _all_divisible_by_16(encoder: nn.Module) -> bool:
    ok = True
    for m in encoder.modules():
        if isinstance(m, nn.Linear):
            if (m.in_features % 16 != 0) or (m.out_features % 16 != 0):
                ok = False
                break
    return ok


def patch_encoder_with_te(
    encoder: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    use_te: bool,
    use_fp8: bool,
) -> TEStatus:
    """
    Replace Linear and LayerNorm within `encoder` with TE modules in-place and
    indicate whether FP8 autocast should be enabled.

    Returns TEStatus with enabled/fp8_enabled flags and a reason string for logging.
    """
    if not use_te:
        return TEStatus(enabled=False, fp8_enabled=False, reason="use_te disabled")
    if not _TE_AVAILABLE:
        return TEStatus(enabled=False, fp8_enabled=False, reason="transformer_engine not installed")
    if device.type != "cuda":
        return TEStatus(enabled=False, fp8_enabled=False, reason=f"device {device.type} not supported for TE")

    # Traverse immediate and nested children, swapping Linear and LayerNorm
    replaced_linear = 0
    replaced_ln = 0

    def _recurse(module: nn.Module):
        nonlocal replaced_linear, replaced_ln
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                _replace_linear(module, child_name, child, params_dtype=dtype)
                replaced_linear += 1
            # Keep LayerNorm as PyTorch for stability; do not replace for now.
            else:
                _recurse(child)

    _recurse(encoder)

    if replaced_linear == 0 and replaced_ln == 0:
        return TEStatus(enabled=False, fp8_enabled=False, reason="no swappable modules found")

    # FP8 only if requested and likely safe for GEMM dims
    fp8_ok = _all_divisible_by_16(encoder)
    fp8_enabled = bool(use_fp8 and fp8_ok)
    reason = (
        f"TE enabled: linear={replaced_linear}, ln={replaced_ln}; FP8={'on' if fp8_enabled else 'off'}"
        + (" (shape not multiple of 16)" if use_fp8 and not fp8_ok else "")
    )
    return TEStatus(enabled=True, fp8_enabled=fp8_enabled, reason=reason)


def te_fp8_context(enabled: bool):
    """Return an autocast context manager for FP8 or a no-op context if disabled."""
    if enabled and _TE_AVAILABLE and te is not None:
        # Default recipe; users can extend to pass custom recipes if needed in future.
        from transformer_engine.common.recipe import DelayedScaling  # type: ignore

        recipe = DelayedScaling()
        return te.fp8_autocast(enabled=True, fp8_recipe=recipe)

    # Fallback no-op context manager
    from contextlib import nullcontext

    return nullcontext()
