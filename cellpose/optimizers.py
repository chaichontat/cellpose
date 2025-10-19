from __future__ import annotations

from typing import Iterable

import logging

import torch

logger = logging.getLogger(__name__)


def _import_muon_cls() -> type[torch.optim.Optimizer]:
    """Import Muon from native torch if available."""
    try:
        from torch.optim import Muon  # type: ignore attr-defined
    except Exception as exc:  # ImportError or AttributeError on older torch
        raise ImportError(
            "torch.optim.Muon is unavailable; install PyTorch >= 2.9 or choose a different optimizer."
        ) from exc
    return Muon


def create_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Instantiate an optimizer by name with consistent logging."""

    lname = name.lower()
    if lname == "adamw":
        logger.info(
            f">>> AdamW, learning_rate={lr:0.5f}, weight_decay={weight_decay:0.5f}"
        )
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if lname == "sgd":
        logger.info(
            f">>> SGD, learning_rate={lr:0.5f}, weight_decay={weight_decay:0.5f}, momentum={momentum:0.3f}"
        )
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum
        )

    if lname == "muon":
        muon_cls = _import_muon_cls()
        logger.info(
            f">>> Muon, learning_rate={lr:0.5f}, weight_decay={weight_decay:0.5f}"
        )
        return muon_cls(params, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"unknown optimizer '{name}'")


def optimizer_choices() -> list[str]:
    return ["adamw", "sgd", "muon"]
