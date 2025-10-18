from __future__ import annotations

from typing import Iterable

import logging

import torch

logger = logging.getLogger(__name__)


def _import_muon_cls():
    """Import Muon from native torch (PyTorch >= 2.9)."""
    try:
        from torch.optim import Muon
    except Exception as e:  # ImportError or AttributeError on older torch
        raise ImportError(
            "torch.optim.Muon is unavailable; please install/upgrade PyTorch to >= 2.9 or choose a different optimizer."
        ) from e
    return Muon


def create_optimizer(
    params: Iterable[torch.nn.Parameter],
    name: str,
    lr: float,
    *,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
):
    """Create an optimizer by name with consistent arguments.

    Supported names: 'adamw', 'sgd', 'muon'.
    For 'muon', a Muon optimizer implementation must be importable.
    """
    lname = name.lower()
    if lname == "adamw":
        logger.info(f">>> AdamW, learning_rate={lr:0.5f}, weight_decay={weight_decay:0.5f}")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if lname == "sgd":
        logger.info(f">>> SGD, learning_rate={lr:0.5f}, weight_decay={weight_decay:0.5f}, momentum={momentum:0.3f}")
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    if lname == "muon":
        MuonCls = _import_muon_cls()
        logger.info(f">>> Muon, learning_rate={lr:0.5f}, weight_decay={weight_decay:0.5f}")
        return MuonCls(params, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"unknown optimizer '{name}', expected one of: adamw, sgd, muon")


def optimizer_choices():
    return ["adamw", "sgd", "muon"]
