from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from cellpose import core as _core
from cellpose.core import run_net


@dataclass(frozen=True)
class StripeLayout:
    K: int
    guard: int
    starts: tuple[int, ...]
    stripe_height: int
    slot_height: int
    border: int
    bsize: int


MIN_GUARD = 5


def compute_stripe_layout(
    Ly: int,
    *,
    bsize: int = 256,
    border: int = 0,
) -> StripeLayout | None:
    """Auto-select K to maximize packing while ensuring guard >= MIN_GUARD.

    Parameters
    ----------
    Ly : int
        Height of each data stripe (in pixels) before border padding.
    bsize : int
        Size of the square tile used for packing.
    border : int
        Pixels of padding applied above and below each stripe.

    Returns
    -------
    StripeLayout or None
        Layout configuration, or None if packing with guard >= MIN_GUARD is not possible.
    """
    if Ly <= 0 or bsize <= 0:
        raise ValueError(f"Invalid dimensions Ly={Ly}, bsize={bsize}")
    border = max(0, int(border))
    slot_height = Ly + 2 * border
    if slot_height <= 0:
        return None

    # Find largest K where guard >= MIN_GUARD
    # Constraint: K * slot_height + (K-1) * MIN_GUARD <= bsize
    Kmax = 1
    for K in range(2, 10):  # reasonable upper bound
        required = K * slot_height + (K - 1) * MIN_GUARD
        if required <= bsize:
            Kmax = K
        else:
            break

    if Kmax < 2:
        return None  # Can't pack with guard >= MIN_GUARD

    # Compute actual guard (maximize it)
    guard = (bsize - Kmax * slot_height) // (Kmax - 1)

    starts = tuple(i * (slot_height + guard) for i in range(Kmax))
    return StripeLayout(
        K=Kmax,
        guard=guard,
        starts=starts,
        stripe_height=Ly,
        slot_height=slot_height,
        border=border,
        bsize=bsize,
    )


def pack_planes_to_stripes(
    x: np.ndarray, layout: StripeLayout
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    if x.ndim != 4:
        raise ValueError(f"Expected x shape [Lz, Ly, Lx, C], got {x.shape}")
    Lz, Ly, Lx, C = x.shape
    if Ly != layout.stripe_height:
        raise ValueError(
            f"Ly mismatch: x has Ly={Ly}, layout.stripe_height={layout.stripe_height}"
        )
    K = layout.K
    G = (Lz + K - 1) // K
    packed = np.zeros((G, layout.bsize, Lx, C), dtype=x.dtype)
    mapping: list[tuple[int, int, int]] = []
    for z in range(Lz):
        g = z // K
        i = z % K
        slot_start = layout.starts[i]
        y0 = slot_start + layout.border
        y1 = y0 + Ly
        packed[g, y0:y1, :, :] = x[z]
        mapping.append((g, y0, y1))
    return packed, mapping


def unpack_stripes_to_planes(
    y_packed: np.ndarray, mapping: Sequence[tuple[int, int, int]], Lz: int, Ly: int
) -> np.ndarray:
    if y_packed.ndim != 4:
        raise ValueError(
            f"Expected y_packed shape [G, bsize, Lx, Cout], got {y_packed.shape}"
        )
    G, bsize, Lx, Cout = y_packed.shape
    y = np.zeros((Lz, Ly, Lx, Cout), dtype=y_packed.dtype)
    for z in range(Lz):
        g, y0, y1 = mapping[z]
        if not (0 <= g < G and 0 <= y0 < y1 <= bsize):
            raise ValueError(f"Invalid mapping entry for z={z}: {(g, y0, y1)}")
        y[z, :, :, :] = y_packed[g, y0:y1, :, :]
    return y


@contextmanager
def _forward_counter() -> dict[str, float]:
    orig_forward = _core._forward
    stats = {"batches": 0, "tiles": 0}

    def wrapped(net, x):
        stats["batches"] += 1
        stats["tiles"] += int(x.shape[0])
        return orig_forward(net, x)

    _core._forward = wrapped  # type: ignore[assignment]
    try:
        yield stats
    finally:
        _core._forward = orig_forward  # type: ignore[assignment]


def _maybe_count(return_stats: bool):
    return _forward_counter() if return_stats else nullcontext({})


def run_net_with_stats(
    net,
    stack: np.ndarray,
    *,
    bsize: int = 256,
    batch_size: int = 4,
    augment: bool = False,
    tile_overlap: float = 0.1,
    return_stats: bool = False,
    single_tile_if_fit: bool = False,
):
    with _maybe_count(return_stats) as stats:
        yf, styles = run_net(
            net,
            stack,
            bsize=bsize,
            batch_size=batch_size,
            augment=augment,
            tile_overlap=tile_overlap,
            single_tile_if_fit=single_tile_if_fit,
        )
    if return_stats:
        stats["tiles_per_batch"] = stats["tiles"] / max(1, stats["batches"])
        return yf, styles, stats
    return yf, styles


def forward_packed_2d(
    net,
    stack: np.ndarray,
    *,
    bsize: int = 256,
    batch_size: int = 4,
    augment: bool = False,
    tile_overlap: float = 0.1,
    pack_border: int = 0,
    return_stats: bool = False,
):
    if stack.ndim != 4:
        raise ValueError(f"stack must be [Lz, Ly, Lx, C], got {stack.shape}")
    Lz, Ly, Lx, C = stack.shape
    layout = compute_stripe_layout(
        Ly,
        bsize=bsize,
        border=pack_border,
    )
    if layout is None:
        result = run_net_with_stats(
            net,
            stack,
            batch_size=batch_size,
            augment=augment,
            tile_overlap=tile_overlap,
            bsize=bsize,
            return_stats=return_stats,
            single_tile_if_fit=True,
        )
        if return_stats:
            return result  # type: ignore[return-value]
        return result

    packed, mapping = pack_planes_to_stripes(stack, layout)

    # Always route through run_net for consistent preprocessing/tiling
    result = run_net_with_stats(
        net,
        packed,
        batch_size=batch_size,
        augment=augment,
        tile_overlap=tile_overlap,
        bsize=bsize,
        return_stats=return_stats,
        single_tile_if_fit=True,
    )

    if return_stats:
        y_packed, styles, stats = result
    else:
        y_packed, styles = result  # type: ignore[misc]

    yf = unpack_stripes_to_planes(y_packed, mapping, Lz=Lz, Ly=Ly)

    # Do not try to reconstruct per-stripe styles; keep what run_net returned.
    if return_stats:
        return yf, styles, stats  # type: ignore[return-value]
    return yf, styles
