from __future__ import annotations

import numpy as np

from cellpose import models
from cellpose.core import run_net

from .pack_utils import (
    StripeLayout,
    compute_stripe_layout,
    pack_planes_to_stripes,
    unpack_stripes_to_planes,
)


def _run_3d_with_packing(
    net,
    imgs: np.ndarray,
    *,
    batch_size: int,
    augment: bool,
    tile_overlap: float,
    bsize: int,
    pack_k: int,
    guard: int,
    pack_border: int,
    plane_weights: np.ndarray | None = None,
):
    sstr = ["YX", "ZY", "ZX"]
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    ipm = [(0, 1, 2), (1, 0, 2), (1, 2, 0)]
    cp = [(1, 2), (0, 2), (0, 1)]
    cpy = [(0, 1), (0, 1), (0, 1)]
    shape = imgs.shape[:-1]
    yf = np.zeros((*shape, 4), "float32")
    styles_last = None
    if plane_weights is None:
        weights = np.ones(3, dtype=np.float32)
    else:
        weights = np.asarray(plane_weights, dtype=np.float32)
        if weights.shape != (3,):
            raise ValueError("plane_weights must contain three elements for (XY, YZ, ZX).")
        if np.any(weights < 0):
            raise ValueError("plane_weights entries must be non-negative.")
        if np.all(weights == 0):
            raise ValueError("At least one plane weight must be positive.")
    flow_weight_totals = np.zeros(3, dtype=np.float32)
    cellprob_weight_total = 0.0

    for p in range(3):
        weight = float(weights[p])
        xsl = imgs.transpose(pm[p])  # [Z', Y', X', C]
        Lzp, Lyp, Lxp = xsl.shape[:3]

        # Use maximum guard that fits for the realized Ly to mirror training
        if pack_k >= 2:
            slot_height = int(Lyp) + 2 * int(pack_border)
            guard_eff = max(0, (int(bsize) - int(pack_k) * slot_height) // (int(pack_k) - 1))
        else:
            guard_eff = int(guard)
        layout = compute_stripe_layout(
            Lyp,
            bsize=bsize,
            pack_k=pack_k,
            guard=guard_eff,
            border=pack_border,
        )
        if layout is not None:
            packed, mapping = pack_planes_to_stripes(xsl, layout)
            y_packed, styles = run_net(
                net, packed, batch_size=batch_size, augment=augment, tile_overlap=tile_overlap, bsize=bsize
            )
            y = unpack_stripes_to_planes(y_packed, mapping, Lz=Lzp, Ly=Lyp)
        else:
            y, styles = run_net(
                net, xsl, batch_size=batch_size, augment=augment, tile_overlap=tile_overlap, bsize=bsize
            )

        styles_last = styles
        yf[..., -1] += weight * y[..., -1].transpose(ipm[p])
        cellprob_weight_total += weight
        for j in range(2):
            axis_idx = cp[p][j]
            yf[..., axis_idx] += weight * y[..., cpy[p][j]].transpose(ipm[p])
            flow_weight_totals[axis_idx] += weight

    for axis_idx in range(3):
        if flow_weight_totals[axis_idx] > 0:
            yf[..., axis_idx] /= flow_weight_totals[axis_idx]
    if cellprob_weight_total > 0:
        yf[..., -1] /= cellprob_weight_total

    return yf, styles_last


class PackedCellposeModel(models.CellposeModel):
    """Packed ortho inference in 3D mode; 2D path unchanged."""

    def __init__(
        self,
        *args,
        pack_z_stripes: bool = True,
        pack_k: int = 3,
        pack_guard: int = 16,
        pack_min_Ly: int = 1,
        pack_border: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._pack_enabled = bool(pack_z_stripes)
        self._pack_k = int(pack_k)
        self._pack_guard = int(pack_guard)
        self._pack_min_Ly = int(pack_min_Ly)
        self._pack_border = int(pack_border)

    def _run_net(
        self,
        x: np.ndarray,
        augment: bool = False,
        batch_size: int = 8,
        tile_overlap: float = 0.1,
        bsize: int = 256,
        anisotropy: float = 1.0,
        do_3D: bool = False,
    ):
        if do_3D and self._pack_enabled:
            if isinstance(anisotropy, (float, int)) and anisotropy not in (None, 1.0):
                return super()._run_net(
                    x,
                    augment=augment,
                    batch_size=batch_size,
                    tile_overlap=tile_overlap,
                    bsize=bsize,
                    anisotropy=anisotropy,
                    do_3D=do_3D,
                )

            yf, styles = _run_3d_with_packing(
                self.net,
                x,
                batch_size=batch_size,
                augment=augment,
                tile_overlap=tile_overlap,
                bsize=bsize,
                pack_k=self._pack_k,
                guard=self._pack_guard,
                pack_border=self._pack_border,
                plane_weights=None,
            )
            cellprob = yf[..., -1]
            dP = yf[..., :-1].transpose((3, 0, 1, 2))
            return dP, cellprob, styles

        return super()._run_net(
            x,
            augment=augment,
            batch_size=batch_size,
            tile_overlap=tile_overlap,
            bsize=bsize,
            anisotropy=anisotropy,
            do_3D=do_3D,
        )


try:
    from cellpose.contrib.cellposetrt import CellposeModelTRT as _CellposeModelTRT

    class PackedCellposeModelTRT(_CellposeModelTRT):
        def __init__(
            self,
            *args,
            pack_z_stripes: bool = True,
            pack_k: int = 3,
            pack_guard: int = 16,
            pack_min_Ly: int = 1,
            pack_border: int = 5,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self._pack_enabled = bool(pack_z_stripes)
            self._pack_k = int(pack_k)
            self._pack_guard = int(pack_guard)
            self._pack_min_Ly = int(pack_min_Ly)
            self._pack_border = int(pack_border)

        def _run_net(
            self,
            x: np.ndarray,
            augment: bool = False,
            batch_size: int = 8,
            tile_overlap: float = 0.1,
            bsize: int = 256,
            anisotropy: float = 1.0,
            do_3D: bool = False,
        ):
            if do_3D and self._pack_enabled:
                if isinstance(anisotropy, (float, int)) and anisotropy not in (None, 1.0):
                    return super()._run_net(
                        x,
                        augment=augment,
                        batch_size=batch_size,
                        tile_overlap=tile_overlap,
                        bsize=bsize,
                        anisotropy=anisotropy,
                        do_3D=do_3D,
                    )
                yf, styles = _run_3d_with_packing(
                    self.net,
                    x,
                    batch_size=batch_size,
                    augment=augment,
                    tile_overlap=tile_overlap,
                    bsize=bsize,
                    pack_k=self._pack_k,
                    guard=self._pack_guard,
                    pack_border=self._pack_border,
                    plane_weights=None,
                )
                cellprob = yf[..., -1]
                dP = yf[..., :-1].transpose((3, 0, 1, 2))
                return dP, cellprob, styles

            return super()._run_net(
                x,
                augment=augment,
                batch_size=batch_size,
                tile_overlap=tile_overlap,
                bsize=bsize,
                anisotropy=anisotropy,
                do_3D=do_3D,
            )
except Exception:
    pass
