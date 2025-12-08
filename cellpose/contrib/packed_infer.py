from __future__ import annotations

import logging

import numpy as np

from cellpose import models, transforms
from cellpose.contrib.cellposetrt import CellposeModelTRT as _CellposeModelTRT
from cellpose.core import _compute_variance_weights, _smooth_flows_2d, run_net
from cellpose.train import _PACK_STRIPE_BORDER
from cellpose.unet import CellposeUNetModel

from .pack_utils import (
    compute_stripe_layout,
    pack_planes_to_stripes,
    unpack_stripes_to_planes,
)

logger = logging.getLogger(__name__)


def _run_3d_with_packing(
    net,
    imgs: np.ndarray,
    *,
    batch_size: int,
    augment: bool,
    tile_overlap: float,
    bsize: int,
    pack_border: int,
    plane_weights: np.ndarray | None,
    return_raw_3d: bool = False,
    flow2D_smooth: float = 0.0,
    use_variance_fusion: bool = False,
    variance_alpha_flow: float = 0.5,
    variance_alpha_cellprob: float = 1e-5,
):
    # No **kwargs - all parameters must be explicitly defined
    sstr = ["YX", "ZY", "ZX"]
    orient_keys = ["xy", "xz", "yz"]  # align with core.run_3D return_raw contract
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
            raise ValueError(
                "plane_weights must contain three elements for (XY, YZ, ZX)."
            )
        if np.any(weights < 0):
            raise ValueError("plane_weights entries must be non-negative.")
        if np.all(weights == 0):
            raise ValueError("At least one plane weight must be positive.")

    # Initialize weight accumulators - arrays when using variance fusion, scalars otherwise
    if use_variance_fusion:
        flow_weight_totals = [np.zeros(shape, dtype=np.float32) for _ in range(3)]
        cellprob_weight_total = np.zeros(shape, dtype=np.float32)
    else:
        flow_weight_totals = np.zeros(3, dtype=np.float32)
        cellprob_weight_total = 0.0
    raw_outputs = {}

    for p in range(3):
        weight = float(weights[p])
        xsl = imgs.transpose(pm[p])  # [Z', Y', X', C]
        Lzp, Lyp, Lxp = xsl.shape[:3]

        # Only pack orthogonal planes; keep XY (YX orientation) on the baseline path
        use_pack = sstr[p] != "YX"
        layout = None
        if use_pack:
            layout = compute_stripe_layout(
                Lyp,
                bsize=bsize,
                border=pack_border,
            )
        if use_pack and layout is not None:
            logger.info(
                "PackedCellposeModel packing orientation=%s Lz=%d Ly=%d Lx=%d K=%d guard=%d border=%d bsize=%d slot=%d",
                sstr[p],
                Lzp,
                Lyp,
                Lxp,
                layout.K,
                layout.guard,
                pack_border,
                bsize,
                layout.slot_height,
            )
            packed, mapping = pack_planes_to_stripes(xsl, layout)
            y_packed, styles = run_net(
                net,
                packed,
                batch_size=batch_size,
                augment=augment,
                tile_overlap=tile_overlap,
                bsize=bsize,
                single_tile_if_fit=True,
            )
            y = unpack_stripes_to_planes(y_packed, mapping, Lz=Lzp, Ly=Lyp)
        else:
            y, styles = run_net(
                net,
                xsl,
                batch_size=batch_size,
                augment=augment,
                tile_overlap=tile_overlap,
                bsize=bsize,
            )

        # Apply 2D pre-smoothing before aggregation (u-Segment3D)
        if flow2D_smooth > 0:
            y = _smooth_flows_2d(y, sigma=flow2D_smooth, use_gpu=True)

        if return_raw_3d:
            raw_outputs[orient_keys[p]] = {"y": y, "style": styles}
            continue

        styles_last = styles

        if use_variance_fusion:
            # Variance-weighted fusion: weight by inverse local variance
            cellprob_3d = y[..., -1].transpose(ipm[p])
            w_cellprob = _compute_variance_weights(
                cellprob_3d, alpha=variance_alpha_cellprob, use_gpu=True)
            yf[..., -1] += weight * w_cellprob * cellprob_3d
            cellprob_weight_total += weight * w_cellprob

            for j in range(2):
                axis_idx = cp[p][j]
                flow_3d = y[..., cpy[p][j]].transpose(ipm[p])
                w_flow = _compute_variance_weights(
                    flow_3d, alpha=variance_alpha_flow, use_gpu=True)
                yf[..., axis_idx] += weight * w_flow * flow_3d
                flow_weight_totals[axis_idx] += weight * w_flow
        else:
            # Original simple weighted accumulation
            yf[..., -1] += weight * y[..., -1].transpose(ipm[p])
            cellprob_weight_total += weight
            for j in range(2):
                axis_idx = cp[p][j]
                yf[..., axis_idx] += weight * y[..., cpy[p][j]].transpose(ipm[p])
                flow_weight_totals[axis_idx] += weight

    if return_raw_3d:
        return raw_outputs

    # Normalize by accumulated weights
    if use_variance_fusion:
        for axis_idx in range(3):
            mask = flow_weight_totals[axis_idx] > 0
            yf[..., axis_idx][mask] /= flow_weight_totals[axis_idx][mask]
        mask = cellprob_weight_total > 0
        yf[..., -1][mask] /= cellprob_weight_total[mask]
    else:
        for axis_idx in range(3):
            if flow_weight_totals[axis_idx] > 0:
                yf[..., axis_idx] /= flow_weight_totals[axis_idx]
        if cellprob_weight_total > 0:
            yf[..., -1] /= cellprob_weight_total

    return yf, styles_last


class Packed3DMixin:
    """Shared helpers for Cellpose models that support 3D packing."""

    _pack_enabled: bool
    _pack_border: int

    def _should_use_packing(self, do_3D: bool, anisotropy: float | int | None) -> bool:
        # Allow packing regardless of anisotropy; handle resize explicitly in _run_packed_3d
        if not do_3D or not getattr(self, "_pack_enabled", False):
            return False
        return True

    def _run_packed_3d(
        self,
        net,
        x: np.ndarray,
        *,
        batch_size: int,
        augment: bool,
        tile_overlap: float,
        bsize: int,
        anisotropy: float | int | None = 1.0,
        return_raw_3d: bool = False,
        **kwargs,
    ):
        # Mirror baseline behavior: if anisotropy is provided and != 1.0, resize Y accordingly
        if isinstance(anisotropy, (float, int)) and anisotropy not in (None, 1.0):
            Lz, Ly, Lx = x.shape[:-1]
            x = transforms.resize_image(
                x.transpose(1, 0, 2, 3),
                Ly=int(Lz * float(anisotropy)),
                Lx=int(Lx),
            ).transpose(1, 0, 2, 3)
        # _run_3d_with_packing has explicit params - will error on unknown kwargs
        res = _run_3d_with_packing(
            net,
            x,
            batch_size=batch_size,
            augment=augment,
            tile_overlap=tile_overlap,
            bsize=bsize,
            pack_border=getattr(self, "_pack_border", _PACK_STRIPE_BORDER),
            return_raw_3d=return_raw_3d,
            **kwargs,
        )
        if return_raw_3d:
            return res

        yf, styles = res
        cellprob = yf[..., -1]
        dP = yf[..., :-1].transpose((3, 0, 1, 2))
        return dP, cellprob, styles


class PackedCellposeModel(Packed3DMixin, models.CellposeModel):
    """Packed ortho inference in 3D mode; 2D path unchanged."""

    def __init__(
        self,
        *args,
        pack_z_stripes: bool = True,
        pack_border: int = _PACK_STRIPE_BORDER,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._pack_enabled = bool(pack_z_stripes)
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
        plane_weights=None,
        return_raw_3d: bool = False,
        **kwargs,
    ):
        if self._should_use_packing(do_3D, anisotropy):
            return self._run_packed_3d(
                self.net,
                x,
                batch_size=batch_size,
                augment=augment,
                tile_overlap=tile_overlap,
                bsize=bsize,
                anisotropy=anisotropy,
                plane_weights=plane_weights,
                return_raw_3d=return_raw_3d,
                **kwargs,
            )

        return super()._run_net(
            x,
            augment=augment,
            batch_size=batch_size,
            tile_overlap=tile_overlap,
            bsize=bsize,
            anisotropy=anisotropy,
            do_3D=do_3D,
            plane_weights=plane_weights,
            return_raw_3d=return_raw_3d,
            **kwargs,
        )


class PackedCellposeModelTRT(Packed3DMixin, _CellposeModelTRT):
    def __init__(
        self,
        *args,
        pack_z_stripes: bool = True,
        pack_border: int = _PACK_STRIPE_BORDER,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._pack_enabled = bool(pack_z_stripes)
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
        plane_weights=None,
        return_raw_3d: bool = False,
        **kwargs,
    ):
        if self._should_use_packing(do_3D, anisotropy):
            return self._run_packed_3d(
                self.net,
                x,
                batch_size=batch_size,
                augment=augment,
                tile_overlap=tile_overlap,
                bsize=bsize,
                anisotropy=anisotropy,
                plane_weights=plane_weights,
                return_raw_3d=return_raw_3d,
                **kwargs,
            )

        return super()._run_net(
            x,
            augment=augment,
            batch_size=batch_size,
            tile_overlap=tile_overlap,
            bsize=bsize,
            anisotropy=anisotropy,
            do_3D=do_3D,
            plane_weights=plane_weights,
            return_raw_3d=return_raw_3d,
            **kwargs,
        )


class PackedCellposeUNetModel(Packed3DMixin, CellposeUNetModel):
    """UNet model with packed 3D ortho paths; 2D path unchanged."""

    def __init__(
        self,
        *args,
        pack_z_stripes: bool = True,
        pack_border: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._pack_enabled = bool(pack_z_stripes)
        self._pack_border = int(pack_border)

    def _run_net(
        self,
        x,
        rescale=1.0,
        resample=True,
        augment=False,
        batch_size=8,
        tile_overlap=0.1,
        bsize=224,
        anisotropy=1.0,
        do_3D=False,
        plane_weights=None,
        return_raw_3d=False,
        **kwargs,
    ):
        if self._should_use_packing(do_3D, anisotropy):
            return self._run_packed_3d(
                self.net,
                x,
                batch_size=batch_size,
                augment=augment,
                tile_overlap=tile_overlap,
                bsize=bsize,
                anisotropy=anisotropy,
                plane_weights=plane_weights,
                return_raw_3d=return_raw_3d,
                **kwargs,
            )
        return super()._run_net(
            x,
            rescale=rescale,
            resample=resample,
            augment=augment,
            batch_size=batch_size,
            tile_overlap=tile_overlap,
            bsize=bsize,
            anisotropy=anisotropy,
            do_3D=do_3D,
            plane_weights=plane_weights,
            return_raw_3d=return_raw_3d,
        )


class PackedCellposeUNetModelTRT(Packed3DMixin, CellposeUNetModel):
    """UNet model using TRTEngineModule plus packed 3D ortho paths."""

    def __init__(
        self,
        *args,
        pretrained_model: str,
        device=None,
        pack_z_stripes: bool = True,
        pack_border: int = _PACK_STRIPE_BORDER,
        **kwargs,
    ):
        super().__init__(*args, device=device, **kwargs)
        from cellpose.contrib.cellposetrt import TRTEngineModule

        dev = device if device is not None else self.device
        self.net = TRTEngineModule(pretrained_model, device=dev)
        self._pack_enabled = bool(pack_z_stripes)
        self._pack_border = int(pack_border)

    def _run_net(
        self,
        x,
        rescale=1.0,
        resample=True,
        augment=False,
        batch_size=8,
        tile_overlap=0.1,
        bsize=224,
        anisotropy=1.0,
        do_3D=False,
        plane_weights=None,
        return_raw_3d=False,
        **kwargs,
    ):
        if self._should_use_packing(do_3D, anisotropy):
            return self._run_packed_3d(
                self.net,
                x,
                batch_size=batch_size,
                augment=augment,
                tile_overlap=tile_overlap,
                bsize=bsize,
                anisotropy=anisotropy,
                plane_weights=plane_weights,
                return_raw_3d=return_raw_3d,
                **kwargs,
            )
        return super()._run_net(
            x,
            rescale=rescale,
            resample=resample,
            augment=augment,
            batch_size=batch_size,
            tile_overlap=tile_overlap,
            bsize=bsize,
            anisotropy=anisotropy,
            do_3D=do_3D,
            plane_weights=plane_weights,
            return_raw_3d=return_raw_3d,
        )


class CellposeUNetModelTRT(Packed3DMixin, CellposeUNetModel):
    """UNet model using TRTEngineModule plus packed 3D ortho paths."""

    def __init__(
        self,
        *args,
        pretrained_model: str,
        device=None,
        pack_z_stripes: bool = True,
        pack_border: int = _PACK_STRIPE_BORDER,
        **kwargs,
    ):
        super().__init__(*args, device=device, **kwargs)
        from cellpose.contrib.cellposetrt import TRTEngineModule

        dev = device if device is not None else self.device
        self.net = TRTEngineModule(pretrained_model, device=dev)
        self._pack_enabled = bool(pack_z_stripes)
        self._pack_border = int(pack_border)

    def _run_net(
        self,
        x,
        rescale=1.0,
        resample=True,
        augment=False,
        batch_size=8,
        tile_overlap=0.1,
        bsize=224,
        anisotropy=1.0,
        do_3D=False,
        plane_weights=None,
        return_raw_3d=False,
        **kwargs,
    ):
        return super()._run_net(
            x,
            rescale=rescale,
            resample=resample,
            augment=augment,
            batch_size=batch_size,
            tile_overlap=tile_overlap,
            bsize=bsize,
            anisotropy=anisotropy,
            do_3D=do_3D,
            plane_weights=plane_weights,
            return_raw_3d=return_raw_3d,
            **kwargs,
        )
