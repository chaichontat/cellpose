"""
Legacy Cellpose UNet model ported from Cellpose v3.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import trange

from . import dynamics, plot, transforms, utils
from .core import assign_device
from .core import run_3D as run_3D_core
from .unet_net import CPnet

models_logger = logging.getLogger(__name__)

_MODEL_URL = "https://www.cellpose.org/models"
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = Path.home().joinpath(".cellpose", "models")
MODEL_DIR = Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT

# These are the UNet-backed model names supported in Cellpose v3.
MODEL_NAMES = [
    "cyto3", "nuclei", "cyto2_cp3", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3",
    "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3", "cyto2", "cyto", "CPx",
    "transformer_cp3", "neurips_cellpose_default", "neurips_cellpose_transformer",
    "neurips_grayscale_cyto2",
    "CP", "CPx", "TN1", "TN2", "TN3", "LC1", "LC2", "LC3", "LC4"
]

MODEL_LIST_PATH = os.fspath(MODEL_DIR.joinpath("gui_models.txt"))


def model_path(model_type: str, model_index: int = 0) -> str:
    torch_str = "torch"
    if model_type in {"cyto", "cyto2", "nuclei"}:
        basename = f"{model_type}{torch_str}_{model_index}"
    else:
        basename = model_type
    return cache_model_path(basename)


def size_model_path(model_type: str) -> str:
    torch_str = "torch"
    if model_type in {"cyto", "nuclei", "cyto2", "cyto3"}:
        if model_type == "cyto3":
            basename = f"size_{model_type}.npy"
        else:
            basename = f"size_{model_type}{torch_str}_0.npy"
        return cache_model_path(basename)
    if os.path.exists(model_type) and os.path.exists(model_type + "_size.npy"):
        return model_type + "_size.npy"
    raise FileNotFoundError(f"size model not found ({model_type + '_size.npy'})")


def cache_model_path(basename: str) -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_MODEL_URL}/{basename}"
    cached_file = os.fspath(MODEL_DIR.joinpath(basename))
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "%s" to %s', url, cached_file)
        utils.download_url_to_file(url, cached_file, progress=True)
    return cached_file


def get_user_models() -> List[str]:
    model_strings: List[str] = []
    if os.path.exists(MODEL_LIST_PATH):
        with open(MODEL_LIST_PATH, "r") as textfile:
            lines = [line.rstrip() for line in textfile]
            if lines:
                model_strings.extend(lines)
    return model_strings


def check_mkl(use_torch: bool = True) -> bool:
    try:
        mkl_enabled = torch.backends.mkldnn.is_available()
    except AttributeError:
        mkl_enabled = False
    if not mkl_enabled:
        models_logger.info(
            "WARNING: MKL version on torch not working/installed - CPU version will be slightly slower."
        )
        models_logger.info("see https://pytorch.org/docs/stable/backends.html?highlight=mkl")
    return mkl_enabled


def _to_device(x: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return torch.from_numpy(x).to(device, dtype=torch.float32)
    return x


def _from_device(X: torch.Tensor) -> np.ndarray:
    tensor = X.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return tensor.numpy()


def _forward(net: torch.nn.Module, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = _to_device(x, net.device)
    net.eval()
    with torch.no_grad():
        y, style = net(X)[:2]
    del X
    return _from_device(y), _from_device(style)


def run_net(net: CPnet, imgi: np.ndarray, batch_size: int = 8, augment: bool = False,
            tile_overlap: float = 0.1, bsize: int = 224, rsz: Optional[Union[float, Sequence[float]]] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
    nout = net.nout if hasattr(net, "nout") else 3
    Lz, Ly0, Lx0, nchan = imgi.shape
    if rsz is not None:
        if not isinstance(rsz, (list, tuple, np.ndarray)):
            rsz = [rsz, rsz]
        Lyr, Lxr = int(Ly0 * rsz[0]), int(Lx0 * rsz[1])
    else:
        Lyr, Lxr = Ly0, Lx0

    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(Lyr, Lxr, min_size=(bsize, bsize))
    Ly, Lx = Lyr + ypad1 + ypad2, Lxr + xpad1 + xpad2
    pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])

    if augment:
        ny = max(2, int(np.ceil(2.0 * Ly / bsize)))
        nx = max(2, int(np.ceil(2.0 * Lx / bsize)))
        ly, lx = bsize, bsize
    else:
        ny = 1 if Ly <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Lx / bsize))
        ly, lx = min(bsize, Ly), min(bsize, Lx)

    yf = np.zeros((Lz, nout, Ly, Lx), "float32")
    styles = np.zeros((Lz, 256), "float32")
    ntiles = ny * nx
    nimgs = max(1, batch_size // ntiles)
    niter = int(np.ceil(Lz / nimgs))
    models_logger.info(
        "tiling decision: Lz=%d, Ly=%d, Lx=%d, bsize=%d, augment=%s, tile_overlap=%.3f -> "
        "ny=%d, nx=%d, ntiles=%d, nimgs_per_batch=%d, niter=%d",
        Lz,
        Ly,
        Lx,
        bsize,
        augment,
        tile_overlap,
        ny,
        nx,
        ntiles,
        nimgs,
        niter,
    )
    ziterator = (trange(niter, file=utils.TqdmToLogger(models_logger, level=logging.INFO), mininterval=30)
                 if niter > 10 or Lz > 1 else range(niter))

    for k in ziterator:
        inds = np.arange(k * nimgs, min(Lz, (k + 1) * nimgs))
        IMGa = np.zeros((ntiles * len(inds), nchan, ly, lx), "float32")
        for i, b in enumerate(inds):
            imgb = transforms.resize_image(imgi[b], rsz=rsz) if rsz is not None else imgi[b].copy()
            imgb = np.pad(imgb.transpose(2, 0, 1), pads, mode="constant")
            IMG, ysub, xsub, Lyt, Lxt = transforms.make_tiles(
                imgb, bsize=bsize, augment=augment, tile_overlap=tile_overlap)
            IMGa[i * ntiles:(i + 1) * ntiles] = np.reshape(IMG, (ny * nx, nchan, ly, lx))

        ya = np.zeros((IMGa.shape[0], nout, ly, lx), "float32")
        stylea = np.zeros((IMGa.shape[0], 256), "float32")
        for j in range(0, IMGa.shape[0], batch_size):
            bslc = slice(j, min(j + batch_size, IMGa.shape[0]))
            ya[bslc], stylea[bslc] = _forward(net, IMGa[bslc])
        for i, b in enumerate(inds):
            y = ya[i * ntiles:(i + 1) * ntiles]
            if augment:
                y = np.reshape(y, (ny, nx, nout, ly, lx))
                y = transforms.unaugment_tiles(y)
                y = np.reshape(y, (-1, nout, ly, lx))
            yfi = transforms.average_tiles(y, ysub, xsub, Lyt, Lxt)
            yf[b] = yfi[:, :imgb.shape[-2], :imgb.shape[-1]]
            stylei = stylea[i * ntiles:(i + 1) * ntiles].sum(axis=0)
            norm = (stylei**2).sum()**0.5
            if norm > 0:
                stylei /= norm
            styles[b] = stylei

    yf = yf[:, :, ypad1:Ly - ypad2, xpad1:Lx - xpad2]
    yf = yf.transpose(0, 2, 3, 1)
    return yf, styles


def run_3D(net: CPnet, imgs: np.ndarray, batch_size: int = 8, augment: bool = False,
           tile_overlap: float = 0.1, bsize: int = 224, net_ortho: Optional[CPnet] = None,
           progress=None) -> Tuple[np.ndarray, np.ndarray]:
    sstr = ["YX", "ZY", "ZX"]
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    ipm = [(0, 1, 2), (1, 0, 2), (1, 2, 0)]
    cp = [(1, 2), (0, 2), (0, 1)]
    cpy = [(0, 1), (0, 1), (0, 1)]
    shape = imgs.shape[:-1]
    yf = np.zeros((*shape, 4), "float32")
    for p in range(3):
        xsl = imgs.transpose(pm[p])
        models_logger.info("running %s: %d planes of size (%d, %d)",
                           sstr[p], shape[pm[p][0]], shape[pm[p][1]], shape[pm[p][2]])
        y, style = run_net(net if p == 0 or net_ortho is None else net_ortho,
                           xsl, batch_size=batch_size, augment=augment,
                           bsize=bsize, tile_overlap=tile_overlap, rsz=None)
        yf[..., -1] += y[..., -1].transpose(ipm[p])
        for j in range(2):
            yf[..., cp[p][j]] += y[..., cpy[p][j]].transpose(ipm[p])
        if progress is not None:
            progress.setValue(25 + 15 * p)
    yf[..., :-1] /= 3.0
    yf[..., -1] /= 3.0
    return yf, style


def convert_image_legacy(x: np.ndarray, channels: Optional[Sequence[int]],
                         channel_axis: Optional[int] = None, z_axis: Optional[int] = None,
                         do_3D: bool = False, nchan: int = 2) -> np.ndarray:
    ndim = x.ndim
    if torch.is_tensor(x):
        transforms.transforms_logger.warning("torch array used as input, converting to numpy")
        x = x.cpu().numpy()

    if x.ndim > 3:
        to_squeeze = np.array([int(isq) for isq, s in enumerate(x.shape) if s == 1])
        if len(to_squeeze) > 0:
            channel_axis = transforms.update_axis(
                channel_axis, to_squeeze,
                x.ndim) if channel_axis is not None else None
            z_axis = transforms.update_axis(z_axis, to_squeeze,
                                 x.ndim) if z_axis is not None else None
            x = x.squeeze()

    if z_axis is not None and x.ndim > 2 and z_axis != 0:
        x = transforms.move_axis(x, m_axis=z_axis, first=True)
        if channel_axis is not None:
            channel_axis += 1
        z_axis = 0
    elif z_axis is None and x.ndim > 2 and channels is not None and min(x.shape) > 5:
        min_dim = min(x.shape)
        if min_dim != channel_axis:
            z_axis = (x.shape).index(min_dim)
            if z_axis != 0:
                x = transforms.move_axis(x, m_axis=z_axis, first=True)
                if channel_axis is not None:
                    channel_axis += 1
            transforms.transforms_logger.warning(
                "z_axis not specified, assuming it is dim %d", z_axis)
            transforms.transforms_logger.warning(
                "if this is actually the channel_axis, use 'model.eval(channel_axis=%d, ...)'", z_axis)
            z_axis = 0

    if z_axis is not None and x.ndim == 3:
        x = x[..., np.newaxis]

    if channel_axis is not None and x.ndim > 2:
        x = transforms.move_axis(x, m_axis=channel_axis, first=False)
    elif x.ndim == 2:
        x = x[:, :, np.newaxis]

    if do_3D:
        if ndim < 3:
            raise ValueError("ERROR: cannot process 2D images in 3D mode")
        if x.ndim < 4:
            x = x[..., np.newaxis]

    if channel_axis is None:
        x = transforms.move_min_dim(x)

    if x.ndim > 3:
        transforms.transforms_logger.info(
            "multi-stack tiff read in as having %d planes %d channels",
            x.shape[0], x.shape[-1])

    x = x.astype("float32")

    if channels is not None:
        channels = channels[0] if len(channels) == 1 else channels
        if len(channels) < 2:
            transforms.transforms_logger.critical("ERROR: two channels not specified")
            raise ValueError("ERROR: two channels not specified")
        if x.shape[-1] < 2:
            transforms.transforms_logger.critical("ERROR: image has < 2 channels, supply channels")
            raise ValueError("ERROR: image has < 2 channels, supply channels")
        chan = channels
        x = np.stack([x[..., chan[0] - 1] if chan[0] > 0 else x[..., 0] * 0 + 1,
                      x[..., chan[1] - 1] if chan[1] > 0 else x[..., 0] * 0],
                     axis=-1)
        if chan[0] < 1 and chan[1] < 1:
            x[..., 0] = 1
        if chan[1] == 0:
            x[..., 1] = 0
        if x.ndim > 4:
            raise ValueError("ERROR: image has > 4 dimensions, cannot process")
        if x.shape[-1] > nchan:
            x = x[..., :nchan]
        elif x.shape[-1] < nchan:
            x2 = np.zeros((*x.shape[:-1], nchan), dtype=x.dtype)
            x2[..., :x.shape[-1]] = x
            x = x2
    else:
        if x.ndim == 2:
            x = np.stack((x, np.zeros_like(x)), axis=-1)
        elif x.shape[-1] < nchan:
            x2 = np.zeros((*x.shape[:-1], nchan), dtype=x.dtype)
            x2[..., :x.shape[-1]] = x
            x = x2
    return x


def get_model_params(pretrained_model, model_type, pretrained_model_ortho, default_model="cyto3"):
    builtin = False
    use_default = False
    diam_mean = None
    model_strings = get_user_models()
    all_models = MODEL_NAMES.copy()
    all_models.extend(model_strings)

    if (pretrained_model and not Path(pretrained_model).exists() and
            np.any([pretrained_model == s for s in all_models])):
        model_type = pretrained_model

    if model_type is not None and np.any([model_type == s for s in all_models]):
        if np.any([model_type == s for s in MODEL_NAMES]):
            builtin = True
        models_logger.info(">> %s << model set to be used", model_type)
        if model_type == "nuclei":
            diam_mean = 17.
        pretrained_model = model_path(model_type)
    elif model_type is not None:
        if Path(model_type).exists():
            pretrained_model = model_type
        else:
            models_logger.warning("model_type does not exist, using default model")
            use_default = True
    else:
        if pretrained_model and not Path(pretrained_model).exists():
            models_logger.warning("pretrained_model path does not exist, using default model")
            use_default = True
        elif pretrained_model and pretrained_model.endswith("nucleitorch_0"):
            builtin = True
            diam_mean = 17.

    if pretrained_model_ortho:
        if pretrained_model_ortho in all_models:
            pretrained_model_ortho = model_path(pretrained_model_ortho)
        elif Path(pretrained_model_ortho).exists():
            pass
        else:
            pretrained_model_ortho = None

    pretrained_model = model_path(default_model) if use_default else pretrained_model
    builtin = True if use_default else builtin
    return pretrained_model, diam_mean, builtin, pretrained_model_ortho


class CellposeUNetModel:
    def __init__(self, gpu: bool = False, pretrained_model: Union[bool, str] = False,
                 model_type: Optional[str] = None, mkldnn: bool = True, diam_mean: float = 30.,
                 device: Optional[torch.device] = None, nchan: int = 2,
                 pretrained_model_ortho: Optional[str] = None, backbone: str = "default"):
        self.diam_mean = diam_mean
        default_model = "cyto3" if backbone == "default" else "transformer_cp3"
        pretrained_model, diam_mean, builtin, pretrained_model_ortho = get_model_params(
            pretrained_model, model_type, pretrained_model_ortho, default_model)
        self.diam_mean = diam_mean if diam_mean is not None else self.diam_mean
        self.builtin = builtin

        self.mkldnn = None
        self.device = assign_device(gpu=gpu)[0] if device is None else device
        if torch.cuda.is_available():
            device_gpu = self.device.type == "cuda"
        elif torch.backends.mps.is_available():
            device_gpu = self.device.type == "mps"
        else:
            device_gpu = False
        self.gpu = device_gpu
        if not self.gpu:
            self.mkldnn = check_mkl(True) if mkldnn else False

        self.nchan = nchan
        self.nclasses = 3
        nbase = [32, 64, 128, 256]
        self.nbase = [nchan, *nbase]
        self.pretrained_model = pretrained_model
        if backbone != "default":
            raise ValueError("CellposeUNetModel only supports the legacy UNet backbone")
        self.net = CPnet(self.nbase, self.nclasses, sz=3, mkldnn=self.mkldnn,
                         max_pool=True, diam_mean=self.diam_mean).to(self.device)

        self.net_ortho = None
        if pretrained_model_ortho is not None:
            self.net_ortho = CPnet(self.nbase, self.nclasses, sz=3,
                                   mkldnn=self.mkldnn, max_pool=True).to(self.device)

        if isinstance(self.pretrained_model, list):
            if len(self.pretrained_model) == 2:
                self.net.load_model(self.pretrained_model[0], device=self.device)
                self.net_ortho = CPnet(self.nbase, self.nclasses, sz=3,
                                       mkldnn=self.mkldnn, max_pool=True).to(self.device)
                self.net_ortho.load_model(self.pretrained_model[1], device=self.device)
            else:
                models_logger.warning("pretrained_model list must be length 2")
                self.pretrained_model = self.pretrained_model[0]

        if isinstance(self.pretrained_model, str):
            models_logger.info(">>>> loading model %s", self.pretrained_model)
            self.net.load_model(self.pretrained_model, device=self.device)
        elif self.pretrained_model:
            models_logger.warning("Unsupported pretrained_model type %s", type(self.pretrained_model))

        if self.net_ortho is not None and pretrained_model_ortho:
            models_logger.info(">>>> loading ortho model %s", pretrained_model_ortho)
            self.net_ortho.load_model(pretrained_model_ortho, device=self.device)

        if self.pretrained_model:
            if not self.builtin:
                self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]
        else:
            self.diam_labels = self.diam_mean
        if pretrained_model_ortho and self.net_ortho is None and isinstance(pretrained_model_ortho, str):
            models_logger.warning("Ortho model requested but could not be loaded.")

        self.net_type = "cellpose_unet"

    def eval(self, x, batch_size: int = 8, resample: bool = True,
             channels: Optional[Sequence[int]] = None, channel_axis: Optional[int] = None,
             z_axis: Optional[int] = None, normalize: Union[bool, dict] = True,
             invert: bool = False, rescale: Optional[Union[float, Sequence[float]]] = None,
             diameter: Optional[Union[float, Sequence[float]]] = None,
             flow_threshold: float = 0.4, cellprob_threshold: float = 0.0, do_3D: bool = False,
             anisotropy: Optional[float] = None, stitch_threshold: float = 0.0,
             min_size: int = 15, max_size_fraction: float = 0.4, niter: Optional[int] = None,
             augment: bool = False, tile_overlap: float = 0.1, bsize: int = 224,
             interp: bool = True, compute_masks: bool = True, progress=None,
             flow3D_smooth: int = 0, ortho_weights: Optional[Sequence[float]] = None):
        if isinstance(x, list) or x.squeeze().ndim == 5:
            self.timing = []
            masks, styles, flows = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                tic = time.time()
                maski, flowi, stylei = self.eval(
                    x[i],
                    batch_size=batch_size,
                    channels=channels[i] if channels is not None and
                    ((len(channels) == len(x) and
                      (isinstance(channels[i], list) or
                       isinstance(channels[i], np.ndarray)) and len(channels[i]) == 2))
                    else channels, channel_axis=channel_axis, z_axis=z_axis,
                    normalize=normalize, invert=invert,
                    rescale=rescale[i] if isinstance(rescale, (list, np.ndarray)) else rescale,
                    diameter=diameter[i] if isinstance(diameter, (list, np.ndarray)) else diameter,
                    do_3D=do_3D, anisotropy=anisotropy, augment=augment,
                    tile_overlap=tile_overlap, bsize=bsize, resample=resample,
                    flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                    compute_masks=compute_masks, min_size=min_size, ortho_weights=ortho_weights,
                    max_size_fraction=max_size_fraction, stitch_threshold=stitch_threshold,
                    progress=progress, niter=niter)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
                self.timing.append(time.time() - tic)
            return masks, flows, styles

        x = convert_image_legacy(
            x, channels=channels, channel_axis=channel_axis, z_axis=z_axis, do_3D=(do_3D or stitch_threshold > 0),
            nchan=self.nchan)
        if x.ndim < 4:
            x = x[np.newaxis, ...]
        nimg = x.shape[0]
        orig_shape = x.shape

        rescale_args = rescale
        if diameter is not None:
            rescale_args = self.diam_mean / diameter
        elif rescale is None:
            rescale_args = self.diam_mean / self.diam_labels

        # keep anisotropy consistent after resizing XY
        if rescale_args is not None and rescale_args != 1.0 and anisotropy is not None:
            anisotropy = rescale_args * anisotropy

        normalize_params = {
            "lowhigh": None,
            "percentile": None,
            "normalize": True,
            "norm3D": True,
            "sharpen_radius": 0,
            "smooth_radius": 0,
            "tile_norm_blocksize": 0,
            "tile_norm_smooth3D": 1,
            "invert": False,
        }
        if isinstance(normalize, dict):
            normalize_params.update(normalize)
        elif isinstance(normalize, bool):
            normalize_params["normalize"] = normalize
            normalize_params["invert"] = invert
        else:
            raise ValueError("normalize parameter must be a bool or a dict")

        do_normalization = True if normalize_params["normalize"] else False
        if nimg > 1 and do_normalization and (stitch_threshold or do_3D):
            normalize_params["norm3D"] = True if do_3D else normalize_params["norm3D"]
            x = transforms.normalize_img(x, **normalize_params)
            do_normalization = False
        else:
            if normalize_params["norm3D"] and nimg > 1:
                models_logger.warning(
                    "normalize_params['norm3D'] is True but do_3D is False and stitch_threshold=0, so setting to False"
                )
                normalize_params["norm3D"] = False
        if do_normalization:
            x = transforms.normalize_img(x, **normalize_params)

        dP, cellprob, styles = self._run_net(
            x, rescale=rescale_args, resample=resample, augment=augment,
            batch_size=batch_size, tile_overlap=tile_overlap,
            bsize=bsize, anisotropy=anisotropy, do_3D=do_3D,
            plane_weights=ortho_weights)

        if do_3D:
            if flow3D_smooth > 0:
                models_logger.info("smoothing flows with sigma=%s", flow3D_smooth)
                dP = gaussian_filter(dP, (0, flow3D_smooth, flow3D_smooth, flow3D_smooth))
            torch.cuda.empty_cache()
            gc.collect()

        if resample:
            dP = self._resize_gradients(dP, to_y_size=x.shape[1], to_x_size=x.shape[2],
                                        to_z_size=x.shape[0] if do_3D else None)
            cellprob = self._resize_cellprob(cellprob, to_x_size=x.shape[2],
                                             to_y_size=x.shape[1], to_z_size=x.shape[0] if do_3D else None)

        if compute_masks:
                niter0 = 200
                niter = niter0 if niter is None or niter == 0 else niter
                masks = self._compute_masks(x.shape, dP, cellprob, flow_threshold=flow_threshold,
                               cellprob_threshold=cellprob_threshold, interp=interp, min_size=min_size,
                            max_size_fraction=max_size_fraction, niter=niter,
                            stitch_threshold=stitch_threshold, do_3D=do_3D)
        else:
            masks = np.zeros(0)

        masks, dP, cellprob = masks.squeeze(), dP.squeeze(), cellprob.squeeze()
        if compute_masks and do_3D and rescale_args is not None and rescale_args != 1.0:
            # ensure masks returned at original size even when resample=False
            masks = masks if masks.size == 0 else masks
            if masks.size > 0:
                masks = transforms.resize_image(masks, Ly=orig_shape[1], Lx=orig_shape[2], no_channels=True)
                masks = masks.transpose(1, 0, 2)
                masks = transforms.resize_image(masks, Ly=orig_shape[0], Lx=orig_shape[2], no_channels=True)
                masks = masks.transpose(1, 0, 2)
        # Ensure 2D flow visualization for non-3D runs to avoid expensive 3D normalize
        dP_vis = dP
        if not do_3D:
            if dP_vis.ndim == 4:  # (2, Z, Y, X)
                dP_vis = dP_vis[:, 0]
        return masks, [plot.dx_to_circ(dP_vis), dP, cellprob], styles

    def _run_net(self, x, rescale=1.0, resample=True, augment=False,
                 batch_size=8, tile_overlap=0.1,
                 bsize=224, anisotropy=1.0, do_3D=False,
                 plane_weights: Optional[Sequence[float]] = None,
                 return_raw_3d: bool = False, **kwargs):
        tic = time.time()
        shape = x.shape
        nimg = shape[0]

        if do_3D:
            Lz, Ly, Lx = shape[:-1]

            # First apply diameter-based rescaling (isotropic in XY)
            if rescale != 1.0:
                Ly_r = int(Ly * rescale)
                Lx_r = int(Lx * rescale)
                x = np.stack(
                    [
                        transforms.resize_image(x[z], Ly=Ly_r, Lx=Lx_r, no_channels=False)
                        for z in range(Lz)
                    ],
                    axis=0,
                )
                Ly, Lx = Ly_r, Lx_r

            # Then apply anisotropy adjustment on already rescaled XY
            if anisotropy is not None and anisotropy != 1.0:
                models_logger.info("resizing 3D image with anisotropy=%s", anisotropy)
                Lz_r, Ly_r, Lx_r = x.shape[:-1]
                x = transforms.resize_image(
                    x.transpose(1, 0, 2, 3),
                    Ly=int(Lz_r * anisotropy),
                    Lx=int(Lx_r),
                ).transpose(1, 0, 2, 3)

            if return_raw_3d:
                return run_3D_core(
                    self.net,
                    x,
                    batch_size=batch_size,
                    augment=augment,
                    tile_overlap=tile_overlap,
                    bsize=bsize,
                    net_ortho=self.net_ortho,
                    plane_weights=plane_weights,
                    return_raw=True,
                )

            # Mirror SAM behavior: use core.run_3D with optional ortho net and plane weights (None -> uniform)
            yf, styles = run_3D_core(
                self.net,
                x,
                batch_size=batch_size,
                augment=augment,
                tile_overlap=tile_overlap,
                bsize=bsize,
                net_ortho=self.net_ortho,
                plane_weights=plane_weights,
            )
            cellprob = yf[..., -1]
            dP = yf[..., :-1].transpose((3, 0, 1, 2))

            # Resize outputs back to original shape if we rescaled
            if resample and rescale != 1.0:
                dP = self._resize_gradients(dP, to_y_size=shape[1], to_x_size=shape[2], to_z_size=shape[0])
                cellprob = self._resize_cellprob(cellprob, to_x_size=shape[2], to_y_size=shape[1], to_z_size=shape[0])
        else:
            yf, styles = run_net(self.net, x, bsize=bsize, augment=augment,
                                 batch_size=batch_size, tile_overlap=tile_overlap,
                                 rsz=rescale if rescale != 1.0 else None)
            if resample and rescale != 1.0:
                yf = transforms.resize_image(yf, shape[1], shape[2])
            cellprob = yf[..., 2]
            dP = yf[..., :2].transpose((3, 0, 1, 2))

        styles = styles.squeeze()
        if nimg > 1:
            models_logger.info("network run in %2.2fs", time.time() - tic)

        return dP, cellprob, styles

    def _resize_gradients(self, grads, to_y_size, to_x_size, to_z_size=None):
        grads_shape = grads.shape
        grads = grads.squeeze()
        squeeze_happened = grads.shape != grads_shape
        grads_shape = np.array(grads_shape)

        if grads.ndim == 3:
            # 2D case, with XY flows in 2 channels:
            grads = np.moveaxis(grads, 0, -1)  # Put gradients last
            grads = transforms.resize_image(grads, Ly=to_y_size, Lx=to_x_size, no_channels=False)
            grads = np.moveaxis(grads, -1, 0)  # Put gradients first

            if squeeze_happened:
                grads = np.expand_dims(grads, int(np.argwhere(grads_shape == 1)))  # add back empty axis for compatibility
        elif grads.ndim == 4 and to_z_size is not None:
            # dP has gradients that can be treated as channels:
            grads = grads.transpose(1, 2, 3, 0)  # move gradients last:
            grads = transforms.resize_image(grads, Ly=to_y_size, Lx=to_x_size, no_channels=False)
            grads = grads.transpose(1, 0, 2, 3)  # switch axes to resize again
            grads = transforms.resize_image(grads, Ly=to_z_size, Lx=to_x_size, no_channels=False)
            grads = grads.transpose(3, 1, 0, 2)  # undo transposition
        else:
            return transforms.resize_image(grads, Ly=to_y_size, Lx=to_x_size, interpolation=cv2.INTER_LINEAR)

        return grads

    def _resize_cellprob(self, prob: np.ndarray, to_y_size: int, to_x_size: int, to_z_size: int = None) -> np.ndarray:
        prob = prob.squeeze()
        if prob.ndim == 2:
            return transforms.resize_image(prob, Ly=to_y_size, Lx=to_x_size, no_channels=True)
        if prob.ndim == 3 and to_z_size is not None:
            prob = transforms.resize_image(prob, Ly=to_y_size, Lx=to_x_size, no_channels=True)
            prob = prob.transpose(1, 0, 2)
            prob = transforms.resize_image(prob, Ly=to_z_size, Lx=to_x_size, no_channels=True)
            prob = prob.transpose(1, 0, 2)
            return prob
        return transforms.resize_image(prob, Ly=to_y_size, Lx=to_x_size, no_channels=True)

    def _compute_masks(self, shape, dP, cellprob, flow_threshold=0.4, cellprob_threshold=0.0,
                       interp=True, min_size=15, max_size_fraction=0.4, niter=None,
                       do_3D=False, stitch_threshold=0.0):
        """ compute masks from flows and cell probability """
        Lz, Ly, Lx = shape[:3]
        tic = time.time()
        if do_3D:
            masks = dynamics.resize_and_compute_masks(
                dP, cellprob, niter=niter, cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold, do_3D=do_3D,
                min_size=min_size, max_size_fraction=max_size_fraction,
                resize=shape[:3] if (np.array(dP.shape[-3:])!=np.array(shape[:3])).sum()
                        else None,
                device=self.device)
        else:
            nimg = shape[0]
            Ly0, Lx0 = cellprob[0].shape
            resize = None if Ly0==Ly and Lx0==Lx else [Ly, Lx]
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out,
                            mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                # turn off min_size for 3D stitching
                min_size0 = min_size if stitch_threshold == 0 or nimg == 1 else -1
                outputs = dynamics.resize_and_compute_masks(
                    dP[:, i], cellprob[i],
                    niter=niter, cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold, resize=resize,
                    min_size=min_size0, max_size_fraction=max_size_fraction,
                    device=self.device)
                if i==0 and nimg > 1:
                    masks = np.zeros((nimg, shape[1], shape[2]), outputs.dtype)
                if nimg > 1:
                    masks[i] = outputs
                else:
                    masks = outputs

            if stitch_threshold > 0 and nimg > 1:
                models_logger.info(
                    f"stitching {nimg} planes using stitch_threshold={stitch_threshold:0.3f} to make 3D masks"
                )
                masks = utils.stitch3D(masks, stitch_threshold=stitch_threshold)
                masks = utils.fill_holes_and_remove_small_masks(
                    masks, min_size=min_size)
            elif nimg > 1:
                models_logger.warning(
                    "3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only"
                )

        flow_time = time.time() - tic
        if shape[0] > 1:
            models_logger.info("masks created in %2.2fs" % (flow_time))

        return masks


class SizeModel:
    def __init__(self, cp_model: CellposeUNetModel, device: Optional[torch.device] = None,
                 pretrained_size: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_size = pretrained_size
        self.cp = cp_model
        self.device = self.cp.device if device is None else device
        self.diam_mean = self.cp.diam_mean
        if pretrained_size is not None:
            self.params = np.load(self.pretrained_size, allow_pickle=True).item()
            self.diam_mean = self.params["diam_mean"]
        if not hasattr(self.cp, "pretrained_model"):
            error_message = "no pretrained cellpose model specified, cannot compute size"
            models_logger.critical(error_message)
            raise ValueError(error_message)

    def eval(self, x, channels=None, channel_axis=None, normalize=True, invert=False,
             augment=False, batch_size=8, progress=None):
        if isinstance(x, list):
            self.timing = []
            diams, diams_style = [], []
            nimg = len(x)
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                tic = time.time()
                diam, diam_style = self.eval(
                    x[i], channels=channels[i] if
                    (channels is not None and len(channels) == len(x) and
                     (isinstance(channels[i], list) or
                      isinstance(channels[i], np.ndarray)) and
                     len(channels[i]) == 2) else channels, channel_axis=channel_axis,
                    normalize=normalize, invert=invert, augment=augment,
                    batch_size=batch_size, progress=progress)
                diams.append(diam)
                diams_style.append(diam_style)
                self.timing.append(time.time() - tic)

            return diams, diams_style

        if x.squeeze().ndim > 3:
            models_logger.warning("image is not 2D cannot compute diameter")
            return self.diam_mean, self.diam_mean

        styles = self.cp.eval(x, channels=channels, channel_axis=channel_axis,
                              normalize=normalize, invert=invert, augment=augment,
                              batch_size=batch_size, resample=False,
                              compute_masks=False)[-1]

        diam_style = self._size_estimation(np.array(styles))
        diam_style = self.diam_mean if (diam_style == 0 or
                                        np.isnan(diam_style)) else diam_style

        masks = self.cp.eval(
            x, compute_masks=True, channels=channels, channel_axis=channel_axis,
            normalize=normalize, invert=invert, augment=augment,
            batch_size=batch_size, resample=False,
            rescale=self.diam_mean / diam_style if self.diam_mean > 0 else 1,
            diameter=None, interp=False)[0]

        diam = utils.diameters(masks)[0]
        diam = self.diam_mean if (diam == 0 or np.isnan(diam)) else diam
        return diam, diam_style

    def _size_estimation(self, style):
        szest = np.exp(self.params["A"] @ (style - self.params["smean"]).T +
                       np.log(self.diam_mean) + self.params["ymean"])
        szest = np.maximum(5., szest)
        return szest


__all__ = ["CellposeUNetModel", "SizeModel", "MODEL_NAMES", "model_path", "size_model_path", "get_user_models"]
