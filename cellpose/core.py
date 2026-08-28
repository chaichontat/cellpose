"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import logging

import cupy as cp
import numpy as np
import torch
from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_gpu
from cupyx.scipy.ndimage import uniform_filter as uniform_filter_gpu
from scipy.ndimage import gaussian_filter, uniform_filter
from tqdm import trange

from . import transforms, utils

TORCH_ENABLED = True
CUPY_ENABLED = True

core_logger = logging.getLogger(__name__)
tqdm_out = utils.TqdmToLogger(core_logger, level=logging.INFO)



def _smooth_flows_2d(y, sigma, use_gpu=True):
    """Apply 2D Gaussian smoothing to flow fields in-place when possible.

    Args:
        y: Array of shape [num_slices, H, W, 3] where last dim is [flow_y, flow_x, cellprob].
        sigma: Gaussian sigma for smoothing.
        use_gpu: If True and CuPy available, use GPU acceleration.

    Returns:
        Smoothed array of same shape.
    """
    if sigma <= 0:
        return y

    if use_gpu and CUPY_ENABLED and torch.cuda.is_available():
        # Use CuPy for GPU-accelerated smoothing
        y_gpu = cp.asarray(y)
        del y  # Free input array immediately
        # Smooth only flow components (first 2 channels), not cellprob
        for c in range(2):
            for z in range(y_gpu.shape[0]):
                y_gpu[z, :, :, c] = gaussian_filter_gpu(y_gpu[z, :, :, c], sigma=sigma)
        y_smoothed = cp.asnumpy(y_gpu)
        del y_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU fallback - modify in place
        for c in range(2):
            for z in range(y.shape[0]):
                y[z, :, :, c] = gaussian_filter(y[z, :, :, c], sigma=sigma)
        y_smoothed = y

    return y_smoothed


def _smooth_flows_3d(dP, sigma, use_gpu=True):
    """Apply 3D Gaussian smoothing to flow fields.

    Args:
        dP: Array of shape [3, Z, Y, X] where first dim is flow components (dz, dy, dx).
        sigma: Gaussian sigma for smoothing spatial dimensions.
        use_gpu: If True and CuPy available, use GPU acceleration.

    Returns:
        Smoothed array of same shape.
    """
    if sigma <= 0:
        return dP

    core_logger.info(f"smoothing 3D flows with sigma={sigma}, use_gpu={use_gpu and CUPY_ENABLED}")

    if use_gpu and CUPY_ENABLED and torch.cuda.is_available():
        try:
            dP_gpu = cp.asarray(dP)
            # Smooth each flow component separately in 3D
            for c in range(dP_gpu.shape[0]):
                dP_gpu[c] = gaussian_filter_gpu(dP_gpu[c], sigma=sigma)
            dP_smoothed = cp.asnumpy(dP_gpu)
            del dP_gpu
        finally:
            cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU fallback
        dP_smoothed = np.empty_like(dP)
        for c in range(dP.shape[0]):
            dP_smoothed[c] = gaussian_filter(dP[c], sigma=sigma)

    return dP_smoothed


def _compute_variance_weights(data, alpha=0.5, window_size=3, use_gpu=True):
    """Compute inverse-variance weights for fusion (u-Segment3D).

    Higher weight is given to regions with low local variance (smooth/confident predictions).
    This automatically downweights noisy predictions from orthogonal planes.

    Args:
        data: Array of shape [Z, Y, X] (single channel).
        alpha: Stabilization constant. Small = aggressive (trust smoothest).
               Recommended: 1e-5 for cellprob, 0.5 for flows.
        window_size: Local neighborhood for variance computation.
        use_gpu: Use CuPy if available.

    Returns:
        weights: Array of same shape, inverse-variance weights.
    """
    if use_gpu and CUPY_ENABLED and torch.cuda.is_available():
        try:
            data_gpu = cp.asarray(data)
            mean_data = uniform_filter_gpu(data_gpu, size=window_size)
            sq_mean = uniform_filter_gpu(data_gpu ** 2, size=window_size)
            variance = cp.maximum(sq_mean - mean_data ** 2, 0)
            sigma = cp.sqrt(variance)
            weights = 1.0 / (sigma + alpha)
            weights_np = cp.asnumpy(weights)
            del data_gpu, mean_data, sq_mean, variance, sigma, weights
        finally:
            cp.get_default_memory_pool().free_all_blocks()
        return weights_np
    else:
        mean_data = uniform_filter(data, size=window_size)
        sq_mean = uniform_filter(data ** 2, size=window_size)
        variance = np.maximum(sq_mean - mean_data ** 2, 0)
        sigma = np.sqrt(variance)
        return 1.0 / (sigma + alpha)


def use_gpu(gpu_number=0, use_torch=True):
    """
    Check if GPU is available for use.

    Args:
        gpu_number (int): The index of the GPU to be used. Default is 0.
        use_torch (bool): Whether to use PyTorch for GPU check. Default is True.

    Returns:
        bool: True if GPU is available, False otherwise.

    Raises:
        ValueError: If use_torch is False, as cellpose only runs with PyTorch now.
    """
    if use_torch:
        return _use_gpu_torch(gpu_number)
    else:
        raise ValueError("cellpose only runs with PyTorch now")


def _use_gpu_torch(gpu_number=0):
    """
    Checks if CUDA or MPS is available and working with PyTorch.

    Args:
        gpu_number (int): The GPU device number to use (default is 0).

    Returns:
        bool: True if CUDA or MPS is available and working, False otherwise.
    """
    try:
        device = torch.device("cuda:" + str(gpu_number))
        _ = torch.zeros((1,1)).to(device)
        core_logger.info("** TORCH CUDA version installed and working. **")
        return True
    except:
        pass
    try:
        device = torch.device('mps:' + str(gpu_number))
        _ = torch.zeros((1,1)).to(device)
        core_logger.info('** TORCH MPS version installed and working. **')
        return True
    except:
        core_logger.info('Neither TORCH CUDA nor MPS version not installed/working.')
        return False


def assign_device(use_torch=True, gpu=False, device=0):
    """
    Assigns the device (CPU or GPU or mps) to be used for computation.

    Args:
        use_torch (bool, optional): Whether to use torch for GPU detection. Defaults to True.
        gpu (bool, optional): Whether to use GPU for computation. Defaults to False.
        device (int or str, optional): The device index or name to be used. Defaults to 0.

    Returns:
        torch.device, bool (True if GPU is used, False otherwise)
    """

    if isinstance(device, str):
        if device != "mps" or not(gpu and torch.backends.mps.is_available()):
            device = int(device)
    if gpu and use_gpu(use_torch=True):
        try:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{device}')
                core_logger.info(">>>> using GPU (CUDA)")
                gpu = True
                cpu = False
        except:
            gpu = False
            cpu = True
        try:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                core_logger.info(">>>> using GPU (MPS)")
                gpu = True
                cpu = False
        except:
            gpu = False
            cpu = True
    else:
        device = torch.device('cpu')
        core_logger.info('>>>> using CPU')
        gpu = False
        cpu = True

    if cpu:
        device = torch.device("cpu")
        core_logger.info(">>>> using CPU")
        gpu = False
    return device, gpu


def _to_device(x, device, dtype=torch.float32):
    """
    Converts the input tensor or numpy array to the specified device.

    Args:
        x (torch.Tensor or numpy.ndarray): The input tensor or numpy array.
        device (torch.device): The target device.

    Returns:
        torch.Tensor: The converted tensor on the specified device.
    """
    if not isinstance(x, torch.Tensor):
        X = torch.from_numpy(x).to(device, dtype=dtype)
        return X
    else:
        return x


def _from_device(X):
    """
    Converts a PyTorch tensor from the device to a NumPy array on the CPU.

    Args:
        X (torch.Tensor): The input PyTorch tensor.

    Returns:
        numpy.ndarray: The converted NumPy array.
    """
    # The cast is so numpy conversion always works
    x = X.detach().cpu().to(torch.float32).numpy()
    return x


def _forward(net, x):
    """Converts images to torch tensors, runs the network model, and returns numpy arrays.

    More robust dtype handling: falls back to the dtype of the first parameter
    if the module does not expose a `dtype` attribute (e.g., CPnet).

    Args:
        net (torch.nn.Module): The network model.
        x (numpy.ndarray): The input images.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The output predictions (flows and cellprob) and style features.
    """
    try:
        dtype = net.dtype
    except Exception:
        try:
            dtype = next(net.parameters()).dtype
        except Exception:
            dtype = torch.float32
    X = _to_device(x, device=net.device, dtype=dtype)
    net.eval()
    with torch.inference_mode():
        y, style = net(X)[:2]
    del X
    y = _from_device(y)
    style = _from_device(style)
    return y, style


def run_net(net, imgi, batch_size=8, augment=False, tile_overlap=0.1, bsize=224,
            rsz=None, single_tile_if_fit=False, skip_empty_tiles=False):
    """
    Run network on stack of images.
    (faster if augment is False)

    Args:
        net (class): cellpose network (model.net)
        imgi (np.ndarray): The input image or stack of images of size [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
        rsz (float, optional): Resize coefficient(s) for image. Defaults to 1.0.
        augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.
        skip_empty_tiles (bool, optional): Skip all-zero network tiles and scatter
            zero outputs back into their positions. Defaults to False.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: outputs of network y and style. If tiled `y` is averaged in tile overlaps. Size of [Ly x Lx x 3] or [Lz x Ly x Lx x 3].
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability.
            style is a 1D array of size 256 summarizing the style of the image, if tiled `style` is averaged over tiles.
    """
    # run network
    Lz, Ly0, Lx0, nchan = imgi.shape
    if rsz is not None:
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        Lyr, Lxr = int(Ly0 * rsz[0]), int(Lx0 * rsz[1])
    else:
        Lyr, Lxr = Ly0, Lx0

    ly, lx = bsize, bsize
    # default padding matches legacy behavior (ensures >= bsize and multiples of 16 with extra margins)
    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(Lyr, Lxr, min_size=(bsize, bsize))

    # Optional optimization: if both dimensions can fit in a single tile and we are not
    # augmenting, avoid the extra 16-pixel padding on X to keep nx=1.
    if single_tile_if_fit and not augment:
        # Minimize padding to keep single tile along any dimension that already fits bsize
        # while preserving 16px alignment and without exceeding bsize.
        div = 16
        # Y dimension
        if Lyr <= bsize:
            LpadY = (div - (Lyr % div)) % div
            if Lyr + LpadY <= bsize:
                ypad1 = LpadY // 2
                ypad2 = LpadY - ypad1
        # X dimension: keep legacy pads to preserve tile shapes
    Ly, Lx = Lyr + ypad1 + ypad2, Lxr + xpad1 + xpad2
    pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])

    if augment:
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
    else:
        ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize))

    # run multiple slices at the same time
    ntiles = ny * nx
    nimgs = max(1, batch_size // ntiles)  # number of imgs to run in the same batch
    niter = int(np.ceil(Lz / nimgs))
    core_logger.info(
        "tiling decision: Lz=%d, Ly=%d, Lx=%d, bsize=%d, augment=%s, tile_overlap=%.3f, "
        "single_tile_if_fit=%s -> ny=%d, nx=%d, ntiles=%d, nimgs_per_batch=%d, niter=%d",
        Lz,
        Ly,
        Lx,
        bsize,
        augment,
        tile_overlap,
        single_tile_if_fit,
        ny,
        nx,
        ntiles,
        nimgs,
        niter,
    )
    ziterator = (trange(niter, file=tqdm_out, mininterval=30)
                    if niter > 10 or Lz > 1 else range(niter))
    nout = net.nout if hasattr(net, "nout") else 3
    yf = np.zeros((Lz, nout, Ly, Lx), "float32")
    styles = np.zeros((Lz, 256), "float32")
    skipped_tiles = 0
    for k in ziterator:
        inds = np.arange(k * nimgs, min(Lz, (k + 1) * nimgs))
        IMGa = np.zeros((ntiles * len(inds), nchan, ly, lx), "float32")
        for i, b in enumerate(inds):
            # pad image for net so Ly and Lx are divisible by 4
            imgb = transforms.resize_image(imgi[b], rsz=rsz) if rsz is not None else imgi[b].copy()
            imgb = np.pad(imgb.transpose(2,0,1), pads, mode="constant")
            IMG, ysub, xsub, Lyt, Lxt = transforms.make_tiles(
                imgb, bsize=bsize, augment=augment,
                tile_overlap=(0.0 if (single_tile_if_fit and not augment and Lx <= bsize and Ly <= bsize) else tile_overlap))
            IMGa[i * ntiles : (i+1) * ntiles] = np.reshape(IMG,
                                            (ny * nx, nchan, ly, lx))
        active_tiles = (
            np.flatnonzero(np.any(IMGa != 0, axis=(1, 2, 3)))
            if skip_empty_tiles
            else np.arange(IMGa.shape[0])
        )
        skipped_tiles += IMGa.shape[0] - len(active_tiles)
        ya = np.zeros((IMGa.shape[0], nout, ly, lx), "float32")
        stylea = np.zeros((IMGa.shape[0], 256), "float32")
        for j in range(0, len(active_tiles), batch_size):
            batch_indices = active_tiles[j : j + batch_size]
            ya0, stylea0 = _forward(net, IMGa[batch_indices])
            if ya0.shape[1] != nout:
                raise ValueError(
                    f"Network returned {ya0.shape[1]} channels; expected {nout}."
                )
            ya[batch_indices] = ya0
            stylea[batch_indices] = stylea0

        # average tiles
        for i, b in enumerate(inds):
            y = ya[i * ntiles : (i + 1) * ntiles]
            if augment:
                y = np.reshape(y, (ny, nx, 3, ly, lx))
                y = transforms.unaugment_tiles(y)
                y = np.reshape(y, (-1, 3, ly, lx))
            yfi = transforms.average_tiles(y, ysub, xsub, Lyt, Lxt)
            yf[b] = yfi[:, :imgb.shape[-2], :imgb.shape[-1]]
            # stylei = stylea[i * ntiles:(i + 1) * ntiles].sum(axis=0)
            # stylei /= (stylei**2).sum()**0.5
            # styles[b] = stylei
    if skip_empty_tiles:
        core_logger.info("skipped %d all-zero network tiles", skipped_tiles)
    # slices from padding
    yf = yf[:, :, ypad1 : Ly-ypad2, xpad1 : Lx-xpad2]
    yf = yf.transpose(0,2,3,1)
    return yf, np.array(styles)


def run_3D(net, imgs, batch_size=8, augment=False,
           tile_overlap=0.1, bsize=224, net_ortho=None,
           progress=None, plane_weights=None,
           return_raw=False, flow2D_smooth=0.0,
           use_variance_fusion=False, variance_alpha_flow=0.5, variance_alpha_cellprob=1e-5,
           skip_empty_tiles=False):
    """
    Run network on image z-stack.

    (faster if augment is False)

    Args:
        imgs (np.ndarray): The input image stack of size [Lz x Ly x Lx x nchan].
        batch_size (int, optional): Number of tiles to run in a batch. Defaults to 8.
        rsz (float, optional): Resize coefficient(s) for image. Defaults to 1.0.
        anisotropy (float, optional): for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y). Defaults to None.
        augment (bool, optional): Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        bsize (int, optional): Size of tiles to use in pixels [bsize x bsize]. Defaults to 224.
        net_ortho (class, optional): cellpose network for orthogonal ZY and ZX planes. Defaults to None.
        progress (QProgressBar, optional): pyqt progress bar. Defaults to None.
        return_raw (bool, optional): If True, return raw per-axis outputs before aggregation. Defaults to False.
        flow2D_smooth (float, optional): Gaussian sigma for pre-smoothing 2D flows before 3D aggregation
            (u-Segment3D recommends 1.0). Defaults to 0.0 (no smoothing).
        use_variance_fusion (bool, optional): Use inverse-variance weighted fusion (u-Segment3D).
            Automatically downweights noisy planes. Defaults to False.
        variance_alpha_flow (float, optional): Alpha for flow variance weighting. Larger = more averaging.
            Defaults to 0.5 (conservative).
        variance_alpha_cellprob (float, optional): Alpha for cellprob variance weighting.
            Defaults to 1e-5 (aggressive, strongly trust smooth predictions).
        skip_empty_tiles (bool, optional): Skip all-zero 2D network tiles and
            scatter zero outputs into their positions. Defaults to False.

    Returns:
        If `return_raw` is True:
            dict: Raw per-axis outputs with keys "xy", "xz", "yz". Each value is a dict with:
                - "y": np.ndarray of shape [num_slices, H, W, 3] (2D flows + cellprob)
                - "style": np.ndarray style vector
    """
    sstr = ["YX", "ZY", "ZX"]
    orient_keys = ["xy", "xz", "yz"]  # match user-facing plane names
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    ipm = [(0, 1, 2), (1, 0, 2), (1, 2, 0)]
    cp = [(1, 2), (0, 2), (0, 1)]
    cpy = [(0, 1), (0, 1), (0, 1)]
    shape = imgs.shape[:-1]
    yf = np.zeros((*shape, 4), "float32")
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
    # Initialize weight accumulators - arrays when using variance fusion, scalars otherwise
    core_logger.info(f"run_3D: use_variance_fusion={use_variance_fusion}")
    if use_variance_fusion:
        flow_weight_totals = [np.zeros(shape, dtype=np.float32) for _ in range(3)]
        cellprob_weight_total = np.zeros(shape, dtype=np.float32)
        core_logger.info(f"Using variance-weighted fusion: alpha_flow={variance_alpha_flow}, alpha_cellprob={variance_alpha_cellprob}")
    else:
        flow_weight_totals = np.zeros(3, dtype=np.float32)
        cellprob_weight_total = 0.0

    raw_outputs = {} if return_raw else None
    for p in range(3):
        weight = float(weights[p])
        xsl = imgs.transpose(pm[p])
        # per image
        core_logger.info("running %s: %d planes of size (%d, %d)" %
                         (sstr[p], shape[pm[p][0]], shape[pm[p][1]], shape[pm[p][2]]))
        active_net = net if p == 0 or net_ortho is None else net_ortho
        y, style = run_net(active_net,
                           xsl, batch_size=batch_size, augment=augment,
                           bsize=bsize, tile_overlap=tile_overlap,
                           rsz=None, skip_empty_tiles=skip_empty_tiles)

        # Apply 2D pre-smoothing before aggregation (u-Segment3D)
        if flow2D_smooth > 0:
            y = _smooth_flows_2d(y, sigma=flow2D_smooth, use_gpu=True)

        if return_raw:
            raw_outputs[orient_keys[p]] = {"y": y.copy(), "style": style.copy()}

        if use_variance_fusion:
            # Variance-weighted fusion: weight by inverse local variance
            # Transpose cellprob to 3D volume space
            cellprob_3d = y[..., -1].transpose(ipm[p])
            w_cellprob = _compute_variance_weights(
                cellprob_3d, alpha=variance_alpha_cellprob, use_gpu=True)
            yf[..., -1] += weight * w_cellprob * cellprob_3d
            cellprob_weight_total += weight * w_cellprob

            # Variance weights for each flow component
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

        y = None; del y

        if progress is not None:
            progress.setValue(25 + 15 * p)

    if return_raw:
        return raw_outputs

    # Normalize by accumulated weights
    if use_variance_fusion:
        # Per-voxel normalization for variance-weighted fusion
        for axis_idx in range(3):
            mask = flow_weight_totals[axis_idx] > 0
            yf[..., axis_idx][mask] /= flow_weight_totals[axis_idx][mask]
        mask = cellprob_weight_total > 0
        yf[..., -1][mask] /= cellprob_weight_total[mask]
    else:
        # Scalar normalization for simple weighted average
        for axis_idx in range(3):
            if flow_weight_totals[axis_idx] > 0:
                yf[..., axis_idx] /= flow_weight_totals[axis_idx]
        if cellprob_weight_total > 0:
            yf[..., -1] /= cellprob_weight_total

    return yf, style
