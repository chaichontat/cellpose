import time
import os
import numpy as np
from typing import Callable, Optional
from cellpose import io, utils, models, dynamics
from cellpose.transforms import normalize_img, random_rotate_and_resize
from cellpose.contrib.pack_utils import compute_stripe_layout, compute_max_guard, pack_planes_to_stripes
from pathlib import Path
import torch
from torch import nn
from tqdm import trange

import logging

train_logger = logging.getLogger(__name__)
_PACK_TRAIN_LOGGED = False
_THIN_HEIGHT_PX = 100
_THIN_PACK_RATIO = 0.8  # fraction of thin stripes that stay packed
_SQUARE_STRIPE_RATIO = 0.10  # fraction of square tiles converted into synthetic stripes
_SQUARE_STRIPE_MIN_HEIGHT = 30
_SQUARE_STRIPE_MAX_HEIGHT = 90
_PACK_STRIPE_BORDER = 0  # pixels of padding applied above/below each packed stripe


def _resolve_pack_stripe_height(
    stripe_height: int | None,
    *,
    pack_k: int,
    guard: int,
    bsize: int,
    border: int,
    max_height: int | None = None,
) -> tuple[int | None, bool]:
    """Determine a stripe height that satisfies the packing constraint.

    Returns
    -------
    (height, auto_adjusted)
        height is ``None`` if packing is impossible for the provided params.
        auto_adjusted is True when the requested height had to be reduced.
    """

    if pack_k <= 1:
        return None, False
    border = max(0, int(border))

    max_h = None
    if max_height is not None and max_height > 0:
        max_h = int(max_height)

    candidate = stripe_height if stripe_height is not None and stripe_height > 0 else (max_h if max_h is not None else bsize)
    if max_h is not None:
        candidate = min(candidate, max_h)
    candidate = min(int(candidate), int(bsize))
    
    layout = compute_stripe_layout(
        candidate,
        bsize=bsize,
        pack_k=pack_k,
        guard=guard,
        border=border,
    )
    if layout is not None:
        return layout.stripe_height, False

    max_k = max(2, int(pack_k))
    slot_available = int(bsize) - max(0, int(guard)) * (max_k - 1)
    if slot_available <= 0:
        return None, False
    slot_per_stripe = slot_available // max_k
    stripe = slot_per_stripe - 2 * border
    if max_h is not None:
        stripe = min(stripe, max_h)
    if stripe <= 0:
        return None, False
    # Quantize auto-selected height down to a multiple of 16 to honor conv strides
    div = 16
    stripe_q = max(div, (int(stripe) // div) * div)
    if max_h is not None and stripe_q > max_h:
        stripe_q = (max_h // div) * div
        if stripe_q <= 0:
            stripe_q = max_h
    return stripe_q, True

def _loss_fn_class(lbl, y, class_weights=None):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).

    Returns:
        torch.Tensor: Loss value.

    """

    criterion3 = nn.CrossEntropyLoss(reduction="mean", weight=class_weights)
    loss3 = criterion3(y[:, :-3], lbl[:, 0].long())

    return loss3

def _loss_fn_seg(lbl, y, device):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        device (torch.device): Device on which the tensors are located.

    Returns:
        torch.Tensor: Loss value.

    """
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = 5. * lbl[:, -2:]
    loss = criterion(y[:, -3:-1], veci)
    loss /= 2.
    loss2 = criterion2(y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype))
    loss = loss + loss2
    return loss

def _reshape_norm(data, channel_axis=None, normalize_params={"normalize": False}):
    """
    Reshapes and normalizes the input data.

    Args:
        data (list): List of input data, with channels axis first or last.
        normalize_params (dict, optional): Dictionary of normalization parameters. Defaults to {"normalize": False}.

    Returns:
        list: List of reshaped and normalized data.
    """
    if (np.array([td.ndim!=3 for td in data]).sum() > 0 or
        np.array([td.shape[0]!=3 for td in data]).sum() > 0):
        data_new = []
        for td in data:
            if td.ndim == 3:
                channel_axis0 = channel_axis if channel_axis is not None else np.array(td.shape).argmin()
                # put channel axis first
                td = np.moveaxis(td, channel_axis0, 0)
                td = td[:3] # keep at most 3 channels
            if td.ndim == 2 or (td.ndim == 3 and td.shape[0] == 1):
                td = np.stack((td, 0*td, 0*td), axis=0)
            elif td.ndim == 3 and td.shape[0] < 3:
                td = np.concatenate((td, 0*td[:1]), axis=0)
            data_new.append(td)
        data = data_new
    if normalize_params["normalize"]:
        data = [
            normalize_img(td, normalize=normalize_params, axis=0)
            for td in data
        ]
    return data


def _pack_training_tiles(
    imgs: np.ndarray,
    lbls: np.ndarray,
    *,
    pack_k: int,
    guard: int,
    bsize: int,
    stripe_height: int | None,
    border: int,
):
    """Pack `pack_k` cropped stripes into a single `bsize×bsize` tile to mirror inference packing.

    Parameters
    ----------
    imgs : np.ndarray
        Array of shape (N, C, H, W) returned by `random_rotate_and_resize`.
    lbls : np.ndarray
        Array of shape (N, L, H, W) containing supervision targets.
    pack_k : int
        Number of stripes per packed tile (same as inference packing).
    guard : int
        Guard rows inserted between stripes.
    bsize : int
        Tile size (typically 256).
    stripe_height : int | None
        Height of each stripe before packing. If ``None`` (or if the requested value
        violates the packing constraint), the height is auto-chosen based on the
        input tile size and ``pack_k``/``guard``.
    border : int
        Pixels of padding inserted above and below each stripe prior to packing.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Packed images and labels. Falls back to the original tensors if packing is not possible.
    """

    if pack_k <= 1:
        return imgs, lbls

    N, C, H, W = imgs.shape
    desired_height = stripe_height if (stripe_height is not None and stripe_height > 0) else H
    stripe_h, auto_adjusted = _resolve_pack_stripe_height(
        desired_height,
        pack_k=pack_k,
        guard=guard,
        bsize=bsize,
        border=border,
        max_height=H,
    )
    if stripe_h is None or stripe_h <= 0:
        train_logger.debug(
            "Packing requested but layout unavailable (stripe_height=%s, pack_k=%d, guard=%d, bsize=%d); "
            "returning original batch of %d tiles",
            stripe_height,
            pack_k,
            guard,
            bsize,
            N,
        )
        return imgs, lbls

    layout = compute_stripe_layout(
        stripe_h,
        bsize=bsize,
        pack_k=pack_k,
        guard=guard,
        border=border,
    )
    if layout is None:
        train_logger.debug(
            "Resolved stripe height %d still cannot satisfy packing (pack_k=%d, guard=%d, bsize=%d); "
            "returning original batch of %d tiles",
            stripe_h,
            pack_k,
            guard,
            bsize,
            N,
        )
        return imgs, lbls
    if auto_adjusted:
        train_logger.debug(
            "Auto-adjusted training stripe height to %d (pack_k=%d, guard=%d, bsize=%d)",
            stripe_h,
            pack_k,
            guard,
            bsize,
        )

    global _PACK_TRAIN_LOGGED
    if not _PACK_TRAIN_LOGGED:
        packed_groups = (N + layout.K - 1) // layout.K
        train_logger.info(
            "Packing training stripes: tiles=%d pack_k=%d guard=%d stripe_height=%d -> groups=%d",
            N,
            layout.K,
            layout.guard,
            stripe_h,
            packed_groups,
        )
        _PACK_TRAIN_LOGGED = True
    train_logger.debug(
        "Packing training batch: tiles=%d -> groups=%d (pack_k=%d, guard=%d, stripe_height=%d, bsize=%d)",
        N,
        (N + layout.K - 1) // layout.K,
        layout.K,
        layout.guard,
        stripe_h,
        bsize,
    )

    img_stripes = []
    lbl_stripes = []
    for i in range(N):
        if H == stripe_h:
            y0 = 0
        else:
            y0 = np.random.randint(0, H - stripe_h + 1)
        y1 = y0 + stripe_h
        img_stripes.append(imgs[i, :, y0:y1, :])
        lbl_stripes.append(lbls[i, :, y0:y1, :])

    img_stripes = np.transpose(np.stack(img_stripes, axis=0), (0, 2, 3, 1))  # [N, stripe_h, W, C]
    lbl_stripes = np.transpose(np.stack(lbl_stripes, axis=0), (0, 2, 3, 1))  # [N, stripe_h, W, L]

    packed_imgs, mapping = pack_planes_to_stripes(img_stripes, layout)
    packed_lbls, _ = pack_planes_to_stripes(lbl_stripes, layout)

    imgs_out = np.transpose(packed_imgs, (0, 3, 1, 2))
    lbls_out = np.transpose(packed_lbls, (0, 3, 1, 2))
    return imgs_out.astype(imgs.dtype, copy=False), lbls_out.astype(lbls.dtype, copy=False)


def _prepad_height(img: np.ndarray) -> int:
    """Return the image height before any padding/warping is applied."""

    arr = np.asarray(img)
    if arr.ndim < 2:
        return int(arr.shape[0])

    # Channel-first tensors use <=4 channels; height sits on axis -2.
    if arr.ndim >= 3 and arr.shape[0] <= 4:
        return int(arr.shape[-2])

    # Fallback: treat the leading axis as height (channel-last or 2D input).
    return int(arr.shape[0])


def _effective_height(img: np.ndarray, rescale_factor: float | int | None) -> float:
    """Compute the pre-pad height after diameter-based rescaling."""

    base = float(_prepad_height(img))
    if rescale_factor is None:
        return base
    try:
        scale = float(rescale_factor)
    except (TypeError, ValueError):
        return base
    if scale <= 0:
        return base
    return base / scale


def _partition_indices(indices: list[int], ratio: float) -> tuple[list[int], list[int]]:
    if not indices:
        return [], []
    if ratio >= 1.0:
        return indices, []
    total = len(indices)
    pack_count = max(0, min(total, int(round(total * ratio))))
    if pack_count <= 0:
        return [], indices
    if pack_count >= total:
        return indices, []
    selection = np.random.choice(indices, size=pack_count, replace=False)
    packed_set = {int(i) for i in selection.tolist()}
    packed = sorted(packed_set)
    fallback = [idx for idx in indices if idx not in packed_set]
    return packed, fallback


def _partition_thin_indices(indices: list[int], ratio: float) -> tuple[list[int], list[int]]:
    return _partition_indices(indices, ratio)


def _slice_variable_height_stripes(
    img: np.ndarray,
    lbl: np.ndarray | None,
    *,
    min_height: int,
    max_height: int,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    if img.ndim != 3:
        raise ValueError(f"Expected channel-first image tensor, got shape {img.shape}")
    _, height, _ = img.shape
    min_h = max(1, int(min_height))
    max_h = max(min_h, int(max_height))
    stripes: list[tuple[np.ndarray, np.ndarray | None]] = []
    y = 0
    rng = np.random.default_rng()
    while y < height:
        remaining = height - y
        stripe_h = int(rng.integers(min_h, max_h + 1))
        stripe_h = min(stripe_h, remaining)
        if remaining - stripe_h < min_h and remaining != stripe_h:
            stripe_h = remaining
        stripe_img = img[:, y : y + stripe_h, :]
        stripe_lbl = lbl[:, y : y + stripe_h, :] if lbl is not None else None
        stripes.append((stripe_img, stripe_lbl))
        y += stripe_h
    return stripes


def _generate_square_stripe_samples(
    indices: list[int],
    imgs: list[np.ndarray],
    lbls: list[np.ndarray] | None,
    rescales: np.ndarray,
    *,
    min_height: int,
    max_height: int,
) -> tuple[list[np.ndarray], list[np.ndarray] | None, list[float]]:
    if not indices:
        return [], ([] if lbls is not None else None), []
    stripe_imgs: list[np.ndarray] = []
    stripe_lbls: list[np.ndarray] | None = [] if lbls is not None else None
    stripe_rescales: list[float] = []
    for idx in indices:
        img = imgs[idx]
        lbl = lbls[idx] if lbls is not None else None
        stripes = _slice_variable_height_stripes(
            img,
            lbl,
            min_height=min_height,
            max_height=max_height,
        )
        for stripe_img, stripe_lbl in stripes:
            stripe_imgs.append(stripe_img)
            if stripe_lbls is not None and stripe_lbl is not None:
                stripe_lbls.append(stripe_lbl)
            stripe_rescales.append(float(rescales[idx]))
    return stripe_imgs, stripe_lbls, stripe_rescales


def _prepare_dimension_packed_batch(
    imgs: list[np.ndarray],
    lbls: list[np.ndarray] | None,
    rsc: np.ndarray,
    *,
    scale_range,
    bsize: int,
    pack_enabled: bool,
    pack_height: int | None,
    pack_k: int,
    pack_guard: int,
    pack_border: int,
    phot_aug: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Shared batch builder for train/eval loops with optional packing."""

    def _standard_path() -> tuple[np.ndarray, np.ndarray, int, int]:
        imgi, lbl = random_rotate_and_resize(
            imgs,
            Y=lbls,
            rescale=rsc,
            scale_range=scale_range,
            xy=(bsize, bsize),
            rotate=True,
            do_flip=True,
        )[:2]
        if phot_aug is not None:
            imgi = phot_aug(imgi)
        return imgi, lbl, 0, imgi.shape[0]

    if not pack_enabled or pack_height is None or pack_k <= 1:
        return _standard_path()

    img_heights = [_effective_height(img, rsc[i]) for i, img in enumerate(imgs)]
    thin_indices = [i for i, h in enumerate(img_heights) if h < _THIN_HEIGHT_PX]
    large_indices = [i for i, h in enumerate(img_heights) if h >= _THIN_HEIGHT_PX]

    thin_indices, thin_as_std = _partition_thin_indices(thin_indices, _THIN_PACK_RATIO)
    if thin_as_std:
        large_indices = sorted(large_indices + thin_as_std)

    square_indices, large_indices = _partition_indices(large_indices, _SQUARE_STRIPE_RATIO)
    stripe_imgs: list[np.ndarray] = []
    stripe_lbls: list[np.ndarray] | None = [] if lbls is not None else None
    stripe_rsc: list[float] = []
    if square_indices and pack_height is not None:
        s_imgs, s_lbls, s_rsc = _generate_square_stripe_samples(
            square_indices,
            imgs,
            lbls,
            rsc,
            min_height=_SQUARE_STRIPE_MIN_HEIGHT,
            max_height=_SQUARE_STRIPE_MAX_HEIGHT,
        )
        stripe_imgs = s_imgs
        if stripe_lbls is not None and s_lbls is not None:
            stripe_lbls = s_lbls
        else:
            stripe_lbls = None
        stripe_rsc = s_rsc

    thin_indices = sorted(thin_indices)

    batches: list[np.ndarray] = []
    labels_batches: list[np.ndarray] = []
    batch_packed_tiles = 0
    batch_std_tiles = 0

    if len(thin_indices) > 0 or stripe_imgs:
        imgs_thin = [imgs[i] for i in thin_indices]
        imgs_thin.extend(stripe_imgs)
        if lbls is not None:
            lbls_thin = [lbls[i] for i in thin_indices]
            if stripe_lbls:
                lbls_thin.extend(stripe_lbls)
        else:
            lbls_thin = None
        thin_rescales = [float(rsc[i]) for i in thin_indices]
        thin_rescales.extend(stripe_rsc)
        rsc_thin = np.asarray(thin_rescales, dtype=rsc.dtype) if thin_rescales else np.zeros((0,), dtype=rsc.dtype)

        imgi_p, lbl_p = random_rotate_and_resize(
            imgs_thin,
            Y=lbls_thin,
            rescale=rsc_thin,
            scale_range=scale_range,
            xy=(pack_height, bsize),
            rotate=False,
            do_flip=True,
        )[:2]
        if phot_aug is not None:
            imgi_p = phot_aug(imgi_p)
        imgi_p, lbl_p = _pack_training_tiles(
            imgi_p,
            lbl_p,
            pack_k=pack_k,
            guard=pack_guard,
            bsize=bsize,
            stripe_height=pack_height,
            border=pack_border,
        )
        if imgi_p.shape[2:] == (bsize, bsize) and lbl_p.shape[2:] == (bsize, bsize):
            batches.append(imgi_p)
            labels_batches.append(lbl_p)
            batch_packed_tiles = len(imgi_p)

    if len(large_indices) > 0:
        imgs_large = [imgs[i] for i in large_indices]
        lbls_large = [lbls[i] for i in large_indices] if lbls is not None else None
        rsc_large = rsc[large_indices]
        imgi_s, lbl_s = random_rotate_and_resize(
            imgs_large,
            Y=lbls_large,
            rescale=rsc_large,
            scale_range=scale_range,
            xy=(bsize, bsize),
            rotate=True,
            do_flip=True,
        )[:2]
        if phot_aug is not None:
            imgi_s = phot_aug(imgi_s)
        batches.append(imgi_s)
        labels_batches.append(lbl_s)
        batch_std_tiles = len(imgi_s)

    if batches:
        imgi = np.concatenate(batches, axis=0) if len(batches) > 1 else batches[0]
        lbl = np.concatenate(labels_batches, axis=0) if len(labels_batches) > 1 else labels_batches[0]
        return imgi, lbl, batch_packed_tiles, batch_std_tiles

    return None


class _ScaledTileSaver:
    def __init__(self, root: Path | str, limit: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit = max(0, int(limit))
        self.saved = 0

    def save_batch(self, tiles: np.ndarray, *, phase: str, epoch: int, batch_index: int) -> None:
        if self.limit == 0 or self.saved >= self.limit:
            return
        remaining = self.limit - self.saved
        subset = tiles[:remaining]
        for idx, tile in enumerate(subset):
            arr = self._to_uint8(tile)
            filename = self.root / (
                f"{phase}_epoch{epoch + 1:04d}_batch{batch_index:04d}_tile{self.saved:05d}.png"
            )
            io.imsave(filename.as_posix(), arr)
            self.saved += 1
            if self.saved >= self.limit:
                break

    @staticmethod
    def _to_uint8(tile: np.ndarray) -> np.ndarray:
        arr = np.asarray(tile, dtype=np.float32)
        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        arr_min = arr.min(initial=0.0)
        arr -= arr_min
        arr_max = arr.max(initial=0.0)
        if arr_max > 0:
            arr /= arr_max
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return arr

def _get_batch(inds, data=None, labels=None, files=None, labels_files=None,
               normalize_params={"normalize": False}):
    """
    Get a batch of images and labels.

    Args:
        inds (list): List of indices indicating which images and labels to retrieve.
        data (list or None): List of image data. If None, images will be loaded from files.
        labels (list or None): List of label data. If None, labels will be loaded from files.
        files (list or None): List of file paths for images.
        labels_files (list or None): List of file paths for labels.
        normalize_params (dict): Dictionary of parameters for image normalization (will be faster, if loading from files to pre-normalize).

    Returns:
        tuple: A tuple containing two lists: the batch of images and the batch of labels.
    """
    if data is None:
        lbls = None
        imgs = [io.imread(files[i]) for i in inds]
        imgs = _reshape_norm(imgs, normalize_params=normalize_params)
        if labels_files is not None:
            lbls = [io.imread(labels_files[i])[1:] for i in inds]
    else:
        imgs = [data[i] for i in inds]
        lbls = [labels[i][1:] for i in inds]
    return imgs, lbls

def _reshape_norm_save(files, channels=None, channel_axis=None,
                       normalize_params={"normalize": False}):
    """ not currently used -- normalization happening on each batch if not load_files """
    files_new = []
    for f in trange(files):
        td = io.imread(f)
        if channels is not None:
            td = convert_image(td, channels=channels,
                                          channel_axis=channel_axis)
            td = td.transpose(2, 0, 1)
        if normalize_params["normalize"]:
            td = normalize_img(td, normalize=normalize_params, axis=0)
        fnew = os.path.splitext(str(f))[0] + "_cpnorm.tif"
        io.imsave(fnew, td)
        files_new.append(fnew)
    return files_new
    # else:
    #     train_files = reshape_norm_save(train_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)
    # elif test_files is not None:
    #     test_files = reshape_norm_save(test_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)


def _process_train_test(train_data=None, train_labels=None, train_files=None,
                        train_labels_files=None, train_probs=None, test_data=None,
                        test_labels=None, test_files=None, test_labels_files=None,
                        test_probs=None, load_files=True, min_train_masks=5,
                        compute_flows=False, normalize_params={"normalize": False},
                        channel_axis=None, device=None):
    """
    Process train and test data.

    Args:
        train_data (list or None): List of training data arrays.
        train_labels (list or None): List of training label arrays.
        train_files (list or None): List of training file paths.
        train_labels_files (list or None): List of training label file paths.
        train_probs (ndarray or None): Array of training probabilities.
        test_data (list or None): List of test data arrays.
        test_labels (list or None): List of test label arrays.
        test_files (list or None): List of test file paths.
        test_labels_files (list or None): List of test label file paths.
        test_probs (ndarray or None): Array of test probabilities.
        load_files (bool): Whether to load data from files.
        min_train_masks (int): Minimum number of masks required for training images.
        compute_flows (bool): Whether to compute flows.
        channels (list or None): List of channel indices to use.
        channel_axis (int or None): Axis of channel dimension.
        rgb (bool): Convert training/testing images to RGB.
        normalize_params (dict): Dictionary of normalization parameters.
        device (torch.device): Device to use for computation.

    Returns:
        tuple: A tuple containing the processed train and test data and sampling probabilities and diameters.
    """
    if device == None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else None

    if train_data is not None and train_labels is not None:
        # if data is loaded
        nimg = len(train_data)
        nimg_test = len(test_data) if test_data is not None else None
    else:
        # otherwise use files
        nimg = len(train_files)
        if train_labels_files is None:
            train_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in train_files
            ]
            train_labels_files = [tf for tf in train_labels_files if os.path.exists(tf)]
        if (test_data is not None or
                test_files is not None) and test_labels_files is None:
            test_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in test_files
            ]
            test_labels_files = [tf for tf in test_labels_files if os.path.exists(tf)]
        if not load_files:
            train_logger.info(">>> using files instead of loading dataset")
        else:
            # load all images
            train_logger.info(">>> loading images and labels")
            train_data = [io.imread(train_files[i]) for i in trange(nimg)]
            train_labels = [io.imread(train_labels_files[i]) for i in trange(nimg)]
        nimg_test = len(test_files) if test_files is not None else None
        if load_files and nimg_test:
            test_data = [io.imread(test_files[i]) for i in trange(nimg_test)]
            test_labels = [io.imread(test_labels_files[i]) for i in trange(nimg_test)]

    ### check that arrays are correct size
    if ((train_labels is not None and nimg != len(train_labels)) or
        (train_labels_files is not None and nimg != len(train_labels_files))):
        error_message = "train data and labels not same length"
        train_logger.critical(error_message)
        raise ValueError(error_message)
    if ((test_labels is not None and nimg_test != len(test_labels)) or
        (test_labels_files is not None and nimg_test != len(test_labels_files))):
        train_logger.warning("test data and labels not same length, not using")
        test_data, test_files = None, None
    if train_labels is not None:
        if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
            error_message = "training data or labels are not at least two-dimensional"
            train_logger.critical(error_message)
            raise ValueError(error_message)
        if train_data[0].ndim > 3:
            error_message = "training data is more than three-dimensional (should be 2D or 3D array)"
            train_logger.critical(error_message)
            raise ValueError(error_message)

    ### check that flows are computed
    if train_labels is not None:
        train_labels = dynamics.labels_to_flows(train_labels, files=train_files,
                                                device=device)
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(test_labels, files=test_files,
                                                   device=device)
    elif compute_flows:
        for k in trange(nimg):
            tl = dynamics.labels_to_flows(io.imread(train_labels_files),
                                          files=train_files, device=device)
        if test_files is not None:
            for k in trange(nimg_test):
                tl = dynamics.labels_to_flows(io.imread(test_labels_files),
                                              files=test_files, device=device)

    ### compute diameters
    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    train_logger.info(">>> computing diameters")
    for k in trange(nimg):
        tl = (train_labels[k][0]
              if train_labels is not None else io.imread(train_labels_files[k])[0])
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train < 5] = 5.
    if test_data is not None:
        diam_test = np.array(
            [utils.diameters(test_labels[k][0])[0] for k in trange(len(test_labels))])
        diam_test[diam_test < 5] = 5.
    elif test_labels_files is not None:
        diam_test = np.array([
            utils.diameters(io.imread(test_labels_files[k])[0])[0]
            for k in trange(len(test_labels_files))
        ])
        diam_test[diam_test < 5] = 5.
    else:
        diam_test = None

    ### check to remove training images with too few masks
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            train_logger.warning(
                f"{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set"
            )
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            if train_data is not None:
                train_data = [train_data[i] for i in ikeep]
                train_labels = [train_labels[i] for i in ikeep]
            if train_files is not None:
                train_files = [train_files[i] for i in ikeep]
            if train_labels_files is not None:
                train_labels_files = [train_labels_files[i] for i in ikeep]
            if train_probs is not None:
                train_probs = train_probs[ikeep]
            diam_train = diam_train[ikeep]
            nimg = len(train_data)

    ### normalize probabilities
    train_probs = 1. / nimg * np.ones(nimg,
                                      "float64") if train_probs is None else train_probs
    train_probs /= train_probs.sum()
    if test_files is not None or test_data is not None:
        test_probs = 1. / nimg_test * np.ones(
            nimg_test, "float64") if test_probs is None else test_probs
        test_probs /= test_probs.sum()

    ### reshape and normalize train / test data
    normed = False
    if normalize_params["normalize"]:
        train_logger.info(f">>> normalizing {normalize_params}")
    if train_data is not None:
        train_data = _reshape_norm(train_data, channel_axis=channel_axis,
                                   normalize_params=normalize_params)
        normed = True
    if test_data is not None:
        test_data = _reshape_norm(test_data, channel_axis=channel_axis,
                                  normalize_params=normalize_params)

    return (train_data, train_labels, train_files, train_labels_files, train_probs,
            diam_train, test_data, test_labels, test_files, test_labels_files,
            test_probs, diam_test, normed)


def train_seg(net, train_data=None, train_labels=None, train_files=None,
              train_labels_files=None, train_probs=None, test_data=None,
              test_labels=None, test_files=None, test_labels_files=None,
              test_probs=None, channel_axis=None,
              load_files=True, batch_size=1, learning_rate=5e-5, SGD=False,
              n_epochs=100, weight_decay=0.1, normalize=True, compute_flows=False,
              save_path=None, save_every=100, save_each=False, nimg_per_epoch=None,
              nimg_test_per_epoch=None, rescale=False, scale_range=None, bsize=256,
              min_train_masks=5, model_name=None, class_weights=None,
              pack_to_single_tile=False, pack_k=3, pack_guard=16, pack_stripe_height: int | None = 68,
              pack_stripe_border: int = _PACK_STRIPE_BORDER,
              debug_save_scaled_dir: str | os.PathLike[str] | None = None,
              debug_save_scaled_limit: int = 32,
              batch_photom_augment: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    """
    Train the network with images for segmentation.

    Args:
        net (object): The network model to train. If `net` is a bfloat16 model on MPS, it will be converted to float32 for training. The saved models will be in float32, but the original model will be returned in bfloat16 for consistency. CUDA/CPU will train in bfloat16 if that is the provided net dtype.
        train_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for training. Defaults to None.
        train_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for train_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        train_files (List[str], optional): List of strings - file names for images in train_data (to save flows for future runs). Defaults to None.
        train_labels_files (list or None): List of training label file paths. Defaults to None.
        train_probs (List[float], optional): List of floats - probabilities for each image to be selected during training. Defaults to None.
        test_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for testing. Defaults to None.
        test_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for test_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        test_files (List[str], optional): List of strings - file names for images in test_data (to save flows for future runs). Defaults to None.
        test_labels_files (list or None): List of test label file paths. Defaults to None.
        test_probs (List[float], optional): List of floats - probabilities for each image to be selected during testing. Defaults to None.
        load_files (bool, optional): Boolean - whether to load images and labels from files. Defaults to True.
        batch_size (int, optional): Integer - number of patches to run simultaneously on the GPU. Defaults to 8.
        learning_rate (float or List[float], optional): Float or list/np.ndarray - learning rate for training. Defaults to 0.005.
        n_epochs (int, optional): Integer - number of times to go through the whole training set during training. Defaults to 2000.
        weight_decay (float, optional): Float - weight decay for the optimizer. Defaults to 1e-5.
        momentum (float, optional): Float - momentum for the optimizer. Defaults to 0.9.
        SGD (bool, optional): Deprecated in v4.0.1+ - AdamW always used.
        normalize (bool or dict, optional): Boolean or dictionary - whether to normalize the data. Defaults to True.
        compute_flows (bool, optional): Boolean - whether to compute flows during training. Defaults to False.
        save_path (str, optional): String - where to save the trained model. Defaults to None.
        save_every (int, optional): Integer - save the network every [save_every] epochs. Defaults to 100.
        save_each (bool, optional): Boolean - save the network to a new filename at every [save_each] epoch. Defaults to False.
        nimg_per_epoch (int, optional): Integer - minimum number of images to train on per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): Integer - minimum number of images to test on per epoch. Defaults to None.
        rescale (bool, optional): Boolean - whether or not to rescale images during training. Defaults to True.
        min_train_masks (int, optional): Integer - minimum number of masks an image must have to use in the training set. Defaults to 5.
        model_name (str, optional): String - name of the network. Defaults to None.
        pack_to_single_tile (bool, optional): When True, crops stripes and packs `pack_k` of them
            into a single `bsize×bsize` tile with guard rows so that training matches packed
            inference. Defaults to False.
        pack_k (int, optional): Number of stripes per packed tile when packing is enabled.
            Defaults to 3.
        pack_guard (int, optional): Guard rows inserted between stripes during packing. Defaults to 16.
        pack_stripe_height (int or None, optional): Stripe height (in pixels) cropped from each training
            tile before packing. Defaults to 68. Set to ``None`` (or a non-positive value) to
            auto-select the maximum permissible height given ``pack_k`` and ``guard``.
            Images with height < 100px are automatically routed to the packed path; others use
            standard training with full augmentation.
        pack_stripe_border (int, optional): Pixels of zero padding applied above and below each stripe
            when packing so that stripes stay away from the tile edges. Defaults to 5.
        debug_save_scaled_dir (str or Path, optional): Directory to save PNG copies of the
            scaled tiles immediately before they are fed to the network. When None (default),
            no images are written.
        debug_save_scaled_limit (int, optional): Maximum number of PNGs to save when
            ``debug_save_scaled_dir`` is provided. Defaults to 32.

    Returns:
        tuple: A tuple containing the path to the saved model weights, training losses, and test losses.

    """
    if SGD:
        train_logger.warning("SGD is deprecated, using AdamW instead")

    device = net.device

    original_net_dtype = None
    if device.type == 'mps' and net.dtype == torch.bfloat16:
        # NOTE: this produces a side effect of returning a network that is not of a guaranteed dtype \
        original_net_dtype = torch.bfloat16
        train_logger.warning("Training with bfloat16 on MPS is not supported, using float32 network instead")
        net.dtype = torch.float32
        net.to(torch.float32)

    scale_range = 0.5 if scale_range is None else scale_range

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    pack_height_effective: int | None = None
    pack_guard_effective: int = int(pack_guard)
    pack_border_effective: int = max(0, int(pack_stripe_border))
    if pack_to_single_tile and pack_k > 1:
        pack_height_effective, auto_adjusted = _resolve_pack_stripe_height(
            pack_stripe_height,
            pack_k=pack_k,
            guard=pack_guard,
            bsize=bsize,
            border=pack_border_effective,
        )
        if pack_height_effective is None:
            train_logger.warning(
                "Packing requested but no admissible stripe height fits pack_k=%d guard=%d within bsize=%d; disabling packing.",
                pack_k,
                pack_guard,
                bsize,
            )
            pack_to_single_tile = False
        else:
            # Compute max guard for the resolved stripe height
            pack_guard_effective = compute_max_guard(
                pack_height_effective,
                bsize=bsize,
                pack_k=pack_k,
                border=pack_border_effective,
            )
            train_logger.info(
                "Packed training enabled (dimension-based routing): stripe_h=%d (+%d border), k=%d, guard=%d (max), bsize=%d",
                pack_height_effective,
                pack_border_effective,
                pack_k,
                pack_guard_effective,
                bsize,
            )
            if auto_adjusted:
                train_logger.info(
                    "Auto-adjusted training stripe height to %d (border=%d, pack_k=%d, guard=%d, bsize=%d)",
                    pack_height_effective,
                    pack_border_effective,
                    pack_k,
                    pack_guard_effective,
                    bsize,
                )

    out = _process_train_test(train_data=train_data, train_labels=train_labels,
                              train_files=train_files, train_labels_files=train_labels_files,
                              train_probs=train_probs,
                              test_data=test_data, test_labels=test_labels,
                              test_files=test_files, test_labels_files=test_labels_files,
                              test_probs=test_probs,
                              load_files=load_files, min_train_masks=min_train_masks,
                              compute_flows=compute_flows, channel_axis=channel_axis,
                              normalize_params=normalize_params, device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {"normalize_params": normalize_params, "channel_axis": channel_axis}

    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    env_debug_dir = os.environ.get("CELLPOSE_SAVE_SCALED_DIR")
    if debug_save_scaled_dir is None:
        debug_save_scaled_dir = env_debug_dir
    env_limit = os.environ.get("CELLPOSE_SAVE_SCALED_LIMIT")
    if env_limit is not None:
        try:
            debug_save_scaled_limit = int(env_limit)
        except ValueError:
            pass
    tile_saver: _ScaledTileSaver | None = None
    if debug_save_scaled_dir:
        tile_saver = _ScaledTileSaver(debug_save_scaled_dir, debug_save_scaled_limit)
        train_logger.info(
            "Saving up to %d scaled tiles to %s",
            debug_save_scaled_limit,
            debug_save_scaled_dir,
        )

    if class_weights is not None and isinstance(class_weights, (list, np.ndarray, tuple)):
        class_weights = torch.from_numpy(class_weights).to(device).float()
        print(class_weights)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    # learning rate schedule
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")
    train_logger.info(
        f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)

    t0 = time.time()
    phot_aug = batch_photom_augment
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {filename}")

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    for iepoch in range(n_epochs):
        packed_tiles_epoch = 0
        std_tiles_epoch = 0
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            # choose random images for epoch with probability train_probs
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            # otherwise use all images
            rperm = np.random.permutation(np.arange(0, nimg))
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch] # set learning rate
        net.train()
        for k in range(0, nimg_per_epoch, batch_size):
            kend = min(k + batch_size, nimg_per_epoch)
            inds = rperm[k:kend]
            imgs, lbls = _get_batch(inds, data=train_data, labels=train_labels,
                                    files=train_files, labels_files=train_labels_files,
                                    **kwargs)
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item() if rescale else np.ones(
                len(diams), "float32")

            result = _prepare_dimension_packed_batch(
                imgs,
                lbls,
                rsc,
                scale_range=scale_range,
                bsize=bsize,
                pack_enabled=pack_to_single_tile,
                pack_height=pack_height_effective,
                pack_k=pack_k,
                pack_guard=pack_guard_effective,
                pack_border=pack_border_effective,
                phot_aug=phot_aug,
            )
            if result is None:
                continue
            imgi, lbl, batch_packed_tiles, batch_std_tiles = result
            packed_tiles_epoch += batch_packed_tiles
            std_tiles_epoch += batch_std_tiles
            if tile_saver is not None:
                tile_saver.save_batch(
                    imgi, phase="train", epoch=iepoch, batch_index=k // batch_size
                )

            # network and loss optimization
            X = torch.from_numpy(imgi).to(device)
            lbl = torch.from_numpy(lbl).to(device)

            if X.dtype != net.dtype:
                X = X.to(net.dtype)
                lbl = lbl.to(net.dtype)

            y = net(X)[0]
            loss = _loss_fn_seg(lbl, y, device)
            if y.shape[1] > 3:
                loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
                loss += loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_loss *= len(imgi)

            # keep track of average training loss across epochs
            lavg += train_loss
            nsum += len(imgi)
            # per epoch training loss
            train_losses[iepoch] += train_loss
        train_losses[iepoch] /= nimg_per_epoch
        if pack_to_single_tile and pack_k > 1 and pack_height_effective is not None:
            total_epoch_tiles = packed_tiles_epoch + std_tiles_epoch
            if total_epoch_tiles > 0:
                train_logger.info(
                    "Epoch %d: thin_tiles=%d (H<%d, packed), large_tiles=%d (H≥%d, standard) — stripe_h=%d (+%d border), k=%d, guard=%d",
                    iepoch + 1,
                    packed_tiles_epoch,
                    _THIN_HEIGHT_PX,
                    std_tiles_epoch,
                    _THIN_HEIGHT_PX,
                    pack_height_effective,
                    pack_border_effective,
                    pack_k,
                    pack_guard_effective,
                )

        if iepoch == 5 or iepoch % 10 == 0:
            lavgt = 0.
            if test_data is not None or test_files is not None:
                np.random.seed(42)
                if nimg_test != nimg_test_per_epoch:
                    rperm = np.random.choice(np.arange(0, nimg_test),
                                             size=(nimg_test_per_epoch,), p=test_probs)
                else:
                    rperm = np.random.permutation(np.arange(0, nimg_test))
                packed_tiles_epoch = 0
                std_tiles_epoch = 0
                for ibatch in range(0, len(rperm), batch_size):
                    with torch.no_grad():
                        net.eval()
                        inds = rperm[ibatch:ibatch + batch_size]
                        imgs, lbls = _get_batch(inds, data=test_data,
                                                labels=test_labels, files=test_files,
                                                labels_files=test_labels_files,
                                                **kwargs)
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = diams / net.diam_mean.item() if rescale else np.ones(
                            len(diams), "float32")

                        result = _prepare_dimension_packed_batch(
                            imgs,
                            lbls,
                            rsc,
                            scale_range=scale_range,
                            bsize=bsize,
                            pack_enabled=pack_to_single_tile,
                            pack_height=pack_height_effective,
                            pack_k=pack_k,
                            pack_guard=pack_guard_effective,
                            pack_border=pack_border_effective,
                        )
                        if result is None:
                            continue
                        imgi, lbl, batch_packed_tiles, batch_std_tiles = result
                        packed_tiles_epoch += batch_packed_tiles
                        std_tiles_epoch += batch_std_tiles
                        if tile_saver is not None:
                            tile_saver.save_batch(
                                imgi, phase="eval", epoch=iepoch, batch_index=ibatch // batch_size
                            )

                        X = torch.from_numpy(imgi).to(device)
                        lbl = torch.from_numpy(lbl).to(device)

                        if X.dtype != net.dtype:
                            X = X.to(net.dtype)
                            lbl = lbl.to(net.dtype)

                        y = net(X)[0]
                        loss = _loss_fn_seg(lbl, y, device)
                        if y.shape[1] > 3:
                            loss3 = _loss_fn_class(lbl, y, class_weights=class_weights)
                            loss += loss3
                        test_loss = loss.item()
                        test_loss *= len(imgi)
                        lavgt += test_loss
                lavgt /= len(rperm)
                if pack_to_single_tile and pack_k > 1 and pack_height_effective is not None:
                    total_epoch_tiles = packed_tiles_epoch + std_tiles_epoch
                    if total_epoch_tiles > 0:
                            train_logger.info(
                                "Eval: thin_tiles=%d (H<%d, packed), large_tiles=%d (H≥%d, standard) — stripe_h=%d (+%d border), k=%d, guard=%d",
                                packed_tiles_epoch,
                                _THIN_HEIGHT_PX,
                                std_tiles_epoch,
                                _THIN_HEIGHT_PX,
                                pack_height_effective,
                                pack_border_effective,
                                pack_k,
                                pack_guard_effective,
                            )
                test_losses[iepoch] = lavgt
            lavg /= nsum
            train_logger.info(
                f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
            )
            lavg, nsum = 0, 0

        if iepoch == n_epochs - 1 or (iepoch % save_every == 0 and iepoch != 0):
            if save_each and iepoch != n_epochs - 1:  #separate files as model progresses
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
            else:
                filename0 = filename
            train_logger.info(f"saving network parameters to {filename0}")
            net.save_model(filename0)

    net.save_model(filename)

    if original_net_dtype is not None:
        net.dtype = original_net_dtype
        net.to(original_net_dtype)

    return filename, train_losses, test_losses
