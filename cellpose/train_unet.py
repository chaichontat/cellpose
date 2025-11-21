import logging
import datetime
import os
import time
import json
from pathlib import Path
from .schedules import build_lr_schedule

import numpy as np
import torch
from torch import nn
from tqdm import trange

from cellpose import dynamics, io, models, transforms, utils
from cellpose.contrib.pack_utils import compute_max_guard
from cellpose.transforms import normalize_img, random_rotate_and_resize
from cellpose.train import (
    _prepare_dimension_packed_batch as _prepare_packed_batch,
    _resolve_pack_stripe_height as _resolve_pack_height,
)

train_logger = logging.getLogger(__name__)


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
    target_dtype = y.dtype
    veci = 5.0 * torch.from_numpy(lbl[:, 1:]).to(device=device, dtype=target_dtype)
    loss = criterion(y[:, :2], veci)
    loss /= 2.0
    target_mask = torch.from_numpy(lbl[:, 0] > 0.5).to(
        device=device,
        dtype=target_dtype,
    )
    loss2 = criterion2(y[:, -1], target_mask)
    loss = loss + loss2
    return loss


def _get_batch(
    inds,
    data=None,
    labels=None,
    files=None,
    labels_files=None,
    channels=None,
    channel_axis=None,
    rgb=False,
    normalize_params={"normalize": False},
):
    """
    Get a batch of images and labels.

    Args:
        inds (list): List of indices indicating which images and labels to retrieve.
        data (list or None): List of image data. If None, images will be loaded from files.
        labels (list or None): List of label data. If None, labels will be loaded from files.
        files (list or None): List of file paths for images.
        labels_files (list or None): List of file paths for labels.
        channels (list or None): List of channel indices to extract from images.
        channel_axis (int or None): Axis along which the channels are located.
        normalize_params (dict): Dictionary of parameters for image normalization (will be faster, if loading from files to pre-normalize).

    Returns:
        tuple: A tuple containing two lists: the batch of images and the batch of labels.
    """
    if data is None:
        lbls = None
        imgs = [io.imread(files[i]) for i in inds]
        imgs = _reshape_norm(
            imgs,
            channels=channels,
            channel_axis=channel_axis,
            rgb=rgb,
            normalize_params=normalize_params,
        )
        if labels_files is not None:
            lbls = [io.imread(labels_files[i])[1:] for i in inds]
    else:
        imgs = [data[i] for i in inds]
        lbls = [labels[i][1:] for i in inds]
    return imgs, lbls


def pad_to_rgb(img):
    if img.ndim == 2 or np.ptp(img[1]) < 1e-3:
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        img = np.tile(img[:1], (3, 1, 1))
    elif img.shape[0] < 3:
        nc, Ly, Lx = img.shape
        # randomly flip channels
        if np.random.rand() > 0.5:
            img = img[::-1]
        # randomly insert blank channel
        ic = np.random.randint(3)
        img = np.insert(img, ic, np.zeros((3 - nc, Ly, Lx), dtype=img.dtype), axis=0)
    return img


def convert_to_rgb(img):
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
        img = np.tile(img, (3, 1, 1))
    elif img.shape[0] < 3:
        img = img.mean(axis=0, keepdims=True)
        img = transforms.normalize99(img)
        img = np.tile(img, (3, 1, 1))
    return img


def _reshape_norm(
    data,
    channels=None,
    channel_axis=None,
    rgb=False,
    normalize_params={"normalize": False},
):
    """
    Reshapes and normalizes the input data.

    Args:
        data (list): List of input data.
        channels (int or list, optional): Number of channels or list of channel indices to keep. Defaults to None.
        channel_axis (int, optional): Axis along which the channels are located. Defaults to None.
        normalize_params (dict, optional): Dictionary of normalization parameters. Defaults to {"normalize": False}.

    Returns:
        list: List of reshaped and normalized data.
    """
    if channels is not None or channel_axis is not None:
        data = [
            transforms.convert_image_unet(
                td, channels=channels, channel_axis=channel_axis
            )
            for td in data
        ]
        data = [td.transpose(2, 0, 1) for td in data]
    if normalize_params["normalize"]:
        data = [
            transforms.normalize_img(td, normalize=normalize_params, axis=0)
            for td in data
        ]
    if rgb:
        data = [pad_to_rgb(td) for td in data]
    return data


def _reshape_norm_save(
    files, channels=None, channel_axis=None, normalize_params={"normalize": False}
):
    """not currently used -- normalization happening on each batch if not load_files"""
    files_new = []
    for f in trange(files):
        td = io.imread(f)
        if channels is not None:
            td = transforms.convert_image(
                td, channels=channels, channel_axis=channel_axis
            )
            td = td.transpose(2, 0, 1)
        if normalize_params["normalize"]:
            td = transforms.normalize_img(td, normalize=normalize_params, axis=0)
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


def _process_train_test(
    train_data=None,
    train_labels=None,
    train_files=None,
    train_labels_files=None,
    train_probs=None,
    test_data=None,
    test_labels=None,
    test_files=None,
    test_labels_files=None,
    test_probs=None,
    load_files=True,
    min_train_masks=5,
    compute_flows=False,
    channels=None,
    channel_axis=None,
    rgb=False,
    normalize_params={"normalize": False},
    device=None,
):
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
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else None
        )

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
        if (
            test_data is not None or test_files is not None
        ) and test_labels_files is None:
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
    if (train_labels is not None and nimg != len(train_labels)) or (
        train_labels_files is not None and nimg != len(train_labels_files)
    ):
        error_message = "train data and labels not same length"
        train_logger.critical(error_message)
        raise ValueError(error_message)
    if (test_labels is not None and nimg_test != len(test_labels)) or (
        test_labels_files is not None and nimg_test != len(test_labels_files)
    ):
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
        train_labels = dynamics.labels_to_flows(
            train_labels, files=train_files, device=device
        )
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(
                test_labels, files=test_files, device=device
            )
    elif compute_flows:
        for k in trange(nimg):
            tl = dynamics.labels_to_flows(
                io.imread(train_labels_files), files=train_files, device=device
            )
        if test_files is not None:
            for k in trange(nimg_test):
                tl = dynamics.labels_to_flows(
                    io.imread(test_labels_files), files=test_files, device=device
                )

    ### compute diameters
    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    train_logger.info(">>> computing diameters")
    for k in trange(nimg):
        tl = (
            train_labels[k][0]
            if train_labels is not None
            else io.imread(train_labels_files[k])[0]
        )
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train < 5] = 5.0
    if test_data is not None:
        diam_test = np.array([
            utils.diameters(test_labels[k][0])[0] for k in trange(len(test_labels))
        ])
        diam_test[diam_test < 5] = 5.0
    elif test_labels_files is not None:
        diam_test = np.array([
            utils.diameters(io.imread(test_labels_files[k])[0])[0]
            for k in trange(len(test_labels_files))
        ])
        diam_test[diam_test < 5] = 5.0
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
    train_probs = (
        1.0 / nimg * np.ones(nimg, "float64") if train_probs is None else train_probs
    )
    train_probs /= train_probs.sum()
    if test_files is not None or test_data is not None:
        test_probs = (
            1.0 / nimg_test * np.ones(nimg_test, "float64")
            if test_probs is None
            else test_probs
        )
        test_probs /= test_probs.sum()

    ### reshape and normalize train / test data
    normed = False
    if channels is not None or normalize_params["normalize"]:
        if channels:
            train_logger.info(f">>> using channels {channels}")
        if normalize_params["normalize"]:
            train_logger.info(f">>> normalizing {normalize_params}")
        if train_data is not None:
            train_data = _reshape_norm(
                train_data,
                channels=channels,
                channel_axis=channel_axis,
                rgb=rgb,
                normalize_params=normalize_params,
            )
            normed = True
        if test_data is not None:
            test_data = _reshape_norm(
                test_data,
                channels=channels,
                channel_axis=channel_axis,
                rgb=rgb,
                normalize_params=normalize_params,
            )

    return (
        train_data,
        train_labels,
        train_files,
        train_labels_files,
        train_probs,
        diam_train,
        test_data,
        test_labels,
        test_files,
        test_labels_files,
        test_probs,
        diam_test,
        normed,
    )


def train_seg(
    net,
    train_data=None,
    train_labels=None,
    train_files=None,
    train_labels_files=None,
    train_probs=None,
    test_data=None,
    test_labels=None,
    test_files=None,
    test_labels_files=None,
    test_probs=None,
    load_files=True,
    batch_size=8,
    learning_rate=0.005,
    n_epochs=2000,
    weight_decay=1e-5,
    momentum=0.9,
    SGD=False,
    channels=None,
    channel_axis=None,
    rgb=False,
    normalize=True,
    compute_flows=False,
    save_path=None,
    save_every=100,
    save_each=False,
    nimg_per_epoch=None,
    nimg_test_per_epoch=None,
    rescale=True,
    scale_range=None,
    bsize=224,
    min_train_masks=5,
    model_name=None,
    batch_photom_augment=None,
    pack_to_single_tile=False,
    pack_k=3,
    pack_guard=16,
    pack_stripe_height=68,
    pack_stripe_border=0,
    lr_schedule: str = "cosine",
    warmup_epochs: int = 10,
    cosine_hold_epochs: int | None = None,
    cosine_min_lr: float | None = None,
):
    """
    Train the network with images for segmentation.

    Args:
        net (object): The network model to train.
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
        SGD (bool, optional): Boolean - whether to use SGD as optimization instead of RAdam. Defaults to False.
        channels (List[int], optional): List of ints - channels to use for training. Defaults to None.
        channel_axis (int, optional): Integer - axis of the channel dimension in the input data. Defaults to None.
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
        batch_photom_augment (Callable, optional): Photometric augmentation applied to the batch after geometric transforms. Defaults to None.
        pack_to_single_tile (bool, optional): Enable stripe packing for thin tiles. Defaults to False.
        pack_k (int, optional): Maximum number of stripes to pack into one tile. Defaults to 3.
        pack_guard (int, optional): Guard pixels between packed stripes. Defaults to 16.
        pack_stripe_height (int, optional): Preferred stripe height before packing. Defaults to 68.
        pack_stripe_border (int, optional): Border padding (pixels) above/below each stripe. Defaults to 0.

    Returns:
        tuple: A tuple containing the path to the saved model weights, training losses, and test losses.

    """
    device = net.device

    # Enforce BF16 training to align with SAM backend defaults.
    if not hasattr(net, "dtype"):
        first_param = next(net.parameters(), None)
        inferred_dtype = (
            first_param.dtype if first_param is not None else torch.bfloat16
        )
        net.dtype = inferred_dtype
    if net.dtype != torch.bfloat16:
        net.to(dtype=torch.bfloat16)
        net.dtype = torch.bfloat16
    else:
        net.to(dtype=torch.bfloat16)

    scale_range0 = 0.5 if rescale else 1.0
    scale_range = scale_range if scale_range is not None else scale_range0

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    out = _process_train_test(
        train_data=train_data,
        train_labels=train_labels,
        train_files=train_files,
        train_labels_files=train_labels_files,
        train_probs=train_probs,
        test_data=test_data,
        test_labels=test_labels,
        test_files=test_files,
        test_labels_files=test_labels_files,
        test_probs=test_probs,
        load_files=load_files,
        min_train_masks=min_train_masks,
        compute_flows=compute_flows,
        channels=channels,
        channel_axis=channel_axis,
        rgb=rgb,
        normalize_params=normalize_params,
        device=net.device,
    )
    (
        train_data,
        train_labels,
        train_files,
        train_labels_files,
        train_probs,
        diam_train,
        test_data,
        test_labels,
        test_files,
        test_labels_files,
        test_probs,
        diam_test,
        normed,
    ) = out
    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {
            "normalize_params": normalize_params,
            "channels": channels,
            "channel_axis": channel_axis,
            "rgb": rgb,
        }

    net.diam_labels.data = torch.tensor(
        [diam_train.mean()],
        device=device,
        dtype=net.dtype,
    )

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = (
        nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch
    )

    # learning rate schedule
    LR = build_lr_schedule(
        n_epochs,
        learning_rate,
        schedule=lr_schedule,
        warmup_epochs=warmup_epochs,
        hold_epochs=cosine_hold_epochs,
        min_lr=cosine_min_lr,
    )

    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")

    if not SGD:
        train_logger.info(
            f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}, lr_schedule={lr_schedule}, warmup_epochs={warmup_epochs}, hold_epochs={cosine_hold_epochs}, final_lr={LR[-1]:0.5f}"
        )
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        train_logger.info(
            f">>> SGD, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}, momentum={momentum:0.3f}"
        )
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {filename}")

    # training report setup
    logs_dir = save_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    start_ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    report_path = logs_dir / f"{model_name}-{start_ts}.jsonl"
    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            start_payload = {
                "type": "start",
                "started_at": start_ts,
                "model_name": model_name,
                "model_path": str(filename),
                "device": str(device),
                "optimizer": "SGD" if SGD else "AdamW",
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "n_epochs": int(n_epochs),
                "batch_size": int(batch_size),
                "bsize": int(bsize),
                "lr_schedule": lr_schedule,
                "warmup_epochs": int(warmup_epochs),
                "cosine_hold_epochs": None if cosine_hold_epochs is None else int(cosine_hold_epochs),
                "cosine_min_lr": None if cosine_min_lr is None else float(cosine_min_lr),
                "final_lr": float(LR[-1]) if len(LR) else None,
                "train_files": list(map(str, train_files)) if train_files is not None else None,
                "test_files": list(map(str, test_files)) if test_files is not None else None,
            }
            rf.write(json.dumps(start_payload, ensure_ascii=False) + "\n")
        train_logger.info(f"Training report: {report_path}")
    except Exception as _e:
        train_logger.warning(f"could not create training report at {report_path}: {_e}")

    phot_aug = batch_photom_augment

    pack_guard_effective = max(0, int(pack_guard))
    pack_border_effective = max(0, int(pack_stripe_border))
    pack_height_effective: int | None = None

    if pack_to_single_tile and pack_k > 1:
        pack_height_effective, auto_adjusted = _resolve_pack_height(
            pack_stripe_height,
            pack_k=pack_k,
            guard=pack_guard_effective,
            bsize=bsize,
            border=pack_border_effective,
        )
        if pack_height_effective is None:
            train_logger.warning(
                "Packing requested but no valid stripe height fits pack_k=%d guard=%d within bsize=%d; disabling packing.",
                pack_k,
                pack_guard_effective,
                bsize,
            )
            pack_to_single_tile = False
        else:
            pack_guard_effective = compute_max_guard(
                pack_height_effective,
                bsize=bsize,
                pack_k=pack_k,
                border=pack_border_effective,
            )
            train_logger.info(
                "Packed training enabled: stripe_h=%d (+%d border), k=%d, guard=%d, bsize=%d",
                pack_height_effective,
                pack_border_effective,
                pack_k,
                pack_guard_effective,
                bsize,
            )
            if auto_adjusted:
                train_logger.info(
                    "Auto-adjusted stripe height to %d for packing (requested=%s).",
                    pack_height_effective,
                    pack_stripe_height,
                )

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    for iepoch in range(n_epochs):
        packed_tiles_epoch = 0
        std_tiles_epoch = 0
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            # choose random images for epoch with probability train_probs
            rperm = np.random.choice(
                np.arange(0, nimg), size=(nimg_per_epoch,), p=train_probs
            )
        else:
            # otherwise use all images
            rperm = np.random.permutation(np.arange(0, nimg))
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]  # set learning rate
        net.train()
        for k in range(0, nimg_per_epoch, batch_size):
            kend = min(k + batch_size, nimg_per_epoch)
            inds = rperm[k:kend]
            imgs, lbls = _get_batch(
                inds,
                data=train_data,
                labels=train_labels,
                files=train_files,
                labels_files=train_labels_files,
                **kwargs,
            )
            diams = np.array([diam_train[i] for i in inds])
            rsc = (
                diams / net.diam_mean.item()
                if rescale
                else np.ones(len(diams), "float32")
            )
            result = _prepare_packed_batch(
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
            # network and loss optimization
            X = torch.from_numpy(imgi).to(device=device, dtype=net.dtype)
            y = net(X)[0]
            loss = _loss_fn_seg(lbl, y, device)
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

        if (
            pack_to_single_tile
            and pack_k > 1
            and pack_height_effective is not None
            and (packed_tiles_epoch + std_tiles_epoch) > 0
        ):
            train_logger.info(
                "Epoch %d packing stats — packed_tiles=%d, standard_tiles=%d (stripe_h=%d, border=%d, guard=%d, k=%d)",
                iepoch + 1,
                packed_tiles_epoch,
                std_tiles_epoch,
                pack_height_effective,
                pack_border_effective,
                pack_guard_effective,
                pack_k,
            )

        # Log metrics every epoch instead of sparsely
        if True:
            lavgt = 0.0
            if test_data is not None or test_files is not None:
                np.random.seed(42)
                if nimg_test != nimg_test_per_epoch:
                    rperm = np.random.choice(
                        np.arange(0, nimg_test),
                        size=(nimg_test_per_epoch,),
                        p=test_probs,
                    )
                else:
                    rperm = np.random.permutation(np.arange(0, nimg_test))
                packed_tiles_eval = 0
                std_tiles_eval = 0
                for ibatch in range(0, len(rperm), batch_size):
                    with torch.no_grad():
                        net.eval()
                        inds = rperm[ibatch : ibatch + batch_size]
                        imgs, lbls = _get_batch(
                            inds,
                            data=test_data,
                            labels=test_labels,
                            files=test_files,
                            labels_files=test_labels_files,
                            **kwargs,
                        )
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = (
                            diams / net.diam_mean.item()
                            if rescale
                            else np.ones(len(diams), "float32")
                        )
                        result = _prepare_packed_batch(
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
                        packed_tiles_eval += batch_packed_tiles
                        std_tiles_eval += batch_std_tiles
                        X = torch.from_numpy(imgi).to(device=device, dtype=net.dtype)
                        y = net(X)[0]
                        loss = _loss_fn_seg(lbl, y, device)
                        test_loss = loss.item()
                        test_loss *= len(imgi)
                        lavgt += test_loss
                lavgt /= len(rperm)
                if (
                    pack_to_single_tile
                    and pack_k > 1
                    and pack_height_effective is not None
                    and (packed_tiles_eval + std_tiles_eval) > 0
                ):
                    train_logger.info(
                        "Eval packing stats — packed_tiles=%d, standard_tiles=%d (stripe_h=%d, border=%d, guard=%d, k=%d)",
                        packed_tiles_eval,
                        std_tiles_eval,
                        pack_height_effective,
                        pack_border_effective,
                        pack_guard_effective,
                        pack_k,
                    )
                test_losses[iepoch] = lavgt
            lavg /= nsum
            elapsed = time.time() - t0
            train_logger.info(
                f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {elapsed:.2f}s"
            )
            # append to training report each epoch (JSONL)
            try:
                with open(report_path, "a", encoding="utf-8") as rf:
                    rf.write(
                        json.dumps(
                            {
                                "type": "epoch",
                                "epoch": int(iepoch + 1),
                                "lr": float(LR[iepoch]),
                                "train_loss": float(lavg),
                                "test_loss": float(lavgt),
                                "elapsed_seconds": float(elapsed),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception as _e:
                train_logger.warning(f"could not update training report at {report_path}: {_e}")
            lavg, nsum = 0, 0

        if iepoch == n_epochs - 1 or (iepoch % save_every == 0 and iepoch != 0):
            if (
                save_each and iepoch != n_epochs - 1
            ):  # separate files as model progresses
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
            else:
                filename0 = filename
            train_logger.info(f"saving network parameters to {filename0}")
            net.save_model(filename0)

    net.save_model(filename)
    # write summary footer with total training time (JSONL)
    try:
        total_elapsed = time.time() - t0
        with open(report_path, "a", encoding="utf-8") as rf:
            rf.write(
                json.dumps(
                    {
                        "type": "end",
                        "ended_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                        "total_elapsed_seconds": float(total_elapsed),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass

    return filename, train_losses, test_losses

