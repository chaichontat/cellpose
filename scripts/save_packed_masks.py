from pathlib import Path

import numpy as np
from tifffile import imread

from cellpose import models
from cellpose.contrib.pack_utils import forward_packed_2d


class PackedCellposeModel(models.CellposeModel):
    def __init__(self, *args, pack_k: int = 3, pack_guard: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self._pack_k = pack_k
        self._pack_guard = pack_guard

    def _run_net(
        self,
        x: np.ndarray,
        augment: bool = False,
        batch_size: int = 8,
        tile_overlap: float = 0.0,
        bsize: int = 256,
        anisotropy: float = 1.0,
        do_3D: bool = False,
    ):
        if do_3D:
            return super()._run_net(
                x,
                augment=augment,
                batch_size=batch_size,
                tile_overlap=tile_overlap,
                bsize=bsize,
                anisotropy=anisotropy,
                do_3D=do_3D,
            )
        yf, styles = forward_packed_2d(
            self.net,
            x,
            bsize=bsize,
            batch_size=batch_size,
            augment=augment,
            tile_overlap=tile_overlap,
            pack_k=self._pack_k,
            guard=self._pack_guard,
            return_stats=False,
        )
        cellprob = yf[..., -1]
        dP = yf[..., -3:-1].transpose((3, 0, 1, 2))
        if yf.shape[-1] > 3:
            styles = yf[..., :-3]
        styles = styles.squeeze()
        return dP, cellprob, styles


def prepare_images(path: Path, target_hw=(68, 256), count: int = 10):
    arr = imread(path.as_posix())
    if arr.ndim == 4:
        arr2d = arr[5, :, ::2, ::2]
    elif arr.ndim == 3:
        arr2d = arr[:, ::2, ::2]
    else:
        raise ValueError(f"Unexpected image ndim {arr.ndim}")
    plane = arr2d.astype(np.float32)
    Ly, Lx = target_hw
    C, H, W = plane.shape
    pad_y = max(0, Ly - H)
    pad_x = max(0, Lx - W)
    if pad_y or pad_x:
        plane = np.pad(plane, (0, (pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)))
        C, H, W = plane.shape
    y0 = max(0, (H - Ly) // 2)
    x0 = max(0, (W - Lx) // 2)
    return plane[:, y0:y0 + Ly, x0:x0 + Lx]



def run_model(model, images):
    kwargs = dict(
        do_3D=False,
        channel_axis=0,
        z_axis=None,
        compute_masks=True,
        bsize=256,
        batch_size=1,
        augment=False,
        tile_overlap=0.0,
    )
    return model.eval(images, **kwargs)


def main():
    path = Path("/working/20251001_JaxA3_Coro11/analysis/deconv/registered--3r+pi/reg-0076.tif")
    images = prepare_images(path)
    print("Prepared", len(images), "images with shape", images[0].shape, flush=True)

    baseline = models.CellposeModel(gpu=True, pretrained_model="/working/cellpose-training/models/embryonicsam")
    packed = PackedCellposeModel(gpu=True, pretrained_model="/working/cellpose-training/models/embryonicsam")

    masks_b, _, _ = run_model(baseline, images)
    print("Baseline done", flush=True)
    masks_p, _, _ = run_model(packed, images)
    print("Packed done", flush=True)

    out_dir = Path("/home/chaichontat/mask_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    images_np = np.asarray(images, dtype=np.float32)
    np.save(out_dir / "images.npy", images_np)
    np.save(out_dir / "images_channel0.npy", images_np)
    np.save(out_dir / "masks_baseline.npy", np.asarray(masks_b))
    np.save(out_dir / "masks_packed.npy", np.asarray(masks_p))
    print("Results saved to", out_dir, flush=True)


if __name__ == "__main__":
    main()
