from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cellpose.gui import guiortho, io


def test_main_view_records_its_display_transform():
    image = np.array(
        [
            [[2, 2, 2], [6, 6, 6]],
            [[8, 8, 8], [10, 10, 10]],
        ],
        dtype=np.uint16,
    )
    parent = SimpleNamespace(
        load_3D=False,
        restore=None,
        autobtn=SimpleNamespace(isChecked=lambda: False),
        clear_all=lambda: None,
        compute_scale=lambda: None,
    )

    io._initialize_images(parent, image, load_3D=False)

    assert parent._display_range == (2.0, 10.0)
    np.testing.assert_allclose(parent.stack[0], (image - 2) * (255 / 8))


def test_ortho_stack_uses_main_display_transform(tmp_path, monkeypatch):
    paths = [tmp_path / "sample_z0.tif", tmp_path / "sample_z1.tif"]
    for path in paths:
        path.touch()

    raw_images = {}
    for path, value in zip(paths, (2, 3)):
        image = np.zeros((4, 5, 3), dtype=np.uint16)
        image[..., 0] = value
        raw_images[path.name] = image
    parent = SimpleNamespace(
        load_3D=False,
        loaded=False,
        ortho_nz=0,
        nchan=3,
        orthobtn=SimpleNamespace(isChecked=lambda: False),
        enable_buttons=lambda: None,
        update_plot=lambda: None,
        update_layer=lambda: None,
        update_scale=lambda: None,
    )

    def load_main(parent, filename, load_seg, load_3D):
        parent.loaded = True
        parent.filename = filename
        parent._display_range = (0.0, 10.0)
        parent.stack = raw_images[Path(filename).name][None].astype(np.float32) * 25.5
        parent.Ly, parent.Lx = 4, 5

    def load_plane(filename):
        return raw_images[Path(filename).name]

    monkeypatch.setattr(guiortho.io, "_load_image", load_main)
    monkeypatch.setattr(guiortho.io, "imread_2D", load_plane)

    guiortho.MainW_ortho2D._load_image_ortho2D(
        parent, filename=str(paths[1]), load_seg=False
    )

    np.testing.assert_allclose(parent.stack_ortho[parent.zc_ortho], parent.stack[0])
    np.testing.assert_allclose(parent.stack_ortho[0, ..., 0], 51.0)
    np.testing.assert_allclose(parent.stack_ortho[0, ..., 1:], 0.0)
