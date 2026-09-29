from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cellpose.gui import guiortho


def test_ortho_loader_groups_trailing_axis_coordinate(tmp_path, monkeypatch):
    names = [
        "sample_orthozx-z440-x3419-y3899.tif",
        "sample_orthozx-z440-x3419-y3901.tif",
        "sample_orthozx-z440-x3419-y3903.tif",
        "sample_orthozx-z440-x3420-y3901.tif",
    ]
    for name in names:
        (tmp_path / name).touch()

    target = tmp_path / names[1]
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
        parent._display_range = (0.0, 0.0)
        parent.stack = np.zeros((1, 4, 5, 3), dtype=np.float32)
        parent.Ly, parent.Lx = 4, 5

    monkeypatch.setattr(guiortho.io, "_load_image", load_main)
    monkeypatch.setattr(
        guiortho.io, "imread_2D", lambda _: np.zeros((4, 5, 3), dtype=np.uint16)
    )

    guiortho.MainW_ortho2D._load_image_ortho2D(
        parent, filename=str(target), load_seg=False
    )

    assert parent.ortho_used_z_indices == [3899, 3901, 3903]
    assert parent.zc_ortho == 1
    assert [Path(path).name for path in parent.ortho_files_sorted] == names[:3]
