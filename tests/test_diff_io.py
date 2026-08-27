from types import SimpleNamespace

import numpy as np

from cellpose.gui import io


class _Count:

    def __init__(self) -> None:
        self.value = 0

    def get(self) -> int:
        return self.value

    def set(self, value: int) -> None:
        self.value = int(value)


def _parent(events: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        load_3D=False,
        autoloadMasks=SimpleNamespace(isChecked=lambda: False),
        loaded=False,
        ncells=_Count(),
        colormap=np.zeros((4, 4), dtype=np.uint8),
        reset=lambda: events.append("reset"),
        draw_layer=lambda: events.append("draw"),
        clear_all=lambda: events.append("clear"),
        enable_buttons=lambda: events.append("enable"),
        update_layer=lambda: events.append("update"),
        _diff_cache_before_image_change=lambda: events.append("cache"),
        _diff_restore_after_image_load=lambda: events.append("restore"),
    )


def test_seg_load_calls_diff_restore_once(tmp_path, monkeypatch):
    events: list[str] = []
    parent = _parent(events)
    image_path = tmp_path / "sample.tif"
    image_path.touch()
    seg_path = tmp_path / "sample_seg.npy"
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    masks = np.zeros((1, 4, 5), dtype=np.uint16)
    np.save(seg_path, {"masks": masks, "outlines": masks.copy()})

    monkeypatch.setattr(io, "imread_2D", lambda _: image)

    def initialize(parent, loaded_image, load_3D=False):
        parent.stack = loaded_image[np.newaxis, ...]
        parent.NZ = 1

    monkeypatch.setattr(io, "_initialize_images", initialize)
    monkeypatch.setattr(io, "_masks_to_gui", lambda *args, **kwargs: None)

    io._load_image(parent, filename=str(image_path), load_seg=True)

    assert events.count("cache") == 1
    assert events.count("restore") == 1
    assert events[-1] == "restore"
    assert parent.loaded


def test_autoload_masks_does_not_override_seg_npy(tmp_path, monkeypatch):
    events: list[str] = []
    parent = _parent(events)
    parent.autoloadMasks = SimpleNamespace(isChecked=lambda: True)
    image_path = tmp_path / "sample.tif"
    image_path.touch()
    (tmp_path / "sample_seg.npy").touch()
    (tmp_path / "sample_masks.tif").touch()
    monkeypatch.setattr(
        io, "imread_2D", lambda filename: np.zeros((4, 5, 3), dtype=np.uint8)
    )

    monkeypatch.setattr(
        io, "_load_seg", lambda *args, **kwargs: events.append("load_seg")
    )
    monkeypatch.setattr(
        io, "_load_masks", lambda *args, **kwargs: events.append("load_masks")
    )

    io._load_image(parent, filename=str(image_path), load_seg=True)

    assert "load_seg" in events
    assert "load_masks" not in events


def test_failed_seg_load_does_not_restore_diff(tmp_path):
    events: list[str] = []
    parent = _parent(events)

    io._load_seg(parent, filename=str(tmp_path / "missing_seg.npy"))

    assert "restore" not in events
    assert not parent.loaded


def test_seg_without_image_does_not_restore_diff(tmp_path):
    events: list[str] = []
    parent = _parent(events)
    seg_path = tmp_path / "orphan_seg.npy"
    masks = np.zeros((1, 4, 5), dtype=np.uint16)
    np.save(seg_path, {"masks": masks, "outlines": masks.copy()})

    io._load_seg(parent, filename=str(seg_path))

    assert "restore" not in events


def test_direct_seg_load_caches_outgoing_state(tmp_path, monkeypatch):
    events: list[str] = []
    parent = _parent(events)
    image_path = tmp_path / "sample.tif"
    image_path.touch()
    seg_path = tmp_path / "sample_seg.npy"
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    masks = np.zeros((1, 4, 5), dtype=np.uint16)
    np.save(
        seg_path,
        {
            "filename": str(image_path),
            "masks": masks,
            "outlines": masks.copy(),
        },
    )

    monkeypatch.setattr(io, "imread_2D", lambda _: image)
    monkeypatch.setattr(
        io,
        "_initialize_images",
        lambda parent, image, load_3D=False: setattr(parent, "stack", image),
    )
    monkeypatch.setattr(io, "_masks_to_gui", lambda *args, **kwargs: None)

    io._load_seg(parent, filename=str(seg_path))

    assert events.count("cache") == 1
    assert events.count("restore") == 1
    assert events.index("cache") < events.index("reset")
