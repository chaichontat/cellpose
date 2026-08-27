import numpy as np

from cellpose.gui import guiortho


def test_read_ortho_mask_plane_uses_autoload_naming(tmp_path, monkeypatch):
    image = tmp_path / "sample-001.png"
    mask = tmp_path / "sample-001_masks.png"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    labels = np.zeros((6, 8), dtype=np.uint16)
    labels[2:5, 3:7] = 4

    monkeypatch.setattr(guiortho, "imread", lambda filename: labels)

    loaded = guiortho._read_ortho_mask_plane(image, (6, 8))

    np.testing.assert_array_equal(loaded, labels)


def test_read_ortho_mask_plane_accepts_tif_fallback_and_encoded_channel(
        tmp_path, monkeypatch):
    image = tmp_path / "sample-002.png"
    mask = tmp_path / "sample-002_masks.tif"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    encoded = np.zeros((5, 7, 3), dtype=np.uint16)
    encoded[1:4, 2:6, 0] = 2
    encoded[..., 1] = 99

    monkeypatch.setattr(guiortho, "imread", lambda filename: encoded)

    loaded = guiortho._read_ortho_mask_plane(image, (5, 7))

    np.testing.assert_array_equal(loaded, encoded[..., 0])


def test_read_ortho_mask_plane_rejects_mismatched_shape(tmp_path, monkeypatch):
    image = tmp_path / "sample-003.tif"
    mask = tmp_path / "sample-003_masks.tif"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    monkeypatch.setattr(
        guiortho, "imread", lambda filename: np.zeros((4, 5), dtype=np.uint16)
    )

    assert guiortho._read_ortho_mask_plane(image, (5, 5)) is None


def test_live_mask_replaces_fixed_main_ortho_plane():
    ortho_masks = np.zeros((3, 5, 7), dtype=np.uint16)
    ortho_outlines = np.zeros_like(ortho_masks)
    live_masks = np.zeros((1, 5, 7), dtype=np.uint16)
    live_outlines = np.zeros_like(live_masks)
    live_masks[0, 2:4, 3:6] = 8
    live_outlines[0, 2, 3:6] = 8

    synced_masks, synced_outlines = guiortho._sync_live_ortho_mask_plane(
        ortho_masks, ortho_outlines, live_masks, live_outlines,
        ortho_index=1, current_z=0,
    )

    np.testing.assert_array_equal(synced_masks[1], live_masks[0])
    np.testing.assert_array_equal(synced_outlines[1], live_outlines[0])
    assert not np.any(synced_masks[0])
    assert not np.any(synced_masks[2])


def test_ortho_fill_and_internal_outline_are_literal_slices():
    masks = np.zeros((5, 7), dtype=np.uint16)
    masks[1:4, 1:6] = 1
    masks[1:4, 4:6] = 2
    outlines = np.zeros_like(masks)
    outlines[1:4, 4] = 2  # an internal cut already present in the outline slice
    colors = np.array([[255, 255, 255], [10, 20, 30], [40, 50, 60]], np.uint8)

    layer = guiortho._render_ortho_mask_overlay(
        masks, outlines, True, True, colors, colors, 128,
        [200, 200, 255, 200], outline_opacity=160,
    )

    np.testing.assert_array_equal(layer[2, 2], [10, 20, 30, 128])
    np.testing.assert_array_equal(layer[2, 5], [40, 50, 60, 128])
    np.testing.assert_array_equal(layer[2, 4], [200, 200, 255, 160])
    np.testing.assert_array_equal(layer[0, 0], [0, 0, 0, 0])


def test_ortho_fill_toggle_does_not_change_outline_slice():
    masks = np.ones((3, 4), dtype=np.uint16)
    outlines = np.zeros_like(masks)
    outlines[:, 2] = 1
    colors = np.array([[255, 255, 255], [10, 20, 30]], np.uint8)

    layer = guiortho._render_ortho_mask_overlay(
        masks, outlines, False, True, colors, colors, 128,
        [200, 200, 255, 200], outline_opacity=160,
    )

    assert not np.any(layer[:, :2])
    np.testing.assert_array_equal(layer[1, 2], [200, 200, 255, 160])
