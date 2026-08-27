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
