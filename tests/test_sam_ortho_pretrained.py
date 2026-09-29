import types

import numpy as np
import pytest

import cellpose.models as models
import cellpose.core as core


class _StubTransformer:
    def __init__(self, dtype=None):
        self.dtype = dtype
        self.device = None
        self.loaded = []

    def to(self, device):
        self.device = device
        return self

    def load_model(self, path, device=None):
        self.loaded.append((path, device))


def _fake_exists(_path: str) -> bool:
    # Treat all paths as existing to avoid filesystem / download interactions.
    return True


@pytest.mark.parametrize("has_ortho", [False, True])
def test_sam_constructor_creates_optional_ortho_net(monkeypatch, has_ortho: bool) -> None:
    monkeypatch.setattr(models, "Transformer", _StubTransformer)
    monkeypatch.setattr(models.os.path, "exists", _fake_exists)

    ortho_path = "/tmp/ortho.model" if has_ortho else None

    model = models.CellposeModel(
        pretrained_model="cpsam",
        pretrained_model_ortho=ortho_path,
    )

    # Main net is always created and loaded.
    assert isinstance(model.net, _StubTransformer)
    assert model.net.loaded, "main SAM net should be loaded"

    if has_ortho:
        assert model.net_ortho is not None
        assert isinstance(model.net_ortho, _StubTransformer)
        # Expect a separate network with its own load call.
        assert model.net_ortho is not model.net
        assert model.net_ortho.loaded, "ortho SAM net should be loaded"
    else:
        assert model.net_ortho is None


def test_sam_3d_uses_ortho_net_for_yz_zx(monkeypatch) -> None:
    monkeypatch.setattr(models, "Transformer", _StubTransformer)
    monkeypatch.setattr(models.os.path, "exists", _fake_exists)

    calls = []

    def fake_run_net(net, imgi, *args, **kwargs):
        calls.append(net)
        Lz, Ly, Lx, nchan = imgi.shape
        y = np.zeros((Lz, Ly, Lx, 3), dtype=np.float32)
        styles = np.zeros((Lz, 256), dtype=np.float32)
        return y, styles

    monkeypatch.setattr(core, "run_net", fake_run_net)

    model = models.CellposeModel(
        pretrained_model="cpsam",
        pretrained_model_ortho="/tmp/ortho.model",
    )

    # Small synthetic 3D stack: [Z, Y, X, C].
    x = np.zeros((3, 8, 8, 2), dtype=np.float32)

    dP, cellprob, styles = model._run_net(
        x,
        augment=False,
        batch_size=1,
        tile_overlap=0.1,
        bsize=64,
        anisotropy=1.0,
        do_3D=True,
        plane_weights=None,
    )

    # Expect three passes (XY, YZ, ZX).
    assert len(calls) == 3

    # XY should use the main net; YZ/ZX should use the ortho net when present.
    assert calls[0] is model.net
    assert calls[1] is model.net_ortho
    assert calls[2] is model.net_ortho

    # Basic shape sanity checks for downstream expectations.
    assert dP.ndim == 4 and dP.shape[0] == 3
    assert cellprob.shape == (3, 8, 8)
    assert styles.shape[-1] == 256


def test_sam_3d_falls_back_to_main_net_when_no_ortho(monkeypatch) -> None:
    monkeypatch.setattr(models, "Transformer", _StubTransformer)
    monkeypatch.setattr(models.os.path, "exists", _fake_exists)

    calls = []

    def fake_run_net(net, imgi, *args, **kwargs):
        calls.append(net)
        Lz, Ly, Lx, nchan = imgi.shape
        y = np.zeros((Lz, Ly, Lx, 3), dtype=np.float32)
        styles = np.zeros((Lz, 256), dtype=np.float32)
        return y, styles

    monkeypatch.setattr(core, "run_net", fake_run_net)

    model = models.CellposeModel(
        pretrained_model="cpsam",
        pretrained_model_ortho=None,
    )

    x = np.zeros((3, 8, 8, 2), dtype=np.float32)

    model._run_net(
        x,
        augment=False,
        batch_size=1,
        tile_overlap=0.1,
        bsize=64,
        anisotropy=1.0,
        do_3D=True,
        plane_weights=None,
    )

    # All three passes should use the main net when no ortho net is configured.
    assert len(calls) == 3
    assert all(net is model.net for net in calls)

