from types import SimpleNamespace

import numpy as np
import pytest

from cellpose.contrib import packed_infer
from cellpose.contrib import cellposetrt
from cellpose import models


@pytest.mark.parametrize("use_packing", [False, True])
@pytest.mark.parametrize("use_ortho_model", [False, True])
def test_packed_3d_dispatches_models_by_plane(
    monkeypatch: pytest.MonkeyPatch,
    use_packing: bool,
    use_ortho_model: bool,
) -> None:
    primary = object()
    ortho = object() if use_ortho_model else None
    calls = []

    def fake_run_net(net, images, **_kwargs):
        calls.append(net)
        return (
            np.zeros((*images.shape[:-1], 3), dtype=np.float32),
            np.zeros((images.shape[0], 256), dtype=np.float32),
        )

    layout = SimpleNamespace(K=1, guard=0, slot_height=1)
    monkeypatch.setattr(
        packed_infer,
        "compute_stripe_layout",
        lambda *_args, **_kwargs: layout if use_packing else None,
    )
    monkeypatch.setattr(packed_infer, "run_net", fake_run_net)
    monkeypatch.setattr(
        packed_infer,
        "pack_planes_to_stripes",
        lambda images, _layout: (images, object()),
    )
    monkeypatch.setattr(
        packed_infer,
        "unpack_stripes_to_planes",
        lambda outputs, _mapping, **_kwargs: outputs,
    )

    packed_infer._run_3d_with_packing(
        primary,
        np.zeros((2, 3, 4, 1), dtype=np.float32),
        batch_size=1,
        augment=False,
        tile_overlap=0.1,
        bsize=256,
        pack_border=0,
        plane_weights=None,
        net_ortho=ortho,
        return_raw_3d=True,
    )

    expected_ortho = ortho if ortho is not None else primary
    assert calls == [primary, expected_ortho, expected_ortho]


def test_trt_models_require_matching_input_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_model_init(self, **kwargs) -> None:
        self.net = None
        self.pretrained_model_ortho = kwargs["pretrained_model_ortho"]

    def fake_engine(path, **_kwargs):
        bsize = 256 if path == "xy.plan" else 224
        return SimpleNamespace(_in_dims=(1, 3, bsize, bsize))

    monkeypatch.setattr(models.CellposeModel, "__init__", fake_model_init)
    monkeypatch.setattr(cellposetrt, "TRTEngineModule", fake_engine)

    with pytest.raises(ValueError, match="matching input profiles"):
        cellposetrt.CellposeModelTRT(
            pretrained_model="xy.plan",
            pretrained_model_ortho="ortho.plan",
        )
