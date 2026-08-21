import numpy as np

from cellpose import core, models


def _stub_model(mask: np.ndarray) -> models.CellposeModel:
    model = object.__new__(models.CellposeModel)

    def run_net(image: np.ndarray, **_kwargs: object):
        shape = image.shape[:-1]
        return (
            np.zeros((3, *shape), dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(256, dtype=np.float32),
        )

    model._run_net = run_net
    model._compute_masks = lambda *_args, **_kwargs: mask.copy()
    return model


def test_mask_only_skips_identity_resizes_and_flow_render(monkeypatch) -> None:
    image = np.zeros((2, 4, 5, 3), dtype=np.float32)
    expected = np.arange(2 * 4 * 5, dtype=np.uint32).reshape(2, 4, 5)
    model = _stub_model(expected)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("identity resize or flow rendering should be skipped")

    monkeypatch.setattr(models.transforms, "resize_image", unexpected)
    monkeypatch.setattr(models.plot, "dx_to_circ", unexpected)

    masks, flows, styles = model.eval(
        image,
        diameter=30,
        anisotropy=1,
        do_3D=True,
        channel_axis=3,
        z_axis=0,
        normalize=False,
        resample=False,
        return_flows=False,
    )

    np.testing.assert_array_equal(masks, expected)
    assert flows is None
    assert styles.shape == (256,)


def test_mask_only_restores_scaled_mask_shape() -> None:
    image = np.zeros((2, 8, 10, 3), dtype=np.float32)
    model = _stub_model(np.zeros((2, 4, 5), dtype=np.uint32))

    masks, flows, _styles = model.eval(
        image,
        diameter=60,
        anisotropy=1,
        do_3D=True,
        channel_axis=3,
        z_axis=0,
        normalize=False,
        resample=False,
        return_flows=False,
    )

    assert masks.shape == image.shape[:3]
    assert flows is None


def test_mask_only_restores_scaled_2d_mask_shape() -> None:
    image = np.zeros((8, 10, 3), dtype=np.float32)
    model = _stub_model(np.zeros((4, 5), dtype=np.uint32))

    masks, flows, _styles = model.eval(
        image,
        diameter=60,
        channel_axis=2,
        normalize=False,
        resample=False,
        return_flows=False,
    )

    assert masks.shape == image.shape[:2]
    assert flows is None


def test_mask_only_matches_default_for_scaled_3d_labels() -> None:
    image = np.zeros((2, 8, 10, 3), dtype=np.float32)
    source = np.arange(2 * 4 * 5, dtype=np.uint32).reshape(2, 4, 5)

    default_masks, _flows, _styles = _stub_model(source).eval(
        image,
        diameter=60,
        anisotropy=1,
        do_3D=True,
        channel_axis=3,
        z_axis=0,
        normalize=False,
        resample=False,
    )
    mask_only, flows, _styles = _stub_model(source).eval(
        image,
        diameter=60,
        anisotropy=1,
        do_3D=True,
        channel_axis=3,
        z_axis=0,
        normalize=False,
        resample=False,
        return_flows=False,
    )

    np.testing.assert_array_equal(mask_only, default_masks)
    assert flows is None


def test_mask_only_matches_default_for_scaled_2d_labels() -> None:
    image = np.zeros((8, 10, 3), dtype=np.float32)
    source = np.arange(4 * 5, dtype=np.uint32).reshape(4, 5)

    default_masks, _flows, _styles = _stub_model(source).eval(
        image,
        diameter=60,
        channel_axis=2,
        normalize=False,
        resample=False,
    )
    mask_only, flows, _styles = _stub_model(source).eval(
        image,
        diameter=60,
        channel_axis=2,
        normalize=False,
        resample=False,
        return_flows=False,
    )

    np.testing.assert_array_equal(mask_only, default_masks)
    assert flows is None


def test_flow_diagnostics_remain_enabled_by_default(monkeypatch) -> None:
    image = np.zeros((2, 4, 5, 3), dtype=np.float32)
    expected = np.ones((2, 4, 5), dtype=np.uint32)
    model = _stub_model(expected)
    rendered = np.array([123], dtype=np.uint8)
    monkeypatch.setattr(models.plot, "dx_to_circ", lambda _flow: rendered)

    masks, flows, _styles = model.eval(
        image,
        diameter=30,
        anisotropy=1,
        do_3D=True,
        channel_axis=3,
        z_axis=0,
        normalize=False,
        resample=False,
    )

    np.testing.assert_array_equal(masks, expected)
    np.testing.assert_array_equal(flows[0], rendered)
    assert flows[1].shape == (3, *image.shape[:3])
    assert flows[2].shape == image.shape[:3]


def test_run_3d_does_not_print_diagnostics(monkeypatch, capsys) -> None:
    image = np.zeros((2, 3, 4, 1), dtype=np.float32)

    def fake_run_net(_net, stack: np.ndarray, **_kwargs: object):
        return (
            np.zeros((*stack.shape[:-1], 3), dtype=np.float32),
            np.zeros(256, dtype=np.float32),
        )

    monkeypatch.setattr(core, "run_net", fake_run_net)

    core.run_3D(object(), image)

    assert capsys.readouterr().out == ""
