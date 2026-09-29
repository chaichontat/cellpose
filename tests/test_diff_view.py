from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cellpose.gui import gui as gui_module
from cellpose.contrib.diff import contour_diff_rgb
from cellpose.gui.gui import MainW


class _Button:

    def __init__(self):
        self.enabled = False
        self.text = ""
        self.tooltip = ""

    def setEnabled(self, enabled):
        self.enabled = enabled

    def setText(self, text):
        self.text = text

    def setToolTip(self, tooltip):
        self.tooltip = tooltip


def test_store_new_state_preserves_saved_override():
    parent = SimpleNamespace(
        cellpix=np.array([[[0, 1]]], dtype=np.uint16),
        outpix=np.array([[[0, 1]]], dtype=np.uint16),
        cellcolors=np.array([[255, 255, 255], [1, 2, 3]], dtype=np.uint8),
        ncells=SimpleNamespace(get=lambda: 1),
        _diff_state_old_manual_override=True,
        _diff_state_new=None,
    )

    MainW._diff_store_current_as_new(parent)

    assert parent._diff_state_old_manual_override is True
    np.testing.assert_array_equal(
        parent._diff_state_new["masks"], parent.cellpix)


def test_loaded_mask_baseline_enables_diff_and_reset_without_seg_npy():
    saved = {"masks": np.zeros((1, 2, 2), dtype=np.uint16)}
    parent = SimpleNamespace(
        diffButton=_Button(),
        maskToggleButton=_Button(),
        _diff_seg_path=None,
        _diff_state_new={"masks": np.ones((1, 2, 2), dtype=np.uint16)},
        _diff_showing_restored=False,
        _diff_get_saved_state=lambda reload=False: saved,
        _diff_can_reset=lambda: (True, ""),
        NZ=1,
    )

    MainW._diff_update_button_state(parent)

    assert parent.diffButton.enabled is True
    assert parent.maskToggleButton.enabled is True


def test_button_refresh_preserves_tif_mask_baseline(tmp_path):
    saved = {"masks": np.zeros((1, 2, 2), dtype=np.uint16)}
    refreshes = []
    parent = SimpleNamespace(
        filename=str(tmp_path / "image.tif"),
        _diff_seg_path=None,
        _diff_state_old=saved,
        _diff_state_old_manual_override=True,
        _diff_update_button_state=lambda: refreshes.append(True),
    )

    MainW._diff_refresh_seg_path(parent)

    assert parent._diff_state_old is saved
    assert parent._diff_state_old_manual_override is True
    assert refreshes == [True]


def test_accept_new_uses_fresh_saved_side_label():
    saved = np.array([[1, 0], [0, 0]], dtype=np.uint16)
    current = np.array([[0, 2], [0, 0]], dtype=np.uint16)
    refreshes = []
    parent = SimpleNamespace(
        _diff_get_planes=lambda: (saved, current, 0),
        _find_nonzero_label_near=MainW._find_nonzero_label_near,
        _diff_state_old={"masks": saved[np.newaxis], "outlines": None,
                         "colors": None},
        _diff_state_old_manual_override=False,
        _diff_showing_restored=False,
        _refresh_comparison_viewers=lambda: refreshes.append(True),
        _diff_log=lambda message: None,
    )

    changed = MainW._diff_accept_new_at(parent, 0, 1)

    assert changed is True
    assert parent._diff_state_old["masks"][0, 0, 1] == 3
    assert parent._diff_state_old_manual_override is True
    assert refreshes == [True]


def test_set_crosshair_updates_both_viewers_once():
    updates = []

    class Parent:
        _set_crosshair = MainW._set_crosshair

        def __init__(self):
            self._diff_last_crosshair = None

        def _diff_update_crosshair_lines(self, coords):
            updates.append(("diff", coords))

        def _gradxy_update_crosshair_lines(self, coords):
            updates.append(("gradxy", coords))

    parent = Parent()
    parent._set_crosshair((2, 4))
    parent._set_crosshair((2, 4))

    assert parent._diff_last_crosshair == (2.0, 4.0)
    assert updates == [
        ("diff", (2.0, 4.0)),
        ("gradxy", (2.0, 4.0)),
    ]


def test_cache_miss_closes_stale_auxiliary_viewers():
    closed = []
    parent = SimpleNamespace(
        _diff_state_cache=SimpleNamespace(retrieve=lambda key: None),
        _diff_cache_key=lambda: "image.tif",
        _diff_close_existing=lambda: closed.append("diff"),
        _gradxy_close_existing=lambda: closed.append("gradxy"),
        _diff_last_crosshair=(1, 2),
        _diff_update_button_state=lambda: None,
    )

    MainW._diff_restore_after_image_load(parent)

    assert closed == ["diff", "gradxy"]
    assert parent._diff_last_crosshair is None


def test_cache_transition_roundtrips_saved_override():
    saved = {"masks": np.array([[[0, 7]]], dtype=np.uint16)}
    new = {"masks": np.array([[[0, 2]]], dtype=np.uint16)}
    stored = {}

    class Cache:

        def store(self, key, **entry):
            stored.update(entry)

        def retrieve(self, key):
            return stored

    parent = SimpleNamespace(
        _diff_state_cache=Cache(),
        _diff_cache_key=lambda: "image.tif",
        _diff_state_new=new,
        _diff_state_old=saved,
        _diff_state_old_manual_override=True,
        _diff_showing_restored=False,
        _diff_last_crosshair=None,
        flows=[np.zeros((1, 1, 2, 3), dtype=np.uint8)],
        _diff_apply_state=lambda state: None,
        _refresh_comparison_viewers=lambda: None,
        _diff_update_button_state=lambda: None,
        Ly=1,
        Lx=2,
    )
    parent._set_crosshair = lambda coords: setattr(
        parent, "_diff_last_crosshair", coords)

    MainW._diff_cache_before_image_change(parent)
    parent._diff_state_old = None
    parent._diff_state_old_manual_override = False
    MainW._diff_restore_after_image_load(parent)

    np.testing.assert_array_equal(parent._diff_state_old["masks"], saved["masks"])
    assert parent._diff_state_old_manual_override is True


def test_prediction_invalidation_clears_masks_and_flows_together():
    transitions = []
    parent = SimpleNamespace(
        _diff_state_new={"masks": np.ones((1, 2, 2))},
        _diff_showing_restored=True,
        flows=[np.ones((1, 2, 2, 3))],
        _diff_update_button_state=lambda: transitions.append("buttons"),
        _refresh_comparison_viewers=lambda: transitions.append("viewers"),
    )

    MainW._invalidate_prediction_state(parent)

    assert parent._diff_state_new is None
    assert parent._diff_showing_restored is False
    assert parent.flows == [[], [], []]
    assert transitions == ["buttons", "viewers"]


def test_diff_planes_reject_mismatched_z_stacks():
    parent = SimpleNamespace(
        _diff_get_saved_state=lambda reload=False: {
            "masks": np.zeros((2, 4, 5), dtype=np.uint16)
        },
        _diff_state_new={"masks": np.zeros((3, 4, 5), dtype=np.uint16)},
        currentZ=1,
    )

    with pytest.raises(ValueError, match="Z-planes"):
        MainW._diff_get_planes(parent)


def test_toggle_failure_keeps_displayed_side(monkeypatch):
    warnings = []
    parent = SimpleNamespace(
        _diff_can_reset=lambda: (True, ""),
        _diff_showing_restored=True,
        _diff_state_new={"masks": np.ones((1, 2, 2), dtype=np.uint16)},
        _diff_apply_state=lambda state: (_ for _ in ()).throw(
            ValueError("invalid state")
        ),
        _diff_update_button_state=lambda: None,
        _refresh_comparison_viewers=lambda: None,
    )
    monkeypatch.setattr(
        gui_module.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    MainW.toggle_mask_restore(parent)

    assert parent._diff_showing_restored is True
    assert len(warnings) == 1


def test_clear_all_action_records_manual_edit():
    events = []
    parent = SimpleNamespace(
        clear_all=lambda: events.append("clear"),
        _diff_showing_restored=False,
        _diff_state_old=None,
        _diff_store_current_as_new=lambda: events.append("store"),
        _diff_update_button_state=lambda: events.append("buttons"),
        _refresh_comparison_viewers=lambda: events.append("refresh"),
    )

    MainW.clear_all_action(parent)

    assert events == ["clear", "store", "buttons", "refresh"]


def test_dead_diff_figure_resets_viewer_state(monkeypatch):
    resets = []
    parent = SimpleNamespace(
        _diff_fig=SimpleNamespace(number=7),
        _diff_ax=object(),
        _diff_reset_state=lambda: resets.append(True),
    )
    monkeypatch.setattr(gui_module.plt, "fignum_exists", lambda number: False)

    MainW._diff_refresh_overlay(parent)

    assert resets == [True]


def test_gradxy_click_requires_primary_button():
    axis = object()
    updates = []
    parent = SimpleNamespace(
        _gradxy_ax=axis,
        _set_crosshair=lambda coords: updates.append(coords),
    )
    event = SimpleNamespace(
        inaxes=axis,
        button=3,
        xdata=4.0,
        ydata=2.0,
    )

    MainW._on_gradxy_click(parent, event)
    event.button = 1
    MainW._on_gradxy_click(parent, event)

    assert updates == [(2.0, 4.0)]


def test_tif_baseline_diff_window_opens_without_seg_path(monkeypatch):
    subplots_kwargs = {}
    reload_values = []
    window_state = SimpleNamespace(normal_calls=0)
    window_state.showNormal = lambda: setattr(
        window_state, "normal_calls", window_state.normal_calls + 1)
    manager = SimpleNamespace(
        window=window_state,
        set_window_title=lambda title: None,
    )
    canvas = SimpleNamespace(
        manager=manager,
        mpl_connect=lambda event, callback: 1,
    )
    figure = SimpleNamespace(
        canvas=canvas,
        tight_layout=lambda: None,
        show=lambda: None,
    )
    axes = SimpleNamespace(
        imshow=lambda *args, **kwargs: object(),
        axis=lambda value: None,
        set_xlim=lambda *args: None,
        set_ylim=lambda *args: None,
    )
    parent = SimpleNamespace(
        _diff_seg_path=None,
        _diff_state_new={"masks": np.zeros((1, 2, 2), dtype=np.uint16)},
        _diff_get_saved_state=lambda reload: (
            reload_values.append(reload) or
            {"masks": np.zeros((1, 2, 2), dtype=np.uint16)}
        ),
        _diff_recompute_overlay=lambda: np.zeros((2, 2, 3), dtype=np.uint8),
        _diff_close_existing=lambda: None,
        _diff_update_crosshair_lines=lambda: None,
        _diff_on_close=lambda event: None,
        _on_diff_click=lambda event: None,
        _on_diff_scroll=lambda event: None,
        _diff_log=lambda message: None,
    )
    def subplots(**kwargs):
        subplots_kwargs.update(kwargs)
        return figure, axes

    monkeypatch.setattr(gui_module.plt, "subplots", subplots)
    monkeypatch.setattr(gui_module.plt, "show", lambda **kwargs: None)

    MainW.show_segmentation_diff(parent)

    assert subplots_kwargs["figsize"] == (6, 6)
    assert reload_values == [False]
    assert window_state.normal_calls == 1


def test_diff_contours_are_one_pixel_after_2x_upsampling():
    saved = np.array([[0, 1], [0, 1]], dtype=np.int32)
    current = np.zeros_like(saved)

    rgb = contour_diff_rgb(
        saved,
        current,
        upsample=2,
        tol_pixels=0,
        min_component_size=1,
        connectivity=4,
        emphasize_nodes=False,
    )

    assert rgb.shape == (4, 4, 3)
    boundary = np.any(rgb != 0, axis=-1)
    assert np.flatnonzero(np.any(boundary, axis=0)).tolist() == [1]


def test_diff_click_maps_2x_render_pixel_to_mask_pixel():
    axis = object()
    accepted = []
    diff_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    diff_rgb[1, 1] = [255, 80, 255]
    image = SimpleNamespace(get_extent=lambda: (-0.5, 1.5, 1.5, -0.5))
    parent = SimpleNamespace(
        _diff_ax=axis,
        _diff_img_im=image,
        _diff_diff_rgb=diff_rgb,
        _diff_last_shape=(2, 2),
        _diff_click_to_indices=lambda x, y: MainW._diff_click_to_indices(
            parent, x, y),
        _diff_color_kind=MainW._diff_color_kind,
        _diff_accept_old_at=lambda y, x: accepted.append((y, x)) or True,
        _diff_accept_new_at=lambda y, x: False,
        _diff_log=lambda message: None,
    )
    event = SimpleNamespace(inaxes=axis, button=1, xdata=0.0, ydata=0.0)

    MainW._on_diff_click(parent, event)

    assert accepted == [(0, 0)]
