from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cellpose.gui.diffcache import DiffStateCache
from cellpose.gui.diffhooks import note_manual_edit


def test_cache_roundtrip_isolates_arrays_and_flows():
    cache = DiffStateCache()
    masks = np.array([[1, 0]], dtype=np.uint16)
    outlines = np.array([[0, 1]], dtype=np.uint16)
    saved_masks = np.array([[2, 0]], dtype=np.uint16)
    flow = np.array([[1.0, 2.0]], dtype=np.float32)
    flows = [flow, ["metadata"]]

    cache.store(
        "image.tif",
        new_state={
            "masks": masks,
            "outlines": outlines,
            "metadata": {"labels": [1]},
        },
        saved_state={"masks": saved_masks, "source": {"kind": "memory"}},
        saved_state_manual_override=True,
        showing_restored=True,
        crosshair=(3, 4),
        flows=flows,
    )
    masks[0, 0] = 9
    outlines[0, 1] = 9
    saved_masks[0, 0] = 9
    flow[0, 0] = 9
    flows[1].append("changed")

    restored = cache.retrieve("image.tif")

    assert restored is not None
    np.testing.assert_array_equal(restored["new_state"]["masks"], [[1, 0]])
    np.testing.assert_array_equal(restored["saved_state"]["masks"], [[2, 0]])
    np.testing.assert_array_equal(restored["new_state"]["outlines"], [[0, 1]])
    np.testing.assert_array_equal(restored["flows"][0], [[1.0, 2.0]])
    assert restored["flows"][1] == ["metadata"]
    assert restored["crosshair"] == (3, 4)
    assert restored["showing_restored"] is True
    assert restored["saved_state_manual_override"] is True

    restored["new_state"]["metadata"]["labels"].append(2)
    restored["saved_state"]["source"]["kind"] = "changed"
    restored["flows"][0][0, 0] = 7
    second_restore = cache.retrieve("image.tif")
    assert second_restore["new_state"]["metadata"]["labels"] == [1]
    assert second_restore["saved_state"]["source"] == {"kind": "memory"}
    assert second_restore["flows"][0][0, 0] == 1.0


def test_cache_discards_entry():
    cache = DiffStateCache()
    cache.store("image.tif", new_state={"masks": np.ones((1, 1))})

    cache.discard("image.tif")

    assert cache.retrieve("image.tif") is None


def test_manual_edit_of_new_state_pins_saved_baseline_and_refreshes():
    events = []
    saved_state = {"masks": np.array([[[4]]], dtype=np.uint16)}
    parent = SimpleNamespace(
        _diff_showing_restored=False,
        _diff_state_old=saved_state,
        _diff_state_old_manual_override=False,
        _diff_update_button_state=lambda: events.append("buttons"),
        _refresh_comparison_viewers=lambda: events.append("refresh"),
    )

    def store():
        parent._diff_state_old_manual_override = False
        events.append("store")

    parent._diff_store_current_as_new = store

    note_manual_edit(parent)

    assert events == ["store", "buttons", "refresh"]
    assert parent._diff_showing_restored is False
    assert parent._diff_state_old is saved_state
    assert parent._diff_state_old_manual_override is True


def test_manual_edit_of_restored_state_snapshots_saved_side():
    events = []
    masks = np.array([[[0, 1]]], dtype=np.uint16)
    outlines = np.array([[[0, 1]]], dtype=np.uint16)
    colors = np.array([[0, 0, 0], [3, 4, 5]], dtype=np.uint8)
    parent = SimpleNamespace(
        cellpix=masks,
        outpix=outlines,
        cellcolors=colors,
        ncells=SimpleNamespace(get=lambda: 1),
        _diff_showing_restored=True,
        _diff_store_current_as_new=lambda: events.append("store"),
        _diff_update_button_state=lambda: events.append("buttons"),
        _refresh_comparison_viewers=lambda: events.append("refresh"),
    )

    note_manual_edit(parent)
    masks[0, 0, 1] = 9
    outlines[0, 0, 1] = 9
    colors[1, 0] = 9

    assert events == ["buttons", "refresh"]
    np.testing.assert_array_equal(parent._diff_state_old["masks"], [[[0, 1]]])
    np.testing.assert_array_equal(parent._diff_state_old["outlines"], [[[0, 1]]])
    np.testing.assert_array_equal(parent._diff_state_old["colors"], [[3, 4, 5]])
    assert parent._diff_state_old_manual_override is True
