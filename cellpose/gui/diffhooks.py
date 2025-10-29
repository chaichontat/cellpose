"""
Helpers that keep the GUI diff state synchronized with manual edits.
These utilities avoid importing PyQt so they can be tested headlessly.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class _ToggleLike(Protocol):
    def setText(self, text: str) -> Any: ...


def _call_if_possible(obj: Any, attr: str) -> None:
    """Invoke obj.attr() if it exists and is callable."""
    if obj is None:
        return
    method = getattr(obj, attr, None)
    if callable(method):
        method()


def _snapshot_current_masks(parent: Any) -> dict[str, Any] | None:
    """Return a snapshot of the GUI masks/outlines/colors or None on failure."""
    masks_attr = getattr(parent, "cellpix", None)
    if masks_attr is None:
        return None
    try:
        masks = np.asarray(masks_attr).copy()
    except Exception:
        return None

    outlines_attr = getattr(parent, "outpix", None)
    outlines = None
    if outlines_attr is not None:
        try:
            outlines = np.asarray(outlines_attr).copy()
        except Exception:
            outlines = None

    colors = None
    cellcolors = getattr(parent, "cellcolors", None)
    ncells = 0
    ncells_attr = getattr(parent, "ncells", None)
    if hasattr(ncells_attr, "get") and callable(ncells_attr.get):
        try:
            ncells = int(ncells_attr.get())
        except Exception:
            ncells = 0
    else:
        try:
            ncells = int(ncells_attr)
        except Exception:
            ncells = 0

    if ncells > 0 and isinstance(cellcolors, np.ndarray):
        colors_slice = cellcolors[1:ncells + 1]
        if colors_slice.size > 0:
            colors = colors_slice.copy()

    return {
        "masks": masks,
        "outlines": outlines,
        "colors": colors,
    }


def note_manual_edit(parent: Any) -> None:
    """
    Snapshot the current masks/outlines into the diff cache after a manual edit.

    Parameters
    ----------
    parent : Any
        GUI main window instance. Must expose ``_diff_store_current_as_new``,
        ``_diff_update_button_state`` and ``_diff_refresh_overlay`` callables.
        A ``maskToggleButton`` attribute is optional.
    """
    if parent is None:
        return

    store = getattr(parent, "_diff_store_current_as_new", None)
    if not callable(store):
        return

    editing_restored = bool(getattr(parent, "_diff_showing_restored", False))
    overlay_active = bool(getattr(parent, "_diff_overlay_reference_active", False))
    toggle_text = None

    if editing_restored:
        snapshot = _snapshot_current_masks(parent)
        if snapshot is None:
            return
        setattr(parent, "_diff_state_old", snapshot)
        setattr(parent, "_diff_state_old_manual_override", True)
        toggle_text = "show new mask"
    else:
        preserve_overlay = overlay_active
        if preserve_overlay:
            setattr(parent, "_diff_preserve_overlay_reference", True)
        try:
            store()
        except Exception:
            if preserve_overlay:
                setattr(parent, "_diff_preserve_overlay_reference", False)
            return
        finally:
            if preserve_overlay:
                setattr(parent, "_diff_preserve_overlay_reference", False)
        setattr(parent, "_diff_showing_restored", False)
        setattr(parent, "_diff_state_old_manual_override", False)
        toggle_text = "reset mask"

    if toggle_text is not None:
        toggle: _ToggleLike | None = getattr(parent, "maskToggleButton", None)
        if toggle is not None and hasattr(toggle, "setText"):
            try:
                toggle.setText(toggle_text)
            except Exception:
                pass

    _call_if_possible(parent, "_diff_update_button_state")
    try:
        _call_if_possible(parent, "_diff_refresh_overlay")
    except Exception:
        # Overlay refresh is opportunistic; ignore rendering errors.
        pass
