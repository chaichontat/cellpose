"""Synchronize GUI diff state after manual mask edits."""

from __future__ import annotations

from typing import Any

import numpy as np


def snapshot_current_masks(parent: Any) -> dict[str, Any]:
    """Copy the active masks and matching display metadata."""
    ncells = int(parent.ncells.get())
    colors = parent.cellcolors[1:ncells + 1].copy() if ncells else None
    return {
        "masks": np.asarray(parent.cellpix).copy(),
        "outlines": np.asarray(parent.outpix).copy(),
        "colors": colors,
    }


def note_manual_edit(parent: Any) -> None:
    """Synchronize the edited side of the saved-versus-new diff state."""
    if parent._diff_showing_restored:
        parent._diff_state_old = snapshot_current_masks(parent)
        parent._diff_state_old_manual_override = True
    else:
        parent._diff_store_current_as_new()
        parent._diff_state_old_manual_override = parent._diff_state_old is not None

    parent._diff_update_button_state()
    parent._refresh_comparison_viewers()
