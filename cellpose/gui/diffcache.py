from __future__ import annotations

import copy
from typing import Any

import numpy as np


DiffState = dict[str, Any]
CacheEntry = dict[str, Any]


def _clone_state(state: DiffState | None) -> DiffState | None:
    if state is None:
        return None
    cloned: DiffState = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            cloned[key] = np.array(value, copy=True)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _clone_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.array(value, copy=True)


def _clone_entry(entry: CacheEntry | None) -> CacheEntry | None:
    if entry is None:
        return None
    cloned: CacheEntry = {
        "new_state": _clone_state(entry.get("new_state")),
        "active_state": _clone_state(entry.get("active_state")),
        "latest_masks": _clone_array(entry.get("latest_masks")),
        "showing_restored": bool(entry.get("showing_restored", False)),
        "crosshair": tuple(entry["crosshair"]) if entry.get("crosshair") is not None else None,
        "z_index": entry.get("z_index"),
        "last_shape": entry.get("last_shape"),
    }
    return cloned


class DiffStateCache:
    """Cache segmentation and diff state per image filename."""

    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}

    def store(
        self,
        key: str | None,
        *,
        new_state: DiffState | None,
        latest_masks: np.ndarray | None = None,
        active_state: DiffState | None = None,
        showing_restored: bool = False,
        crosshair: tuple[float, float] | None = None,
        z_index: int | None = None,
        last_shape: tuple[int, int] | None = None,
    ) -> None:
        if not key:
            return
        if new_state is None:
            self._store.pop(key, None)
            return

        entry: CacheEntry = {
            "new_state": _clone_state(new_state),
            "active_state": _clone_state(active_state),
            "latest_masks": _clone_array(latest_masks),
            "showing_restored": bool(showing_restored),
            "crosshair": tuple(crosshair) if crosshair is not None else None,
            "z_index": z_index,
            "last_shape": last_shape,
        }
        self._store[key] = entry

    def retrieve(self, key: str | None) -> CacheEntry | None:
        if not key:
            return None
        entry = self._store.get(key)
        return _clone_entry(entry)

    def discard(self, key: str | None) -> None:
        if not key:
            return
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
