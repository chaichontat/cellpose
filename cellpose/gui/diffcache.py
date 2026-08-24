from __future__ import annotations

import copy
from typing import Any

DiffState = dict[str, Any]
CacheEntry = dict[str, Any]


class DiffStateCache:
    """Cache segmentation and diff state per image filename."""

    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}

    def store(
        self,
        key: str,
        *,
        new_state: DiffState,
        saved_state: DiffState | None = None,
        saved_state_manual_override: bool = False,
        showing_restored: bool = False,
        crosshair: tuple[float, float] | None = None,
        flows: list[Any] | None = None,
    ) -> None:
        entry = {
            "new_state": new_state,
            "saved_state": saved_state,
            "saved_state_manual_override": saved_state_manual_override,
            "showing_restored": showing_restored,
            "crosshair": tuple(crosshair) if crosshair is not None else None,
            "flows": flows,
        }
        self._store[key] = copy.deepcopy(entry)

    def retrieve(self, key: str) -> CacheEntry | None:
        entry = self._store.get(key)
        return copy.deepcopy(entry)

    def discard(self, key: str) -> None:
        self._store.pop(key, None)
