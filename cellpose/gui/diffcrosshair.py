"""
Global crosshair synchronization for segmentation diff viewers.

The GUI maintains a single Matplotlib diff window at a time, but users can
switch between adjacent tiles/images while keeping the diff window open.
This helper keeps the diff crosshair position consistent across image loads
and across any future windows that opt into the protocol.
"""

from __future__ import annotations

import weakref
from typing import Optional, Protocol, Tuple


Coords = Optional[Tuple[float, float]]


class _Listener(Protocol):
    def diff_crosshair_updated(
        self,
        coords: Coords,
        *,
        source: object | None = None,
        reason: str | None = None,
    ) -> None:
        ...


class DiffCrosshairHub:
    """Singleton dispatcher that keeps diff crosshair positions in sync."""

    _instance: "DiffCrosshairHub | None" = None

    def __init__(self) -> None:
        self._listeners: list[weakref.ReferenceType[_Listener]] = []
        self._coords: Coords = None

    @classmethod
    def instance(cls) -> "DiffCrosshairHub":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, listener: _Listener) -> None:
        """Track *listener*; immediately emit the current coordinates if set."""
        self._listeners.append(weakref.ref(listener))
        self._prune()
        if self._coords is not None:
            try:
                listener.diff_crosshair_updated(
                    self._coords, source=None, reason="register"
                )
            except Exception:
                # Consumers are responsible for their own error reporting.
                pass

    def unregister(self, listener: _Listener) -> None:
        """Stop tracking *listener*."""
        self._listeners = [
            ref
            for ref in self._listeners
            if (obj := ref()) is not None and obj is not listener
        ]

    def current(self) -> Coords:
        """Return the most recent coordinates broadcast to the hub."""
        return self._coords

    def set_coords(
        self,
        coords: Coords,
        *,
        source: object | None = None,
        reason: str | None = None,
    ) -> None:
        """Persist *coords* (y, x) and notify all listeners except *source*."""
        if coords is None:
            self._coords = None
        else:
            y, x = coords
            self._coords = (float(y), float(x))
        self._broadcast(source=source, reason=reason)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _prune(self) -> None:
        self._listeners = [ref for ref in self._listeners if ref() is not None]

    def _broadcast(
        self,
        *,
        source: object | None = None,
        reason: str | None = None,
    ) -> None:
        if not self._listeners:
            return
        for ref in list(self._listeners):
            listener = ref()
            if listener is None:
                self._listeners.remove(ref)
                continue
            if listener is source:
                continue
            try:
                listener.diff_crosshair_updated(
                    self._coords, source=source, reason=reason
                )
            except Exception:
                # Keep other listeners alive even if one misbehaves.
                continue
