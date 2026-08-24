import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy.QtWidgets import QApplication

from cellpose.gui.guiortho import MainW_ortho2D


class _Line:

    def __init__(self) -> None:
        self.positions = []

    def setPos(self, position) -> None:
        self.positions.append(position)

    @property
    def position(self):
        return self.positions[-1] if self.positions else None


class _OrthoHarness(MainW_ortho2D):

    def __init__(self) -> None:
        pass


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _window(qt_app) -> _OrthoHarness:
    window = _OrthoHarness()
    window.Ly = 10
    window.Lx = 20
    window.zc = 4
    window.vLine = _Line()
    window.hLine = _Line()
    window.vLineOrtho = [_Line(), _Line()]
    window.hLineOrtho = [_Line(), _Line()]
    return window


def test_ortho_crosshair_updates_once(qt_app):
    window = _window(qt_app)
    viewer_updates = []
    window.yortho = 12
    window.xortho = -3
    window._diff_last_crosshair = None
    window._diff_update_crosshair_lines = lambda coords: viewer_updates.append(
        ("diff", coords))
    window._gradxy_update_crosshair_lines = lambda coords: viewer_updates.append(
        ("gradxy", coords))

    window.update_crosshairs()

    assert viewer_updates == [
        ("diff", (9.0, 0.0)),
        ("gradxy", (9.0, 0.0)),
    ]
    assert window.hLine.position == 9
    assert window.vLine.position == 0
    assert len(window.hLine.positions) == 1
    assert len(window.vLine.positions) == 1


def test_crosshair_uses_clamped_ortho_coordinates(qt_app):
    window = _window(qt_app)
    viewer_updates = []
    window._diff_last_crosshair = None
    window._diff_update_crosshair_lines = lambda coords: viewer_updates.append(
        ("diff", coords))
    window._gradxy_update_crosshair_lines = lambda coords: viewer_updates.append(
        ("gradxy", coords))

    window._set_crosshair((12.5, -3.5))

    assert (window.yortho, window.xortho) == (9, 0)
    assert window.hLine.position == 9
    assert window.vLine.position == 0
    assert viewer_updates == [
        ("diff", (9.0, 0.0)),
        ("gradxy", (9.0, 0.0)),
    ]
    assert window.get_crosshair_coords() == (9.0, 0.0)


def test_cleared_crosshair_is_not_restored_from_ortho_position(qt_app):
    window = _window(qt_app)
    viewer_updates = []
    window._diff_last_crosshair = None
    window._diff_update_crosshair_lines = lambda coords: viewer_updates.append(
        ("diff", coords))
    window._gradxy_update_crosshair_lines = lambda coords: viewer_updates.append(
        ("gradxy", coords))

    window._set_crosshair((3, 4))
    window._set_crosshair(None)

    assert window.get_crosshair_coords() is None
    assert viewer_updates[-2:] == [("diff", None), ("gradxy", None)]
