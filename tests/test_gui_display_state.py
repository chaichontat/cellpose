from cellpose.gui.gui import GUI


class _ComboBoxStub:
    def __init__(self, index=0, count=6):
        self._index = index
        self._count = count
        self.block_calls = []

    def currentIndex(self):
        return self._index

    def setCurrentIndex(self, index):
        self._index = index

    def count(self):
        return self._count

    def blockSignals(self, flag):
        self.block_calls.append(flag)


class _GUIStub:
    def __init__(self, color=0, index=0, count=6):
        self.color = color
        self.RGBDropDown = _ComboBoxStub(index=index, count=count)


def test_capture_and_restore_display_state_roundtrip():
    stub = _GUIStub(color=3, index=3)
    state = GUI._capture_display_state(stub)

    stub.color = 0
    stub.RGBDropDown.setCurrentIndex(0)

    GUI._restore_display_state(stub, state)

    assert stub.color == 3
    assert stub.RGBDropDown.currentIndex() == 3
    assert stub.RGBDropDown.block_calls == [True, False]


def test_restore_display_state_clamps_to_valid_range():
    stub = _GUIStub(color=0, index=0, count=4)
    state = {"color": 12, "rgb_index": 12}

    GUI._restore_display_state(stub, state)

    assert stub.color == 3
    assert stub.RGBDropDown.currentIndex() == 3


def test_restore_display_state_handles_missing_state():
    stub = _GUIStub(color=2, index=2)

    GUI._restore_display_state(stub, None)

    assert stub.color == 2
    assert stub.RGBDropDown.currentIndex() == 2
