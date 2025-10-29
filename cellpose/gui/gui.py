"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

import sys, os, pathlib, warnings, datetime, time, copy

import copy
from qtpy import QtGui, QtCore
from superqt import QRangeSlider, QCollapsible
from qtpy.QtWidgets import QScrollArea, QMainWindow, QApplication, QWidget, QScrollBar, \
    QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, \
        QLineEdit, QMessageBox, QGroupBox
import pyqtgraph as pg

import numpy as np
import cv2

from . import guiparts, menus, io
from .diffhooks import note_manual_edit
from .diffcache import DiffStateCache
from .diffcrosshair import DiffCrosshairHub
from .. import models, core, dynamics, version, train
from ..utils import download_url_to_file, masks_to_outlines, diameters
from ..io import get_image_files, imsave, imread
from ..transforms import resize_image, normalize99, normalize99_tile, smooth_sharpen_img
from ..models import normalize_default
from ..plot import disk
from ..contrib.diff import contour_diff_rgb

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

Horizontal = QtCore.Qt.Orientation.Horizontal


class Slider(QRangeSlider):

    def __init__(self, parent, name, color):
        super().__init__(Horizontal)
        self.setEnabled(False)
        self.valueChanged.connect(lambda: self.levelChanged(parent))
        self.name = name

        self.setStyleSheet(""" QSlider{
                             background-color: transparent;
                             }
        """)
        self.show()

    def levelChanged(self, parent):
        parent.level_change(self.name)


class QHLine(QFrame):

    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setLineWidth(8)


def make_bwr():
    # make a bwr colormap
    b = np.append(255 * np.ones(128), np.linspace(0, 255, 128)[::-1])[:, np.newaxis]
    r = np.append(np.linspace(0, 255, 128), 255 * np.ones(128))[:, np.newaxis]
    g = np.append(np.linspace(0, 255, 128),
                  np.linspace(0, 255, 128)[::-1])[:, np.newaxis]
    color = np.concatenate((r, g, b), axis=-1).astype(np.uint8)
    bwr = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return bwr


def make_spectral():
    # make spectral colormap
    r = np.array([
        0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
        84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 128, 128, 128, 128, 128,
        128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 120, 112, 104, 96, 88,
        80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 11, 15, 19, 23,
        27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103,
        107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167,
        171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231,
        235, 239, 243, 247, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255
    ])
    g = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 4, 4, 3, 3,
        2, 2, 1, 1, 0, 0, 0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111,
        119, 127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239,
        247, 255, 247, 239, 231, 223, 215, 207, 199, 191, 183, 175, 167, 159, 151, 143,
        135, 128, 129, 131, 132, 134, 135, 137, 139, 140, 142, 143, 145, 147, 148, 150,
        151, 153, 154, 156, 158, 159, 161, 162, 164, 166, 167, 169, 170, 172, 174, 175,
        177, 178, 180, 181, 183, 185, 186, 188, 189, 191, 193, 194, 196, 197, 199, 201,
        202, 204, 205, 207, 208, 210, 212, 213, 215, 216, 218, 220, 221, 223, 224, 226,
        228, 229, 231, 232, 234, 235, 237, 239, 240, 242, 243, 245, 247, 248, 250, 251,
        253, 255, 251, 247, 243, 239, 235, 231, 227, 223, 219, 215, 211, 207, 203, 199,
        195, 191, 187, 183, 179, 175, 171, 167, 163, 159, 155, 151, 147, 143, 139, 135,
        131, 127, 123, 119, 115, 111, 107, 103, 99, 95, 91, 87, 83, 79, 75, 71, 67, 63,
        59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 15, 11, 7, 3, 0, 8, 16, 24, 32, 41,
        49, 57, 65, 74, 82, 90, 98, 106, 115, 123, 131, 139, 148, 156, 164, 172, 180,
        189, 197, 205, 213, 222, 230, 238, 246, 254
    ])
    b = np.array([
        0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127, 135, 143,
        151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239, 247, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 247,
        243, 239, 235, 231, 227, 223, 219, 215, 211, 207, 203, 199, 195, 191, 187, 183,
        179, 175, 171, 167, 163, 159, 155, 151, 147, 143, 139, 135, 131, 128, 126, 124,
        122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90,
        88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50,
        48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10,
        8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 16, 24, 32, 41, 49, 57, 65, 74,
        82, 90, 98, 106, 115, 123, 131, 139, 148, 156, 164, 172, 180, 189, 197, 205,
        213, 222, 230, 238, 246, 254
    ])
    color = (np.vstack((r, g, b)).T).astype(np.uint8)
    spectral = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return spectral


def make_cmap(cm=0):
    # make a single channel colormap
    r = np.arange(0, 256)
    color = np.zeros((256, 3))
    color[:, cm] = r
    color = color.astype(np.uint8)
    cmap = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return cmap


def run(image=None):
    from ..io import logger_setup
    logger, log_file = logger_setup()
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
    guip_path = pathlib.Path.home().joinpath(".cellpose", "cellposeSAM_gui.png")
    if not icon_path.is_file():
        cp_dir = pathlib.Path.home().joinpath(".cellpose")
        cp_dir.mkdir(exist_ok=True)
        print("downloading logo")
        download_url_to_file(
            "https://www.cellpose.org/static/images/cellpose_transparent.png",
            icon_path, progress=True)
    if not guip_path.is_file():
        print("downloading help window image")
        download_url_to_file("https://www.cellpose.org/static/images/cellposeSAM_gui.png",
                             guip_path, progress=True)
    icon_path = str(icon_path.resolve())
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(64, 64))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    app.setStyle("Fusion")
    app.setPalette(guiparts.DarkPalette())
    MainW(image=image, logger=logger)
    ret = app.exec_()
    sys.exit(ret)


class MainW(QMainWindow):

    def __init__(self, image=None, logger=None):
        super(MainW, self).__init__()

        self.logger = logger
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, 1200, 1000)
        self.setWindowTitle(f"cellpose v{version}")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))
        app_icon = QtGui.QIcon()
        icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
        icon_path = str(icon_path.resolve())
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(64, 64))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        # rgb(150,255,150)
        self.setStyleSheet(guiparts.stylesheet())

        menus.mainmenu(self)
        menus.editmenu(self)
        menus.modelmenu(self)
        menus.helpmenu(self)

        self.stylePressed = """QPushButton {Text-align: center;
                             background-color: rgb(150,50,150);
                             border-color: white;
                             color:white;}
                            QToolTip {
                           background-color: black;
                           color: white;
                           border: black solid 1px
                           }"""
        self.styleUnpressed = """QPushButton {Text-align: center;
                               background-color: rgb(50,50,50);
                                border-color: white;
                               color:white;}
                                QToolTip {
                           background-color: black;
                           color: white;
                           border: black solid 1px
                           }"""
        self.loaded = False

        # ---- MAIN WIDGET LAYOUT ---- #
        self.cwidget = QWidget(self)
        self.lmain = QGridLayout()
        self.cwidget.setLayout(self.lmain)
        self.setCentralWidget(self.cwidget)
        self.lmain.setVerticalSpacing(0)
        self.lmain.setContentsMargins(0, 0, 0, 10)

        self.imask = 0
        self.scrollarea = QScrollArea()
        self.scrollarea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollarea.setStyleSheet("""QScrollArea { border: none }""")
        self.scrollarea.setWidgetResizable(True)
        self.swidget = QWidget(self)
        self.scrollarea.setWidget(self.swidget)
        self.l0 = QGridLayout()
        self.swidget.setLayout(self.l0)
        b = self.make_buttons()
        self.lmain.addWidget(self.scrollarea, 0, 0, 39, 9)

        # ---- drawing area ---- #
        self.win = pg.GraphicsLayoutWidget()

        self.lmain.addWidget(self.win, 0, 9, 40, 30)

        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        self.lmain.setColumnStretch(10, 1)
        bwrmap = make_bwr()
        self.bwr = bwrmap.getLookupTable(start=0.0, stop=255.0, alpha=False)
        self.cmap = []
        # spectral colormap
        self.cmap.append(make_spectral().getLookupTable(start=0.0, stop=255.0,
                                                        alpha=False))
        # single channel colormaps
        for i in range(3):
            self.cmap.append(
                make_cmap(i).getLookupTable(start=0.0, stop=255.0, alpha=False))

        if MATPLOTLIB:
            self.colormap = (plt.get_cmap("gist_ncar")(np.linspace(0.0, .9, 1000000)) *
                             255).astype(np.uint8)
            np.random.seed(42)  # make colors stable
            self.colormap = self.colormap[np.random.permutation(1000000)]
        else:
            np.random.seed(42)  # make colors stable
            self.colormap = ((np.random.rand(1000000, 3) * 0.8 + 0.1) * 255).astype(
                np.uint8)
        self.NZ = 1
        self.restore = None
        self.ratio = 1.
        self.reset()

        # This needs to go after .reset() is called to get state fully set up:
        self.autobtn.checkStateChanged.connect(self.compute_saturation_if_checked)

        self.load_3D = False
        # Fallback resample flag for legacy code paths that expect a checkbox.
        # Some GUI configurations don't create the control, so default to True.
        self.resample = True

        # if called with image, load it
        if image is not None:
            self.filename = image
            io._load_image(self, self.filename)

        # training settings
        d = datetime.datetime.now()
        self.training_params = {
            "model_index": 0,
            "learning_rate": 1e-5,
            "weight_decay": 0.1,
            "n_epochs": 100,
            "model_name": "cpsam" + d.strftime("_%Y%m%d_%H%M%S"),
        }

        self.stitch_threshold = 0.
        self.flow3D_smooth = 0.
        self.anisotropy = 1.
        self.min_size = 15

        self.setAcceptDrops(True)
        self.win.show()
        self.show()

    def help_window(self):
        HW = guiparts.HelpWindow(self)
        HW.show()

    def train_help_window(self):
        THW = guiparts.TrainHelpWindow(self)
        THW.show()

    def gui_window(self):
        EG = guiparts.ExampleGUI(self)
        EG.show()

    def make_buttons(self):
        self.boldfont = QtGui.QFont("Arial", 11, QtGui.QFont.Bold)
        self.boldmedfont = QtGui.QFont("Arial", 9, QtGui.QFont.Bold)
        self.medfont = QtGui.QFont("Arial", 9)
        self.smallfont = QtGui.QFont("Arial", 8)

        b = 0
        self.satBox = QGroupBox("Views")
        self.satBox.setFont(self.boldfont)
        self.satBoxG = QGridLayout()
        self.satBox.setLayout(self.satBoxG)
        self.l0.addWidget(self.satBox, b, 0, 1, 9)

        widget_row = 0
        self.view = 0  # 0=image, 1=flowsXY, 2=flowsZ, 3=cellprob
        self.color = 0  # 0=RGB, 1=gray, 2=R, 3=G, 4=B
        self.RGBDropDown = QComboBox()
        self.RGBDropDown.addItems(
            ["RGB", "red=R", "green=G", "blue=B", "gray", "spectral"])
        self.RGBDropDown.setFont(self.medfont)
        self.RGBDropDown.currentIndexChanged.connect(self.color_choose)
        self.satBoxG.addWidget(self.RGBDropDown, widget_row, 0, 1, 3)

        label = QLabel("<p>[&uarr; / &darr; or W/S]</p>")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, widget_row, 3, 1, 3)
        label = QLabel("[R / G / B \n toggles color ]")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, widget_row, 6, 1, 3)

        widget_row += 1
        self.ViewDropDown = QComboBox()
        self.ViewDropDown.addItems(["image", "gradXY", "cellprob", "restored"])
        self.ViewDropDown.setFont(self.medfont)
        self.ViewDropDown.model().item(3).setEnabled(False)
        self.ViewDropDown.currentIndexChanged.connect(self.update_plot)
        self.satBoxG.addWidget(self.ViewDropDown, widget_row, 0, 2, 3)

        label = QLabel("[pageup / pagedown]")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, widget_row, 3, 1, 5)

        widget_row += 2
        label = QLabel("")
        label.setToolTip(
            "NOTE: manually changing the saturation bars does not affect normalization in segmentation"
        )
        self.satBoxG.addWidget(label, widget_row, 0, 1, 5)

        self.autobtn = QCheckBox("auto-adjust saturation")
        self.autobtn.setToolTip("sets scale-bars as normalized for segmentation")
        self.autobtn.setFont(self.medfont)
        self.autobtn.setChecked(True)
        self.satBoxG.addWidget(self.autobtn, widget_row, 1, 1, 8)

        widget_row += 1
        self.sliders = []
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 100, 100]]
        colornames = ["red", "Chartreuse", "DodgerBlue"]
        names = ["red", "green", "blue"]
        for r in range(3):
            widget_row += 1
            if r == 0:
                label = QLabel('<font color="gray">gray/</font><br>red')
            else:
                label = QLabel(names[r] + ":")
            label.setStyleSheet(f"color: {colornames[r]}")
            label.setFont(self.boldmedfont)
            self.satBoxG.addWidget(label, widget_row, 0, 1, 2)
            self.sliders.append(Slider(self, names[r], colors[r]))
            self.sliders[-1].setMinimum(-.1)
            self.sliders[-1].setMaximum(255.1)
            self.sliders[-1].setValue([0, 255])
            self.sliders[-1].setToolTip(
                "NOTE: manually changing the saturation bars does not affect normalization in segmentation"
            )
            self.satBoxG.addWidget(self.sliders[-1], widget_row, 2, 1, 7)

        b += 1
        self.drawBox = QGroupBox("Drawing")
        self.drawBox.setFont(self.boldfont)
        self.drawBoxG = QGridLayout()
        self.drawBox.setLayout(self.drawBoxG)
        self.l0.addWidget(self.drawBox, b, 0, 1, 9)
        self.autosave = True

        widget_row = 0
        self.brush_size = 3
        self.BrushChoose = QComboBox()
        self.BrushChoose.addItems(["1", "3", "5", "7", "9"])
        self.BrushChoose.currentIndexChanged.connect(self.brush_choose)
        self.BrushChoose.setFixedWidth(40)
        self.BrushChoose.setFont(self.medfont)
        self.drawBoxG.addWidget(self.BrushChoose, widget_row, 3, 1, 2)
        label = QLabel("brush size:")
        label.setFont(self.medfont)
        self.drawBoxG.addWidget(label, widget_row, 0, 1, 3)

        widget_row += 1
        # turn off masks
        self.layer_off = False
        self.masksOn = True
        self.MCheckBox = QCheckBox("MASKS ON [X]")
        self.MCheckBox.setFont(self.medfont)
        self.MCheckBox.setChecked(True)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.drawBoxG.addWidget(self.MCheckBox, widget_row, 0, 1, 5)

        widget_row += 1
        # turn off outlines
        self.outlinesOn = False  # turn off by default
        self.OCheckBox = QCheckBox("outlines on [Z]")
        self.OCheckBox.setFont(self.medfont)
        self.drawBoxG.addWidget(self.OCheckBox, widget_row, 0, 1, 5)
        self.OCheckBox.setChecked(False)
        self.OCheckBox.toggled.connect(self.toggle_masks)

        widget_row += 1
        self.SCheckBox = QCheckBox("single stroke")
        self.SCheckBox.setFont(self.medfont)
        self.SCheckBox.setChecked(True)
        self.SCheckBox.toggled.connect(self.autosave_on)
        self.SCheckBox.setEnabled(True)
        self.drawBoxG.addWidget(self.SCheckBox, widget_row, 0, 1, 5)

        # buttons for deleting multiple cells
        self.deleteBox = QGroupBox("delete multiple ROIs")
        self.deleteBox.setStyleSheet("color: rgb(200, 200, 200)")
        self.deleteBox.setFont(self.medfont)
        self.deleteBoxG = QGridLayout()
        self.deleteBox.setLayout(self.deleteBoxG)
        self.drawBoxG.addWidget(self.deleteBox, 0, 5, 4, 4)
        self.MakeDeletionRegionButton = QPushButton("region-select")
        self.MakeDeletionRegionButton.clicked.connect(self.remove_region_cells)
        self.deleteBoxG.addWidget(self.MakeDeletionRegionButton, 0, 0, 1, 4)
        self.MakeDeletionRegionButton.setFont(self.smallfont)
        self.MakeDeletionRegionButton.setFixedWidth(70)
        self.DeleteMultipleROIButton = QPushButton("click-select")
        self.DeleteMultipleROIButton.clicked.connect(self.delete_multiple_cells)
        self.deleteBoxG.addWidget(self.DeleteMultipleROIButton, 1, 0, 1, 4)
        self.DeleteMultipleROIButton.setFont(self.smallfont)
        self.DeleteMultipleROIButton.setFixedWidth(70)
        self.DoneDeleteMultipleROIButton = QPushButton("done")
        self.DoneDeleteMultipleROIButton.clicked.connect(
            self.done_remove_multiple_cells)
        self.deleteBoxG.addWidget(self.DoneDeleteMultipleROIButton, 2, 0, 1, 2)
        self.DoneDeleteMultipleROIButton.setFont(self.smallfont)
        self.DoneDeleteMultipleROIButton.setFixedWidth(35)
        self.CancelDeleteMultipleROIButton = QPushButton("cancel")
        self.CancelDeleteMultipleROIButton.clicked.connect(self.cancel_remove_multiple)
        self.deleteBoxG.addWidget(self.CancelDeleteMultipleROIButton, 2, 2, 1, 2)
        self.CancelDeleteMultipleROIButton.setFont(self.smallfont)
        self.CancelDeleteMultipleROIButton.setFixedWidth(35)

        b += 1
        widget_row = 0
        self.segBox = QGroupBox("Segmentation")
        self.segBoxG = QGridLayout()
        self.segBox.setLayout(self.segBoxG)
        self.l0.addWidget(self.segBox, b, 0, 1, 9)
        self.segBox.setFont(self.boldfont)

        widget_row += 1

        # use GPU
        self.useGPU = QCheckBox("use GPU")
        self.useGPU.setToolTip(
            "if you have specially installed the <i>cuda</i> version of torch, then you can activate this"
        )
        self.useGPU.setFont(self.medfont)
        self.check_gpu()
        self.segBoxG.addWidget(self.useGPU, widget_row, 0, 1, 3)

        # compute segmentation with general models
        self.net_text = ["run CPSAM"]
        nett = ["cellpose super-generalist model"]

        self.StyleButtons = []
        jj = 4
        for j in range(len(self.net_text)):
            self.StyleButtons.append(
                guiparts.ModelButton(self, self.net_text[j], self.net_text[j]))
            w = 5
            self.segBoxG.addWidget(self.StyleButtons[-1], widget_row, jj, 1, w)
            jj += w
            self.StyleButtons[-1].setToolTip(nett[j])

        widget_row += 1
        self.ncells = guiparts.ObservableVariable(0)
        self.roi_count = QLabel()
        self.roi_count.setFont(self.boldfont)
        self.roi_count.setAlignment(QtCore.Qt.AlignLeft)
        self.ncells.valueChanged.connect(
            lambda n: self.roi_count.setText(f'{str(n)} ROIs')
        )

        self.segBoxG.addWidget(self.roi_count, widget_row, 0, 1, 4)

        self.progress = QProgressBar(self)
        self.segBoxG.addWidget(self.progress, widget_row, 4, 1, 5)

        widget_row += 1

        self.diffButton = QPushButton("diff")
        self.diffButton.setFont(self.medfont)
        self.diffButton.setEnabled(False)
        self.diffButton.clicked.connect(self.show_segmentation_diff)
        self.segBoxG.addWidget(self.diffButton, widget_row, 0, 1, 3)

        self.maskToggleButton = QPushButton("reset mask")
        self.maskToggleButton.setFont(self.medfont)
        self.maskToggleButton.setEnabled(False)
        self.maskToggleButton.clicked.connect(self.toggle_mask_restore)
        self.segBoxG.addWidget(self.maskToggleButton, widget_row, 3, 1, 3)

        self._diff_state_cache = DiffStateCache()
        self._diff_fig = None
        self._diff_ax = None
        self._diff_img_im = None
        self._diff_diff_rgb = None
        self._diff_click_cid = None
        self._diff_scroll_cid = None
        self._diff_z_index = None
        self._diff_crosshair_lines = None
        self._diff_last_shape = None
        self._diff_last_crosshair = None
        self._diff_drag_active = False
        self._diff_drag_last_update = 0.0
        self._diff_drag_interval = 1.0 / 15.0
        self._diff_zoom_min_span = 5.0

        self._diff_state_old = None
        self._diff_state_new = None
        self._diff_showing_restored = False
        self._diff_state_old_manual_override = False
        self._diff_overlay_reference_masks = None
        self._diff_overlay_reference_active = False
        self._diff_preserve_overlay_reference = False
        self._diff_crosshair_suppress_broadcast = False
        self._diff_crosshair_hub = DiffCrosshairHub.instance()
        self._diff_crosshair_hub.register(self)

        widget_row += 1

        ############################### Segmentation settings ###############################
        self.additional_seg_settings_qcollapsible = QCollapsible("additional settings")
        self.additional_seg_settings_qcollapsible.setFont(self.medfont)
        self.additional_seg_settings_qcollapsible._toggle_btn.setFont(self.medfont)
        self.segmentation_settings = guiparts.SegmentationSettings(self.medfont)
        self.additional_seg_settings_qcollapsible.setContent(self.segmentation_settings)
        self.segBoxG.addWidget(self.additional_seg_settings_qcollapsible, widget_row, 0, 1, 9)

        # connect edits to image processing steps:
        self.segmentation_settings.diameter_box.editingFinished.connect(self.update_scale)
        self.segmentation_settings.flow_threshold_box.returnPressed.connect(self.compute_cprob)
        self.segmentation_settings.cellprob_threshold_box.returnPressed.connect(self.compute_cprob)
        self.segmentation_settings.niter_box.returnPressed.connect(self.compute_cprob)

        # Needed to do this for the drop down to not be open on startup
        self.additional_seg_settings_qcollapsible._toggle_btn.setChecked(True)
        self.additional_seg_settings_qcollapsible._toggle_btn.setChecked(False)

        b += 1
        self.modelBox = QGroupBox("user-trained models")
        self.modelBoxG = QGridLayout()
        self.modelBox.setLayout(self.modelBoxG)
        self.l0.addWidget(self.modelBox, b, 0, 1, 9)
        self.modelBox.setFont(self.boldfont)
        # choose models
        self.ModelChooseC = QComboBox()
        self.ModelChooseC.setFont(self.medfont)
        current_index = 0
        self.ModelChooseC.addItems(["custom models"])
        if len(self.model_strings) > 0:
            self.ModelChooseC.addItems(self.model_strings)
        self.ModelChooseC.setFixedWidth(175)
        self.ModelChooseC.setCurrentIndex(current_index)
        tipstr = 'add or train your own models in the "Models" file menu and choose model here'
        self.ModelChooseC.setToolTip(tipstr)
        self.ModelChooseC.activated.connect(lambda: self.model_choose(custom=True))
        self.modelBoxG.addWidget(self.ModelChooseC, widget_row, 0, 1, 8)

        # compute segmentation w/ custom model
        self.ModelButtonC = QPushButton(u"run")
        self.ModelButtonC.setFont(self.medfont)
        self.ModelButtonC.setFixedWidth(35)
        self.ModelButtonC.clicked.connect(
            lambda: self.compute_segmentation(custom=True))
        self.modelBoxG.addWidget(self.ModelButtonC, widget_row, 8, 1, 1)
        self.ModelButtonC.setEnabled(False)


        b += 1
        self.filterBox = QGroupBox("Image filtering")
        self.filterBox.setFont(self.boldfont)
        self.filterBox_grid_layout = QGridLayout()
        self.filterBox.setLayout(self.filterBox_grid_layout)
        self.l0.addWidget(self.filterBox, b, 0, 1, 9)

        b0 = 0

        # DENOISING
        self.DenoiseButtons = []
        widget_row = 0

        # Filtering
        self.FilterButtons = []
        nett = [
            "clear restore/filter",
            "filter image (settings below)",
        ]
        self.filter_text = ["none",
                             "filter",
                             ]
        self.restore = None
        self.ratio = 1.
        jj = 0
        w = 3
        for j in range(len(self.filter_text)):
            self.FilterButtons.append(
                guiparts.FilterButton(self, self.filter_text[j]))
            self.filterBox_grid_layout.addWidget(self.FilterButtons[-1], widget_row, jj, 1, w)
            self.FilterButtons[-1].setFixedWidth(75)
            self.FilterButtons[-1].setToolTip(nett[j])
            self.FilterButtons[-1].setFont(self.medfont)
            widget_row += 1 if j%2==1 else 0
            jj = 0 if j%2==1 else jj + w

        self.save_norm = QCheckBox("save restored/filtered image")
        self.save_norm.setFont(self.medfont)
        self.save_norm.setToolTip("save restored/filtered image in _seg.npy file")
        self.save_norm.setChecked(True)

        widget_row += 2

        self.filtBox = QCollapsible("custom filter settings")
        self.filtBox._toggle_btn.setFont(self.medfont)
        self.filtBoxG = QGridLayout()
        _content = QWidget()
        _content.setLayout(self.filtBoxG)
        _content.setMaximumHeight(0)
        _content.setMinimumHeight(0)
        self.filtBox.setContent(_content)
        self.filterBox_grid_layout.addWidget(self.filtBox, widget_row, 0, 1, 9)

        self.filt_vals = [0., 0., 0., 0.]
        self.filt_edits = []
        labels = [
            "sharpen\nradius", "smooth\nradius", "tile_norm\nblocksize",
            "tile_norm\nsmooth3D"
        ]
        tooltips = [
            "set size of surround-subtraction filter for sharpening image",
            "set size of gaussian filter for smoothing image",
            "set size of tiles to use to normalize image",
            "set amount of smoothing of normalization values across planes"
        ]

        for p in range(4):
            label = QLabel(f"{labels[p]}:")
            label.setToolTip(tooltips[p])
            label.setFont(self.medfont)
            self.filtBoxG.addWidget(label, widget_row + p // 2, 4 * (p % 2), 1, 2)
            self.filt_edits.append(QLineEdit())
            self.filt_edits[p].setText(str(self.filt_vals[p]))
            self.filt_edits[p].setFixedWidth(40)
            self.filt_edits[p].setFont(self.medfont)
            self.filtBoxG.addWidget(self.filt_edits[p], widget_row + p // 2, 4 * (p % 2) + 2, 1,
                                    2)
            self.filt_edits[p].setToolTip(tooltips[p])

        widget_row += 3
        self.norm3D_cb = QCheckBox("norm3D")
        self.norm3D_cb.setFont(self.medfont)
        self.norm3D_cb.setChecked(True)
        self.norm3D_cb.setToolTip("run same normalization across planes")
        self.filtBoxG.addWidget(self.norm3D_cb, widget_row, 0, 1, 3)


        return b

    def level_change(self, r):
        r = ["red", "green", "blue"].index(r)
        if self.loaded:
            sval = self.sliders[r].value()
            self.saturation[r][self.currentZ] = sval
            if not self.autobtn.isChecked():
                for r in range(3):
                    for i in range(len(self.saturation[r])):
                        self.saturation[r][i] = self.saturation[r][self.currentZ]
            self.update_plot()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space and not event.isAutoRepeat():
            self._diff_drag_active = True
            self._diff_drag_last_update = 0.0
            event.accept()
            return
        if (event.key() == QtCore.Qt.Key_X and
                event.modifiers() & QtCore.Qt.ShiftModifier and
                not event.isAutoRepeat()):
            if self.loaded:
                self._clear_anchor_points()
            event.accept()
            return
        if self.loaded:
            if not (event.modifiers() &
                    (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier |
                     QtCore.Qt.AltModifier) or self.in_stroke):
                updated = False
                if len(self.current_point_set) > 0:
                    if event.key() == QtCore.Qt.Key_Return:
                        self.add_set()
                else:
                    nviews = self.ViewDropDown.count() - 1
                    nviews += int(
                        self.ViewDropDown.model().item(self.ViewDropDown.count() -
                                                       1).isEnabled())
                    if event.key() == QtCore.Qt.Key_X:
                        self.MCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_Z:
                        self.OCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_Left or event.key(
                    ) == QtCore.Qt.Key_A:
                        self.get_prev_image()
                    elif event.key() == QtCore.Qt.Key_Right or event.key(
                    ) == QtCore.Qt.Key_D:
                        self.get_next_image()
                    elif event.key() == QtCore.Qt.Key_PageDown:
                        self.view = (self.view + 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)
                    elif event.key() == QtCore.Qt.Key_PageUp:
                        self.view = (self.view - 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)

                # can change background or stroke size if cell not finished
                if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_W:
                    self.color = (self.color - 1) % (6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_Down or event.key(
                ) == QtCore.Qt.Key_S:
                    self.color = (self.color + 1) % (6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_R:
                    if self.color != 1:
                        self.color = 1
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_G:
                    if self.color != 2:
                        self.color = 2
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_B:
                    if self.color != 3:
                        self.color = 3
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif (event.key() == QtCore.Qt.Key_Comma or
                      event.key() == QtCore.Qt.Key_Period):
                    count = self.BrushChoose.count()
                    gci = self.BrushChoose.currentIndex()
                    if event.key() == QtCore.Qt.Key_Comma:
                        gci = max(0, gci - 1)
                    else:
                        gci = min(count - 1, gci + 1)
                    self.BrushChoose.setCurrentIndex(gci)
                    self.brush_choose()
                if not updated:
                    self.update_plot()
        if event.key() == QtCore.Qt.Key_Minus or event.key() == QtCore.Qt.Key_Equal:
            self.p0.keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space and not event.isAutoRepeat():
            self._diff_drag_active = False
            event.accept()
            return
        super().keyReleaseEvent(event)

    def autosave_on(self):
        if self.SCheckBox.isChecked():
            self.autosave = True
        else:
            self.autosave = False

    def check_gpu(self, torch=True):
        # also decide whether or not to use torch
        self.useGPU.setChecked(False)
        self.useGPU.setEnabled(False)
        if core.use_gpu(use_torch=True):
            self.useGPU.setEnabled(True)
            self.useGPU.setChecked(True)
        else:
            self.useGPU.setStyleSheet("color: rgb(80,80,80);")


    def model_choose(self, custom=False):
        index = self.ModelChooseC.currentIndex(
        ) if custom else self.ModelChooseB.currentIndex()
        if index > 0:
            if custom:
                model_name = self.ModelChooseC.currentText()
            else:
                model_name = self.net_names[index - 1]
            print(f"GUI_INFO: selected model {model_name}, loading now")
            self.initialize_model(model_name=model_name, custom=custom)

    def toggle_scale(self):
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True

    def enable_buttons(self):
        if len(self.model_strings) > 0:
            self.ModelButtonC.setEnabled(True)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(True)

        for i in range(len(self.FilterButtons)):
            self.FilterButtons[i].setEnabled(True)
        if self.load_3D:
            self.FilterButtons[-2].setEnabled(False)

        self.newmodel.setEnabled(True)
        self.loadMasks.setEnabled(True)

        for n in range(self.nchan):
            self.sliders[n].setEnabled(True)
        for n in range(self.nchan, 3):
            self.sliders[n].setEnabled(True)

        self.toggle_mask_ops()

        self.update_plot()
        self.setWindowTitle(self.filename)
        self._diff_refresh_seg_path()

    def disable_buttons_removeROIs(self):
        if len(self.model_strings) > 0:
            self.ModelButtonC.setEnabled(False)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(False)
        self.newmodel.setEnabled(False)
        self.loadMasks.setEnabled(False)
        self.saveSet.setEnabled(False)
        self.savePNG.setEnabled(False)
        self.saveFlows.setEnabled(False)
        self.saveOutlines.setEnabled(False)
        self.saveROIs.setEnabled(False)

        self.MakeDeletionRegionButton.setEnabled(False)
        self.DeleteMultipleROIButton.setEnabled(False)
        self.DoneDeleteMultipleROIButton.setEnabled(True)
        self.CancelDeleteMultipleROIButton.setEnabled(True)

    def toggle_mask_ops(self):
        self.update_layer()
        self.toggle_saving()
        self.toggle_removals()
        self._diff_update_button_state()

    def toggle_saving(self):
        if self.ncells > 0:
            self.saveSet.setEnabled(True)
            self.savePNG.setEnabled(True)
            self.saveFlows.setEnabled(True)
            self.saveOutlines.setEnabled(True)
            self.saveROIs.setEnabled(True)
        else:
            self.saveSet.setEnabled(False)
            self.savePNG.setEnabled(False)
            self.saveFlows.setEnabled(False)
            self.saveOutlines.setEnabled(False)
            self.saveROIs.setEnabled(False)

    def _diff_refresh_seg_path(self):
        self._diff_state_old = None
        self._diff_state_old_manual_override = False
        if isinstance(self.filename, str) and self.filename:
            candidate = os.path.splitext(self.filename)[0] + "_seg.npy"
            if os.path.exists(candidate):
                self._diff_seg_path = candidate
                self._diff_get_saved_state(reload=True)
            else:
                self._diff_seg_path = None
        else:
            self._diff_seg_path = None
        self._diff_update_button_state()

    def _diff_update_button_state(self):
        if not hasattr(self, "diffButton"):
            return
        seg_path = self._diff_seg_file()
        has_saved = seg_path is not None and os.path.exists(seg_path)
        has_prediction = isinstance(self._diff_latest_masks, np.ndarray)
        prediction_dims_ok = False
        if has_prediction:
            try:
                prediction_dims_ok = np.asarray(self._diff_latest_masks).ndim >= 2
            except Exception:
                prediction_dims_ok = False
        enabled = bool(has_saved and has_prediction and prediction_dims_ok)

        self.diffButton.setEnabled(enabled)
        if not has_saved:
            diff_tip = "Load an image with an existing _seg.npy to enable diff."
        elif not has_prediction or not prediction_dims_ok:
            diff_tip = "Run a segmentation model to enable diff."
        else:
            diff_tip = ("Compare current Z plane against saved _seg.npy."
                        if getattr(self, "NZ", 1) > 1
                        else "Open contour diff against saved _seg.npy.")
            if not MATPLOTLIB:
                diff_tip += " (Matplotlib required; click to see installation hint.)"
        self.diffButton.setToolTip(diff_tip)

        if hasattr(self, "maskToggleButton"):
            can_reset, reset_tip = self._diff_can_reset()
            self.maskToggleButton.setEnabled(can_reset)
            if self._diff_showing_restored:
                self.maskToggleButton.setText("show new mask")
                base_tip = "Show latest model prediction"
            else:
                self.maskToggleButton.setText("reset mask")
                base_tip = "Show saved _seg.npy segmentation"
            self.maskToggleButton.setToolTip(base_tip if can_reset else reset_tip)

    def _diff_cache_key(self, filename=None):
        path = filename if filename is not None else getattr(self, "filename", None)
        if isinstance(path, str) and path:
            try:
                return os.path.abspath(path)
            except Exception:
                return path
        return None

    def _diff_cache_before_image_change(self):
        cache = getattr(self, "_diff_state_cache", None)
        if cache is None:
            return
        key = self._diff_cache_key()
        if key is None:
            return
        state_new = getattr(self, "_diff_state_new", None)
        has_masks = isinstance(state_new, dict) and state_new.get("masks") is not None
        if not has_masks:
            cache.discard(key)
            return
        latest_masks = self._diff_latest_masks if isinstance(self._diff_latest_masks, np.ndarray) else None
        active_state = state_new if not self._diff_showing_restored else self._diff_get_saved_state(reload=False)
        cache.store(
            key,
            new_state=state_new,
            latest_masks=latest_masks,
            active_state=active_state,
            showing_restored=self._diff_showing_restored,
            crosshair=self._diff_last_crosshair,
            z_index=self._diff_z_index,
            last_shape=self._diff_last_shape,
        )

    def _diff_restore_after_image_load(self):
        cache = getattr(self, "_diff_state_cache", None)
        if cache is None:
            self._diff_update_button_state()
            return
        key = self._diff_cache_key()
        entry = cache.retrieve(key)
        if not entry:
            self._diff_update_button_state()
            return
        showing_restored = bool(entry.get("showing_restored", False))
        new_state = entry.get("new_state")
        active_state = entry.get("active_state")
        latest_masks = entry.get("latest_masks")
        crosshair = entry.get("crosshair")
        z_index = entry.get("z_index")
        last_shape = entry.get("last_shape")
        try:
            if showing_restored:
                if active_state is not None:
                    self._diff_apply_state(active_state, treat_as_new=False)
                else:
                    state_old = self._diff_get_saved_state(reload=True)
                    if state_old is not None:
                        self._diff_apply_state(state_old, treat_as_new=False)
                self._diff_state_new = new_state
                if isinstance(latest_masks, np.ndarray):
                    self._diff_latest_masks = np.array(latest_masks, copy=True)
                elif isinstance(new_state, dict) and new_state.get("masks") is not None:
                    self._diff_latest_masks = np.array(new_state["masks"], copy=True)
                else:
                    self._diff_latest_masks = None
            else:
                if active_state is None:
                    active_state = new_state
                if active_state is not None:
                    self._diff_apply_state(active_state, treat_as_new=True)
                else:
                    self._diff_state_new = None
                    self._diff_latest_masks = None
            self._diff_showing_restored = showing_restored
            self._diff_last_crosshair = tuple(crosshair) if crosshair is not None else None
            self._diff_last_shape = last_shape
            self._diff_z_index = z_index
            if self._diff_last_crosshair is not None:
                self._diff_update_crosshair_lines(
                    self._diff_last_crosshair, reason="restore"
                )
        except Exception as exc:
            print(f"GUI_WARNING: failed to restore diff state for {key}: {exc}")
            cache.discard(key)
        finally:
            self._diff_update_button_state()

    def get_crosshair_coords(self):
        return self._diff_last_crosshair

    def _diff_reset_state(self):
        self._diff_fig = None
        self._diff_ax = None
        self._diff_img_im = None
        self._diff_diff_rgb = None
        self._diff_click_cid = None
        self._diff_scroll_cid = None
        self._diff_z_index = None
        self._diff_crosshair_lines = None
        self._diff_last_shape = None
        self._diff_last_crosshair = None
        self._diff_state_old_manual_override = False
        self._diff_crosshair_suppress_broadcast = False
        self._diff_clear_overlay_reference()

    def _diff_clear_overlay_reference(self):
        self._diff_overlay_reference_masks = None
        self._diff_overlay_reference_active = False
        self._diff_preserve_overlay_reference = False

    def _diff_on_close(self, event):
        if getattr(event, "canvas", None) is None:
            return
        if getattr(event.canvas, "figure", None) is not getattr(self, "_diff_fig", None):
            return
        self._diff_disconnect_handlers(event.canvas)
        self._diff_reset_state()

    def _diff_disconnect_handlers(self, canvas):
        if canvas is None:
            return
        if self._diff_click_cid is not None:
            try:
                canvas.mpl_disconnect(self._diff_click_cid)
            except Exception:
                pass
        if self._diff_scroll_cid is not None:
            try:
                canvas.mpl_disconnect(self._diff_scroll_cid)
            except Exception:
                pass

    def _diff_close_existing(self):
        fig = getattr(self, "_diff_fig", None)
        if fig is None:
            return
        canvas = getattr(fig, "canvas", None)
        if canvas is not None:
            self._diff_disconnect_handlers(canvas)
        try:
            plt.close(fig)
        except Exception as exc:
            print(f"GUI_WARNING: failed to close existing diff viewer: {exc}")
        self._diff_reset_state()

    def _diff_update_crosshair_lines(self, coords=None, *, reason=None):
        if not MATPLOTLIB:
            return
        fig = getattr(self, "_diff_fig", None)
        ax = getattr(self, "_diff_ax", None)
        if fig is None or ax is None:
            return
        if not plt.fignum_exists(fig.number):
            self._diff_reset_state()
            return
        if coords is None:
            coords = self.get_crosshair_coords()
        self._diff_last_crosshair = coords
        if coords is None:
            return
        y, x = map(float, coords)
        if self._diff_last_shape:
            h, w = self._diff_last_shape
            if h > 0 and w > 0:
                y = float(np.clip(y, 0, h - 1))
                x = float(np.clip(x, 0, w - 1))
        if self._diff_crosshair_lines is None:
            line_kwargs = {"color": "cyan", "linewidth": 0.8, "alpha": 0.7, "linestyle": "--"}
            hline = ax.axhline(y, **line_kwargs)
            vline = ax.axvline(x, **line_kwargs)
            self._diff_crosshair_lines = (hline, vline)
        else:
            hline, vline = self._diff_crosshair_lines
            hline.set_ydata([y, y])
            vline.set_xdata([x, x])
        fig.canvas.draw_idle()
        if (
            hasattr(self, "_diff_crosshair_hub")
            and self._diff_crosshair_hub is not None
            and not getattr(self, "_diff_crosshair_suppress_broadcast", False)
        ):
            self._diff_crosshair_hub.set_coords(
                (y, x), source=self, reason=reason or "local"
            )

    def diff_crosshair_updated(self, coords, *, source=None, reason=None):
        if coords is None:
            return
        try:
            y, x = coords
        except (TypeError, ValueError):
            return
        self._diff_last_crosshair = (float(y), float(x))
        prev = self._diff_crosshair_suppress_broadcast
        self._diff_crosshair_suppress_broadcast = True
        try:
            self._diff_update_crosshair_lines(
                self._diff_last_crosshair, reason=reason or "hub"
            )
        finally:
            self._diff_crosshair_suppress_broadcast = prev

    def _diff_can_reset(self) -> tuple[bool, str]:
        state_old = self._diff_get_saved_state(reload=False)
        if state_old is None or state_old.get("masks") is None:
            return False, "Saved _seg.npy masks unavailable for reset."
        if self._diff_state_new is None or self._diff_state_new.get("masks") is None:
            return False, "Run a segmentation model before resetting masks."
        try:
            old_masks = np.asarray(state_old["masks"])
            new_masks = np.asarray(self._diff_state_new["masks"])
        except Exception:
            return False, "Masks could not be compared for reset."
        old_masks = np.squeeze(old_masks)
        new_masks = np.squeeze(new_masks)
        if old_masks.ndim == 2:
            old_masks = old_masks[np.newaxis, ...]
        if new_masks.ndim == 2:
            new_masks = new_masks[np.newaxis, ...]
        if old_masks.shape != new_masks.shape:
            return False, "Saved masks shape does not match the model result."
        return True, ""

    def _diff_get_saved_state(self, reload: bool = False):
        manual_override = getattr(self, "_diff_state_old_manual_override", False)
        if manual_override and self._diff_state_old is not None:
            return self._diff_state_old
        seg_path = self._diff_seg_file()
        if seg_path is None:
            self._diff_state_old = None
            self._diff_state_old_manual_override = False
            return None
        if not reload and self._diff_state_old is not None:
            return self._diff_state_old
        try:
            dat = np.load(seg_path, allow_pickle=True).item()
        except Exception as exc:
            print(f"GUI_WARNING: failed to load saved masks for diff: {exc}")
            self._diff_state_old = None
            self._diff_state_old_manual_override = False
            return None
        masks = dat.get("masks")
        if masks is None:
            self._diff_state_old = None
            self._diff_state_old_manual_override = False
            return None
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        outlines = dat.get("outlines")
        if outlines is not None:
            outlines = np.asarray(outlines)
            if outlines.ndim == 2:
                outlines = outlines[np.newaxis, ...]
        colors = dat.get("colors")
        if colors is not None:
            colors = np.asarray(colors)
        state = {
            "masks": masks.astype(masks.dtype, copy=True),
            "outlines": outlines.astype(outlines.dtype, copy=True) if isinstance(outlines, np.ndarray) else outlines,
            "colors": colors.astype(colors.dtype, copy=True) if isinstance(colors, np.ndarray) else colors,
        }
        self._diff_state_old = state
        self._diff_state_old_manual_override = False
        return state

    def _diff_store_current_as_new(self):
        try:
            masks = np.asarray(self.cellpix).copy()
            outlines = np.asarray(self.outpix).copy()
            getter = getattr(self.ncells, "get", None)
            if callable(getter):
                try:
                    ncells = int(getter())
                except Exception:
                    ncells = 0
            else:
                try:
                    ncells = int(self.ncells)
                except Exception:
                    ncells = 0
            colors = None
            if ncells > 0 and isinstance(self.cellcolors, np.ndarray):
                colors_slice = self.cellcolors[1:ncells + 1]
                if colors_slice.size > 0:
                    colors = colors_slice.copy()
        except Exception:
            self._diff_state_new = None
            self._diff_latest_masks = None
            return
        preserve_overlay = bool(getattr(self, "_diff_preserve_overlay_reference", False))
        if not preserve_overlay:
            self._diff_clear_overlay_reference()
        self._diff_state_new = {
            "masks": masks,
            "outlines": outlines,
            "colors": colors,
        }
        self._diff_state_old_manual_override = False
        if not preserve_overlay or self._diff_latest_masks is None:
            self._diff_latest_masks = masks.copy()

    def _diff_note_manual_edit(self):
        """Update diff caches after any manual mask changes."""
        if not hasattr(self, "_diff_seg_path"):
            return
        note_manual_edit(self)

    def _diff_apply_state(self, state: dict, treat_as_new: bool):
        if state is None or state.get("masks") is None:
            raise ValueError("diff state missing masks")
        masks = np.array(state["masks"], copy=True)
        outlines = state.get("outlines")
        colors = state.get("colors")
        outlines_copy = np.array(outlines, copy=True) if isinstance(outlines, np.ndarray) else outlines
        colors_copy = np.array(colors, copy=True) if isinstance(colors, np.ndarray) else colors
        io._masks_to_gui(self, masks, outlines=outlines_copy, colors=colors_copy)
        if treat_as_new:
            self._diff_store_current_as_new()
        # refresh overlay if open
        try:
            self._diff_refresh_overlay()
        except Exception:
            pass

    # ===== Interactive diff viewer helpers =====
    def _diff_get_planes(self):
        """Return (saved_plane, current_plane, z_index) for active Z."""
        state_old = self._diff_get_saved_state(reload=False)
        if state_old is None:
            return None
        state_new = self._diff_state_new
        reference_active = bool(getattr(self, "_diff_overlay_reference_active", False))
        reference_masks = self._diff_overlay_reference_masks if reference_active else None

        old_masks = np.asarray(state_old.get("masks"))

        if reference_active and isinstance(reference_masks, np.ndarray):
            new_masks = np.asarray(reference_masks)
        else:
            if state_new is None or state_new.get("masks") is None:
                return None
            new_masks = np.asarray(state_new.get("masks"))

        if old_masks.ndim == 2:
            z_idx = 0
            saved_plane = old_masks
        else:
            z_idx = int(np.clip(self.currentZ, 0, old_masks.shape[0] - 1))
            saved_plane = old_masks[z_idx]
        if new_masks.ndim == 2:
            current_plane = new_masks
        else:
            z_idx = int(np.clip(self.currentZ, 0, new_masks.shape[0] - 1))
            current_plane = new_masks[z_idx]
        return saved_plane, current_plane, z_idx

    def _diff_recompute_overlay(self):
        planes = self._diff_get_planes()
        if planes is None:
            return None
        saved_plane, current_plane, z_idx = planes
        try:
            diff_rgb = contour_diff_rgb(saved_plane.astype(np.int32),
                                        current_plane.astype(np.int32))
        except Exception:
            return None
        self._diff_diff_rgb = diff_rgb
        self._diff_last_shape = diff_rgb.shape[:2]
        self._diff_z_index = z_idx
        return diff_rgb

    def _diff_refresh_overlay(self):
        if not MATPLOTLIB:
            return
        if self._diff_fig is None or self._diff_ax is None:
            return
        if not plt.fignum_exists(self._diff_fig.number):
            return
        prev_xlim = self._diff_ax.get_xlim() if self._diff_ax is not None else None
        prev_ylim = self._diff_ax.get_ylim() if self._diff_ax is not None else None

        diff_rgb = self._diff_recompute_overlay()
        if diff_rgb is None:
            return
        bounds = self._diff_get_bounds()
        if self._diff_img_im is None:
            self._diff_ax.clear()
            self._diff_img_im = self._diff_ax.imshow(
                diff_rgb,
                interpolation="nearest",
                origin="upper",
                extent=(-0.5, diff_rgb.shape[1] - 0.5, diff_rgb.shape[0] - 0.5, -0.5),
            )
            self._diff_ax.axis("off")
        else:
            self._diff_img_im.set_data(diff_rgb)
            self._diff_img_im.set_extent(
                (-0.5, diff_rgb.shape[1] - 0.5, diff_rgb.shape[0] - 0.5, -0.5)
            )
        if bounds is not None:
            if prev_xlim is None or prev_ylim is None:
                self._diff_ax.set_xlim(bounds[0], bounds[1])
                self._diff_ax.set_ylim(bounds[2], bounds[3])
            else:
                self._diff_ax.set_xlim(*prev_xlim)
                self._diff_ax.set_ylim(*prev_ylim)
                self._diff_clamp_view()
        self._diff_fig.canvas.draw_idle()
        self._diff_update_crosshair_lines()

    def _diff_log(self, message: str):
        logger = getattr(self, "logger", None)
        if logger is not None:
            try:
                logger.info(message)
                return
            except Exception:
                pass
        print(f"GUI_INFO: {message}")

    def _diff_get_bounds(self):
        if self._diff_diff_rgb is None:
            return None
        height, width = self._diff_diff_rgb.shape[:2]
        return (-0.5, width - 0.5, height - 0.5, -0.5)

    def _diff_clamp_view(self):
        bounds = self._diff_get_bounds()
        if bounds is None or self._diff_ax is None:
            return
        x_min, x_max, y_max, y_min = bounds  # note y order
        ax = self._diff_ax
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()

        xlo, xhi = self._diff_clamp_interval(xlo, xhi, x_min, x_max)
        ylo, yhi = self._diff_clamp_interval(ylo, yhi, y_min, y_max)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

    def _diff_clamp_interval(self, lo, hi, min_val, max_val):
        """Clamp [lo, hi] (respecting orientation) into [min_val, max_val]."""
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        length = abs(hi - lo)
        full_length = max_val - min_val
        if length > full_length:
            # reset to full
            return (min_val, max_val) if lo <= hi else (max_val, min_val)
        # shift if out of bounds
        low = min(lo, hi)
        high = max(lo, hi)
        if low < min_val:
            delta = min_val - low
            lo += delta
            hi += delta
        if max(lo, hi) > max_val:
            delta = max(lo, hi) - max_val
            lo -= delta
            hi -= delta
        return lo, hi

    @staticmethod
    def _diff_color_kind(rgb):
        """Classify pixel color into 'old', 'new', or 'other'."""
        only_a = np.array([255, 80, 255], dtype=np.float32)  # magenta (old)
        only_b = np.array([135, 205, 135], dtype=np.float32) # green (new)
        node_a = np.array([255, 0, 255], dtype=np.float32)   # bright magenta
        node_b = np.array([0, 255, 0], dtype=np.float32)     # bright green
        yellow = np.array([255, 255, 0], dtype=np.float32)
        gray = np.array([150, 150, 150], dtype=np.float32)

        v = np.array(rgb, dtype=np.float32)
        def d2(a):
            diff = v - a
            return float(diff.dot(diff))
        tol = 2000
        if min(d2(only_a), d2(node_a)) <= tol and d2(yellow) > tol and d2(gray) > tol:
            return 'old'
        if min(d2(only_b), d2(node_b)) <= tol and d2(yellow) > tol and d2(gray) > tol:
            return 'new'
        return 'other'

    @staticmethod
    def _find_nonzero_label_near(L, y, x, max_radius=10):
        h, w = L.shape
        yi = int(round(float(y)))
        xi = int(round(float(x)))
        yi = max(0, min(h - 1, yi))
        xi = max(0, min(w - 1, xi))
        if L[yi, xi] != 0:
            return int(L[yi, xi])
        for r in range(1, max_radius + 1):
            y0 = max(0, yi - r)
            y1 = min(h, yi + r + 1)
            x0 = max(0, xi - r)
            x1 = min(w, xi + r + 1)
            patch = L[y0:y1, x0:x1]
            nz = patch[patch != 0]
            if nz.size > 0:
                vals, counts = np.unique(nz, return_counts=True)
                return int(vals[np.argmax(counts)])
        return 0

    def _diff_accept_old_at(self, y, x):
        planes = self._diff_get_planes()
        if planes is None:
            return False
        saved_plane, current_plane, z_idx = planes
        old_id = self._find_nonzero_label_near(saved_plane, y, x)
        if old_id == 0:
            return False
        old_mask = (saved_plane == old_id)
        if not np.any(old_mask):
            return False
        state_new = self._diff_state_new
        if state_new is None or state_new.get("masks") is None:
            return False
        new_masks = np.array(state_new["masks"], copy=True)
        if new_masks.ndim == 2:
            new_slice = new_masks
        else:
            new_slice = new_masks[z_idx]
        # Remove conflicting labels entirely from the slice
        overlapping = np.unique(new_slice[old_mask])
        overlapping = overlapping[overlapping != 0]
        for oid in overlapping:
            new_slice[new_slice == oid] = 0
        # Insert old label region as a fresh label id
        new_id = int(new_slice.max()) + 1
        new_slice[old_mask] = new_id
        if new_masks.ndim != 2:
            new_masks[z_idx] = new_slice
        new_state = {
            "masks": new_masks,
            "outlines": None,
            "colors": None,
        }
        self._diff_apply_state(new_state, treat_as_new=True)
        self._diff_log(f"diff viewer accepted saved label -> inserted old ID as {new_id}")
        return True

    def _diff_accept_new_at(self, y, x):
        planes = self._diff_get_planes()
        if planes is None:
            return False
        saved_plane, current_plane, z_idx = planes
        new_id = self._find_nonzero_label_near(current_plane, y, x)
        if new_id == 0:
            return False
        new_mask = (current_plane == new_id)
        if not np.any(new_mask):
            return False
        state_old = self._diff_state_old
        if state_old is None or state_old.get("masks") is None:
            state_old = self._diff_get_saved_state(reload=False)
            if state_old is None or state_old.get("masks") is None:
                return False
        old_masks = np.array(state_old["masks"], copy=True)
        if old_masks.ndim == 2:
            old_slice = old_masks
        else:
            old_slice = old_masks[z_idx]
        overlapping = np.unique(old_slice[new_mask])
        overlapping = overlapping[overlapping != 0]
        for oid in overlapping:
            old_slice[old_slice == oid] = 0
        old_slice[new_mask] = new_id
        if old_masks.ndim != 2:
            old_masks[z_idx] = old_slice
        self._diff_state_old = {
            "masks": old_masks,
            "outlines": None,
            "colors": None,
        }
        self._diff_state_old_manual_override = True
        if self._diff_showing_restored:
            try:
                self._diff_apply_state(self._diff_state_old, treat_as_new=False)
            except Exception:
                pass
        self._diff_log(f"diff viewer accepted model label -> saved mask updated to ID {new_id}")
        return True

    def _diff_click_to_indices(self, x: float, y: float):
        """Map matplotlib data coords (x, y) to array indices (row, col)."""
        if self._diff_img_im is None or self._diff_diff_rgb is None:
            return None
        arr = np.asarray(self._diff_diff_rgb)
        if arr.ndim < 2:
            return None
        height, width = arr.shape[:2]
        if height == 0 or width == 0:
            return None

        extent = self._diff_img_im.get_extent()
        if extent is None:
            extent = (-0.5, width - 0.5, height - 0.5, -0.5)
        x0, x1, y0, y1 = extent
        if x0 == x1 or y0 == y1:
            return None

        # Normalize coordinates in [0, 1]
        u = (x - x0) / (x1 - x0)
        v = (y - y0) / (y1 - y0)

        if np.isnan(u) or np.isnan(v):
            return None

        if x1 < x0:
            u = 1.0 - u
        if y1 < y0:
            v = 1.0 - v

        col = int(np.floor(u * width))
        row = int(np.floor(v * height))

        if col < 0 or col >= width or row < 0 or row >= height:
            return None

        return row, col

    def _on_diff_click(self, event):
        if event is None or event.inaxes is None or event.button != 1:
            return
        if event.inaxes is not self._diff_ax:
            self._diff_log("diff viewer click ignored (outside diff axes)")
            return
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            self._diff_log("diff viewer click ignored (no data coords)")
            return
        if self._diff_diff_rgb is None:
            return
        mapped = self._diff_click_to_indices(float(x), float(y))
        if mapped is None:
            self._diff_log("diff viewer click ignored (outside image extent)")
            return
        yi, xi = mapped
        rgb = self._diff_diff_rgb[yi, xi]
        kind = self._diff_color_kind(rgb)
        self._diff_log(
            f"diff viewer click at (y={yi}, x={xi}) classified as '{kind}' with rgb={rgb.tolist()}"
        )
        if kind == 'old':
            changed = self._diff_accept_old_at(yi, xi)
            if changed:
                self._diff_log("diff viewer overlay refresh after accepting old")
                self._diff_refresh_overlay()
            else:
                self._diff_log("diff viewer old acceptance skipped (no change)")
        elif kind == 'new':
            changed = self._diff_accept_new_at(yi, xi)
            if changed:
                self._diff_log("diff viewer overlay refresh after accepting new")
                self._diff_refresh_overlay()
            else:
                self._diff_log("diff viewer new acceptance skipped (no change)")
        else:
            self._diff_log("diff viewer click classified as 'other'; no action taken")

    def _on_diff_scroll(self, event):
        if event is None or event.inaxes is not self._diff_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        bounds = self._diff_get_bounds()
        if bounds is None:
            return
        x_min, x_max, y_max, y_min = bounds
        ax = self._diff_ax
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        span_x = abs(current_xlim[1] - current_xlim[0])
        span_y = abs(current_ylim[1] - current_ylim[0])
        full_span_x = x_max - x_min
        full_span_y = y_max - y_min
        if span_x <= 0 or span_y <= 0:
            return

        # Determine zoom factor
        step = getattr(event, "step", 0)
        if step == 0:
            step = 1 if getattr(event, "button", "up") == "up" else -1
        zoom_in = step > 0
        scale = 0.8 if zoom_in else 1.25

        min_span = getattr(self, "_diff_zoom_min_span", 5.0)

        def compute_limits(current_range, center, full_min, full_max):
            orientation = 1 if current_range[1] >= current_range[0] else -1
            span = abs(current_range[1] - current_range[0])
            new_span = span * scale
            new_span = max(min_span, min(new_span, full_max - full_min))
            half = new_span / 2.0
            if orientation == 1:
                lo = center - half
                hi = center + half
            else:
                lo = center + half
                hi = center - half

            # Clamp to bounds
            low = min(lo, hi)
            high = max(lo, hi)
            if low < full_min:
                delta = full_min - low
                lo += delta
                hi += delta
            if high > full_max:
                delta = high - full_max
                lo -= delta
                hi -= delta
            return lo, hi

        new_xlim = compute_limits(current_xlim, float(event.xdata), x_min, x_max)
        new_ylim = compute_limits(current_ylim, float(event.ydata), y_min, y_max)
        ax.set_xlim(*new_xlim)
        ax.set_ylim(*new_ylim)
        self._diff_log(
            f"diff viewer scroll zoom to xlim={new_xlim}, ylim={new_ylim}"
        )
        self._diff_update_crosshair_lines()
        ax.figure.canvas.draw_idle()

    def show_segmentation_diff(self):
        if not MATPLOTLIB:
            QMessageBox.warning(self, "Diff viewer unavailable",
                                "Install matplotlib to view segmentation diffs.")
            return
        seg_path = self._diff_seg_file()
        if seg_path is None or not os.path.exists(seg_path):
            QMessageBox.warning(self, "Saved segmentation missing",
                                "Expected _seg.npy not found for this image.")
            return
        overlay_masks = self._diff_overlay_reference_masks if getattr(self, "_diff_overlay_reference_active", False) else None
        current_source = overlay_masks if isinstance(overlay_masks, np.ndarray) else self._diff_latest_masks
        if current_source is None:
            QMessageBox.information(self, "No model result",
                                    "Run a segmentation model before viewing the diff.")
            return

        saved_state = self._diff_get_saved_state(reload=True)
        if saved_state is None:
            QMessageBox.critical(
                self, "Failed to load _seg.npy",
                f"Could not read {os.path.basename(seg_path)}.")
            return
        saved_masks = saved_state.get("masks")
        if saved_masks is None:
            QMessageBox.critical(
                self, "Invalid _seg.npy", "The segmentation file does not contain a 'masks' entry.")
            return

        saved_masks = np.squeeze(np.asarray(saved_masks))
        current_masks = np.squeeze(np.asarray(current_source))

        def _select_plane(arr, label):
            if arr.ndim == 2:
                return arr, None
            if arr.ndim == 3:
                if arr.shape[0] == 0:
                    raise ValueError(f"{label} masks array has zero Z-planes.")
                z_idx = int(np.clip(self.currentZ, 0, arr.shape[0] - 1))
                return arr[z_idx], z_idx
            raise ValueError(f"{label} masks array has unsupported ndim={arr.ndim}.")

        try:
            saved_plane, saved_z = _select_plane(saved_masks, "Saved")
        except ValueError as exc:
            QMessageBox.warning(self, "Unsupported saved masks", str(exc))
            return

        try:
            current_plane, current_z = _select_plane(current_masks, "Current")
        except ValueError as exc:
            QMessageBox.warning(self, "Unsupported current masks", str(exc))
            return

        if saved_masks.ndim == 3 and current_masks.ndim == 3:
            if saved_masks.shape[0] != current_masks.shape[0]:
                QMessageBox.warning(
                    self, "Z-stack mismatch",
                    f"Saved masks have {saved_masks.shape[0]} planes, model result has "
                    f"{current_masks.shape[0]} planes.")
                return
            z_index = int(np.clip(self.currentZ, 0, saved_masks.shape[0] - 1))
            saved_plane = saved_masks[z_index]
            current_plane = current_masks[z_index]
        else:
            z_index = saved_z if saved_z is not None else current_z if current_z is not None else 0

        if saved_plane.shape != current_plane.shape:
            QMessageBox.warning(
                self, "Shape mismatch",
                f"Saved plane shape {saved_plane.shape} does not match model result "
                f"{current_plane.shape}.")
            return

        try:
            diff_rgb = contour_diff_rgb(saved_plane.astype(np.int32),
                                        current_plane.astype(np.int32))
        except Exception as exc:
            QMessageBox.critical(self, "Diff computation failed",
                                 f"Contour comparison failed:\n{exc}")
            return

        self._diff_close_existing()

        fig, ax = plt.subplots(figsize=(8, 8))
        height, width = diff_rgb.shape[:2]
        im = ax.imshow(
            diff_rgb,
            interpolation="nearest",
            origin="upper",
            extent=(-0.5, width - 0.5, height - 0.5, -0.5),
        )
        ax.axis("off")
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        try:
            fig.canvas.manager.set_window_title("Cellpose segmentation diff")
        except Exception:
            pass
        fig.tight_layout()
        self._diff_fig = fig
        self._diff_ax = ax
        self._diff_img_im = im
        self._diff_diff_rgb = diff_rgb
        self._diff_z_index = z_index
        self._diff_crosshair_lines = None
        self._diff_last_shape = diff_rgb.shape[:2]
        self._diff_last_crosshair = None
        if hasattr(fig.canvas, "mpl_connect"):
            fig.canvas.mpl_connect("close_event", self._diff_on_close)
            self._diff_click_cid = fig.canvas.mpl_connect("button_press_event", self._on_diff_click)
            self._diff_scroll_cid = fig.canvas.mpl_connect("scroll_event", self._on_diff_scroll)
        hub_coords = self._diff_crosshair_hub.current()
        if hub_coords is not None:
            self.diff_crosshair_updated(hub_coords, source=None, reason="viewer-open")
        else:
            self._diff_update_crosshair_lines(reason="viewer-open")
        fig.show()
        plt.show(block=False)
        self._diff_log("diff viewer open: click magenta to accept saved label, green to accept model label")

    def toggle_mask_restore(self):
        can_reset, reason = self._diff_can_reset()
        if not can_reset:
            QMessageBox.information(self, "Mask reset unavailable", reason)
            return
        if self._diff_showing_restored:
            if self._diff_state_new is None:
                QMessageBox.warning(self, "Mask toggle failed",
                                    "Model prediction masks are unavailable.")
                return
            try:
                self._diff_clear_overlay_reference()
                self._diff_showing_restored = False
                self._diff_apply_state(self._diff_state_new, treat_as_new=True)
                if hasattr(self, "maskToggleButton"):
                    self.maskToggleButton.setText("reset mask")
            except Exception as exc:
                self._diff_showing_restored = False
                QMessageBox.warning(self, "Mask toggle failed",
                                    f"Could not restore model masks:\n{exc}")
        else:
            state_old = self._diff_get_saved_state(reload=True)
            if state_old is None or state_old.get("masks") is None:
                QMessageBox.warning(self, "Mask toggle failed",
                                    "Saved masks could not be loaded.")
                return
            try:
                self._diff_showing_restored = True
                self._diff_apply_state(state_old, treat_as_new=False)
                if hasattr(self, "maskToggleButton"):
                    self.maskToggleButton.setText("show new mask")
            except Exception as exc:
                self._diff_showing_restored = False
                QMessageBox.warning(self, "Mask toggle failed",
                                    f"Could not apply saved masks:\n{exc}")

        self._diff_update_button_state()
        self._diff_update_crosshair_lines()

    def toggle_removals(self):
        if self.ncells > 0:
            self.ClearButton.setEnabled(True)
            self.remcell.setEnabled(True)
            self.undo.setEnabled(True)
            self.MakeDeletionRegionButton.setEnabled(True)
            self.DeleteMultipleROIButton.setEnabled(True)
            self.DoneDeleteMultipleROIButton.setEnabled(False)
            self.CancelDeleteMultipleROIButton.setEnabled(False)
        else:
            self.ClearButton.setEnabled(False)
            self.remcell.setEnabled(False)
            self.undo.setEnabled(False)
            self.MakeDeletionRegionButton.setEnabled(False)
            self.DeleteMultipleROIButton.setEnabled(False)
            self.DoneDeleteMultipleROIButton.setEnabled(False)
            self.CancelDeleteMultipleROIButton.setEnabled(False)

    def remove_action(self):
        if self.selected > 0:
            self.remove_cell(self.selected)

    def undo_action(self):
        if (len(self.strokes) > 0 and self.strokes[-1][0][0] == self.currentZ):
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells > 0:
                self.remove_cell(self.ncells.get())

    def undo_remove_action(self):
        self.undo_remove_cell()

    def _capture_display_state(self):
        """Store UI state required to restore channel display after image loads."""
        combo = getattr(self, "RGBDropDown", None)
        rgb_index = combo.currentIndex() if combo is not None else None
        return {
            "color": getattr(self, "color", None),
            "rgb_index": rgb_index,
        }

    def _restore_display_state(self, state):
        if not state:
            return

        combo = getattr(self, "RGBDropDown", None)
        if combo is None:
            return

        color = state.get("color") if isinstance(state, dict) else None
        rgb_index = state.get("rgb_index") if isinstance(state, dict) else None
        if rgb_index is None and color is None:
            return

        if rgb_index is None:
            rgb_index = color if color is not None else combo.currentIndex()

        count = combo.count() if hasattr(combo, "count") else None
        if count is not None and count > 0:
            max_index = count - 1
            rgb_index = max(0, min(max_index, rgb_index))
            if color is not None:
                color = max(0, min(max_index, color))

        self.color = color if color is not None else rgb_index

        blocker = getattr(combo, "blockSignals", None)
        should_unblock = False
        if callable(blocker):
            blocker(True)
            should_unblock = True
        try:
            combo.setCurrentIndex(rgb_index)
        finally:
            if should_unblock:
                blocker(False)

    def get_files(self):
        folder = os.path.dirname(self.filename)
        mask_filter = "_masks"
        images = get_image_files(folder, mask_filter)
        fnames = [os.path.split(images[k])[-1] for k in range(len(images))]
        f0 = os.path.split(self.filename)[-1]
        idx = np.nonzero(np.array(fnames) == f0)[0][0]
        return images, idx

    def get_prev_image(self):
        # Store current view state before loading the new image
        current_view_rect = self.p0.viewRect()
        current_saturation = copy.deepcopy(self.saturation)
        current_z_index = self.currentZ
        display_state = self._capture_display_state()

        images, idx = self.get_files()
        idx = (idx - 1) % len(images)
        io._load_image(self, filename=images[idx])

        try:
            # Restore zoom and position
            self.p0.setRange(rect=current_view_rect, padding=0)

            # Restore saturation for the current Z slice
            # Check if the Z index from the previous image is valid for the new image's saturation structure
            if self.currentZ < len(self.saturation[0]) and current_z_index < len(current_saturation[0]):
                 for r in range(len(self.saturation)):
                     if r < len(self.saturation) and r < len(current_saturation):
                         if self.currentZ < len(self.saturation[r]) and current_z_index < len(current_saturation[r]):
                             self.saturation[r][self.currentZ] = current_saturation[r][current_z_index]
            else:
                 print("GUI_WARNING: Z index mismatch or invalid state after loading previous image, applying default/first slice saturation.")
            # Update sliders to reflect restored saturation
            for r_idx, r_name in enumerate(["red", "green", "blue"]):
                 if r_idx < len(self.saturation) and self.currentZ < len(self.saturation[r_idx]):
                     self.sliders[r_idx].setValue(self.saturation[r_idx][self.currentZ])

            self._restore_display_state(display_state)
            # Refresh the plot
            self.update_plot()
        except Exception as e:
            print(f"GUI_ERROR: Could not restore view state: {e}")
            self._restore_display_state(display_state)


    def get_next_image(self, load_seg=True):
         # Store current view state before loading the new image
        current_view_rect = self.p0.viewRect()
        current_saturation = copy.deepcopy(self.saturation)
        current_z_index = self.currentZ
        display_state = self._capture_display_state()

        images, idx = self.get_files()
        idx = (idx + 1) % len(images)
        io._load_image(self, filename=images[idx], load_seg=load_seg)

        # Restore view state after loading the new image
        try:
            self.p0.setRange(rect=current_view_rect, padding=0)
            # Restore saturation for the current Z slice
            # Check if the Z index from the previous image is valid for the new image's saturation structure
            if self.currentZ < len(self.saturation[0]) and current_z_index < len(current_saturation[0]):
                 for r in range(len(self.saturation)):
                     # Ensure channel index 'r' is valid for both current and previous saturation lists
                     if r < len(self.saturation) and r < len(current_saturation):
                         # Ensure Z indices are valid within their respective saturation lists for this channel
                         if self.currentZ < len(self.saturation[r]) and current_z_index < len(current_saturation[r]):
                             self.saturation[r][self.currentZ] = current_saturation[r][current_z_index]
            else:
                 print("GUI_WARNING: Z index mismatch or invalid state after loading next image, applying default/first slice saturation.")

            # Update sliders to reflect restored saturation
            for r_idx, r_name in enumerate(["red", "green", "blue"]):
                 if r_idx < len(self.saturation) and self.currentZ < len(self.saturation[r_idx]):
                     self.sliders[r_idx].setValue(self.saturation[r_idx][self.currentZ])

            self._restore_display_state(display_state)
            self.update_plot()
        except Exception as e:
            print(f"GUI_ERROR: Could not restore view state: {e}")
            self._restore_display_state(display_state)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        target_file = files[0]
        if os.path.splitext(target_file)[-1] == ".npy":
            paired_image = io.resolve_seg_drop_target(target_file)
            if paired_image:
                io._load_image(self, filename=paired_image, load_seg=True, load_3D=self.load_3D)
            else:
                io._load_seg(self, filename=target_file, load_3D=self.load_3D)
        else:
            io._load_image(self, filename=target_file, load_seg=True, load_3D=self.load_3D)

    def toggle_masks(self):
        if self.MCheckBox.isChecked():
            self.masksOn = True
        else:
            self.masksOn = False
        if self.OCheckBox.isChecked():
            self.outlinesOn = True
        else:
            self.outlinesOn = False
        if not self.masksOn and not self.outlinesOn:
            self.p0.removeItem(self.layer)
            self.layer_off = True
        else:
            if self.layer_off:
                self.p0.addItem(self.layer)
            self.draw_layer()
            self.update_layer()
        if self.loaded:
            self.update_plot()
            self.update_layer()

    def make_viewbox(self):
        self.p0 = guiparts.ViewBoxNoRightDrag(parent=self, lockAspect=True,
                                              name="plot1", border=[100, 100,
                                                                    100], invertY=True)
        self.p0.setCursor(QtCore.Qt.CrossCursor)
        self.brush_size = 3
        self.win.addItem(self.p0, 0, 0, rowspan=1, colspan=1)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.layer = guiparts.ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0, 255])
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self)
        self.scale.setLevels([0, 255])
        self.p0.scene().contextMenuItem = self.p0
        self.Ly, self.Lx = 512, 512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)
        self._anchor_brush = pg.mkBrush(0, 255, 255, 255)
        self.anchor_scatter = pg.ScatterPlotItem(
            symbol="s",
            size=8,
            brush=self._anchor_brush,
            pen=None,
            pxMode=True,
        )
        self.anchor_scatter.setZValue(20)
        self.p0.addItem(self.anchor_scatter)
        self._anchor_points_by_plane = {}

    def _anchor_plane_index(self):
        plane = getattr(self, "currentZ", 0)
        if hasattr(self, "zc_ortho"):
            try:
                plane = int(self.zc_ortho)
            except Exception:
                plane = int(plane)
        return int(plane)

    def _refresh_anchor_views(self):
        self._update_anchor_display()
        self._update_ortho_anchor_display()

    def _update_anchor_display(self):
        if not hasattr(self, "anchor_scatter"):
            return
        plane = self._anchor_plane_index()
        anchors = self._anchor_points_by_plane.get(plane, [])
        if not anchors:
            self.anchor_scatter.setData([], [])
            return
        xs = [pt["x"] for pt in anchors]
        ys = [pt["y"] for pt in anchors]
        self.anchor_scatter.setData(
            xs,
            ys,
            symbol="s",
            size=8,
            brush=self._anchor_brush,
            pen=None,
            pxMode=True,
        )

    def _update_ortho_anchor_display(self):
        """Overridden in ortho subclass."""
        return

    def _add_anchor_point(self, x, y):
        plane = self._anchor_plane_index()
        anchor_list = self._anchor_points_by_plane.setdefault(plane, [])
        for anchor in anchor_list:
            if anchor["x"] == x and anchor["y"] == y:
                return
        anchor_list.append({"x": x, "y": y})
        self._refresh_anchor_views()

    def _remove_anchor_at_scene_pos(self, scene_pos):
        if not hasattr(self, "anchor_scatter"):
            return False
        spots = self.anchor_scatter.pointsAt(scene_pos)
        if spots is None or len(spots) == 0:
            return False
        plane = self._anchor_plane_index()
        anchor_list = self._anchor_points_by_plane.get(plane, [])
        if not anchor_list:
            return False
        to_remove = set()
        for spot in spots:
            pos = spot.pos()
            to_remove.add((int(round(pos.x())), int(round(pos.y()))))
        if not to_remove:
            return False
        filtered = [
            anchor for anchor in anchor_list
            if (anchor["x"], anchor["y"]) not in to_remove
        ]
        removed = len(anchor_list) != len(filtered)
        if removed:
            if filtered:
                self._anchor_points_by_plane[plane] = filtered
            else:
                del self._anchor_points_by_plane[plane]
            self._refresh_anchor_views()
        return removed

    def _clear_anchor_points(self):
        self._anchor_points_by_plane.clear()
        self._refresh_anchor_views()

    def reset(self):
        # ---- start sets of points ---- #
        self.selected = 0
        self.nchan = 3
        self.loaded = False
        self.channel = [0, 1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.resize = False
        self.ncells.reset()
        self.zdraw = []
        self.removed_cell = []
        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]
        self._anchor_points_by_plane = {}

        # -- zero out image stack -- #
        self.opacity = 128  # how opaque masks should be
        self.outcolor = [200, 200, 255, 200]
        self.NZ, self.Ly, self.Lx = 1, 256, 256
        self.saturation = self.saturation if hasattr(self, 'saturation') else []

        # only adjust the saturation if auto-adjust is on:
        if self.autobtn.isChecked():
            for r in range(3):
                self.saturation.append([[0, 255] for n in range(self.NZ)])
                self.sliders[r].setValue([0, 255])
                self.sliders[r].setEnabled(False)
                self.sliders[r].show()
        self.currentZ = 0
        self.flows = [[], [], [], [], [[]]]
        # masks matrix
        # image matrix with a scale disk
        self.stack = np.zeros((1, self.Ly, self.Lx, 3))
        self.Lyr, self.Lxr = self.Ly, self.Lx
        self.Ly0, self.Lx0 = self.Ly, self.Lx
        self.radii = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.cellpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        self.outpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        self.ismanual = np.zeros(0, "bool")

        # -- set menus to default -- #
        self.color = 0
        self.RGBDropDown.setCurrentIndex(self.color)
        self.view = 0
        self.ViewDropDown.setCurrentIndex(0)
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        self.delete_restore()

        self.clear_all()

        self.filename = []
        self.loaded = False
        self.recompute_masks = False
        self._diff_seg_path = None
        self._diff_latest_masks = None
        self._diff_state_old = None
        self._diff_state_new = None
        self._diff_fig = None
        self._diff_ax = None
        self._diff_img_im = None
        self._diff_diff_rgb = None
        self._diff_click_cid = None
        self._diff_scroll_cid = None
        self._diff_z_index = None
        self._diff_crosshair_lines = None
        self._diff_last_shape = None
        self._diff_last_crosshair = None
        self._diff_drag_active = False
        self._diff_drag_last_update = 0.0
        self._diff_showing_restored = False
        self._diff_state_old_manual_override = False
        self._diff_crosshair_suppress_broadcast = False
        self._diff_clear_overlay_reference()
        if hasattr(self, "diffButton"):
            self.diffButton.setEnabled(False)
        if hasattr(self, "maskToggleButton"):
            self.maskToggleButton.setEnabled(False)
            self.maskToggleButton.setText("reset mask")

        self.deleting_multiple = False
        self.removing_cells_list = []
        self.removing_region = False
        self.remove_roi_obj = None

    def delete_restore(self):
        """ delete restored imgs but don't reset settings """
        if hasattr(self, "stack_filtered"):
            del self.stack_filtered
        if hasattr(self, "cellpix_orig"):
            self.cellpix = self.cellpix_orig.copy()
            self.outpix = self.outpix_orig.copy()
            del self.outpix_orig, self.outpix_resize
            del self.cellpix_orig, self.cellpix_resize

    def clear_restore(self):
        """ delete restored imgs and reset settings """
        print("GUI_INFO: clearing restored image")
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        if self.ViewDropDown.currentIndex() == self.ViewDropDown.count() - 1:
            self.ViewDropDown.setCurrentIndex(0)
        self.delete_restore()
        self.restore = None
        self.ratio = 1.
        self.set_normalize_params(self.get_normalize_params())

    def brush_choose(self):
        self.brush_size = self.BrushChoose.currentIndex() * 2 + 1
        if self.loaded:
            self.layer.setDrawKernel(kernel_size=self.brush_size)
            self.update_layer()

    def clear_all(self):
        self.prev_selected = 0
        self.selected = 0
        if self.restore and "upsample" in self.restore:
            self.layerz = 0 * np.ones((self.Lyr, self.Lxr, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.cellpix_resize = self.cellpix.copy()
            self.outpix_resize = self.outpix.copy()
            self.cellpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
            self.outpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
        else:
            self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)

        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]
        self.ncells.reset()
        self.toggle_removals()
        self.update_scale()
        self.update_layer()
        self._refresh_anchor_views()
        self._diff_note_manual_edit()

    def select_cell(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity])
            self.update_layer()

    def select_cell_multi(self, idx):
        if idx > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity])
            self.update_layer()

    def unselect_cell(self):
        if self.selected > 0:
            idx = self.selected
            if idx < (self.ncells.get() + 1):
                z = self.currentZ
                self.layerz[self.cellpix[z] == idx] = np.append(
                    self.cellcolors[idx], self.opacity)
                if self.outlinesOn:
                    self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                        np.uint8)
                    #[0,0,0,self.opacity])
                self.update_layer()
        self.selected = 0

    def unselect_cell_multi(self, idx):
        z = self.currentZ
        self.layerz[self.cellpix[z] == idx] = np.append(self.cellcolors[idx],
                                                        self.opacity)
        if self.outlinesOn:
            self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                np.uint8)
            # [0,0,0,self.opacity])
        self.update_layer()

    def remove_cell(self, idx):
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        # because the function remove_single_cell updates the state of the cellpix and outpix arrays
        # by reindexing cells to avoid gaps in the indices, we need to remove the cells in reverse order
        # so that the indices are correct
        idx.sort(reverse=True)
        for i in idx:
            self.remove_single_cell(i)
        self.ncells -= len(idx)  # _save_sets uses ncells
        self.update_layer()

        if self.ncells == 0:
            self.ClearButton.setEnabled(False)
        if self.NZ == 1:
            io._save_sets_with_check(self)
        self._diff_note_manual_edit()


    def remove_single_cell(self, idx):
        # remove from manual array
        self.selected = 0
        if self.NZ > 1:
            zextent = ((self.cellpix == idx).sum(axis=(1, 2)) > 0).nonzero()[0]
        else:
            zextent = [0]
        for z in zextent:
            cp = self.cellpix[z] == idx
            op = self.outpix[z] == idx
            # remove from self.cellpix and self.outpix
            self.cellpix[z, cp] = 0
            self.outpix[z, op] = 0
            if z == self.currentZ:
                # remove from mask layer
                self.layerz[cp] = np.array([0, 0, 0, 0])

        # reduce other pixels by -1
        self.cellpix[self.cellpix > idx] -= 1
        self.outpix[self.outpix > idx] -= 1

        if self.NZ == 1:
            self.removed_cell = [
                self.ismanual[idx - 1], self.cellcolors[idx],
                np.nonzero(cp),
                np.nonzero(op)
            ]
            self.redo.setEnabled(True)
            ar, ac = self.removed_cell[2]
            d = datetime.datetime.now()
            self.track_changes.append(
                [d.strftime("%m/%d/%Y, %H:%M:%S"), "removed mask", [ar, ac]])
        # remove cell from lists
        self.ismanual = np.delete(self.ismanual, idx - 1)
        self.cellcolors = np.delete(self.cellcolors, [idx], axis=0)
        del self.zdraw[idx - 1]
        print("GUI_INFO: removed cell %d" % (idx - 1))

    def remove_region_cells(self):
        if self.removing_cells_list:
            for idx in self.removing_cells_list:
                self.unselect_cell_multi(idx)
            self.removing_cells_list.clear()
        self.disable_buttons_removeROIs()
        self.removing_region = True

        self.clear_multi_selected_cells()

        # make roi region here in center of view, making ROI half the size of the view
        roi_width = self.p0.viewRect().width() / 2
        x_loc = self.p0.viewRect().x() + (roi_width / 2)
        roi_height = self.p0.viewRect().height() / 2
        y_loc = self.p0.viewRect().y() + (roi_height / 2)

        pos = [x_loc, y_loc]
        roi = pg.RectROI(pos, [roi_width, roi_height], pen=pg.mkPen("y", width=2),
                         removable=True)
        roi.sigRemoveRequested.connect(self.remove_roi)
        roi.sigRegionChangeFinished.connect(self.roi_changed)
        self.p0.addItem(roi)
        self.remove_roi_obj = roi
        self.roi_changed(roi)

    def delete_multiple_cells(self):
        self.unselect_cell()
        self.disable_buttons_removeROIs()
        self.DoneDeleteMultipleROIButton.setEnabled(True)
        self.MakeDeletionRegionButton.setEnabled(True)
        self.CancelDeleteMultipleROIButton.setEnabled(True)
        self.deleting_multiple = True

    def done_remove_multiple_cells(self):
        self.deleting_multiple = False
        self.removing_region = False
        self.DoneDeleteMultipleROIButton.setEnabled(False)
        self.MakeDeletionRegionButton.setEnabled(False)
        self.CancelDeleteMultipleROIButton.setEnabled(False)

        if self.removing_cells_list:
            self.removing_cells_list = list(set(self.removing_cells_list))
            display_remove_list = [i - 1 for i in self.removing_cells_list]
            print(f"GUI_INFO: removing cells: {display_remove_list}")
            self.remove_cell(self.removing_cells_list)
            self.removing_cells_list.clear()
            self.unselect_cell()
        self.enable_buttons()

        if self.remove_roi_obj is not None:
            self.remove_roi(self.remove_roi_obj)

    def merge_cells(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected != self.prev_selected:
            for z in range(self.NZ):
                ar0, ac0 = np.nonzero(self.cellpix[z] == self.prev_selected)
                ar1, ac1 = np.nonzero(self.cellpix[z] == self.selected)
                touching = np.logical_and((ar0[:, np.newaxis] - ar1) < 3,
                                          (ac0[:, np.newaxis] - ac1) < 3).sum()
                ar = np.hstack((ar0, ar1))
                ac = np.hstack((ac0, ac1))
                vr0, vc0 = np.nonzero(self.outpix[z] == self.prev_selected)
                vr1, vc1 = np.nonzero(self.outpix[z] == self.selected)
                self.outpix[z, vr0, vc0] = 0
                self.outpix[z, vr1, vc1] = 0
                if touching > 0:
                    mask = np.zeros((np.ptp(ar) + 4, np.ptp(ac) + 4), np.uint8)
                    mask[ar - ar.min() + 2, ac - ac.min() + 2] = 1
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
                    pvc, pvr = contours[-2][0].squeeze().T
                    vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2

                else:
                    vr = np.hstack((vr0, vr1))
                    vc = np.hstack((vc0, vc1))
                color = self.cellcolors[self.prev_selected]
                self.draw_mask(z, ar, ac, vr, vc, color, idx=self.prev_selected)
            self.remove_cell(self.selected)
            print("GUI_INFO: merged two cells")
            self.update_layer()
            io._save_sets_with_check(self)
            self.undo.setEnabled(False)
            self.redo.setEnabled(False)

    def undo_remove_cell(self):
        if len(self.removed_cell) > 0:
            z = 0
            ar, ac = self.removed_cell[2]
            vr, vc = self.removed_cell[3]
            color = self.removed_cell[1]
            self.draw_mask(z, ar, ac, vr, vc, color)
            self.toggle_mask_ops()
            self.cellcolors = np.append(self.cellcolors, color[np.newaxis, :], axis=0)
            self.ncells += 1
            self.ismanual = np.append(self.ismanual, self.removed_cell[0])
            self.zdraw.append([])
            print(">>> added back removed cell")
            self.update_layer()
            io._save_sets_with_check(self)
            self.removed_cell = []
            self.redo.setEnabled(False)
            self._diff_note_manual_edit()

    def remove_stroke(self, delete_points=True, stroke_ind=-1):
        stroke = np.array(self.strokes[stroke_ind])
        cZ = self.currentZ
        inZ = stroke[0, 0] == cZ
        if inZ:
            outpix = self.outpix[cZ, stroke[:, 1], stroke[:, 2]] > 0
            self.layerz[stroke[~outpix, 1], stroke[~outpix, 2]] = np.array([0, 0, 0, 0])
            cellpix = self.cellpix[cZ, stroke[:, 1], stroke[:, 2]]
            ccol = self.cellcolors.copy()
            if self.selected > 0:
                ccol[self.selected] = np.array([255, 255, 255])
            col2mask = ccol[cellpix]
            if self.masksOn:
                col2mask = np.concatenate(
                    (col2mask, self.opacity * (cellpix[:, np.newaxis] > 0)), axis=-1)
            else:
                col2mask = np.concatenate((col2mask, 0 * (cellpix[:, np.newaxis] > 0)),
                                          axis=-1)
            self.layerz[stroke[:, 1], stroke[:, 2], :] = col2mask
            if self.outlinesOn:
                self.layerz[stroke[outpix, 1], stroke[outpix,
                                                      2]] = np.array(self.outcolor)
            if delete_points:
                del self.current_point_set[stroke_ind]
            self.update_layer()

        del self.strokes[stroke_ind]

    def plot_clicked(self, event):
        if event.button()==QtCore.Qt.LeftButton \
                and not event.modifiers() & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier)\
                and not self.removing_region:
            items = self.win.scene().items(event.scenePos())
            on_main_view = (
                self.p0 in items or self.img in items or self.layer in items
            )
            if event.double():
                if self.loaded and not self.in_stroke and on_main_view:
                    pos = self.p0.mapSceneToView(event.scenePos())
                    xx = int(round(pos.x()))
                    yy = int(round(pos.y()))
                    if 0 <= xx < self.Lx and 0 <= yy < self.Ly:
                        self._add_anchor_point(xx, yy)
                return
            if self.loaded and on_main_view and self._remove_anchor_at_scene_pos(event.scenePos()):
                return

    def cancel_remove_multiple(self):
        self.clear_multi_selected_cells()
        self.done_remove_multiple_cells()

    def clear_multi_selected_cells(self):
        # unselect all previously selected cells:
        for idx in self.removing_cells_list:
            self.unselect_cell_multi(idx)
        self.removing_cells_list.clear()

    def add_roi(self, roi):
        self.p0.addItem(roi)
        self.remove_roi_obj = roi

    def remove_roi(self, roi):
        self.clear_multi_selected_cells()
        assert roi == self.remove_roi_obj
        self.remove_roi_obj = None
        self.p0.removeItem(roi)
        self.removing_region = False

    def roi_changed(self, roi):
        # find the overlapping cells and make them selected
        pos = roi.pos()
        size = roi.size()
        x0 = int(pos.x())
        y0 = int(pos.y())
        x1 = int(pos.x() + size.x())
        y1 = int(pos.y() + size.y())
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > self.Lx:
            x1 = self.Lx
        if y1 > self.Ly:
            y1 = self.Ly

        # find cells in that region
        cell_idxs = np.unique(self.cellpix[self.currentZ, y0:y1, x0:x1])
        cell_idxs = np.trim_zeros(cell_idxs)
        # deselect cells not in region by deselecting all and then selecting the ones in the region
        self.clear_multi_selected_cells()

        for idx in cell_idxs:
            self.select_cell_multi(idx)
            self.removing_cells_list.append(idx)

        self.update_layer()

    def mouse_moved(self, pos):
        if self._diff_drag_active and self.loaded:
            now = time.monotonic()
            if now - self._diff_drag_last_update >= self._diff_drag_interval:
                items = self.win.scene().items(pos)
                if (self.p0 in items) or (self.img in items) or (self.layer in items):
                    view_pos = self.p0.mapSceneToView(pos)
                    x = view_pos.x()
                    y = view_pos.y()
                    if 0 <= x < self.Lx and 0 <= y < self.Ly:
                        if getattr(self, "is_ortho2D", False):
                            self.xortho = int(x)
                            self.yortho = int(y)
                            self.update_ortho()
                        else:
                            coords = (float(y), float(x))
                            self._diff_last_crosshair = coords
                            self._diff_update_crosshair_lines(
                                coords, reason="mouse-drag"
                            )
                self._diff_drag_last_update = now
        items = self.win.scene().items(pos)

    def closeEvent(self, event):
        try:
            if hasattr(self, "_diff_crosshair_hub"):
                self._diff_crosshair_hub.unregister(self)
        finally:
            super().closeEvent(event)

    def color_choose(self):
        self.color = self.RGBDropDown.currentIndex()
        self.view = 0
        self.ViewDropDown.setCurrentIndex(self.view)
        self.update_plot()

    def update_plot(self):
        self.view = self.ViewDropDown.currentIndex()
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape

        if self.view == 0 or self.view == self.ViewDropDown.count() - 1:
            image = self.stack[
                self.currentZ] if self.view == 0 else self.stack_filtered[self.currentZ]
            if self.color == 0:
                self.img.setImage(image, autoLevels=False, lut=None)
                if self.nchan > 1:
                    levels = np.array([
                        self.saturation[0][self.currentZ],
                        self.saturation[1][self.currentZ],
                        self.saturation[2][self.currentZ]
                    ])
                    self.img.setLevels(levels)
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color > 0 and self.color < 4:
                if self.nchan > 1:
                    image = image[:, :, self.color - 1]
                self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color])
                if self.nchan > 1:
                    self.img.setLevels(self.saturation[self.color - 1][self.currentZ])
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color == 4:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=None)
                self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color == 5:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=self.cmap[0])
                self.img.setLevels(self.saturation[0][self.currentZ])
        else:
            image = np.zeros((self.Ly, self.Lx), np.uint8)
            if len(self.flows) >= self.view - 1 and len(self.flows[self.view - 1]) > 0:
                image = self.flows[self.view - 1][self.currentZ]
            if self.view > 1:
                self.img.setImage(image, autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(image, autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])

        for r in range(3):
            self.sliders[r].setValue([
                self.saturation[r][self.currentZ][0],
                self.saturation[r][self.currentZ][1]
            ])
        self._refresh_anchor_views()
        self.win.show()
        self.show()


    def update_layer(self):
        if self.masksOn or self.outlinesOn:
            self.layer.setImage(self.layerz, autoLevels=False)
        self.win.show()
        self.show()


    def add_set(self):
        if len(self.current_point_set) > 0:
            while len(self.strokes) > 0:
                self.remove_stroke(delete_points=False)
            if len(self.current_point_set[0]) > 8:
                color = self.colormap[self.ncells.get(), :3]
                median = self.add_mask(points=self.current_point_set, color=color)
                if median is not None:
                    self.removed_cell = []
                    self.toggle_mask_ops()
                    self.cellcolors = np.append(self.cellcolors, color[np.newaxis, :],
                                                axis=0)
                    self.ncells += 1
                    self.ismanual = np.append(self.ismanual, True)
                    if self.NZ == 1:
                        # only save after each cell if single image
                        io._save_sets_with_check(self)
                    self._diff_note_manual_edit()
            else:
                print("GUI_ERROR: cell too small, not drawn")
            self.current_stroke = []
            self.strokes = []
            self.current_point_set = []
            self.update_layer()

    def add_mask(self, points=None, color=(100, 200, 50), dense=True):
        # points is list of strokes
        points_all = np.concatenate(points, axis=0)

        # loop over z values
        median = []
        zdraw = np.unique(points_all[:, 0])
        z = 0
        ars, acs, vrs, vcs = np.zeros(0, "int"), np.zeros(0, "int"), np.zeros(
            0, "int"), np.zeros(0, "int")
        for stroke in points:
            stroke = np.concatenate(stroke, axis=0).reshape(-1, 4)
            vr = stroke[:, 1]
            vc = stroke[:, 2]
            # get points inside drawn points
            mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
            pts = np.stack((vc - vc.min() + 2, vr - vr.min() + 2),
                           axis=-1)[:, np.newaxis, :]
            mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
            ar, ac = np.nonzero(mask)
            ar, ac = ar + vr.min() - 2, ac + vc.min() - 2
            # get dense outline
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0][:,0].T
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            # concatenate all points
            ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
            # if these pixels are overlapping with another cell, reassign them
            ioverlap = self.cellpix[z][ar, ac] > 0
            if (~ioverlap).sum() < 10:
                print("GUI_ERROR: cell < 10 pixels without overlaps, not drawn")
                return None
            elif ioverlap.sum() > 0:
                ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of new mask
                mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
                mask[ar - vr.min() + 2, ac - vc.min() + 2] = 1
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                pvc, pvr = contours[-2][0][:,0].T
                vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            ars = np.concatenate((ars, ar), axis=0)
            acs = np.concatenate((acs, ac), axis=0)
            vrs = np.concatenate((vrs, vr), axis=0)
            vcs = np.concatenate((vcs, vc), axis=0)

        self.draw_mask(z, ars, acs, vrs, vcs, color)
        median.append(np.array([np.median(ars), np.median(acs)]))

        self.zdraw.append(zdraw)
        d = datetime.datetime.now()
        self.track_changes.append(
            [d.strftime("%m/%d/%Y, %H:%M:%S"), "added mask", [ar, ac]])
        return median

    def draw_mask(self, z, ar, ac, vr, vc, color, idx=None):
        """ draw single mask using outlines and area """
        if idx is None:
            idx = self.ncells + 1
        self.cellpix[z, vr, vc] = idx
        self.cellpix[z, ar, ac] = idx
        self.outpix[z, vr, vc] = idx
        if self.restore and "upsample" in self.restore:
            if self.resize:
                self.cellpix_resize[z, vr, vc] = idx
                self.cellpix_resize[z, ar, ac] = idx
                self.outpix_resize[z, vr, vc] = idx
                self.cellpix_orig[z, (vr / self.ratio).astype(int),
                                  (vc / self.ratio).astype(int)] = idx
                self.cellpix_orig[z, (ar / self.ratio).astype(int),
                                  (ac / self.ratio).astype(int)] = idx
                self.outpix_orig[z, (vr / self.ratio).astype(int),
                                 (vc / self.ratio).astype(int)] = idx
            else:
                self.cellpix_orig[z, vr, vc] = idx
                self.cellpix_orig[z, ar, ac] = idx
                self.outpix_orig[z, vr, vc] = idx

                # get upsampled mask
                vrr = (vr.copy() * self.ratio).astype(int)
                vcr = (vc.copy() * self.ratio).astype(int)
                mask = np.zeros((np.ptp(vrr) + 4, np.ptp(vcr) + 4), np.uint8)
                pts = np.stack((vcr - vcr.min() + 2, vrr - vrr.min() + 2),
                               axis=-1)[:, np.newaxis, :]
                mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
                arr, acr = np.nonzero(mask)
                arr, acr = arr + vrr.min() - 2, acr + vcr.min() - 2
                # get dense outline
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                pvc, pvr = contours[-2][0].squeeze().T
                vrr, vcr = pvr + vrr.min() - 2, pvc + vcr.min() - 2
                # concatenate all points
                arr, acr = np.hstack((np.vstack((vrr, vcr)), np.vstack((arr, acr))))
                self.cellpix_resize[z, vrr, vcr] = idx
                self.cellpix_resize[z, arr, acr] = idx
                self.outpix_resize[z, vrr, vcr] = idx

        if z == self.currentZ:
            self.layerz[ar, ac, :3] = color
            if self.masksOn:
                self.layerz[ar, ac, -1] = self.opacity
            if self.outlinesOn:
                self.layerz[vr, vc] = np.array(self.outcolor)

    def compute_scale(self):
        # get diameter from gui
        diameter = self.segmentation_settings.diameter
        if not diameter:
            diameter = 60

        self.pr = int(diameter)
        self.radii_padding = int(self.pr * 1.25)
        self.radii = np.zeros((self.Ly + self.radii_padding, self.Lx, 4), np.uint8)
        yy, xx = disk([self.Ly + self.radii_padding / 2 - 1, self.pr / 2 + 1],
                      self.pr / 2, self.Ly + self.radii_padding, self.Lx)
        # rgb(150,50,150)
        self.radii[yy, xx, 0] = 150
        self.radii[yy, xx, 1] = 50
        self.radii[yy, xx, 2] = 150
        self.radii[yy, xx, 3] = 255
        self.p0.setYRange(0, self.Ly + self.radii_padding)
        self.p0.setXRange(0, self.Lx)

    def update_scale(self):
        self.compute_scale()
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0, 255.0])
        self.win.show()
        self.show()


    def draw_layer(self):
        if self.resize:
            self.Ly, self.Lx = self.Lyr, self.Lxr
        else:
            self.Ly, self.Lx = self.Ly0, self.Lx0

        if self.masksOn or self.outlinesOn:
            if self.restore and "upsample" in self.restore:
                if self.resize:
                    self.cellpix = self.cellpix_resize.copy()
                    self.outpix = self.outpix_resize.copy()
                else:
                    self.cellpix = self.cellpix_orig.copy()
                    self.outpix = self.outpix_orig.copy()

        self.layerz = np.zeros((self.Ly, self.Lx, 4), np.uint8)
        if self.masksOn:
            self.layerz[..., :3] = self.cellcolors[self.cellpix[self.currentZ], :]
            self.layerz[..., 3] = self.opacity * (self.cellpix[self.currentZ]
                                                  > 0).astype(np.uint8)
            if self.selected > 0:
                self.layerz[self.cellpix[self.currentZ] == self.selected] = np.array(
                    [255, 255, 255, self.opacity])
            cZ = self.currentZ
            stroke_z = np.array([s[0][0] for s in self.strokes])
            inZ = np.nonzero(stroke_z == cZ)[0]
            if len(inZ) > 0:
                for i in inZ:
                    stroke = np.array(self.strokes[i])
                    self.layerz[stroke[:, 1], stroke[:,
                                                     2]] = np.array([255, 0, 255, 100])
        else:
            self.layerz[..., 3] = 0

        if self.outlinesOn:
            self.layerz[self.outpix[self.currentZ] > 0] = np.array(
                self.outcolor).astype(np.uint8)


    def set_normalize_params(self, normalize_params):
        from cellpose.models import normalize_default
        if self.restore != "filter":
            keys = list(normalize_params.keys()).copy()
            for key in keys:
                if key != "percentile":
                    normalize_params[key] = normalize_default[key]
        normalize_params = {**normalize_default, **normalize_params}
        out = self.check_filter_params(normalize_params["sharpen_radius"],
                                       normalize_params["smooth_radius"],
                                       normalize_params["tile_norm_blocksize"],
                                       normalize_params["tile_norm_smooth3D"],
                                       normalize_params["norm3D"],
                                       normalize_params["invert"])


    def check_filter_params(self, sharpen, smooth, tile_norm, smooth3D, norm3D, invert):
        tile_norm = 0 if tile_norm < 0 else tile_norm
        sharpen = 0 if sharpen < 0 else sharpen
        smooth = 0 if smooth < 0 else smooth
        smooth3D = 0 if smooth3D < 0 else smooth3D
        norm3D = bool(norm3D)
        invert = bool(invert)
        if tile_norm > self.Ly and tile_norm > self.Lx:
            print(
                "GUI_ERROR: tile size (tile_norm) bigger than both image dimensions, disabling"
            )
            tile_norm = 0
        self.filt_edits[0].setText(str(sharpen))
        self.filt_edits[1].setText(str(smooth))
        self.filt_edits[2].setText(str(tile_norm))
        self.filt_edits[3].setText(str(smooth3D))
        self.norm3D_cb.setChecked(norm3D)
        return sharpen, smooth, tile_norm, smooth3D, norm3D, invert

    def get_normalize_params(self):
        percentile = [
            self.segmentation_settings.low_percentile,
            self.segmentation_settings.high_percentile,
        ]
        normalize_params = {"percentile": percentile}
        norm3D = self.norm3D_cb.isChecked()
        normalize_params["norm3D"] = norm3D
        sharpen = float(self.filt_edits[0].text())
        smooth = float(self.filt_edits[1].text())
        tile_norm = float(self.filt_edits[2].text())
        smooth3D = float(self.filt_edits[3].text())
        invert = False
        out = self.check_filter_params(sharpen, smooth, tile_norm, smooth3D, norm3D,
                                        invert)
        sharpen, smooth, tile_norm, smooth3D, norm3D, invert = out
        normalize_params["sharpen_radius"] = sharpen
        normalize_params["smooth_radius"] = smooth
        normalize_params["tile_norm_blocksize"] = tile_norm
        normalize_params["tile_norm_smooth3D"] = smooth3D
        normalize_params["invert"] = invert

        from cellpose.models import normalize_default
        normalize_params = {**normalize_default, **normalize_params}

        return normalize_params

    def compute_saturation_if_checked(self):
        if self.autobtn.isChecked():
            self.compute_saturation()

    def compute_saturation(self, return_img=False):
        norm = self.get_normalize_params()
        print(norm)
        sharpen, smooth = norm["sharpen_radius"], norm["smooth_radius"]
        percentile = norm["percentile"]
        tile_norm = norm["tile_norm_blocksize"]
        invert = norm["invert"]
        norm3D = norm["norm3D"]
        smooth3D = norm["tile_norm_smooth3D"]
        tile_norm = norm["tile_norm_blocksize"]

        if sharpen > 0 or smooth > 0 or tile_norm > 0:
            img_norm = self.stack.copy()
        else:
            img_norm = self.stack

        if sharpen > 0 or smooth > 0 or tile_norm > 0:
            self.restore = "filter"
            print(
                "GUI_INFO: computing filtered image because sharpen > 0 or tile_norm > 0"
            )
            print(
                "GUI_WARNING: will use memory to create filtered image -- make sure to have RAM for this"
            )
            img_norm = self.stack.copy()
            if sharpen > 0 or smooth > 0:
                img_norm = smooth_sharpen_img(self.stack, sharpen_radius=sharpen,
                                              smooth_radius=smooth)

            if tile_norm > 0:
                img_norm = normalize99_tile(img_norm, blocksize=tile_norm,
                                            lower=percentile[0], upper=percentile[1],
                                            smooth3D=smooth3D, norm3D=norm3D)
            # convert to 0->255
            img_norm_min = img_norm.min()
            img_norm_max = img_norm.max()
            for c in range(img_norm.shape[-1]):
                if np.ptp(img_norm[..., c]) > 1e-3:
                    img_norm[..., c] -= img_norm_min
                    img_norm[..., c] /= (img_norm_max - img_norm_min)
            img_norm *= 255
            self.stack_filtered = img_norm
            self.ViewDropDown.model().item(self.ViewDropDown.count() -
                                           1).setEnabled(True)
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)
        else:
            img_norm = self.stack if self.restore is None or self.restore == "filter" else self.stack_filtered

        if self.autobtn.isChecked():
            self.saturation = []
            for c in range(img_norm.shape[-1]):
                self.saturation.append([])
                if np.ptp(img_norm[..., c]) > 1e-3:
                    if norm3D:
                        x01 = np.percentile(img_norm[..., c], percentile[0])
                        x99 = np.percentile(img_norm[..., c], percentile[1])
                        if invert:
                            x01i = 255. - x99
                            x99i = 255. - x01
                            x01, x99 = x01i, x99i
                        for n in range(self.NZ):
                            self.saturation[-1].append([x01, x99])
                    else:
                        for z in range(self.NZ):
                            if self.NZ > 1:
                                x01 = np.percentile(img_norm[z, :, :, c], percentile[0])
                                x99 = np.percentile(img_norm[z, :, :, c], percentile[1])
                            else:
                                x01 = np.percentile(img_norm[..., c], percentile[0])
                                x99 = np.percentile(img_norm[..., c], percentile[1])
                            if invert:
                                x01i = 255. - x99
                                x99i = 255. - x01
                                x01, x99 = x01i, x99i
                            self.saturation[-1].append([x01, x99])
                else:
                    for n in range(self.NZ):
                        self.saturation[-1].append([0, 255.])
            print(self.saturation[2][self.currentZ])

            if img_norm.shape[-1] == 1:
                self.saturation.append(self.saturation[0])
                self.saturation.append(self.saturation[0])

        # self.autobtn.setChecked(True)
        self.update_plot()


    def get_model_path(self, custom=False):
        if custom:
            self.current_model = self.ModelChooseC.currentText()
            self.current_model_path = os.fspath(
                models.MODEL_DIR.joinpath(self.current_model))
        else:
            self.current_model = "cpsam"
            self.current_model_path = models.model_path(self.current_model)

    def initialize_model(self, model_name=None, custom=False):
        if model_name is None or custom:
            self.get_model_path(custom=custom)
            if not os.path.exists(self.current_model_path):
                raise ValueError("need to specify model (use dropdown)")

        if model_name is None or not isinstance(model_name, str):
            self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                              pretrained_model=self.current_model_path)
        else:
            self.current_model = model_name
            self.current_model_path = os.fspath(
                models.MODEL_DIR.joinpath(self.current_model))

            self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                             pretrained_model=self.current_model)

    def add_model(self):
        io._add_model(self)
        return

    def remove_model(self):
        io._remove_model(self)
        return

    def new_model(self):
        if self.NZ != 1:
            print("ERROR: cannot train model on 3D data")
            return

        # train model
        image_names = self.get_files()[0]
        self.train_data, self.train_labels, self.train_files, restore, normalize_params = io._get_train_set(
            image_names)
        TW = guiparts.TrainWindow(self, models.MODEL_NAMES)
        train = TW.exec_()
        if train:
            self.logger.info(
                f"training with {[os.path.split(f)[1] for f in self.train_files]}")
            self.train_model(restore=restore, normalize_params=normalize_params)
        else:
            print("GUI_INFO: training cancelled")

    def train_model(self, restore=None, normalize_params=None):
        from cellpose.models import normalize_default
        if normalize_params is None:
            normalize_params = copy.deepcopy(normalize_default)
        model_type = models.MODEL_NAMES[self.training_params["model_index"]]
        self.logger.info(f"training new model starting at model {model_type}")
        self.current_model = model_type
        self.channels = self.training_params["channels"]

        self.logger.info(
            f"training with chan = {self.ChannelChoose[0].currentText()}, chan2 = {self.ChannelChoose[1].currentText()}"
        )

        self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                          model_type=model_type)
        save_path = os.path.dirname(self.filename)

        print("GUI_INFO: name of new model: " + self.training_params["model_name"])
        self.new_model_path, train_losses = train.train_seg(
            self.model.net, train_data=self.train_data, train_labels=self.train_labels,
            normalize=normalize_params, min_train_masks=0,
            save_path=save_path, nimg_per_epoch=max(2, len(self.train_data)),
            learning_rate=self.training_params["learning_rate"],
            weight_decay=self.training_params["weight_decay"],
            n_epochs=self.training_params["n_epochs"],
            model_name=self.training_params["model_name"])[:2]
        # save train losses
        np.save(str(self.new_model_path) + "_train_losses.npy", train_losses)
        # run model on next image
        io._add_model(self, self.new_model_path)
        diam_labels = self.model.net.diam_labels.item()  #.copy()
        self.new_model_ind = len(self.model_strings)
        self.autorun = True
        self.clear_all()
        self.restore = restore
        self.set_normalize_params(normalize_params)
        self.get_next_image(load_seg=False)

        self.compute_segmentation(custom=True)
        self.logger.info(
            f"!!! computed masks for {os.path.split(self.filename)[1]} from new model !!!"
        )


    def compute_cprob(self):
        if self.recompute_masks:
            flow_threshold = self.segmentation_settings.flow_threshold
            cellprob_threshold = self.segmentation_settings.cellprob_threshold
            niter = self.segmentation_settings.niter
            min_size = int(self.min_size.text()) if not isinstance(
                self.min_size, int) else self.min_size

            self.logger.info(
                    "computing masks with cell prob=%0.3f, flow error threshold=%0.3f" %
                    (cellprob_threshold, flow_threshold))

            try:
                dP = self.flows[2].squeeze()
                cellprob = self.flows[3].squeeze()
            except IndexError:
                self.logger.error("Flows don't exist, try running model again.")
                return

            maski = dynamics.resize_and_compute_masks(
                dP=dP,
                cellprob=cellprob,
                niter=niter,
                do_3D=self.load_3D,
                min_size=min_size,
                # max_size_fraction=min_size_fraction, # Leave as default
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold)

            self.masksOn = True
            if not self.OCheckBox.isChecked():
                self.MCheckBox.setChecked(True)
            if maski.ndim < 3:
                maski = maski[np.newaxis, ...]
            self.logger.info("%d cells found" % (len(np.unique(maski)[1:])))
            io._masks_to_gui(self, maski, outlines=None)
            self.show()


    def compute_segmentation(self, custom=False, model_name=None, load_model=True):
        self.progress.setValue(0)
        self._diff_latest_masks = None
        self._diff_update_button_state()
        try:
            tic = time.time()
            self.clear_all()
            self.flows = [[], [], []]
            if load_model:
                self.initialize_model(model_name=model_name, custom=custom)
            self.progress.setValue(10)
            do_3D = self.load_3D
            stitch_threshold = float(self.stitch_threshold.text()) if not isinstance(
                self.stitch_threshold, float) else self.stitch_threshold
            anisotropy = float(self.anisotropy.text()) if not isinstance(
                self.anisotropy, float) else self.anisotropy
            flow3D_smooth = float(self.flow3D_smooth.text()) if not isinstance(
                self.flow3D_smooth, float) else self.flow3D_smooth
            min_size = int(self.min_size.text()) if not isinstance(
                self.min_size, int) else self.min_size
            resample = self.resample.isChecked() if not isinstance(
                self.resample, bool) else self.resample

            do_3D = False if stitch_threshold > 0. else do_3D

            if self.restore == "filter":
                data = self.stack_filtered.copy().squeeze()
            else:
                data = self.stack.copy().squeeze()

            flow_threshold = self.segmentation_settings.flow_threshold
            cellprob_threshold = self.segmentation_settings.cellprob_threshold
            diameter = self.segmentation_settings.diameter
            niter = self.segmentation_settings.niter

            normalize_params = self.get_normalize_params()
            print(normalize_params)
            try:
                masks, flows = self.model.eval(
                    data,
                    diameter=diameter,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold, do_3D=do_3D, niter=niter,
                    normalize=normalize_params, stitch_threshold=stitch_threshold,
                    anisotropy=anisotropy, flow3D_smooth=flow3D_smooth,
                    min_size=min_size, channel_axis=-1,
                    progress=self.progress, z_axis=0 if self.NZ > 1 else None)[:2]
            except Exception as e:
                print("NET ERROR: %s" % e)
                self.progress.setValue(0)
                return

            self.progress.setValue(75)

            # convert flows to uint8 and resize to original image size
            flows_new = []
            flows_new.append(flows[0].copy())  # RGB flow
            flows_new.append((np.clip(normalize99(flows[2].copy()), 0, 1) *
                              255).astype("uint8"))  # cellprob
            flows_new.append(flows[1].copy()) # XY flows
            flows_new.append(flows[2].copy()) # original cellprob

            if self.load_3D:
                if stitch_threshold == 0.:
                    flows_new.append((flows[1][0] / 10 * 127 + 127).astype("uint8"))
                else:
                    flows_new.append(np.zeros(flows[1][0].shape, dtype="uint8"))

            if not self.load_3D:
                if self.restore and "upsample" in self.restore:
                    self.Ly, self.Lx = self.Lyr, self.Lxr

                if flows_new[0].shape[-3:-1] != (self.Ly, self.Lx):
                    self.flows = []
                    for j in range(len(flows_new)):
                        self.flows.append(
                            resize_image(flows_new[j], Ly=self.Ly, Lx=self.Lx,
                                        interpolation=cv2.INTER_NEAREST))
                else:
                    self.flows = flows_new
            else:
                if not resample:
                    self.flows = []
                    Lz, Ly, Lx = self.NZ, self.Ly, self.Lx
                    Lz0, Ly0, Lx0 = flows_new[0].shape[:3]
                    print("GUI_INFO: resizing flows to original image size")
                    for j in range(len(flows_new)):
                        flow0 = flows_new[j]
                        if Ly0 != Ly:
                            flow0 = resize_image(flow0, Ly=Ly, Lx=Lx,
                                                no_channels=flow0.ndim==3,
                                                interpolation=cv2.INTER_NEAREST)
                        if Lz0 != Lz:
                            flow0 = np.swapaxes(resize_image(np.swapaxes(flow0, 0, 1),
                                                Ly=Lz, Lx=Lx,
                                                no_channels=flow0.ndim==3,
                                                interpolation=cv2.INTER_NEAREST), 0, 1)
                        self.flows.append(flow0)
                else:
                    self.flows = flows_new

            # add first axis
            if self.NZ == 1:
                masks = masks[np.newaxis, ...]
                self.flows = [
                    self.flows[n][np.newaxis, ...] for n in range(len(self.flows))
                ]

            self.logger.info("%d cells found with model in %0.3f sec" %
                             (len(np.unique(masks)[1:]), time.time() - tic))
            self.progress.setValue(80)
            z = 0

            io._masks_to_gui(self, masks, outlines=None)
            self.masksOn = True
            self.MCheckBox.setChecked(True)
            self._diff_store_current_as_new()
            self._diff_showing_restored = False
            self._diff_update_button_state()
            self.progress.setValue(100)
            if self.restore != "filter" and self.restore is not None and self.autobtn.isChecked():
                self.compute_saturation()
            if not do_3D and not stitch_threshold > 0:
                self.recompute_masks = True
            else:
                self.recompute_masks = False
        except Exception as e:
            print("ERROR: %s" % e)
    def _diff_seg_file(self):
        return getattr(self, "_diff_seg_path", None)
