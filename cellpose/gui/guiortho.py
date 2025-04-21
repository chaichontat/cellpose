"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import re
import sys, os, pathlib, warnings, datetime, time

from qtpy import QtGui, QtCore
from superqt import QRangeSlider
from qtpy.QtWidgets import QScrollArea, QMainWindow, QApplication, QWidget, QScrollBar, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox
import pyqtgraph as pg

import numpy as np
from scipy.stats import mode
import cv2

from . import guiparts, menus, io
from .. import models, core, dynamics, version
from ..utils import download_url_to_file, masks_to_outlines, diameters
from ..io import get_image_files, imsave, imread
from ..transforms import resize_image, normalize99  #fixed import
from ..plot import disk
from ..transforms import normalize99_tile, smooth_sharpen_img
from .gui import MainW

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False



def run(image=None):
    from ..io import logger_setup
    logger, log_file = logger_setup()
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
    guip_path = pathlib.Path.home().joinpath(".cellpose", "cellpose_gui.png")
    style_path = pathlib.Path.home().joinpath(".cellpose", "style_choice.npy")
    if not icon_path.is_file():
        cp_dir = pathlib.Path.home().joinpath(".cellpose")
        cp_dir.mkdir(exist_ok=True)
        print("downloading logo")
        download_url_to_file(
            "https://www.cellpose.org/static/images/cellpose_transparent.png",
            icon_path, progress=True)
    if not guip_path.is_file():
        print("downloading help window image")
        download_url_to_file("https://www.cellpose.org/static/images/cellpose_gui.png",
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
    #app.setStyleSheet("QLineEdit { color: yellow }")

    # models.download_model_weights() # does not exist
    MainW_ortho2D(image=image, logger=logger)
    ret = app.exec_()
    sys.exit(ret)

# -------------------------------------------------------------------------------------
# NEW CLASS: MainW_ortho2D
# Inherits from MainW, adds orthographic views for Z-stack context
# -------------------------------------------------------------------------------------
class MainW_ortho2D(MainW):

    def __init__(self, image=None, logger=None):
        # Initialize base 2D GUI
        super().__init__(image=None, logger=logger) # Load image later in overridden method

        self.setWindowTitle(f"cellpose v{version} [Ortho-2D View]")
        self.is_ortho2D = True # Flag for this mode
        self.load_3D = False # Ensure base class doesn't try 3D loading

        # --- Ortho View Specific Attributes ---
        self.stack_ortho = None # Holds the Z-stack (NZ_ortho, Ly, Lx, C)
        self.NZ = 1
        self.ortho_nz = 0       # Number of Z planes in the ortho stack
        self.xortho = 0         # X coordinate for ortho crosshair
        self.yortho = 0         # Y coordinate for ortho crosshair
        self.zc_ortho = 0       # Z index of the main plane within stack_ortho
        self.dz = 10            # Number of Z planes to show above/below main plane in ortho views
        self.zaspect = 6.0      # Z-aspect ratio for ortho views

        # --- Ortho View UI Elements (copied from gui3d.py structure) ---
        # Ortho crosshair lines
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g')) # Main view V line
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('g')) # Main view H line
        self.vLineOrtho = [
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g')), # YZ view V line (shows Z)
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g'))  # XZ view V line (shows X)
        ]
        self.hLineOrtho = [
            pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('g')), # YZ view H line (shows Y)
            pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('g'))  # XZ view H line (shows Z)
        ]
        self.make_orthoviews() # Create the ortho view boxes and image items

        # Add ortho view controls to the left panel (self.l0)
        # Find the row index 'b' where the scale disk checkbox was added in make_buttons
        # This is fragile; better to add controls programmatically if layout changes.
        # Assuming 'b' from make_buttons is accessible or recalculate.
        # Let's find the ScaleOn widget and insert before it.
        scale_widget_row = -1
        for i in range(self.l0.rowCount()):
             item = self.l0.itemAtPosition(i, 0)
             if item and item.widget() == self.ScaleOn:
                  scale_widget_row = i
                  break
        b = scale_widget_row if scale_widget_row != -1 else self.l0.rowCount()

        # Insert ortho controls before the scale disk toggle
        ortho_row_start = b

        self.orthobtn = QCheckBox("ortho")
        self.orthobtn.setToolTip("activate orthoviews using Z-stack from folder")
        self.orthobtn.setFont(self.medfont)
        self.orthobtn.setChecked(False) # Off by default
        self.l0.addWidget(self.orthobtn, ortho_row_start, 0, 1, 2)
        self.orthobtn.toggled.connect(self.toggle_ortho)

        label = QLabel("dz:")
        label.setToolTip("Number of Z-planes to show around the main plane in ortho views")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setFont(self.medfont)
        self.l0.addWidget(label, ortho_row_start, 2, 1, 1)
        self.dzedit = QLineEdit()
        self.dzedit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.dzedit.setText(str(self.dz))
        self.dzedit.returnPressed.connect(self.update_ortho)
        self.dzedit.setFixedWidth(40)
        self.dzedit.setFont(self.medfont)
        self.l0.addWidget(self.dzedit, ortho_row_start, 3, 1, 2)

        label = QLabel("z-aspect:")
        label.setToolTip("Aspect ratio for Z dimension in ortho views")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setFont(self.medfont)
        self.l0.addWidget(label, ortho_row_start, 5, 1, 2)
        self.zaspectedit = QLineEdit()
        self.zaspectedit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.zaspectedit.setText(str(self.zaspect))
        self.zaspectedit.returnPressed.connect(self.update_ortho)
        self.zaspectedit.setFixedWidth(40)
        self.zaspectedit.setFont(self.medfont)
        self.l0.addWidget(self.zaspectedit, ortho_row_start, 7, 1, 2)

        # Shift the scale disk toggle down by one row
        self.l0.addWidget(self.ScaleOn, ortho_row_start + 1, 0, 1, 5)

        # --- Load Initial Image ---
        if image is not None:
            self.filename = image
            self._load_image_ortho2D(self.filename) # Use the ortho loading method

    # --- Method Overrides for Ortho2D Functionality ---

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if os.path.splitext(files[0])[-1] == ".npy":
            io._load_seg(self, filename=files[0], load_3D=self.load_3D)
        else:
            print(files)
            self._load_image_ortho2D(filename=files[0], load_seg=True)

    def _load_image(self, filename=None, load_seg=True):
        """ Override base _load_image to call the ortho version. """
        self._load_image_ortho2D(filename=filename, load_seg=load_seg)

    def _load_image_ortho2D(self, filename=None, load_seg=True):
        """ Loads the main 2D image and then finds/loads the Z-stack. """
        if filename is None:
            filename = self.filename

        # 1. Load the main 2D image using the standard io function
        # Set load_3D=False explicitly for the base loader
        try:
            # Temporarily set load_3D flag for the call if needed by io._load_image internal logic
            original_load_3d_flag = self.load_3D
            self.load_3D = False
            io._load_image(self, filename=filename, load_seg=load_seg, load_3D=False)
            self.load_3D = False # Restore flag
        except Exception as e:
            print(f"GUI_ERROR: Failed to load main image '{filename}': {e}")
            self.loaded = False
            return # Stop if main image fails to load

        if not self.loaded:
             print("GUI_ERROR: Main image loading failed (self.loaded is False).")
             return

        print(f"GUI_INFO: Loaded main 2D image: {self.filename}")
        print(f"GUI_INFO: Main stack shape: {self.stack.shape}, NZ={self.ortho_nz}, nchan={self.nchan}") # Should be NZ=1

        # 2. Find and load the Z-stack based on filename pattern
        self.stack_ortho = None
        self.ortho_nz = 0
        self.zc_ortho = 0

        try:
            fparts = os.path.splitext(os.path.basename(filename))
            # Try to find the Z index part (e.g., "basename-Z")
            match = re.match(r'(.*)-(\d+)$', fparts[0])
            if match:
                basename = match.group(1) + '-'
                main_z_str = match.group(2)
                main_z_index = int(main_z_str)
                ext = fparts[1]
                folder = os.path.dirname(filename)
                glob_pattern = os.path.join(folder, f"{basename}*{ext}")
                print(f"GUI_INFO: Globbing for Z-stack: {glob_pattern}")

                z_files_dict = {}
                potential_files = io.glob.glob(glob_pattern)
                for fpath in potential_files:
                     fname_only = os.path.splitext(os.path.basename(fpath))[0]
                     z_match = re.match(rf'{re.escape(basename)}(\d+)$', fname_only)
                     if z_match:
                          z_idx = int(z_match.group(1))
                          z_files_dict[z_idx] = fpath

                if not z_files_dict:
                     print("GUI_WARNING: No Z-stack files found matching pattern.")
                     self.ortho_nz = 1
                     self.stack_ortho = self.stack.copy() # Use main stack as ortho stack
                     self.zc_ortho = 0
                else:
                     # Sort files by Z-index
                     sorted_z = sorted(z_files_dict.keys())
                     sorted_files = [z_files_dict[z] for z in sorted_z]
                     print(f"GUI_INFO: Found {len(sorted_files)} Z-planes: {sorted_z}")

                     # Load images into a list first to check dimensions
                     images = []
                     ref_shape = None
                     for i, f in enumerate(sorted_files):
                        try:
                            img = imread(f)
                            # Basic preprocessing (like in io._load_image)
                            if img.ndim == 2: img = img[:,:,np.newaxis]
                            if img.shape[0] < 4 or img.shape[1] < 4: # Handle channel dim placement
                                img = np.transpose(img, (1,2,0))
                            # Check shape consistency
                            current_shape = img.shape[:2] + (img.shape[2],) # Y, X, C
                            if ref_shape is None:
                                    ref_shape = current_shape
                            elif current_shape[:2] != ref_shape[:2]: # Check Y, X only
                                    print(f"GUI_WARNING: Skipping file {f} due to shape mismatch ({current_shape[:2]} vs {ref_shape[:2]})")
                                    continue
                            # Ensure 3 channels if RGB mode is likely
                            if self.nchan > 1 and img.shape[-1] == 1: # If main img is color, expand gray Z
                                    img = np.repeat(img, 3, axis=-1)
                            elif img.shape[-1] > 3: # Take first 3 channels if more exist
                                    img = img[..., :3]
                            images.append(img)
                        except Exception as e_read:
                            print(f"GUI_WARNING: Could not read or process file {f}: {e_read}")

                     if not images:
                          print("GUI_ERROR: Failed to load any valid Z-stack images.")
                          self.ortho_nz = 1
                          self.stack_ortho = self.stack.copy()
                          self.zc_ortho = 0
                     else:
                          # Stack images into numpy array
                          self.stack_ortho = np.stack(images, axis=0) # (NZ, Ly, Lx, C)

                          img_min = self.stack_ortho.min()
                          img_max = self.stack_ortho.max()
                          self.stack_ortho = self.stack_ortho.astype(np.float32)
                          self.stack_ortho -= img_min
                          if img_max > img_min + 1e-3:
                            self.stack_ortho /= (img_max - img_min)
                          self.stack_ortho *= 255
                          self.ortho_nz = self.stack_ortho.shape[0]
                          print(self.stack_ortho.min(), self.stack_ortho.max())

                          if self.stack_ortho.shape[-1] != 3:
                              self.stack_ortho = np.concatenate((self.stack_ortho, np.zeros((self.stack_ortho.shape[0], self.stack_ortho.shape[1], self.stack_ortho.shape[2], 1), dtype=self.stack_ortho.dtype)), axis=-1)
                        #   self.NZ = self.stack_ortho.shape[0]
                          # Find the Z-index of the main loaded image
                          try:
                               self.zc_ortho = sorted_z.index(main_z_index)
                          except ValueError:
                               print(f"GUI_WARNING: Main image Z-index {main_z_index} not found in sorted list {sorted_z}. Using Z=0.")
                               self.zc_ortho = 0
                          print(f"GUI_INFO: Ortho stack loaded. Shape={self.stack_ortho.shape}, Main Z index={self.zc_ortho}")

            else:
                print("GUI_WARNING: Filename does not match expected Z-stack pattern (basename-Z.ext). Cannot load ortho stack.")
                self.ortho_nz = 1
                self.stack_ortho = self.stack.copy() # Use main stack
                self.zc_ortho = 0

        except Exception as e_glob:
            print(f"GUI_ERROR: Failed during Z-stack loading: {e_glob}")
            self.ortho_nz = 1
            self.stack_ortho = self.stack.copy() if self.stack is not None else None
            self.zc_ortho = 0

        # 3. Finalize setup
        # Update saturation array length if NZ_ortho changed

        # Initialize ortho view position to center of main image
        self.yortho = self.Ly // 2
        self.xortho = self.Lx // 2

        # Update UI elements
        self.enable_buttons() # Re-enable buttons after load
        self.update_plot()    # Update main plot and ortho views if active
        self.update_layer()   # Update mask layer display
        self.update_scale()   # Update scale disk

        # Toggle ortho views on if checkbox is checked
        if self.orthobtn.isChecked():
            self.add_orthoviews()
            self.update_ortho()


    def make_orthoviews(self):
        """ Creates the viewboxes and image items for ortho views. """
        self.pOrtho, self.imgOrtho, self.layerOrtho = [], [], []
        for j in range(2):
            self.pOrtho.append(
                pg.ViewBox(lockAspect=True, name=f"plotOrtho{j}",
                           border=[100, 100, 100], invertY=True, enableMouse=False))
            self.pOrtho[j].setMenuEnabled(False)

            self.imgOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self))
            self.imgOrtho[j].autoDownsample = False

            self.layerOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self))
            self.layerOrtho[j].setLevels([0., 255.])

            #self.pOrtho[j].scene().contextMenuItem = self.pOrtho[j]
            self.pOrtho[j].addItem(self.imgOrtho[j])
            self.pOrtho[j].addItem(self.layerOrtho[j])
            self.pOrtho[j].addItem(self.vLineOrtho[j], ignoreBounds=False)
            self.pOrtho[j].addItem(self.hLineOrtho[j], ignoreBounds=False)

        self.pOrtho[0].linkView(self.pOrtho[0].YAxis, self.p0)
        self.pOrtho[1].linkView(self.pOrtho[1].XAxis, self.p0)


    def add_orthoviews(self):
        self.yortho = self.Ly // 2
        self.xortho = self.Lx // 2
        if self.ortho_nz > 1:
            self.update_ortho()

        self.win.addItem(self.pOrtho[0], 0, 1, rowspan=1, colspan=1)
        self.win.addItem(self.pOrtho[1], 1, 0, rowspan=1, colspan=1)

        qGraphicsGridLayout = self.win.ci.layout
        qGraphicsGridLayout.setColumnStretchFactor(0, 2)
        qGraphicsGridLayout.setColumnStretchFactor(1, 1)
        qGraphicsGridLayout.setRowStretchFactor(0, 2)
        qGraphicsGridLayout.setRowStretchFactor(1, 1)

        #self.p0.linkView(self.p0.YAxis, self.pOrtho[0])
        #self.p0.linkView(self.p0.XAxis, self.pOrtho[1])

        self.pOrtho[0].setYRange(0, self.Lx)
        self.pOrtho[0].setXRange(-self.dz / 3, self.dz * 2 + self.dz / 3)
        self.pOrtho[1].setYRange(-self.dz / 3, self.dz * 2 + self.dz / 3)
        self.pOrtho[1].setXRange(0, self.Ly)
        #self.pOrtho[0].setLimits(minXRange=self.dz*2+self.dz/3*2)
        #self.pOrtho[1].setLimits(minYRange=self.dz*2+self.dz/3*2)

        self.p0.addItem(self.vLine, ignoreBounds=False)
        self.p0.addItem(self.hLine, ignoreBounds=False)
        self.p0.setYRange(0, self.Lx)
        self.p0.setXRange(0, self.Ly)

        self.win.show()
        self.show()

        #self.p0.linkView(self.p0.XAxis, self.pOrtho[1])

    def remove_orthoviews(self):
        self.win.removeItem(self.pOrtho[0])
        self.win.removeItem(self.pOrtho[1])
        self.p0.removeItem(self.vLine)
        self.p0.removeItem(self.hLine)
        self.win.show()
        self.show()


    def update_crosshairs(self):
        self.yortho = min(self.Ly - 1, max(0, int(self.yortho)))
        self.xortho = min(self.Lx - 1, max(0, int(self.xortho)))
        self.vLine.setPos(self.xortho)
        self.hLine.setPos(self.yortho)
        self.vLineOrtho[1].setPos(self.xortho)
        self.hLineOrtho[1].setPos(self.zc)
        self.vLineOrtho[0].setPos(self.zc)
        self.hLineOrtho[0].setPos(self.yortho)

    def update_ortho(self):
        if self.ortho_nz > 1 and self.orthobtn.isChecked():
            dzcurrent = self.dz
            self.dz = min(100, max(3, int(self.dzedit.text())))
            self.zaspect = max(0.01, min(100., float(self.zaspectedit.text())))
            self.dzedit.setText(str(self.dz))
            self.zaspectedit.setText(str(self.zaspect))
            if self.dz != dzcurrent:
                self.pOrtho[0].setXRange(-self.dz / 3, self.dz * 2 + self.dz / 3)
                self.pOrtho[1].setYRange(-self.dz / 3, self.dz * 2 + self.dz / 3)
            dztot = min(self.ortho_nz, self.dz * 2)
            y = self.yortho
            x = self.xortho
            z = self.currentZ
            if dztot == self.ortho_nz:
                zmin, zmax = 0, self.ortho_nz
            else:
                if z - self.dz < 0:
                    zmin = 0
                    zmax = zmin + self.dz * 2
                elif z + self.dz >= self.ortho_nz:
                    zmax = self.ortho_nz
                    zmin = zmax - self.dz * 2
                else:
                    zmin, zmax = z - self.dz, z + self.dz
            self.zc = z - zmin
            self.update_crosshairs()
            if self.view == 0 or self.view == 4:
                for j in range(2):
                    if j == 0:
                        if self.view == 0:
                            image = self.stack_ortho[zmin:zmax, :, x].transpose(1, 0, 2).copy()
                        else:
                            image = self.stack_ortho[zmin:zmax, :,
                                                        x].transpose(1, 0, 2).copy()
                    else:
                        image = self.stack_ortho[
                            zmin:zmax,
                            y, :].copy() if self.view == 0 else self.stack_ortho[zmin:zmax,
                                                                             y, :].copy()
                    if self.nchan == 1:
                        # show single channel
                        image = image[..., 0]
                    if self.color == 0:
                        self.imgOrtho[j].setImage(image, autoLevels=False, lut=None)
                        if self.nchan > 1:
                            levels = np.array([
                                self.saturation[0][self.currentZ],
                                self.saturation[1][self.currentZ],
                                self.saturation[2][self.currentZ]
                            ])
                            self.imgOrtho[j].setLevels(levels)
                        else:
                            self.imgOrtho[j].setLevels(
                                self.saturation[0][self.currentZ])
                    elif self.color > 0 and self.color < 4:
                        if self.nchan > 1:
                            image = image[..., self.color - 1]
                        self.imgOrtho[j].setImage(image, autoLevels=False,
                                                  lut=self.cmap[self.color])
                        if self.nchan > 1:
                            self.imgOrtho[j].setLevels(
                                self.saturation[self.color - 1][self.currentZ])
                        else:
                            self.imgOrtho[j].setLevels(
                                self.saturation[0][self.currentZ])
                    elif self.color == 4:
                        if image.ndim > 2:
                            image = image.astype("float32").mean(axis=2).astype("uint8")
                        self.imgOrtho[j].setImage(image, autoLevels=False, lut=None)
                        self.imgOrtho[j].setLevels(self.saturation[0][self.currentZ])
                    elif self.color == 5:
                        if image.ndim > 2:
                            image = image.astype("float32").mean(axis=2).astype("uint8")
                        self.imgOrtho[j].setImage(image, autoLevels=False,
                                                  lut=self.cmap[0])
                        self.imgOrtho[j].setLevels(self.saturation[0][self.currentZ])
                self.pOrtho[0].setAspectLocked(lock=True, ratio=self.zaspect)
                self.pOrtho[1].setAspectLocked(lock=True, ratio=1. / self.zaspect)

            else:
                image = np.zeros((10, 10), "uint8")
                self.imgOrtho[0].setImage(image, autoLevels=False, lut=None)
                self.imgOrtho[0].setLevels([0.0, 255.0])
                self.imgOrtho[1].setImage(image, autoLevels=False, lut=None)
                self.imgOrtho[1].setLevels([0.0, 255.0])

        zrange = zmax - zmin
        self.layer_ortho = [
            np.zeros((self.Ly, zrange, 4), "uint8"),
            np.zeros((zrange, self.Lx, 4), "uint8")
        ]

        for j in range(2):
            self.layerOrtho[j].setImage(self.layer_ortho[j])
        self.win.show()
        self.show()

    def toggle_ortho(self):
        """ Toggles the visibility of the orthographic views. """
        if self.orthobtn.isChecked():
            self.add_orthoviews()
        else:
            self.remove_orthoviews()


    def plot_clicked(self, event):
        if event.button()==QtCore.Qt.LeftButton \
                and not event.modifiers() & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier)\
                and not self.removing_region:
            if event.double():
                try:
                    self.p0.setYRange(0, self.Ly + self.pr)
                except:
                    self.p0.setYRange(0, self.Ly)
                self.p0.setXRange(0, self.Lx)
            elif self.loaded and not self.in_stroke:
                if self.orthobtn.isChecked():
                    items = self.win.scene().items(event.scenePos())
                    for x in items:
                        if x == self.p0:
                            pos = self.p0.mapSceneToView(event.scenePos())
                            x = int(pos.x())
                            y = int(pos.y())
                            if y >= 0 and y < self.Ly and x >= 0 and x < self.Lx:
                                self.yortho = y
                                self.xortho = x
                                self.update_ortho()


    def update_plot(self):
        """ Overrides base update_plot to also update ortho views. """
        super().update_plot() # Update main view first
        if self.loaded and hasattr(self, 'orthobtn') and self.orthobtn.isChecked():
            self.update_ortho() # Update ortho views after main view updates


    def keyPressEvent(self, event):
         """ Override keyPressEvent to handle potential Z navigation keys if needed,
             but primarily rely on base class for 2D controls. """
         # Let the base class handle most keys first
         super().keyPressEvent(event)

         # Add any ortho-specific key handling here if necessary
         # For example, maybe arrow keys could move the ortho crosshair?
         # if self.loaded and self.orthobtn.isChecked() and not self.in_stroke:
         #    # Example: Move crosshair with Alt + Arrows
         #    if event.modifiers() & QtCore.Qt.AltModifier:
         #        step = 5
         #        if event.key() == QtCore.Qt.Key_Up:
         #            self.yortho = max(0, self.yortho - step)
         #            self.update_ortho()
         #        elif event.key() == QtCore.Qt.Key_Down:
         #            self.yortho = min(self.Ly - 1, self.yortho + step)
         #            self.update_ortho()
         #        elif event.key() == QtCore.Qt.Key_Left:
         #            self.xortho = max(0, self.xortho - step)
         #            self.update_ortho()
         #        elif event.key() == QtCore.Qt.Key_Right:
         #            self.xortho = min(self.Lx - 1, self.xortho + step)
         #            self.update_ortho()


    # --- Ensure 2D Behavior for Segmentation/Masking ---
    # Inherited methods like compute_segmentation, add_mask, remove_cell
    # should work correctly as they operate on self.stack[0], self.cellpix[0] etc.
    # The visualization in ortho views is handled by the overridden update_ortho.

# <<< END OF MainW_ortho2D CLASS >>>