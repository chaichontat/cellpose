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
        self._ortho_seg_warned = False  # Track whether we've logged segmentation alignment warnings

        # MainW expects a resample control when handling 3D flows; default to True so
        # downstream logic that checks self.resample works even without a checkbox.
        if not hasattr(self, "resample"):
            self.resample = True

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
        self._ortho_axes_linked = False
        self.make_orthoviews() # Create the ortho view boxes and image items

        # Add ortho view controls to the left panel (self.l0)
        # Find the row index 'b' where the scale disk checkbox was added in make_buttons
        # This is fragile; better to add controls programmatically if layout changes.
        # Assuming 'b' from make_buttons is accessible or recalculate.
        # Let's find the ScaleOn widget and insert before it if it exists (older GUI layout).
        scale_widget = getattr(self, "ScaleOn", None)
        scale_widget_row = -1
        if scale_widget is not None:
            for i in range(self.l0.rowCount()):
                item = self.l0.itemAtPosition(i, 0)
                if item and item.widget() == scale_widget:
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

        # Shift the scale disk toggle down by one row if present
        if scale_widget is not None:
            self.l0.addWidget(scale_widget, ortho_row_start + 1, 0, 1, 5)

        # --- Load Initial Image ---
        if image is not None:
            self.filename = image
            self._load_image_ortho2D(self.filename) # Use the ortho loading method

    # --- Method Overrides for Ortho2D Functionality ---

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if not files:
            return
        target = files[0]
        ext = os.path.splitext(target)[1].lower()
        if ext == ".npy":
            paired_image = io.resolve_seg_drop_target(target)
            if paired_image:
                target = paired_image
            else:
                msg = (
                    "Ortho viewer does not accept .npy segmentation files. "
                    "Use the standard GUI to load segmentations, or drop an image file."
                )
                print(f"GUI_ERROR: {msg}")
                try:
                    QMessageBox.critical(self, "Unsupported file", msg)
                except Exception:
                    pass
                return
        print(files)
        self._load_image_ortho2D(filename=target, load_seg=True)

    def _load_image(self, filename=None, load_seg=True):
        """ Override base _load_image to call the ortho version. """
        self._load_image_ortho2D(filename=filename, load_seg=load_seg)

    def _load_image_ortho2D(self, filename=None, load_seg=True):
        """ Loads the main 2D image and then finds/loads the Z-stack. """
        if filename is None:
            filename = self.filename

        # 1. Load the main 2D image using the standard io function
        # Set load_3D=False explicitly for the base loader
        original_load_3d_flag = self.load_3D
        try:
            # Temporarily force 2D load; guiortho manages its own ortho stack.
            self.load_3D = False
            io._load_image(self, filename=filename, load_seg=load_seg, load_3D=False)
        except Exception as e:
            print(f"GUI_ERROR: Failed to load main image '{filename}': {e}")
            self.loaded = False
        finally:
            self.load_3D = original_load_3d_flag

        if not self.loaded and load_seg:
            # Retry without segmentation to handle stale/missing _seg.npy files gracefully.
            print("GUI_WARNING: Primary load failed; retrying without segmentation file.")
            try:
                self.load_3D = False
                io._load_image(self, filename=filename, load_seg=False, load_3D=False)
            except Exception as e:
                print(f"GUI_ERROR: Fallback load failed for '{filename}': {e}")
                self.loaded = False
            finally:
                self.load_3D = original_load_3d_flag

        if not self.loaded:
             print("GUI_ERROR: Main image loading failed (self.loaded is False).")
             return

        print(f"GUI_INFO: Loaded main 2D image: {self.filename}")
        print(f"GUI_INFO: Main stack shape: {self.stack.shape}, NZ={self.ortho_nz}, nchan={self.nchan}") # Should be NZ=1

        # 2. Find and load the Z-stack based on filename pattern
        self.stack_ortho = None
        self.ortho_nz = 0
        self.zc_ortho = 0
        self._ortho_seg_warned = False

        try:
            from pathlib import Path
            filepath = Path(filename)
            # Split into stem and suffix
            basename = filepath.stem
            ext = filepath.suffix

            # Try to find Z index using supported patterns:
            #   (1) basename_z<idx>.<ext>  e.g., sample_z05.tif
            #   (2) basename-<idx>.<ext>   e.g., sample-5.tif
            folder = filepath.parent
            m_z = re.match(r'^(.*)_z(\d+)$', basename)
            m_dash = re.match(r'^(.*)-(\d+)$', basename)
            if m_z or m_dash:
                if m_z:
                    family = m_z.group(1)
                    main_z_index = int(m_z.group(2))
                    glob_glob = f"{family}_z*{ext}"
                    stem_regex = re.compile(rf'^{re.escape(family)}_z(\d+)$')
                else:
                    family = m_dash.group(1)
                    main_z_index = int(m_dash.group(2))
                    glob_glob = f"{family}-*{ext}"
                    stem_regex = re.compile(rf'^{re.escape(family)}-(\d+)$')
                glob_pattern = str(folder / glob_glob)
                print(f"GUI_INFO: Globbing for Z-stack: {glob_pattern}")

                z_files_dict = {}
                potential_files = list(Path(folder).glob(glob_glob))
                for fpath in potential_files:
                    fname_only = fpath.stem
                    z_match = stem_regex.match(fname_only)
                    if z_match:
                        z_idx = int(z_match.group(1))
                        z_files_dict[z_idx] = str(fpath)

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

                    # Restrict loading to +/- 20 frames around the central index to avoid loading full stacks
                    half_span = 20
                    try:
                        center_pos = sorted_z.index(main_z_index)
                    except ValueError:
                        center_pos = 0
                    lo = max(0, center_pos - half_span)
                    hi = min(len(sorted_z), center_pos + half_span + 1)
                    if lo > 0 or hi < len(sorted_z):
                        print(f"GUI_INFO: Restricting Z window to indices [{sorted_z[lo]}..{sorted_z[hi-1]}] around {main_z_index}")
                    sorted_z = sorted_z[lo:hi]
                    sorted_files = sorted_files[lo:hi]

                    # Load images into a list first to check dimensions
                    images = []
                    used_files = []
                    used_z_indices = []
                    ref_shape = None
                    for i, (z_idx, f) in enumerate(zip(sorted_z, sorted_files)):
                        try:
                            img = imread(f)
                            # Basic preprocessing (like in io._load_image)
                            if img.ndim == 2:
                                img = img[:, :, np.newaxis]
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
                            used_files.append(f)
                            used_z_indices.append(z_idx)
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
                        # Keep file list aligned with stacked images
                        self.ortho_files_sorted = used_files
                        self.ortho_used_z_indices = used_z_indices
                        print(self.stack_ortho.min(), self.stack_ortho.max())

                        if self.stack_ortho.shape[-1] != 3:
                            self.stack_ortho = np.concatenate((self.stack_ortho, np.zeros((self.stack_ortho.shape[0], self.stack_ortho.shape[1], self.stack_ortho.shape[2], 1), dtype=self.stack_ortho.dtype)), axis=-1)
                        #   self.NZ = self.stack_ortho.shape[0]
                        # Find the Z-index of the main loaded image
                        try:
                            self.zc_ortho = used_z_indices.index(main_z_index)
                        except ValueError:
                            print(f"GUI_WARNING: Main image Z-index {main_z_index} not present after filtering. Using Z=0.")
                            self.zc_ortho = 0
                        print(f"GUI_INFO: Ortho stack loaded. Shape={self.stack_ortho.shape}, Main Z index={self.zc_ortho}")

            else:
                print("GUI_WARNING: Filename does not match expected Z-stack pattern (basename_z##.ext or basename-#.ext). Cannot load ortho stack.")
                self.ortho_nz = 1
                self.stack_ortho = self.stack.copy() # Use main stack
                self.zc_ortho = 0
                self.ortho_files_sorted = []

        except Exception as e_glob:
            print(f"GUI_ERROR: Failed during Z-stack loading: {e_glob}")
            self.ortho_nz = 1
            self.stack_ortho = self.stack.copy() if self.stack is not None else None
            self.zc_ortho = 0
            self.ortho_files_sorted = []

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
        self.anchorScatterOrtho = []
        self.labelOrtho = []
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
            anchor_item = pg.ScatterPlotItem(
                symbol="s",
                size=8,
                brush=self._anchor_brush,
                pen=None,
                pxMode=True,
            )
            anchor_item.setZValue(25)
            self.anchorScatterOrtho.append(anchor_item)
            self.pOrtho[j].addItem(anchor_item)

            # Add text label overlay inside each ortho view (top-left)
            lbl = pg.TextItem("")
            try:
                lbl.setAnchor((0, 0))
            except Exception:
                pass
            lbl.setZValue(10)
            self.labelOrtho.append(lbl)
            self.pOrtho[j].addItem(lbl)

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

        z_init = max(1, 2 * max(3, int(self.dz)))

        # Configure ranges before linking to avoid pushing p0 offscreen.
        self.pOrtho[0].setAspectLocked(lock=False)
        self.pOrtho[0].setXRange(0, z_init, padding=0)
        self.pOrtho[0].setYRange(0, self.Ly, padding=0)
        self.pOrtho[0].setAspectLocked(lock=True, ratio=self.zaspect)

        self.pOrtho[1].setAspectLocked(lock=False)
        self.pOrtho[1].setYRange(0, z_init, padding=0)
        self.pOrtho[1].setXRange(0, self.Lx, padding=0)
        self.pOrtho[1].setAspectLocked(lock=True, ratio=1. / self.zaspect)
        if not self._ortho_axes_linked:
            self.pOrtho[0].linkView(self.pOrtho[0].YAxis, self.p0)
            self.pOrtho[1].linkView(self.pOrtho[1].XAxis, self.p0)
            self._ortho_axes_linked = True

        self.p0.addItem(self.vLine, ignoreBounds=False)
        self.p0.addItem(self.hLine, ignoreBounds=False)

        self.win.show()
        self.show()
        self._update_ortho_anchor_display()

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
        self._diff_update_crosshair_lines((self.yortho, self.xortho))
        self._update_ortho_anchor_display()

    def get_crosshair_coords(self):
        return float(self.yortho), float(self.xortho)

    # Developer note (gui_ortho invariants):
    # - Do not change main XY view (self.p0) ranges in this function.
    # - Axis linking: pOrtho[0].YAxis is linked to self.p0 (Y); pOrtho[1].XAxis is linked to self.p0 (X).
    # - When updating orthoviews, adjust only the Z axis (X for YZ, Y for XZ);
    #   preserve the linked axis range and then re-enable aspect lock to prevent implicit zooms.
    # - Z windowing: default is centered +/- dz around zc_ortho. Clicking in YZ/XZ sets
    #   _preserve_window and _next_z_click_local_k so the clicked Z remains at the same on-screen
    #   position across updates; state (_zi_start, _zrange, zc) encodes the current window.
    # - Keep Z ranges tight and non-negative [0, zrange]; avoid padding offsets.
    # - Clamp dz and z-aspect; tolerate invalid text inputs by falling back to prior values.
    def update_ortho(self):
        if not self.orthobtn.isChecked():
            return

        if self.stack_ortho is None or self.ortho_nz == 0:
            return

        # Snapshot main view range to prevent inadvertent zoom/pan changes
        try:
            p0_xr, p0_yr = self.p0.viewRange()
        except Exception:
            p0_xr, p0_yr = None, None

        # Parse inputs defensively
        try:
            requested_dz = int(self.dzedit.text())
        except Exception:
            requested_dz = self.dz
        self.dz = min(100, max(3, requested_dz))
        try:
            requested_zaspect = float(self.zaspectedit.text())
        except Exception:
            requested_zaspect = self.zaspect
        self.zaspect = max(0.01, min(100., requested_zaspect))
        self.dzedit.setText(str(self.dz))
        self.zaspectedit.setText(str(self.zaspect))

        # Build a Z window either centered (default) or preserving on-screen position after an ortho click
        y = self.yortho
        x = self.xortho
        z_center = min(max(self.zc_ortho, 0), max(self.ortho_nz - 1, 0))
        zrange = max(1, 2 * self.dz)
        center_mode = not getattr(self, "_preserve_window", False)

        # Previous window start and a potential local click index
        prev_start = getattr(self, "_zi_start", None)
        click_k = getattr(self, "_next_z_click_local_k", None)

        if center_mode or prev_start is None:
            zi_start = z_center - self.dz
        else:
            # Keep the clicked z at the same screen position k if provided
            if click_k is not None:
                zi_start = z_center - int(click_k)
            else:
                zi_start = prev_start

        # Enforce bounds
        zi_start = max(0, min(zi_start, max(0, self.ortho_nz - zrange)))
        zi_end_excl = zi_start + zrange

        # Local crosshair position within the orthoviews
        if center_mode or click_k is None:
            self.zc = min(max(z_center - zi_start, 0), zrange - 1)
        else:
            self.zc = min(max(int(click_k), 0), zrange - 1)

        # Persist window state and clear one-shot flags
        self._zi_start = zi_start
        self._zrange = zrange
        self._preserve_window = False
        self._next_z_click_local_k = None
        self.update_crosshairs()

        if self.view == 0 or self.view == 4:
            # Build orthoview images using current window (pad implicitly by bounds checks)
            C = self.stack_ortho.shape[-1]
            dtype = self.stack_ortho.dtype
            Ly_target, Lx_target = int(self.Ly), int(self.Lx)
            yz_img = np.zeros((Ly_target, zrange, C), dtype=dtype)
            xz_img = np.zeros((zrange, Lx_target, C), dtype=dtype)

            # Prepare resampling indices if needed (to match main view dims)
            sy, sx = int(self.stack_ortho.shape[1]), int(self.stack_ortho.shape[2])
            idx_y = None
            idx_x = None
            if sy != Ly_target:
                idx_y = np.clip(np.round(np.linspace(0, sy - 1, Ly_target)).astype(int), 0, sy - 1)
            if sx != Lx_target:
                idx_x = np.clip(np.round(np.linspace(0, sx - 1, Lx_target)).astype(int), 0, sx - 1)

            # Determine valid local range to copy
            local_from = max(0, -zi_start)
            local_to_excl = min(zrange, self.ortho_nz - zi_start)
            for k in range(local_from, local_to_excl):
                z_abs = zi_start + k
                if 0 <= z_abs < self.ortho_nz:
                    col = self.stack_ortho[z_abs, :, x, :]
                    row = self.stack_ortho[z_abs, y, :, :]
                    if idx_y is not None:
                        col = col[idx_y, :]
                    if idx_x is not None:
                        row = row[idx_x, :]
                    yz_img[:, k, :] = col
                    xz_img[k, :, :] = row

            # Compute levels to exactly mirror main view
            levels_main = None
            lut_main = None
            image_mode = self.color
            if image_mode == 0:
                if self.nchan > 1:
                    levels_main = np.array([
                        self.saturation[0][self.currentZ],
                        self.saturation[1][self.currentZ],
                        self.saturation[2][self.currentZ],
                    ])
                else:
                    levels_main = self.saturation[0][self.currentZ]
                lut_main = None
            elif 0 < image_mode < 4:
                levels_main = self.saturation[(self.color - 1) if self.nchan > 1 else 0][self.currentZ]
                lut_main = self.cmap[self.color]
            elif image_mode == 4:
                levels_main = self.saturation[0][self.currentZ]
                lut_main = None
            elif image_mode == 5:
                levels_main = self.saturation[0][self.currentZ]
                lut_main = self.cmap[0]

            for j in range(2):
                image = yz_img if j == 0 else xz_img
                if self.nchan == 1:
                    image = image[..., 0]
                if image_mode == 0:
                    self.imgOrtho[j].setImage(image, autoLevels=False, lut=lut_main)
                elif 0 < image_mode < 4:
                    if self.nchan > 1:
                        image = image[..., self.color - 1]
                    self.imgOrtho[j].setImage(image, autoLevels=False, lut=lut_main)
                elif image_mode in (4, 5):
                    if image.ndim > 2:
                        image = image.astype("float32").mean(axis=2).astype("uint8")
                    self.imgOrtho[j].setImage(image, autoLevels=False, lut=lut_main)
                # Apply the exact same levels as the main view
                if levels_main is not None:
                    self.imgOrtho[j].setLevels(levels_main)

            # Tight Z extents without padding so images align at left/top.
            # Prevent linked axes (to p0) from being altered by aspect lock; restore them after setting Z ranges.
            # YZ view: preserve current Y range while updating X (Z) range
            try:
                xr0, yr0 = self.pOrtho[0].viewRange()
            except Exception:
                xr0, yr0 = [0, self.Lx], [0, self.Ly]
            self.pOrtho[0].setAspectLocked(lock=False)
            self.pOrtho[0].setXRange(0, zrange, padding=0)
            self.pOrtho[0].setYRange(yr0[0], yr0[1], padding=0)
            self.pOrtho[0].setAspectLocked(lock=True, ratio=self.zaspect)

            # XZ view: preserve current X range while updating Y (Z) range
            try:
                xr1, yr1 = self.pOrtho[1].viewRange()
            except Exception:
                xr1, yr1 = [0, self.Lx], [0, self.Ly]
            self.pOrtho[1].setAspectLocked(lock=False)
            self.pOrtho[1].setYRange(0, zrange, padding=0)
            self.pOrtho[1].setXRange(xr1[0], xr1[1], padding=0)
            self.pOrtho[1].setAspectLocked(lock=True, ratio=1. / self.zaspect)
        else:
            image = np.zeros((10, 10), "uint8")
            self.imgOrtho[0].setImage(image, autoLevels=False, lut=None)
            self.imgOrtho[0].setLevels([0.0, 255.0])
            self.imgOrtho[1].setImage(image, autoLevels=False, lut=None)
            self.imgOrtho[1].setLevels([0.0, 255.0])

        # Layers sized to match orthoviews
        self.layer_ortho = [
            np.zeros((self.Ly, zrange, 4), "uint8"),
            np.zeros((zrange, self.Lx, 4), "uint8")
        ]

        cellpix = getattr(self, "cellpix", None)
        outpix = getattr(self, "outpix", None)
        masks_available = (
            getattr(self, "masksOn", False)
            and isinstance(cellpix, np.ndarray)
            and cellpix.ndim == 3
            and cellpix.size > 0
        )
        outlines_available = (
            getattr(self, "outlinesOn", False)
            and isinstance(outpix, np.ndarray)
            and outpix.ndim == 3
            and outpix.size > 0
        )

        if masks_available or outlines_available:
            window_width = max(0, local_to_excl - local_from)
            if window_width > 0:
                cp_sections = [None, None]
                op_sections = [None, None]

                if masks_available:
                    cp_sections = [
                        np.zeros((self.Ly, zrange), dtype=cellpix.dtype),
                        np.zeros((zrange, self.Lx), dtype=cellpix.dtype),
                    ]
                if outlines_available:
                    op_sections = [
                        np.zeros((self.Ly, zrange), dtype=outpix.dtype),
                        np.zeros((zrange, self.Lx), dtype=outpix.dtype),
                    ]

                ortho_indices = getattr(self, "ortho_used_z_indices", None)
                if isinstance(ortho_indices, np.ndarray):
                    ortho_indices = ortho_indices.tolist()
                ortho_len = len(ortho_indices) if isinstance(ortho_indices, (list, tuple)) else 0
                center_abs = None
                if ortho_len and 0 <= self.zc_ortho < ortho_len:
                    center_abs = ortho_indices[self.zc_ortho]

                def resolve_seg_index(z_abs: int):
                    if getattr(self, "NZ", 0) <= 0:
                        return None
                    base_idx = int(getattr(self, "currentZ", 0))
                    seg_idx = None
                    if ortho_len and center_abs is not None and 0 <= z_abs < ortho_len:
                        target_abs = ortho_indices[z_abs]
                        seg_idx = base_idx + (target_abs - center_abs)
                    else:
                        seg_idx = base_idx + (z_abs - self.zc_ortho)
                    if 0 <= seg_idx < self.NZ:
                        return int(seg_idx)
                    if not self._ortho_seg_warned:
                        print(f"GUI_WARNING: Ortho overlay skipped (no segmentation slice for z index {z_abs}).")
                        self._ortho_seg_warned = True
                    return None

                # Populate cross-sections for masks/outlines
                for k in range(local_from, local_to_excl):
                    z_abs = zi_start + k
                    seg_idx = resolve_seg_index(z_abs)
                    if seg_idx is None:
                        continue
                    if masks_available and seg_idx < cellpix.shape[0]:
                        if 0 <= x < cellpix.shape[2]:
                            cp_sections[0][:, k] = cellpix[seg_idx, :, x]
                        if 0 <= y < cellpix.shape[1]:
                            cp_sections[1][k, :] = cellpix[seg_idx, y, :]
                    if outlines_available and seg_idx < outpix.shape[0]:
                        if 0 <= x < outpix.shape[2]:
                            op_sections[0][:, k] = outpix[seg_idx, :, x]
                        if 0 <= y < outpix.shape[1]:
                            op_sections[1][k, :] = outpix[seg_idx, y, :]

                if masks_available:
                    cellcolors = np.asarray(self.cellcolors, dtype=np.uint8)
                    selected_idx = int(getattr(self, "selected", 0))
                    opacity_val = int(np.clip(getattr(self, "opacity", 0), 0, 255))
                    z_slice = slice(local_from, local_to_excl)
                    for j, cp_slice in enumerate(cp_sections):
                        if cp_slice is None:
                            continue
                        cp_safe = cp_slice.copy()
                        if cellcolors.shape[0] == 0:
                            cp_safe[:] = 0
                        else:
                            too_large = cp_safe >= cellcolors.shape[0]
                            if np.any(too_large):
                                cp_safe[too_large] = 0
                        if j == 0:
                            cp_segment = cp_safe[:, z_slice]
                            if cp_segment.size > 0:
                                layer_view = self.layer_ortho[j][:, z_slice]
                                layer_view[..., :3] = cellcolors[cp_segment, :]
                                alpha_block = (opacity_val * (cp_segment > 0).astype(np.uint8)).astype(np.uint8)
                                layer_view[..., 3] = alpha_block
                                if selected_idx > 0:
                                    sel_mask = cp_segment == selected_idx
                                    if np.any(sel_mask):
                                        layer_view[sel_mask] = np.array(
                                            [255, 255, 255, opacity_val], dtype=np.uint8
                                        )
                        else:
                            cp_segment = cp_safe[z_slice, :]
                            if cp_segment.size > 0:
                                layer_view = self.layer_ortho[j][z_slice, :]
                                layer_view[..., :3] = cellcolors[cp_segment, :]
                                alpha_block = (opacity_val * (cp_segment > 0).astype(np.uint8)).astype(np.uint8)
                                layer_view[..., 3] = alpha_block
                                if selected_idx > 0:
                                    sel_mask = cp_segment == selected_idx
                                    if np.any(sel_mask):
                                        layer_view[sel_mask] = np.array(
                                            [255, 255, 255, opacity_val], dtype=np.uint8
                                        )

                if outlines_available:
                    outline_rgba = np.array(self.outcolor, dtype=np.uint8)
                    for j, op_slice in enumerate(op_sections):
                        if op_slice is None:
                            continue
                        outline_mask = op_slice > 0
                        if np.any(outline_mask):
                            self.layer_ortho[j][outline_mask] = outline_rgba

        for j in range(2):
            self.layerOrtho[j].setImage(self.layer_ortho[j])

        # Restore main view range (prevents zoom resets from axis-link side effects)
        if p0_xr is not None and p0_yr is not None:
            try:
                self.p0.setRange(xRange=p0_xr, yRange=p0_yr, padding=0)
            except Exception:
                pass

        self._update_anchor_display()
        self._update_ortho_anchor_display()
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
            if self.loaded and not self.in_stroke and self.orthobtn.isChecked():
                if (self.pOrtho[0] in items) or (self.imgOrtho[0] in items) or (self.layerOrtho[0] in items):
                    pos = self.pOrtho[0].mapSceneToView(event.scenePos())
                    k = int(pos.x())
                    zrange = getattr(self, '_zrange', max(1, 2*self.dz))
                    k = max(0, min(k, zrange - 1))
                    zi_start = getattr(self, '_zi_start', max(0, self.zc_ortho - self.dz))
                    z_abs = zi_start + k
                    self.zc_ortho = max(0, min(z_abs, self.ortho_nz - 1))
                    self._preserve_window = True
                    self._next_z_click_local_k = k
                    self.update_ortho()
                    self._update_anchor_display()
                    return
                if (self.pOrtho[1] in items) or (self.imgOrtho[1] in items) or (self.layerOrtho[1] in items):
                    pos = self.pOrtho[1].mapSceneToView(event.scenePos())
                    k = int(pos.y())
                    zrange = getattr(self, '_zrange', max(1, 2*self.dz))
                    k = max(0, min(k, zrange - 1))
                    zi_start = getattr(self, '_zi_start', max(0, self.zc_ortho - self.dz))
                    z_abs = zi_start + k
                    self.zc_ortho = max(0, min(z_abs, self.ortho_nz - 1))
                    self._preserve_window = True
                    self._next_z_click_local_k = k
                    self.update_ortho()
                    self._update_anchor_display()
                    return
                if on_main_view:
                    pos = self.p0.mapSceneToView(event.scenePos())
                    xx = int(pos.x())
                    yy = int(pos.y())
                    if 0 <= yy < self.Ly and 0 <= xx < self.Lx:
                        self.yortho = yy
                        self.xortho = xx
                        self.update_ortho()
                        return
        super().plot_clicked(event)

    def _update_ortho_anchor_display(self):
        if not getattr(self, "anchorScatterOrtho", None):
            return
        if not getattr(self, "orthobtn", None) or not self.orthobtn.isChecked():
            for item in self.anchorScatterOrtho:
                item.setData([], [])
            return
        if self.stack_ortho is None or self.ortho_nz == 0:
            for item in self.anchorScatterOrtho:
                item.setData([], [])
            return
        zi_start = getattr(self, "_zi_start", None)
        zrange = getattr(self, "_zrange", None)
        if zi_start is None or zrange is None:
            for item in self.anchorScatterOrtho:
                item.setData([], [])
            return
        x_cross = int(round(getattr(self, "xortho", 0)))
        y_cross = int(round(getattr(self, "yortho", 0)))
        yz_x, yz_y, xz_x, xz_y = [], [], [], []
        for plane, anchors in self._anchor_points_by_plane.items():
            try:
                plane_idx = int(plane)
            except Exception:
                continue
            local_k = plane_idx - zi_start
            if not (0 <= local_k < zrange):
                continue
            for anchor in anchors:
                if anchor["x"] == x_cross:
                    yz_x.append(local_k)
                    yz_y.append(anchor["y"])
                if anchor["y"] == y_cross:
                    xz_x.append(anchor["x"])
                    xz_y.append(local_k)
        self.anchorScatterOrtho[0].setData(
            yz_x,
            yz_y,
            symbol="s",
            size=8,
            brush=self._anchor_brush,
            pen=None,
            pxMode=True,
        )
        self.anchorScatterOrtho[1].setData(
            xz_x,
            xz_y,
            symbol="s",
            size=8,
            brush=self._anchor_brush,
            pen=None,
            pxMode=True,
        )


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
