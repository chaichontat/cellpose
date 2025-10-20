# Segmentation Diff Viewer Notes

- `cellpose/gui/gui.py`
  - `MainW.make_buttons` creates `self.diffButton` (row under the progress bar) and connects it to `MainW.show_segmentation_diff`.
  - `MainW.reset` clears `_diff_seg_path`, `_diff_state_old`, `_diff_state_new`, and `_diff_latest_masks`, resetting both diff controls whenever a new image is loaded.
  - `_diff_refresh_seg_path()` resolves `<image>_seg.npy` after `enable_buttons()` runs; `_diff_get_saved_state()` caches the saved masks/outlines/colors, and `_diff_update_button_state()` checks both sources (saved + latest model) before enabling diff/reset. Tooltips explain missing prerequisites, flag Matplotlib absence, and note that Z-stacks compare the current plane.
  - `compute_segmentation()` wipes stale results at start and `_diff_store_current_as_new()` snapshots the latest model masks/outlines/colors for later toggling.
  - `maskToggleButton` calls `toggle_mask_restore()`, which swaps between the saved `_seg.npy` state and the model result via `_diff_apply_state(...)`, limited to cases where shapes match exactly.
  - `show_segmentation_diff()` now tracks the Matplotlib figure/axes and overlays a cyan crosshair that follows the active XY crosshair (via `_diff_update_crosshair_lines()` and `get_crosshair_coords()`; ortho mode returns `(yortho, xortho)`).
  - `show_segmentation_diff()` reloads the saved state, validates the `masks` payload, extracts the current Z-plane when stacks are present, calls `contour_diff_rgb(...)`, and displays the overlay in a non-blocking Matplotlib window (with `QMessageBox` guards for missing files, shape mismatches, or computation errors).

- `cellpose/contrib/diff.py`
  - `contour_diff_rgb(labels_a, labels_b, tol_pixels=2.0, ...)` accepts two 2D integer label maps of identical shape and returns an `(H, W, 3)` `np.uint8` visualization. It handles contour extraction, distance-tolerant matching, optional component pruning, and endpoint/junction emphasis.

- Saved segmentation plumbing (`cellpose/gui/io.py`)
  - `_save_sets` writes `<image>_seg.npy` with the `masks` array and associated metadata; `_load_seg` populates GUI state and calls `parent.enable_buttons()`, which triggers `_diff_refresh_seg_path()`.

- Runtime flow
  1. Load an image (and optional `_seg.npy`) → `enable_buttons()` sets `_diff_seg_path`.
  2. Run a segmentation model → `compute_segmentation()` caches the new masks (entire stack) and re-checks prerequisites.
  3. Diff button becomes active once both sources are available (tooltips clarify any missing requirement).
  4. Clicking `Diff` opens the Matplotlib viewer comparing saved contours vs. the latest model output (current `Z` slice for stacks); the adjacent `reset mask` button toggles the GUI between those two states when their shapes match.

- Current constraints & follow-ups
  - Diff viewer requires Matplotlib; if missing, the button remains disabled with guidance.
  - Mask reset is only available when the saved `_seg.npy` masks match the latest model result shape (including Z-depth); otherwise the control stays disabled with explanatory tooltip.
  - The code surfaces user-facing errors instead of raising, keeping the GUI responsive when `_seg.npy` is missing or malformed.
