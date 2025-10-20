# Change Context
- Added diff/mask management enhancements in `cellpose/gui/gui.py`:
  * Introduced `maskToggleButton` alongside the existing diff control to toggle between saved `_seg.npy` masks and the latest model output.
  * Implemented `_diff_state_old` / `_diff_state_new` caching and helper utilities (`_diff_get_saved_state`, `_diff_store_current_as_new`, `_diff_apply_state`, `_diff_can_reset`) to ensure mask toggling works for both 2D and Z-stack data without corrupting state.
  * Updated `compute_segmentation()` to snapshot the freshly computed masks/outlines, and refactored `show_segmentation_diff()` to reuse the cached saved state.
  * Diff viewer now keeps a cyan crosshair synchronized with the active XY crosshair (ortho mode exposes `(yortho, xortho)` via `get_crosshair_coords()` and `update_crosshairs`).
- Refined diff visualization colors in `cellpose/contrib/diff.py` so “old” contours render magenta, “new” contours render green, and shared contours appear dim gray.
- Documented the updated workflow and safeguards in `diff-viewer.md` (diff button prerequisites, mask reset behavior, error handling expectations).

# Scope
- GUI code paths touched: diff button enablement, mask toggling logic, segmentation post-processing, and diff display handling.
- Visualization utility adjustments limited to `contour_diff_rgb` color palette and documentation.
- No changes were made to the ortho GUI logic after the final fixes; ensure future edits preserve the new diff/mask constraints for both classic and ortho modes.

# Instructions for Next Agent
1. When modifying mask toggling or diff behavior, verify `_diff_can_reset()` still blocks shape-mismatched toggles—especially for Z-stacks loaded via `guiortho`.
2. Keep `_diff_state_old`/`_diff_state_new` structures synchronized (keys: `masks`, `outlines`, `colors`), and update `_diff_store_current_as_new()` if mask data structures change.
3. For additional visual tweaks, adjust `contour_diff_rgb` defaults while keeping the docstring legend accurate.
4. Run `python -m compileall cellpose/gui/gui.py cellpose/contrib/diff.py` after edits to catch syntax or indentation issues in the large GUI module.
5. Update `diff-viewer.md` if you alter button placement, tooltips, or the diff/mask workflow so others can follow the current UX.
