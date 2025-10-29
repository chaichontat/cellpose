# Packed Inference Notes

This memo captures the current behaviour of the packed inference path after the latest investigation on `PackedCellposeModel`.

## Code overview
- **Top-level classes** – Inference is driven by `PackedCellposeModel` and `PackedCellposeModelTRT` (`cellpose/contrib/packed_infer.py`). Both inherit the standard Cellpose model and override `_run_net` to dispatch into `_run_3d_with_packing` when `do_3D=True` and packing is enabled.
- **Helpers** – `_run_3d_with_packing` lives in the same module. It loops over orthogonal orientations, uses `compute_max_guard`/`compute_stripe_layout` (`cellpose/contrib/pack_utils.py`) to decide whether stripes fit, calls `pack_planes_to_stripes`/`unpack_stripes_to_planes`, and ultimately runs `cellpose.core.run_net` on either the packed stripes or the baseline stack.
- **Core tiler** – `cellpose/core.py` implements `run_net`. Packing reuses the existing tiler but sets `single_tile_if_fit=True` so the padded Y dimension stays at one tile when it already fits `bsize`.
- **Training analogue** – `_prepare_dimension_packed_batch` and `_pack_training_tiles` in `cellpose/train.py` mirror the inference logic. They crop thin images to `pack_stripe_height` (default 68 px), pack `pack_k` of them vertically into a single tile, and leave larger/square crops unchanged. This keeps the training distribution aligned with packed inference when `pack_to_single_tile=True`.

## Pipeline recap
- **Input ordering** – TIFFs are stored ZCYX. The packing routines expect ZYXC, so we transpose once on load.
- **Diameter scaling** – If `diameter` is provided, Cellpose rescales XY by `scale = 30 / diameter`. For the benchmark stack with `diameter = 60`, XY shrinks from `1968×1968` to `984×984`.
- **Anisotropy** – The effective anisotropy becomes `anisotropy × scale`. With `anisotropy = 4`, this yields `2.0`. Cellpose then:
  1. Transposes to `(Y, Z, X, C)`.
  2. Resizes the first axis to `int(Z × effective_anisotropy)` (34 → 68).
  3. Transposes back to `(68, 984, 984, C)`.
- **Packing guard/border** – Defaults: `pack_k = 3, guard = 16, border = 5`. `compute_max_guard` clamps guard to the largest value that keeps `pack_k` stripes inside `bsize`.

## Orientation geometry
After the steps above, the working stack is `68 × 984 × 984 × C`. The three orthogonal passes see:
- `YX` (original XY planes): 68 planes, each `984×984`. Packing is disabled for this orientation.
- `ZY`: transpose to `(984, 68, 984, C)`. `Lyp = 68`, `Lxp = 984`, groups = `ceil(984 / pack_k) = 328` before packing; with `pack_k=3`, packed groups become `ceil(984 / 3) = 328` stripes spread across X tiles.
- `ZX`: transpose to `(984, 68, 984, C)`. Identical dimensions, so packing yields the same tile counts as `ZY`.

Result: once anisotropy is applied after the diameter resize, `ZY` and `ZX` become perfectly symmetric; any earlier asymmetry came from evaluating the pre-upsampled stack (where the stripe height was still 34).

## Benchmark (CUDA_VISIBLE_DEVICES=1, `diameter = 60`, `anisotropy = 4`, `bsize = 256`, `pack_k = 3`)
- Total 3D tiles: baseline `11 540` → packed `4 980` (≈2.32× reduction).
- Orientation detail:
  - `YX`: `1 700` tiles baseline, packing disabled (unchanged).
  - `ZY`: `4 920` → `1 640` tiles (≈3.00×).
  - `ZX`: `4 920` → `1 640` tiles (≈3.00×).
- 2D path across Z planes: `1 700` → `1 700` tiles (packing doesn’t help because XY remains larger than `bsize`).

## Key takeaways
- Packing now runs after anisotropy scaling, so stripe heights reflect the upsampled Z.
- XY is intentionally left on the legacy path; only ZY/ZX are packed.
- ZY and ZX now produce identical tile counts because the array handed to packing is symmetrical after the anisotropy resize.
- XY is still the limiting factor for the 2D path; packing only helps once XY fits inside `bsize`.

---

## Training path details
Packing during training is optional and controlled by the `pack_to_single_tile` flag in `cellpose/train.py`. The relevant flow is:
1. **Stripe height selection** – `_resolve_pack_stripe_height` checks whether the requested `pack_stripe_height` fits alongside `pack_k`, `pack_guard`, and `pack_stripe_border` inside `bsize`. If not, it auto-selects the largest admissible height or disables packing.
2. **Batch routing** – `_prepare_dimension_packed_batch` computes an “effective height” (post diameter rescale) for each crop. Anything shorter than `_THIN_HEIGHT_PX = 100` is batched as stripes; thicker crops stay square. A small fraction (`_SQUARE_STRIPE_RATIO = 0.10`) of larger tiles are sliced into synthetic stripes to diversify the training set.
3. **Stripe assembly** – Thin crops are resized to `(pack_height, bsize)` using `random_rotate_and_resize`, photometric augments are applied if present, and `_pack_training_tiles` stacks `pack_k` stripes plus guard rows into a single tile using the same `compute_stripe_layout` logic as inference.
4. **Standard tiles** – Remaining crops go through the legacy augmentation path (`random_rotate_and_resize(..., xy=(bsize, bsize))`).
5. **Metrics & logging** – The training loop logs per-epoch counts of packed vs standard tiles so you can verify the ratio of thin to large samples. During evaluation the same packing rules are applied if `pack_to_single_tile=True`.

Because we only disable packing for the XY (in-plane) orientation at inference—and training never generated XY stripes in the first place—no additional training changes are required. Keeping `pack_k`, `pack_guard`, `pack_border`, and `pack_stripe_height` aligned between training and inference ensures the packed orthogonal planes match what the network learned.
