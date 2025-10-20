import numpy as np
from scipy import ndimage as ndi


def _label_to_contour(L: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    Return a 1-pixel-thick boolean mask of region boundaries for a label image.

    Implementation chooses one-sided differences to avoid double-thick edges:
    we mark the 'upper' and 'left' side of each pairwise boundary (and the two
    diagonals if connectivity=8). That yields a consistent single-pixel trace.
    """
    if L.ndim != 2:
        raise ValueError("Label images must be 2D arrays.")
    H, W = L.shape
    B = np.zeros((H, W), dtype=bool)

    # Horizontal edges (compare with right neighbor): mark left pixel
    dr = (L[:, :-1] != L[:, 1:])
    B[:, :-1] |= dr

    # Vertical edges (compare with bottom neighbor): mark upper pixel
    dd = (L[:-1, :] != L[1:, :])
    B[:-1, :] |= dd

    if connectivity == 8:
        # Main diagonal (↘): mark upper-left pixel
        d1 = (L[:-1, :-1] != L[1:, 1:])
        B[:-1, :-1] |= d1
        # Anti-diagonal (↙): mark lower-left pixel
        d2 = (L[1:, :-1] != L[:-1, 1:])
        B[1:, :-1] |= d2

    return B


def _neighbor_degree(mask: np.ndarray) -> np.ndarray:
    """8-neighborhood degree (number of neighbors) for each True pixel."""
    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    # correlate is slightly better semantics than convolve for binary masks
    return ndi.correlate(mask.astype(np.uint8), k, mode='constant', cval=0)


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components (8‑conn) smaller than min_size pixels."""
    if min_size <= 1:
        return mask
    lab, num = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if num == 0:
        return mask
    counts = np.bincount(lab.ravel())
    keep = np.ones_like(counts, dtype=bool)
    keep[0] = False  # background label
    keep[counts < min_size] = False
    # map keep-array back to mask
    out = mask.copy()
    out[~keep[lab]] = False
    return out


def contour_diff_rgb(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    *,
    tol_pixels: float = 2.0,
    min_component_size: int = 20,
    connectivity: int = 8,
    emphasize_nodes: bool = True,
    matched_gray: tuple[int, int, int] = (150, 150, 150),
    only_a_color: tuple[int, int, int] = (255, 80, 255),    # vivid magenta (old)
    only_b_color: tuple[int, int, int] = (135, 205, 135),   # subdued green (new)
    overlap_color: tuple[int, int, int] = (255, 255, 0),    # yellow (rare)
    node_a_color: tuple[int, int, int] = (255, 0, 255),     # bright magenta nodes
    node_b_color: tuple[int, int, int] = (0, 255, 0)        # bright green nodes
) -> np.ndarray:
    """
    Visualize contour differences between two label images with tolerance.

    Parameters
    ----------
    labels_a, labels_b : np.ndarray
        2D integer label images of identical shape. Values are region IDs.
    tol_pixels : float, default 2.0
        Tolerance (in pixels) for matching boundaries. Boundary pixels in A
        within `tol_pixels` of any boundary pixel in B (and vice versa) are
        considered "matched" and drawn in gray; others are highlighted.
    min_component_size : int, default 20
        Suppress small unmatched fragments (likely jitter) by removing 8‑connected
        components smaller than this many pixels from the unmatched sets.
    connectivity : {4, 8}, default 8
        Neighborhood used to extract contours. 8 captures diagonal contacts.
    emphasize_nodes : bool, default True
        If True, endpoints (degree==1) and junctions (degree>=3) that are
        unmatched are dilated slightly and drawn in a brighter color to
        emphasize likely bifurcations / merges / topological changes.
    *_color : RGB triples in 0..255
        Color choices for rendering.

    Returns
    -------
    rgb : (H, W, 3) np.uint8
        Visualization suitable for `plt.imshow(rgb)`.

    Color semantics
    ---------------
    - dim gray:   contours present in both (within tolerance)
    - vivid magenta: contours present only in A (beyond tolerance, “old”)
    - soft green: contours present only in B (beyond tolerance, “new”)
    - yellow:     rare pixels flagged as "only A" and "only B" simultaneously
                  (can occur in unusual geometric configurations)
    - bright magenta/green dots: unmatched endpoints/junctions (node emphasis)
    """
    if labels_a.shape != labels_b.shape:
        raise ValueError("labels_a and labels_b must have the same shape.")
    if labels_a.ndim != 2 or labels_b.ndim != 2:
        raise ValueError("labels_a and labels_b must be 2D arrays.")

    # 1) Extract thin contour sets
    C_a = _label_to_contour(labels_a, connectivity=connectivity)
    C_b = _label_to_contour(labels_b, connectivity=connectivity)

    # 2) Tolerant symmetric difference via distance transforms
    #    distance_transform_edt() returns distance to the nearest *False*
    #    So compute on the negation of the other contour.
    dt_b = ndi.distance_transform_edt(~C_b)
    dt_a = ndi.distance_transform_edt(~C_a)

    only_a = C_a & (dt_b > tol_pixels)
    only_b = C_b & (dt_a > tol_pixels)

    # 3) Suppress tiny unmatched fragments (likely jitter)
    only_a = _remove_small_components(only_a, min_component_size)
    only_b = _remove_small_components(only_b, min_component_size)

    # 4) Matched set (within tolerance) for a neutral reference overlay
    matched = (C_a & (dt_b <= tol_pixels)) | (C_b & (dt_a <= tol_pixels))

    # 5) Structural node emphasis: unmatched endpoints/junctions
    node_a = node_b = np.zeros_like(C_a, dtype=bool)
    if emphasize_nodes:
        deg_a = _neighbor_degree(C_a)
        deg_b = _neighbor_degree(C_b)
        endpoints_a = C_a & (deg_a == 1)
        endpoints_b = C_b & (deg_b == 1)
        junctions_a = C_a & (deg_a >= 3)
        junctions_b = C_b & (deg_b >= 3)

        # Emphasize only if they are not matched (true structural deltas)
        node_a = (endpoints_a | junctions_a) & (dt_b > tol_pixels)
        node_b = (endpoints_b | junctions_b) & (dt_a > tol_pixels)

        # Thicken a touch for visibility
        cross = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)
        node_a = ndi.binary_dilation(node_a, structure=cross)
        node_b = ndi.binary_dilation(node_b, structure=cross)

    # 6) Compose RGB
    H, W = labels_a.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Matched contours first (light gray baseline)
    rgb[matched] = matched_gray

    # Unmatched regions
    both = only_a & only_b  # rare, but handle explicitly
    if np.any(both):
        rgb[both] = overlap_color

    # Exclusive unmatched
    rgb[only_a & ~both] = only_a_color
    rgb[only_b & ~both] = only_b_color

    # Node emphasis on top
    rgb[node_a] = node_a_color
    rgb[node_b] = node_b_color

    return rgb
