"""Image-only containment control.

Question: does the information needed to decide fit survive the model's preprocessing? This check
never sees the true polygons. It takes the top-view pixels exactly as the processor delivers them
under the frozen `flat` grouping (the 2x2 high-resolution tiles, stitched back together), segments
the peg and the socket opening, puts their centroids together (the task contract), and measures a
signed pixel margin at each permitted quarter turn:

  margin > 0: every peg pixel lies inside the opening, at least `margin` pixels from its edge
  margin < 0: some peg pixel lies outside, at most `-margin` pixels from the opening

The pair "fits" if the best turn's margin is positive. Distances run between pixel centres, so a
margin overstates the geometric gap or overlap by about half a pixel (0.034 mm) in magnitude; this
never changes a decision outside the oracle's ambiguous band. Segmentation thresholds are the fixed ones
from the stage-3 render checks (luminance halfway between background and part, and between the
dark cavity floor and the lit block top), chosen before this control existed. Internal holes are
kept: a tube peg's bore is background, and a socket's central pin is not part of the opening.

It is a synthetic-image diagnostic of information content, not a claim about what the model could
learn to use.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

FG_LUMA = 197  # peg pixels are darker than this on the white background (stage 3)
DARK_LUMA = 88  # cavity-floor pixels are darker than this on the lit block top (stage 3)


def stitched_tiles(processor, image) -> np.ndarray:
    """The 2x2 high-resolution tiles the processor makes under flat grouping, as one uint8 image."""
    x = processor.image_processor(images=[image], return_tensors="np")
    pv = x["pixel_values"][0]  # (patches, 3, 384, 384); patch 0 is the downscaled whole image
    if pv.shape[0] != 5:
        raise ValueError(f"expected a base image plus 2x2 tiles, got {pv.shape[0]} patches")
    tiles = [((t.transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1) * 255).round().astype(np.uint8) for t in pv[1:]]
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def luma(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32).mean(-1)


def peg_mask(img: np.ndarray) -> np.ndarray:
    return luma(img) < FG_LUMA


def aperture_mask(img: np.ndarray) -> np.ndarray:
    return luma(img) < DARK_LUMA


def _centred(mask: np.ndarray, size: int) -> np.ndarray:
    """Place `mask` in a size x size canvas with its centroid at the canvas centre (nearest pixel)."""
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    out = np.zeros((size, size), dtype=bool)
    dy, dx = int(round(size / 2 - cy)), int(round(size / 2 - cx))
    ty, tx = ys + dy, xs + dx
    keep = (ty >= 0) & (ty < size) & (tx >= 0) & (tx < size)
    out[ty[keep], tx[keep]] = True
    return out


def signed_margin_px(peg: np.ndarray, aperture: np.ndarray) -> float:
    """Signed containment margin in pixels at one fixed placement."""
    outside = peg & ~aperture
    if not outside.any():
        # distance from each peg pixel to the nearest pixel that is not opening
        return float(ndimage.distance_transform_edt(aperture)[peg].min())
    # distance from each stray peg pixel to the nearest opening pixel
    return -float(ndimage.distance_transform_edt(~aperture)[outside].max())


def best_margin_px(peg_img: np.ndarray, socket_img: np.ndarray) -> tuple[float, int]:
    """Best signed margin over the four quarter turns, centroids aligned; returns (margin, turn)."""
    size = 2 * max(peg_img.shape[:2])
    ap = _centred(aperture_mask(socket_img), size)
    pm = peg_mask(peg_img)
    best_m, best_deg = -np.inf, 0
    for k, deg in enumerate((0, 90, 180, 270)):
        m = signed_margin_px(_centred(np.rot90(pm, k), size), ap)
        if m > best_m:  # strict, so ties keep the smaller turn
            best_m, best_deg = m, deg
    return best_m, best_deg
