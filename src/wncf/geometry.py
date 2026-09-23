"""Stage 2: the fit oracle (the referee).

Contract: a rigid constant-profile peg and socket. The peg's centroid is
placed on the bore's centroid, and the peg may be turned by 0, 90, 180 or 270 degrees (the paper's
four yaw classes) before a straight insertion along the axis. No tilt, no deformation and no other
rotation. Because both parts have a constant cross-section, a 2D containment test on the profiles
decides the full-depth 3D question.

For one quarter turn R, the signed margin is the largest d for which bore.buffer(-d) covers R(peg):
the smallest clearance around the peg when positive, minus the interference when negative. The
pair's margin is the best over the four turns. With the numerical tolerance EPSILON:

- compatible:   margin >= EPSILON
- ambiguous:    -EPSILON <= margin < EPSILON                          reason "boundary"
- ambiguous:    margin < -EPSILON, but the conservative raster search finds a potential shifted fit
                                                                      reason "off_centre"
- incompatible: no quarter turn fits, even with a free sideways shift

The off-centre flag excludes possible fits that depend on whether a robot may nudge the peg (the
paper uses spiral search for small position errors). The search expands the bore to avoid missing
such fits, so this flag also includes some pairs that cannot actually fit after any translation.
It is a conservative exclusion, not a certificate of a real shifted fit. Excluding these pairs
makes every compatible and incompatible label hold under both conventions.

The margin is the *tightest* gap, so it says whether the peg fits, not whether the shapes match: a
small square in a gear-shaped bore touches the gear's root circle yet rattles in the teeth. For
compatible pairs the oracle also measures the *largest* gap, the distance from the farthest point of
the bore to the peg. A matching pair has a small, even gap everywhere. Pairs whose largest gap
exceeds LOOSE are flagged loose: insertable, but not the same shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.signal import correlate
from shapely import Polygon, box
from shapely.affinity import rotate
from shapely.ops import polylabel

from wncf.parts import CLEARANCE, QUAD_SEGS

QUARTER_TURNS = (0, 90, 180, 270)
EPSILON = 0.05  # mm; numerical exclusion band, >10x the polygon-approximation error (see tests)
LOOSE = 2 * CLEARANCE  # mm; a largest gap above twice the nominal clearance counts as loose
MARGIN_CAP = 2.0  # mm; interference beyond this is reported as -MARGIN_CAP
BISECT_TOL = 1e-3  # mm
RASTER_H = 0.1  # mm; grid step of the sideways-shift search


@dataclass(frozen=True)
class PairLabel:
    label: str  # compatible | incompatible | ambiguous
    reason: str  # "" | boundary | off_centre
    margin_mm: float  # signed margin at the best quarter turn, centroids aligned; clipped at -MARGIN_CAP
    best_rot: int  # quarter turn (deg) giving margin_mm
    fit_rots: tuple[int, ...]  # quarter turns with margin >= EPSILON
    max_gap_mm: float  # compatible pairs: largest gap at the best-matching fitting turn; NaN otherwise
    loose: bool  # max_gap_mm > LOOSE


def quarter_turns(peg: Polygon) -> list[tuple[int, Polygon]]:
    return [(deg, rotate(peg, deg, origin=(0, 0))) for deg in QUARTER_TURNS]


def label_pair(peg: Polygon, bore: Polygon) -> PairLabel:
    return BoreOracle(bore).label(peg)


def label_matrix(pegs: dict[str, Polygon], bores: dict[str, Polygon]) -> dict[tuple[str, str], PairLabel]:
    out = {}
    for sid, bore in bores.items():
        oracle = BoreOracle(bore)  # per-bore geometry is reused across every peg
        for pid, peg in pegs.items():
            out[pid, sid] = oracle.label(peg)
    return out


class BoreOracle:
    def __init__(self, bore: Polygon):
        self.bore = bore
        self.outside = box(*bore.buffer(MARGIN_CAP + 1).bounds).difference(bore)
        self.within_cap = bore.buffer(MARGIN_CAP, quad_segs=QUAD_SEGS)
        shapely.prepare(self.bore)
        shapely.prepare(self.within_cap)
        self._shift = None  # lazily built raster for the sideways-shift search

    def label(self, peg: Polygon) -> PairLabel:
        turns = quarter_turns(peg)
        margins = {deg: self._margin_if_contained(rp) for deg, rp in turns}
        if all(m is None for m in margins.values()):  # no turn fits: measure the interference
            margins = {deg: self._interference_margin(rp) for deg, rp in turns}
        else:
            margins = {deg: (-math.inf if m is None else m) for deg, m in margins.items()}

        best = max(QUARTER_TURNS, key=lambda d: margins[d])  # ties go to the smaller turn
        m = margins[best]
        fit = tuple(d for d in QUARTER_TURNS if margins[d] >= EPSILON)
        if m >= EPSILON:
            rotated = dict(turns)
            gap = min(self._max_gap(rotated[d]) for d in fit)
            return PairLabel("compatible", "", m, best, fit, gap, gap > LOOSE)
        nan = math.nan
        if m >= -EPSILON:
            return PairLabel("ambiguous", "boundary", m, best, fit, nan, False)
        if self.fits_with_shift(turns):
            return PairLabel("ambiguous", "off_centre", m, best, fit, nan, False)
        return PairLabel("incompatible", "", m, best, fit, nan, False)

    def _margin_if_contained(self, rp: Polygon) -> float | None:
        # exact: bore.buffer(-d) covers rp  <=>  every point of rp is at least d from the outside
        return rp.distance(self.outside) if self.bore.contains(rp) else None

    def _interference_margin(self, rp: Polygon) -> float:
        if not self.within_cap.contains(rp):
            return -MARGIN_CAP
        lo, hi = 0.0, MARGIN_CAP  # the needed expansion lies in (lo, hi]
        while hi - lo > BISECT_TOL:
            mid = (lo + hi) / 2
            if self.bore.buffer(mid, quad_segs=QUAD_SEGS).contains(rp):
                hi = mid
            else:
                lo = mid
        return -hi

    def _max_gap(self, rp: Polygon) -> float:
        # smallest g for which rp grown by g covers the whole bore: the bore's farthest point from the peg
        hi = 1.0
        while not rp.buffer(hi, quad_segs=QUAD_SEGS).contains(self.bore):
            hi *= 2
        lo = 0.0 if hi == 1.0 else hi / 2
        while hi - lo > BISECT_TOL:
            mid = (lo + hi) / 2
            if rp.buffer(mid, quad_segs=QUAD_SEGS).contains(self.bore):
                hi = mid
            else:
                lo = mid
        return hi

    def fits_with_shift(self, turns: list[tuple[int, Polygon]]) -> bool:
        """Could a quarter turn fit after a sideways shift under the conservative raster tolerance?

        Raster search on a RASTER_H grid against the bore grown by EPSILON + RASTER_H. It has no false
        negatives for fits inside bore.buffer(EPSILON): any such placement is within one grid step of a
        tested one, and the extra RASTER_H of growth covers that step. It can flag pairs that are tight
        by up to about EPSILON + 1.7 * RASTER_H, which is acceptable for an "ambiguous" flag.
        """
        if self._shift is None:
            self._shift = _ShiftSearch(self.bore.buffer(EPSILON + RASTER_H, quad_segs=QUAD_SEGS))
        return any(self._shift.fits(rp) for _, rp in turns)


class _ShiftSearch:
    def __init__(self, region: Polygon):
        self.region = region
        self.area = region.area
        x0, y0, x1, y1 = region.bounds
        self.w, self.h = x1 - x0, y1 - y0
        self.inradius = _inradius(region)
        self.outside_mask = ~_raster(region, pad=2 * RASTER_H)

    def fits(self, rp: Polygon) -> bool:
        x0, y0, x1, y1 = rp.bounds
        # necessary conditions for any translation: cheap and exact, they skip most pairs
        if rp.area > self.area or x1 - x0 > self.w or y1 - y0 > self.h:
            return False
        if _inradius(rp) > self.inradius + 0.02:
            return False
        kernel = _raster(rp)
        if kernel.shape[0] > self.outside_mask.shape[0] or kernel.shape[1] > self.outside_mask.shape[1]:
            return False
        # count of peg grid points that land outside the region, for every grid offset
        hits = correlate(self.outside_mask.astype(np.float32), kernel.astype(np.float32), mode="valid", method="fft")
        return bool((hits < 0.5).any())


def _raster(g: Polygon, pad: float = 0.0) -> np.ndarray:
    # grid points at integer multiples of RASTER_H, so every raster shares one lattice
    x0, y0, x1, y1 = g.bounds
    xs = np.arange(math.floor((x0 - pad) / RASTER_H), math.ceil((x1 + pad) / RASTER_H) + 1) * RASTER_H
    ys = np.arange(math.floor((y0 - pad) / RASTER_H), math.ceil((y1 + pad) / RASTER_H) + 1) * RASTER_H
    X, Y = np.meshgrid(xs, ys)
    return shapely.contains_xy(g, X, Y)


def _inradius(g: Polygon) -> float:
    return polylabel(g, tolerance=0.01).distance(g.boundary)
