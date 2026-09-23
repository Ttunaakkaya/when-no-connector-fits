"""Stage 1: procedural constant-profile pegs and sockets.

Every part is a straight extrusion of a 2D profile (a shapely polygon, in mm). The profile is the
source of truth: the fit oracle works on these polygons, and the render meshes are built from them,
so the cross-section is constant through the full insertion depth by construction.

Frames (mm, insertion axis z, profile area centroid at the origin):
- Peg: the profile extruded from z=0 to z=PEG_LENGTH. Its mating face is the bottom face (z=0);
  it is inserted along -z.
- Socket: the same BLOCK x BLOCK block for every part, so sockets differ only in their bore. A solid
  floor (z=0..FLOOR) under walls (z=FLOOR..FLOOR+CAVITY_DEPTH); the bore opens upward (+z) and is
  deeper than the peg is long. The nominal bore is the peg profile grown by CLEARANCE on every side
  (round joins), re-centred on its own centroid.

Profiles are drawn in a canonical orientation with flat edges along the x/y axes, because the
paper's yaw estimation aligns flat edges with the image axes and then allows only quarter turns.
Shapes without an obvious mirror symmetry also get a mirrored variant in the same family. Whether a
mirror image is really a different part under quarter turns is decided by the fit oracle: the
offset-notch, single-chamfer and unequal-arm L variants are truly chiral, but an equal-arm L's mirror
image is the same L turned 90 degrees, so those twins fit each other.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import trimesh
from shapely import LineString, MultiPolygon, Point, Polygon, box, to_wkt
from shapely.affinity import rotate, scale, translate
from shapely.geometry.base import BaseGeometry

CLEARANCE = 0.3  # mm per side, 0.6 mm diametral
PEG_LENGTH = 20.0
CAVITY_DEPTH = 22.0
FLOOR = 4.0
BLOCK = 40.0
MIN_WALL = 5.0
MAX_EXTENT = BLOCK - 2 * (MIN_WALL + CLEARANCE)  # largest allowed profile width or height
QUAD_SEGS = 32  # segments per quarter circle


# --- profile templates (canonical orientation; centring happens in build_catalog) ---

def circle(d):
    return Point(0, 0).buffer(d / 2, quad_segs=QUAD_SEGS)


def square(s):
    return box(-s / 2, -s / 2, s / 2, s / 2)


def rect(w, h):
    return box(-w / 2, -h / 2, w / 2, h / 2)


def obround(w, h):
    r = h / 2
    return LineString([(-w / 2 + r, 0), (w / 2 - r, 0)]).buffer(r, quad_segs=QUAD_SEGS)


def ellipse(w, h):
    return scale(circle(2), w / 2, h / 2, origin=(0, 0))


def regular(n, r):
    # flat edge at the bottom: vertices symmetric about -90 degrees
    start = -math.pi / 2 + math.pi / n
    return Polygon([(r * math.cos(start + 2 * math.pi * k / n), r * math.sin(start + 2 * math.pi * k / n)) for k in range(n)])


def d_shape(d, flat):
    return circle(d).intersection(box(-d, -d, d, d / 2 - flat))


def double_d(d, flat):
    return circle(d).intersection(box(-d, -d / 2 + flat, d, d / 2 - flat))


def cross(w, h, arm):
    return box(-w / 2, -arm / 2, w / 2, arm / 2).union(box(-arm / 2, -h / 2, arm / 2, h / 2))


def tee(w, h, t):
    return box(-w / 2, h / 2 - t, w / 2, h / 2).union(box(-t / 2, -h / 2, t / 2, h / 2))


def ell(w, h, t):
    return box(-w / 2, -h / 2, -w / 2 + t, h / 2).union(box(-w / 2, -h / 2, w / 2, -h / 2 + t))


def star(n, r_out, r_in):
    pts = []
    for k in range(2 * n):
        a = math.pi / 2 + math.pi * k / n  # first point straight up
        r = r_out if k % 2 == 0 else r_in
        pts.append((r * math.cos(a), r * math.sin(a)))
    return Polygon(pts)


def keyed_circle(d, key_w, key_h, keys):
    # rectangular tabs (the key) on the peg; the socket gets matching notches
    shape = circle(d)
    for k in keys:
        tab = box(-key_w / 2, d / 2 - 1, key_w / 2, d / 2 + key_h)
        angle = {"n": 0, "w": 90, "s": 180, "e": 270}[k]
        shape = shape.union(rotate(tab, angle, origin=(0, 0)))
    return shape


def notched_rect(w, h, nw, nh, dx):
    return rect(w, h).difference(box(dx - nw / 2, h / 2 - nh, dx + nw / 2, h / 2 + 1))


def trapezoid(top, bottom, h):
    return Polygon([(-bottom / 2, -h / 2), (bottom / 2, -h / 2), (top / 2, h / 2), (-top / 2, h / 2)])


def chamfer_rect(w, h, c):
    # one chamfered corner (top right), DisplayPort-like; chiral
    return Polygon([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2 - c), (w / 2 - c, h / 2), (-w / 2, h / 2)])


def gear(n, r_tip, r_root):
    p = 2 * math.pi / n
    pts = []
    for k in range(n):
        a = math.pi / 2 + k * p  # first tooth straight up
        for r, da in ((r_root, -0.3 * p), (r_tip, -0.15 * p), (r_tip, 0.15 * p), (r_root, 0.3 * p)):
            pts.append((r * math.cos(a + da), r * math.sin(a + da)))
    return Polygon(pts)


def tube(d_out, d_in):
    # a hollow peg; its socket has an annular groove around a central pin
    return circle(d_out).difference(circle(d_in))


def _m(**kw):
    return {**kw, "mirror": True}


# family -> (template, variants). One family = one template; its variants are near neighbours
# (size, aspect, feature size, mirror image) and stay in the same data split.
FAMILIES = {
    "circle": (circle, [dict(d=d) for d in (10, 13, 16, 20, 24)]),
    "square": (square, [dict(s=s) for s in (10, 13, 16, 20)]),
    "rect": (rect, [dict(w=20, h=8), dict(w=20, h=10), dict(w=16, h=8), dict(w=24, h=12), dict(w=12, h=6)]),
    "obround": (obround, [dict(w=20, h=8), dict(w=16, h=8), dict(w=24, h=10), dict(w=14, h=6)]),
    "ellipse": (ellipse, [dict(w=20, h=12), dict(w=18, h=10), dict(w=24, h=16)]),
    "polygon": (regular, [dict(n=3, r=11), dict(n=5, r=10), dict(n=6, r=10), dict(n=6, r=8), dict(n=8, r=10)]),
    "dshape": (d_shape, [dict(d=16, flat=2), dict(d=16, flat=4), dict(d=16, flat=6), dict(d=20, flat=4)]),
    "double_d": (double_d, [dict(d=16, flat=2), dict(d=16, flat=4), dict(d=20, flat=3)]),
    "cross": (cross, [dict(w=20, h=20, arm=5), dict(w=20, h=20, arm=8), dict(w=16, h=16, arm=6), dict(w=20, h=14, arm=6)]),
    "tee": (tee, [dict(w=20, h=16, t=6), dict(w=20, h=16, t=4), dict(w=16, h=20, t=6)]),
    "ell": (ell, [dict(w=18, h=18, t=6), _m(w=18, h=18, t=6), dict(w=20, h=14, t=6), _m(w=20, h=14, t=6),
                  dict(w=18, h=18, t=4), _m(w=18, h=18, t=4)]),
    "star": (star, [dict(n=4, r_out=11, r_in=5.5), dict(n=5, r_out=11, r_in=5.5), dict(n=6, r_out=11, r_in=5.5),
                    dict(n=5, r_out=11, r_in=7)]),
    "keyed": (keyed_circle, [dict(d=16, key_w=3, key_h=2, keys="n"), dict(d=16, key_w=5, key_h=2, keys="n"),
                             dict(d=16, key_w=3, key_h=2, keys="ns"), dict(d=16, key_w=3, key_h=2, keys="ne"),
                             dict(d=20, key_w=4, key_h=2, keys="n")]),
    "notched": (notched_rect, [dict(w=16, h=12, nw=6, nh=3, dx=0), dict(w=16, h=12, nw=6, nh=3, dx=-4),
                               _m(w=16, h=12, nw=6, nh=3, dx=-4), dict(w=16, h=12, nw=8, nh=4, dx=0)]),
    "trapezoid": (trapezoid, [dict(top=20, bottom=14, h=8), dict(top=20, bottom=16, h=8), dict(top=16, bottom=12, h=8)]),
    "chamfered": (chamfer_rect, [dict(w=18, h=8, c=3), _m(w=18, h=8, c=3), dict(w=18, h=8, c=5), dict(w=14, h=8, c=3)]),
    "gear": (gear, [dict(n=6, r_tip=11, r_root=8.5), dict(n=8, r_tip=11, r_root=8.5), dict(n=12, r_tip=11, r_root=9)]),
    "tube": (tube, [dict(d_out=14, d_in=6), dict(d_out=14, d_in=9), dict(d_out=18, d_in=10)]),
}


# --- catalog ---

@dataclass(frozen=True)
class Part:
    part_id: str
    family: str
    params: dict
    peg: Polygon  # peg cross-section, centroid at the origin
    bore: Polygon  # nominal socket opening, centroid at the origin

    @property
    def params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True)


def centred(g: BaseGeometry) -> BaseGeometry:
    c = g.centroid
    return translate(g, -c.x, -c.y)


def nominal_bore(peg: Polygon, clearance: float = CLEARANCE) -> Polygon:
    return centred(peg.buffer(clearance, quad_segs=QUAD_SEGS, join_style="round"))


def build_catalog() -> list[Part]:
    parts = []
    for family, (template, variants) in FAMILIES.items():
        for i, params in enumerate(variants, 1):
            kw = {k: v for k, v in params.items() if k != "mirror"}
            profile = template(**kw)
            if params.get("mirror"):
                profile = scale(profile, -1, 1, origin=(0, 0))
            peg = centred(profile)
            part = Part(f"{family}_{i:02d}", family, params, peg, nominal_bore(peg))
            check_part(part)
            parts.append(part)
    return parts


def check_part(part: Part) -> None:
    for name, g in (("peg", part.peg), ("bore", part.bore)):
        if not (isinstance(g, Polygon) and g.is_valid and g.area > 0):
            raise ValueError(f"{part.part_id}: {name} is not a single valid polygon")
    x0, y0, x1, y1 = part.peg.bounds
    if max(x1 - x0, y1 - y0) > MAX_EXTENT:
        raise ValueError(f"{part.part_id}: profile {x1 - x0:.1f}x{y1 - y0:.1f} mm exceeds {MAX_EXTENT} mm")
    if not block_outline().buffer(-MIN_WALL).contains(part.bore):
        raise ValueError(f"{part.part_id}: socket wall thinner than {MIN_WALL} mm")


def wkt(g: BaseGeometry) -> str:
    return to_wkt(g, rounding_precision=6)


# --- meshes (for rendering) ---

def block_outline() -> Polygon:
    return box(-BLOCK / 2, -BLOCK / 2, BLOCK / 2, BLOCK / 2)


def _extrude(face: BaseGeometry, z0: float, z1: float) -> list[trimesh.Trimesh]:
    geoms = face.geoms if isinstance(face, MultiPolygon) else [face]
    meshes = []
    for g in geoms:
        m = trimesh.creation.extrude_polygon(g, z1 - z0)
        m.apply_translation([0, 0, z0])
        meshes.append(m)
    return meshes


def peg_mesh(peg: Polygon) -> trimesh.Trimesh:
    return trimesh.util.concatenate(_extrude(peg, 0.0, PEG_LENGTH))


def socket_mesh(bore: Polygon) -> trimesh.Trimesh:
    """Fixed block with `bore` cut to CAVITY_DEPTH. Takes the bore, not the Part, so a perturbed bore
    (stage 9) gets exactly the same exterior."""
    walls = block_outline().difference(bore)  # MultiPolygon when the bore has a central pin
    solids = _extrude(block_outline(), 0.0, FLOOR) + _extrude(walls, FLOOR, FLOOR + CAVITY_DEPTH)
    return trimesh.util.concatenate(solids)
