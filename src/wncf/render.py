"""Stage 3: render pegs and sockets with one fixed orthographic camera.

Every image in an experiment shares the same camera, field of view, pixel size, material and
lighting, so apparent size is real size: a 10 mm peg covers the same pixels in every image.

Views (the paper's two):
- v1: straight down the insertion axis, showing the peg's cross-section or the socket's opening
- v2: tilted TILT_DEG from the axis toward -y, showing the top face and the front side

The peg is shown in its insertion pose from the socket camera's viewpoint, so a matching pair has
the same outline in both images. (Photographing the peg's mating face head-on would mirror it
relative to the socket; that is invisible for symmetric shapes but would set a systematic trap for
chiral ones.) A peg is a constant-profile prism, so its top face has the cross-section's outline.

Looking straight down, the block's top face and the cavity floor face the same way and would shade
identically. The cavity floor is therefore dark, as real socket interiors appear in photographs.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pyvista as pv
import trimesh
import vtk
from shapely import Polygon

from wncf.parts import FLOOR, peg_mesh, socket_mesh

IMAGE_PX = 768  # 2x the model's 384 px base tile: the tiled path gets native pixels, the base path a clean 2x downsample
FOV_MM = 52.0  # holds the 40 mm block in the tilted view (projected height 47.6 mm)
MM_PER_PX = FOV_MM / IMAGE_PX
TILT_DEG = 30.0
VIEWS = {"v1": 0.0, "v2": TILT_DEG}
BACKGROUND = "#ffffff"
PART_COLOR = "#9aa3ad"
FLOOR_COLOR = "#2b2b2b"
MATERIAL = dict(ambient=0.25, diffuse=0.75, specular=0.15, specular_power=20)
FEATURE_ANGLE = 30.0  # smooth-shade across polygonised curves, keep prism edges sharp


def settings() -> dict:
    return {
        "image_px": IMAGE_PX, "fov_mm": FOV_MM, "mm_per_px": MM_PER_PX, "views_tilt_deg": VIEWS,
        "tilt_toward": "-y", "projection": "orthographic", "lighting": "pyvista light kit (camera-attached)",
        "background": BACKGROUND, "part_color": PART_COLOR, "floor_color": FLOOR_COLOR, "material": MATERIAL,
        "smooth_shading_feature_angle": FEATURE_ANGLE, "anti_aliasing": "ssaa",
        "peg_pose": "insertion pose, viewed from the socket camera", "pyvista": pv.__version__,
        "vtk": vtk.vtkVersion.GetVTKVersion(),
    }


def settings_sha() -> str:
    return hashlib.sha256(json.dumps(settings(), sort_keys=True).encode()).hexdigest()[:12]


def peg_scene(peg: Polygon, color: str = PART_COLOR) -> list[tuple[trimesh.Trimesh, str]]:
    return [(peg_mesh(peg), color)]


def socket_scene(bore: Polygon, color: str = PART_COLOR, floor_color: str = FLOOR_COLOR) -> list[tuple[trimesh.Trimesh, str]]:
    # a thin dark disk on the cavity floor, 0.02 mm above the floor slab so the two never z-fight
    floor = trimesh.creation.extrude_polygon(bore, 0.02)
    floor.apply_translation([0, 0, FLOOR])
    return [(socket_mesh(bore), color), (floor, floor_color)]


class Renderer:
    """One off-screen plotter reused for every image (same lights, camera model and window)."""

    def __init__(self, background: str = BACKGROUND):
        self.pl = pv.Plotter(off_screen=True, window_size=(IMAGE_PX, IMAGE_PX), lighting="light kit")
        self.pl.set_background(background)
        self.pl.enable_parallel_projection()
        self.pl.enable_anti_aliasing("ssaa")

    def render(self, scene: list[tuple[trimesh.Trimesh, str]], view: str) -> np.ndarray:
        actors = [
            self.pl.add_mesh(_to_pv(m), color=c, smooth_shading=True, split_sharp_edges=True,
                             feature_angle=FEATURE_ANGLE, **MATERIAL)
            for m, c in scene
        ]
        lo = np.min([m.bounds[0] for m, _ in scene], axis=0)
        hi = np.max([m.bounds[1] for m, _ in scene], axis=0)
        focal = (lo + hi) / 2
        focal[:2] = 0.0  # parts are centred on their profile centroid; keep that on the optical axis
        tilt = math.radians(VIEWS[view])
        direction = np.array([0.0, -math.sin(tilt), math.cos(tilt)])  # from the focal point toward the camera
        cam = self.pl.camera
        cam.focal_point = focal
        cam.position = focal + 200.0 * direction
        cam.up = (0.0, 1.0, 0.0)
        cam.parallel_scale = FOV_MM / 2
        self.pl.reset_camera_clipping_range()
        self.pl.render()  # without this the SSAA screenshot returns the previous camera's frame
        img = self.pl.screenshot(return_img=True)
        for a in actors:
            self.pl.remove_actor(a)
        return img

    def close(self):
        self.pl.close()


def _to_pv(m: trimesh.Trimesh) -> pv.PolyData:
    faces = np.hstack([np.full((len(m.faces), 1), 3), m.faces]).ravel()
    return pv.PolyData(np.asarray(m.vertices, dtype=float), faces)
