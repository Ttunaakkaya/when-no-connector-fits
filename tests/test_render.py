"""Stage 3 renderer: scale, orientation, framing and determinism. Needs an OpenGL context."""

import numpy as np
import pytest

from wncf.parts import build_catalog
from wncf.render import MM_PER_PX, Renderer, peg_scene, socket_scene

PARTS = {p.part_id: p for p in build_catalog()}
FG_LUMA, DARK_LUMA = 197, 88


@pytest.fixture(scope="module")
def r():
    renderer = Renderer()
    yield renderer
    renderer.close()


def luma(img):
    return img.astype(np.float32).mean(-1)


def test_pixel_areas_match_the_geometry(r):
    p = PARTS["square_03"]  # 16 mm square
    peg = (luma(r.render(peg_scene(p.peg), "v1")) < FG_LUMA).sum() * MM_PER_PX**2
    bore = (luma(r.render(socket_scene(p.bore), "v1")) < DARK_LUMA).sum() * MM_PER_PX**2
    assert peg == pytest.approx(p.peg.area, rel=0.01)
    assert bore == pytest.approx(p.bore.area, rel=0.01)


def test_peg_outline_lines_up_with_its_own_bore_not_its_mirror(r):
    # chiral L: the peg is drawn in its insertion pose, so it overlays its own opening
    peg = luma(r.render(peg_scene(PARTS["ell_03"].peg), "v1")) < FG_LUMA
    own = luma(r.render(socket_scene(PARTS["ell_03"].bore), "v1")) < DARK_LUMA
    twin = luma(r.render(socket_scene(PARTS["ell_04"].bore), "v1")) < DARK_LUMA
    assert (peg & own).sum() / peg.sum() > 0.99
    assert (peg & twin).sum() / peg.sum() < 0.8


def test_largest_socket_is_not_clipped_in_the_tilted_view(r):
    img = r.render(socket_scene(PARTS["circle_05"].bore), "v2")
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    assert (border >= 250).all()


def test_views_differ_and_rendering_is_deterministic(r):
    scene = socket_scene(PARTS["tube_03"].bore)
    a, b, c = r.render(scene, "v1"), r.render(scene, "v2"), r.render(scene, "v1")
    assert np.array_equal(a, c)
    assert not np.array_equal(a, b)
