"""Image-only containment control: pixels in, geometry-consistent margins out."""

import numpy as np
import pytest

import wncf  # noqa: F401  (pins the HF cache before transformers)
from wncf import REPO_ROOT
from wncf.image_control import best_margin_px, signed_margin_px, stitched_tiles
from wncf.render import MM_PER_PX

R = REPO_ROOT / "data" / "renders" / "procedural"


@pytest.fixture(scope="module")
def proc():
    from transformers import AutoProcessor

    model_id, rev = wncf.MODELS["7b"]
    try:
        return AutoProcessor.from_pretrained(model_id, revision=rev, local_files_only=True)
    except OSError:
        pytest.skip("processor not downloaded")


def tiles(proc, name):
    from PIL import Image

    path = R / name
    if not path.exists():
        pytest.skip("renders not built")
    return stitched_tiles(proc, Image.open(path).convert("RGB"))


def test_stitched_tiles_are_the_render_itself(proc):
    from PIL import Image

    raw = np.asarray(Image.open(R / "cross_01_peg_v1.png").convert("RGB"))
    assert np.array_equal(tiles(proc, "cross_01_peg_v1.png"), raw)


def test_own_socket_reads_as_the_nominal_clearance(proc):
    m, _ = best_margin_px(tiles(proc, "circle_03_peg_v1.png"), tiles(proc, "circle_03_socket_v1.png"))
    assert m * MM_PER_PX == pytest.approx(0.3, abs=0.07)


def test_smaller_socket_reads_as_the_interference(proc):
    # a 16 mm circle against a 13.6 mm bore interferes by 1.2 mm per side
    m, _ = best_margin_px(tiles(proc, "circle_03_peg_v1.png"), tiles(proc, "circle_02_socket_v1.png"))
    assert m * MM_PER_PX == pytest.approx(-1.2, abs=0.07)


def test_signed_margin_on_synthetic_masks():
    ap = np.zeros((40, 40), bool)
    ap[10:30, 10:30] = True
    inside = np.zeros_like(ap)
    inside[14:26, 14:26] = True
    # distances run between pixel centres: row 14 is 5 centres from row 9, the first non-opening row,
    # although the geometric gap between the peg's edge and the opening's edge is 4 px
    assert signed_margin_px(inside, ap) == pytest.approx(5.0)
    poking = np.zeros_like(ap)
    poking[14:33, 14:26] = True
    assert signed_margin_px(poking, ap) == pytest.approx(-3.0)  # row 32 is 3 centres past row 29
