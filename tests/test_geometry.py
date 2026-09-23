"""Stage 2 oracle on cases whose answer is known in advance."""

import pytest
from shapely import box
from shapely.affinity import rotate, scale

from wncf.geometry import EPSILON, LOOSE, label_pair
from wncf.parts import CLEARANCE, build_catalog, centred, circle, ell, gear, nominal_bore, rect, square, tube


def test_exact_fit_has_the_nominal_clearance_in_every_turn():
    lab = label_pair(square(10), nominal_bore(square(10)))
    assert lab.label == "compatible" and not lab.loose
    assert lab.margin_mm == pytest.approx(CLEARANCE, abs=1e-3)  # rounded bore corners are polygonised
    assert lab.fit_rots == (0, 90, 180, 270)


def test_collision_is_incompatible_with_the_interference_as_margin():
    lab = label_pair(square(10), square(9.5))
    assert lab.label == "incompatible"
    # 0.25 mm per side, but the worst point is the corner, which pokes out diagonally
    assert lab.margin_mm == pytest.approx(-0.25 * 2**0.5, abs=2e-3)


def test_quarter_turn_is_allowed():
    lab = label_pair(rect(20, 8), nominal_bore(rect(8, 20)))
    assert lab.label == "compatible"
    assert lab.fit_rots == (90, 270)


def test_non_quarter_turn_is_not_allowed():
    # same square, but the hole is turned 45 degrees: the paper's yaw classes cannot align them
    lab = label_pair(square(10), rotate(nominal_bore(square(10)), 45, origin=(0, 0)))
    assert lab.label == "incompatible"


def test_mirror_image_fits_only_when_it_is_also_a_rotation():
    sym, asym = centred(ell(18, 18, 6)), centred(ell(20, 14, 6))
    mirror = lambda g: scale(g, -1, 1, origin=(0, 0))  # noqa: E731
    # an L with equal arms is its own mirror image after a quarter turn
    assert label_pair(sym, nominal_bore(mirror(sym))).label == "compatible"
    # with unequal arms it is truly chiral
    assert label_pair(asym, nominal_bore(mirror(asym))).label == "incompatible"


def test_loose_fit_is_compatible_and_flagged():
    lab = label_pair(circle(10), circle(13.6))
    assert lab.label == "compatible" and lab.loose
    assert lab.margin_mm == pytest.approx(1.8, abs=5e-3)
    assert lab.max_gap_mm == pytest.approx(1.8, abs=5e-3)


def test_tight_somewhere_but_wrong_shape_is_loose():
    # a square touches a gear bore's root circle (small margin) but leaves the teeth empty (large gap)
    lab = label_pair(square(13), nominal_bore(gear(6, 11, 8.5)))
    assert lab.label == "compatible"
    assert lab.margin_mm < LOOSE < lab.max_gap_mm and lab.loose


def test_knife_edge_is_ambiguous_on_both_sides():
    tight = label_pair(square(10), square(10.04))  # +0.02 mm per side
    assert (tight.label, tight.reason) == ("ambiguous", "boundary")
    pinch = label_pair(square(10), square(9.96))  # -0.02 mm per side
    assert (pinch.label, pinch.reason) == ("ambiguous", "boundary")


def test_fit_that_needs_a_sideways_shift_is_ambiguous():
    # an L's centroid lies in its inner corner, outside the L; a 12x6 bar fits along one arm only off-centre
    lab = label_pair(rect(12, 6), nominal_bore(centred(ell(18, 18, 6))))
    assert (lab.label, lab.reason) == ("ambiguous", "off_centre")
    assert lab.margin_mm < -EPSILON


def test_socket_pin_blocks_a_solid_peg():
    tube_bore = nominal_bore(tube(14, 6))
    assert label_pair(circle(14), tube_bore).label == "incompatible"
    assert label_pair(tube(14, 6), tube_bore).label == "compatible"


def test_polygon_approximation_error_is_far_below_epsilon():
    # the peg circle and the buffered bore are both polygons; the analytic margin is exactly CLEARANCE
    for d in (10, 16, 24):
        err = abs(label_pair(circle(d), nominal_bore(circle(d))).margin_mm - CLEARANCE)
        assert err < EPSILON / 10


def test_every_nominal_pair_fits_snugly():
    for p in build_catalog():
        lab = label_pair(p.peg, p.bore)
        assert lab.label == "compatible" and not lab.loose, p.part_id
        assert EPSILON <= lab.margin_mm <= CLEARANCE + 1e-6, p.part_id
        assert CLEARANCE - 1e-3 <= lab.max_gap_mm <= LOOSE, p.part_id


def test_growing_the_bore_never_lowers_the_margin():
    peg = centred(ell(20, 14, 6))
    margins = [label_pair(peg, nominal_bore(peg, c)).margin_mm for c in (0.1, 0.2, 0.4, 0.8)]
    assert margins == sorted(margins)


def test_a_big_peg_never_fits_a_small_hole_anywhere():
    lab = label_pair(box(-10, -10, 10, 10), circle(10))
    assert lab.label == "incompatible" and lab.margin_mm == pytest.approx(-2.0)
