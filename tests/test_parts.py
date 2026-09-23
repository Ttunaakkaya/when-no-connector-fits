"""Stage 1 invariants: every profile is valid, centred and fits the fixed socket block; meshes are
closed solids whose volumes match their profiles; the catalog is deterministic."""

import pytest
from shapely.affinity import scale

from wncf.parts import (BLOCK, CAVITY_DEPTH, FLOOR, MAX_EXTENT, MIN_WALL, PEG_LENGTH, block_outline,
                        build_catalog, peg_mesh, socket_mesh, wkt)

PARTS = build_catalog()


def test_catalog_size_and_unique_ids():
    assert len(PARTS) >= 60
    assert len({p.family for p in PARTS}) >= 15
    assert len({p.part_id for p in PARTS}) == len(PARTS)


def test_catalog_is_deterministic():
    assert [wkt(p.peg) for p in build_catalog()] == [wkt(p.peg) for p in PARTS]


@pytest.mark.parametrize("part", PARTS, ids=lambda p: p.part_id)
def test_profile_valid_centred_and_within_block(part):
    for g in (part.peg, part.bore):
        assert g.is_valid and g.geom_type == "Polygon"
        assert abs(g.centroid.x) < 1e-9 and abs(g.centroid.y) < 1e-9
    x0, y0, x1, y1 = part.peg.bounds
    assert max(x1 - x0, y1 - y0) <= MAX_EXTENT
    assert block_outline().buffer(-MIN_WALL).contains(part.bore)


@pytest.mark.parametrize("part", PARTS, ids=lambda p: p.part_id)
def test_nominal_pair_fits_with_centroids_aligned(part):
    # Sanity only; the real fit labels come from the stage-2 oracle.
    assert part.bore.contains(part.peg)


@pytest.mark.parametrize("part", PARTS, ids=lambda p: p.part_id)
def test_meshes_are_closed_and_match_profiles(part):
    peg = peg_mesh(part.peg)
    assert peg.is_watertight
    assert peg.volume == pytest.approx(part.peg.area * PEG_LENGTH, rel=1e-6)

    sock = socket_mesh(part.bore)
    assert sock.is_watertight
    expected = BLOCK**2 * FLOOR + (BLOCK**2 - part.bore.area) * CAVITY_DEPTH
    assert sock.volume == pytest.approx(expected, rel=1e-6)
    assert sock.bounds[1][2] == pytest.approx(FLOOR + CAVITY_DEPTH)


def test_mirrored_variants_differ_from_their_originals():
    by_key = {}
    for p in PARTS:
        key = (p.family, tuple(sorted((k, v) for k, v in p.params.items() if k != "mirror")))
        by_key.setdefault(key, []).append(p)
    pairs = [ps for ps in by_key.values() if len(ps) == 2]
    assert len(pairs) >= 5
    for a, b in pairs:
        assert a.peg.symmetric_difference(b.peg).area > 1.0
        assert a.peg.symmetric_difference(scale(b.peg, -1, 1, origin=(0, 0))).area < 1e-6
