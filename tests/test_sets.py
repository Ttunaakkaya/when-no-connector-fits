"""Stage 5 invariants, checked on the written manifests and on a small in-memory rebuild."""

import csv
from collections import defaultdict

import pytest

from wncf import REPO_ROOT
from wncf.geometry import label_pair
from wncf.parts import FAMILIES, build_catalog
from wncf.sets import MIN_DIFFICULTY_GAP, MIN_HARD_SIM, REGIMES, build_blocks, similarity, verify
from wncf.splits import family_splits

BLOCKS = REPO_ROOT / "data" / "candidate_sets" / "blocks.csv"
PARTS = build_catalog()
SPLITS = family_splits(FAMILIES)


def read(path):
    if not path.exists():
        pytest.skip(f"{path.name} not built; run scripts/build_sets.py")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def blocks():
    rows = read(BLOCKS)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["block_id"]].append(r)
    for rs in grouped.values():
        rs.sort(key=lambda r: int(r["position"]))
    return grouped


@pytest.fixture(scope="module")
def labels():
    return {(r["peg_id"], r["socket_id"]): r["label"]
            for r in read(REPO_ROOT / "data" / "parts" / "compat.csv")}


def test_every_block_has_three_distinct_candidates_and_the_right_mate_count(blocks):
    for bid, rs in blocks.items():
        assert len(rs) == 3, bid
        assert len({r["cand_id"] for r in rs}) == 3, bid
        mates = [r for r in rs if r["label"] == "compatible"]
        assert len(mates) == (1 if "present" in bid else 0), bid
        assert sum(int(r["is_mate"]) for r in rs) == len(mates), bid


def test_each_peg_has_all_four_regimes_sharing_one_candidate(blocks):
    by_peg = defaultdict(dict)
    for bid, rs in blocks.items():
        by_peg[rs[0]["peg_id"]][rs[0]["regime"]] = rs
    for peg, regimes in by_peg.items():
        assert set(regimes) == set(REGIMES), peg
        shared = set.intersection(*({r["cand_id"] for r in rs} for rs in regimes.values()))
        assert len(shared) == 1, f"{peg}: blocks should share exactly E2, shared={shared}"
        roles = {rs[0]["regime"]: {r["role"] for r in rs} for rs in regimes.values()}
        assert roles["present_easy"] == {"M", "E1", "E2"}
        assert roles["present_hard"] == {"M", "H", "E2"}
        assert roles["absent_easy"] == {"E3", "E1", "E2"}
        assert roles["absent_hard"] == {"E3", "H", "E2"}


def test_candidates_stay_in_the_pegs_partition_and_exclude_ambiguous_or_loose(blocks, labels):
    split_of = {p.part_id: SPLITS[p.family] for p in PARTS}
    for bid, rs in blocks.items():
        for r in rs:
            assert r["split"] == split_of[r["peg_id"]]
            if r["kind"] == "catalog":
                assert split_of[r["cand_id"]] == r["split"], f"{bid}: {r['cand_id']} outside the partition"
                assert labels[r["peg_id"], r["cand_id"]] in ("compatible", "incompatible"), bid
            else:
                assert r["cand_id"].startswith(r["peg_id"]), f"{bid}: constructed candidate not from this peg"
            assert r["label"] in ("compatible", "incompatible")


def test_difficulty_tiers_separate_for_every_peg(blocks):
    for bid, rs in blocks.items():
        if not bid.endswith("hard"):
            continue
        hard = next(r for r in rs if r["role"] == "H")
        assert float(hard["sim_to_mate"]) >= MIN_HARD_SIM, bid
        assert hard["label"] == "incompatible", bid
    easy_max = defaultdict(float)
    hard_sim = {}
    for bid, rs in blocks.items():
        for r in rs:
            if r["role"].startswith("E"):
                easy_max[r["peg_id"]] = max(easy_max[r["peg_id"]], float(r["sim_to_mate"]))
            elif r["role"] == "H":
                hard_sim[r["peg_id"]] = float(r["sim_to_mate"])
    for peg, sim in hard_sim.items():
        assert sim - easy_max[peg] >= MIN_DIFFICULTY_GAP, f"{peg}: tiers overlap"


def test_manifest_labels_match_the_live_oracle(blocks):
    pegs = {p.part_id: p for p in PARTS}
    cands = read(REPO_ROOT / "data" / "candidate_sets" / "candidates.csv")
    from shapely import from_wkt

    bore = {r["cand_id"]: from_wkt(r["bore_wkt"]) for r in cands}
    checked = 0
    for bid in sorted(blocks)[::17]:  # spread over the manifest, keep the test quick
        for r in blocks[bid]:
            live = label_pair(pegs[r["peg_id"]].peg, bore[r["cand_id"]])
            assert live.label == r["label"], f"{bid}/{r['cand_id']}"
            assert live.margin_mm == pytest.approx(float(r["margin_mm"]), abs=1e-4)
            checked += 1
    assert checked > 10


def test_similarity_is_rotation_aware_and_scale_sensitive():
    from shapely import box
    from shapely.affinity import rotate, scale

    a = box(-10, -5, 10, 5)
    assert similarity(a, rotate(a, 90, origin=(0, 0))) == pytest.approx(1.0)  # a quarter turn is free
    assert similarity(a, rotate(a, 30, origin=(0, 0))) < 0.8  # other angles are not
    assert similarity(a, scale(a, 0.5, 0.5, origin=(0, 0))) == pytest.approx(0.25, abs=0.01)  # size counts


def test_build_is_deterministic_on_the_dev_split(labels):
    dev = [p for p in PARTS if SPLITS[p.family] == "dev"]
    lab = {k: (v, 0.0) for k, v in labels.items()}
    a, log_a = build_blocks(dev, SPLITS, lab)
    b, _ = build_blocks(dev, SPLITS, lab)
    assert [x.block_id for x in a] == [x.block_id for x in b]
    assert [[c.cand_id for c in x.candidates] for x in a] == [[c.cand_id for c in x.candidates] for x in b]
    for block in a:
        verify(block)
    assert {r["peg_id"] for r in log_a} == {p.part_id for p in dev}
