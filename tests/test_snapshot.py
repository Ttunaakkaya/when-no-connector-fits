"""Offline analysis must not need regenerated images, an installed model, or mutable cache."""

import pytest

from wncf.queries import load_queries


def test_snapshot_replay_needs_no_live_identity_or_image_hashing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline replay must not resolve the live model/image identity")

    monkeypatch.setattr("wncf.queries.frozen_identity", forbidden)
    monkeypatch.setattr("wncf.queries.pair_key", forbidden)
    queries = load_queries({"test"}, snapshot=True)
    assert len(queries) == 116
    assert len({query.family for query in queries}) == 7


@pytest.mark.parametrize("arm", [None, "appearance", "prompt", "int8"])
def test_snapshot_scores_equal_live_scores_for_calibration(arm):
    from wncf import REPO_ROOT
    from wncf.config import arm_image_path

    if not (REPO_ROOT / arm_image_path("data/renders/procedural/ell_01_peg_v1.png", arm)).exists():
        pytest.skip("live render comparison needs generated images; offline replay is tested separately")
    assert load_queries({"calib"}, arm, snapshot=True) == load_queries({"calib"}, arm)


def test_modified_snapshot_manifest_is_rejected(monkeypatch):
    import wncf.snapshot as snapshot
    from wncf.provenance import ProvenanceError

    actual = snapshot.read_record(snapshot.MANIFEST)
    actual["files"]["data/candidate_sets/blocks.csv"] = "wrong"
    monkeypatch.setattr(snapshot, "read_record", lambda _: actual)
    with pytest.raises(ProvenanceError, match="snapshot source mismatch"):
        snapshot.verify_snapshot()


def test_live_evaluation_refuses_changed_images_even_with_an_existing_cache_key(monkeypatch):
    from wncf.provenance import ProvenanceError

    monkeypatch.setattr("wncf.queries.pair_key", lambda *args: ("existing-key", ["changed"] * 4))
    with pytest.raises(ProvenanceError, match="live images do not match"):
        load_queries({"test"})
