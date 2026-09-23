"""Score cache identity and the scoring driver's guards; no model is loaded."""

import importlib.util

import pytest

from wncf import REPO_ROOT
from wncf.cache import ScoreCache, cache_key, frozen_identity


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_key_changes_with_every_part_of_the_identity():
    base = {"model_id": "m", "revision": "r", "quant": "nf4", "grouping": "flat", "chat": "qwen_1_5",
            "prompt_sha": "p", "versions": {"torch": "1"}, "image_sha": ["a", "b", "c", "d"]}
    k = cache_key(base)
    assert cache_key(dict(base)) == k  # stable
    for field, value in [("revision", "r2"), ("quant", "int8"), ("grouping", "nested"), ("prompt_sha", "q"),
                         ("versions", {"torch": "2"}), ("image_sha", ["a", "b", "d", "c"])]:
        assert cache_key({**base, field: value}) != k, field


def test_frozen_identity_matches_the_protocol():
    ident = frozen_identity()
    assert ident["revision"] == "0d50680527681998e456c7b78950205bedd8a068"
    assert (ident["quant"], ident["grouping"], ident["chat"], ident["prompt_id"]) == ("nf4", "flat", "qwen_1_5", "controlled")


def test_cache_appends_reloads_and_ignores_duplicates(tmp_path):
    path = tmp_path / "scores.jsonl"
    c = ScoreCache(path)
    c.add("k1", {"p_yes": 0.7})
    c.add("k1", {"p_yes": 0.1})  # a second write for the same key is ignored
    c.add("k2", {"p_yes": 0.4})
    reloaded = ScoreCache(path)
    assert "k1" in reloaded and reloaded.get("k1")["p_yes"] == 0.7
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_driver_scores_five_pairs_per_eligible_peg():
    score = load_script("score")
    if not score.BLOCKS.exists():
        pytest.skip("blocks not built")
    for split, pegs in (("dev", 14), ("calib", 28), ("test", 29)):
        pairs = score.pairs_for({split})
        assert len(pairs) == 5 * pegs, split
        assert all(len(p["paths"]) == 4 for p in pairs)


def test_arm_paths_swap_only_the_render_folder():
    from wncf.config import arm_image_path

    p = "data/renders/procedural/ell_01_socket_v2.png"
    assert arm_image_path(p, None) == p
    assert arm_image_path(p, "appearance") == "data/renders/arm_appearance/ell_01_socket_v2.png"
    assert arm_image_path("data/renders/blocks/ell_01_shrunk0.20_socket_v1.png", "appearance") \
        == "data/renders/arm_appearance/ell_01_shrunk0.20_socket_v1.png"


def test_driver_refuses_arms_outside_calibration(monkeypatch):
    score = load_script("score")
    for split in ("dev", "test"):
        monkeypatch.setattr("sys.argv", ["score.py", "--split", split, "--arm", "appearance"])
        with pytest.raises(SystemExit):
            score.main()


def test_driver_refuses_the_test_split_without_a_freeze(tmp_path, monkeypatch):
    score = load_script("score")
    with pytest.raises(SystemExit):
        score.check_test_allowed(False)
    monkeypatch.setattr(score, "FREEZE", tmp_path / "missing.json")
    with pytest.raises(SystemExit):
        score.check_test_allowed(True)
    stale = tmp_path / "freeze.json"
    stale.write_text('{"blocks_sha256": "not-the-current-manifest"}', encoding="utf-8")
    monkeypatch.setattr(score, "FREEZE", stale)
    with pytest.raises(SystemExit):
        score.check_test_allowed(True)
