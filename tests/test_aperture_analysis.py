"""The corrected descriptive analysis must not silently turn censored margins into measurements."""

import importlib.util
import hashlib
import json

import pytest

from wncf import REPO_ROOT

spec = importlib.util.spec_from_file_location("analyze_aperture", REPO_ROOT / "scripts/analyze_aperture.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(offset, margin, score, label="incompatible", family="f", peg="p"):
    return {"split": "test", "family": family, "peg_id": peg, "cand_id": f"{peg}_{offset}",
            "offset_mm": str(offset), "margin_mm": str(margin), "p_yes": str(score), "label": label,
            "reason": "boundary" if label == "ambiguous" else "", "answer": "yes", "image_fits": "0"}


def test_capped_margins_are_separate_and_never_averaged_into_measured_bins():
    rows = analysis.prepare_rows([row(-1.2, -2, .8), row(-.6, -1.99, .6), row(.1, .08, .7, "compatible")], .7)
    bins = analysis.summarize_bins(rows)
    capped = next(b for b in bins if b["n_left_censored"])
    assert capped["n"] == 1 and capped["mean_margin_mm"] is None
    assert capped["observed_min_margin_mm"] is None and capped["observed_max_margin_mm"] is None
    assert rows[0]["margin_relation"] == "<=" and rows[1]["margin_relation"] == "="
    assert sum(b["n"] for b in bins) == 3
    assert rows[2]["accept_tau"] == 1  # frozen gate is inclusive


def test_crossings_remain_observed_intervals_and_censored_steps_are_unresolved():
    rows = analysis.prepare_rows([row(-1.2, -2, .9), row(-.6, -1, .6), row(-.3, -.4, .8),
                                  row(.1, .08, .7, "compatible")], .75)
    result = analysis.summarize_peg(rows)
    assert result["tau_response"] == "multiple_observed_transitions"
    assert result["n_observed_gate_transitions"] == 3
    assert result["unresolved_margin_steps"] == 1
    assert result["resolved_score_rises"] == result["resolved_score_falls"] == 1
    assert result["nonmonotonic_on_resolved_steps"]
    assert result["transitions"][0]["from_margin_relation"] == "<="
    assert result["transitions"][1]["from_margin_mm"] == -1
    assert result["transitions"][1]["to_margin_mm"] == -.4


def test_no_crossing_is_reported_explicitly_instead_of_inventing_a_boundary():
    rows = analysis.prepare_rows([row(-.3, -.4, .8), row(.1, .08, .85, "compatible")], .7)
    result = analysis.summarize_peg(rows)
    assert result["tau_response"] == "always_accepts"
    assert result["n_observed_gate_transitions"] == 0 and result["transitions"] == []
    assert not result["nonmonotonic_on_resolved_steps"]


def test_ambiguous_rows_are_excluded_from_binary_rate_and_family_counts_are_weighted_before_ratio():
    raw = [row(-.3, -.4, .9, family="large", peg=str(i)) for i in range(3)]
    raw += [row(0, 0, .9, "ambiguous", family="large", peg="a"), row(-.3, -.4, .1, family="small", peg="b")]
    rs = analysis.prepare_rows(raw, .5)
    rate = analysis.rate_summary(rs, "incompatible")
    assert rate["accepted"] == 3 and rate["n"] == 4
    assert rate["raw_rate"] == .75
    # Large family contributes .75 accepted / .75 incompatible; small contributes 0 / 1.
    assert rate["family_balanced_rate"] == pytest.approx(.75 / 1.75)
    assert rate["n_families"] == 2


def test_nonfinite_or_out_of_contract_saved_values_fail_loudly():
    with pytest.raises(ValueError, match="non-finite"):
        analysis.prepare_rows([row(-.3, -.4, float("nan"))], .7)
    with pytest.raises(ValueError, match="reporting cap"):
        analysis.prepare_rows([row(-.3, -3, .8)], .7)


def test_snapshot_replay_uses_verified_archived_image_hashes_without_png_files(tmp_path, monkeypatch):
    identity = {"model_id": "model", "revision": "revision", "quant": "nf4", "grouping": "flat",
                "chat": "qwen_1_5", "prompt_id": "controlled", "prompt_sha": "prompt-hash",
                "views": ["v1", "v2"], "versions": {"torch": "saved-version"}}
    paths = {k: f"missing-renders/{k}.png" for k in ("peg_v1", "peg_v2", "socket_v1", "socket_v2")}
    image_hashes = {path: hashlib.sha256(path.encode()).hexdigest() for path in paths.values()}
    shas = [image_hashes[path] for path in paths.values()]
    key = hashlib.sha256(json.dumps({**identity, "image_sha": shas}, sort_keys=True).encode()).hexdigest()
    snapshot = tmp_path / "scores.jsonl"
    snapshot.write_text(json.dumps({"key": key, "p_yes": .8, "answer": "yes"}) + "\n")
    verified = []

    def verify(*, images=False):
        assert not images
        verified.append(True)
        return {"images": image_hashes}

    def forbidden_hash(path):
        raise AssertionError(f"offline score identity must not open or hash PNGs: {path}")

    monkeypatch.setattr(analysis, "SNAPSHOT", snapshot)
    monkeypatch.setattr(analysis, "verify_snapshot", verify)
    monkeypatch.setattr(analysis, "sha256", forbidden_hash)
    rows = analysis.prepare_rows([{**row(-.3, -.4, .8), **paths}], .7)
    assert analysis.verify_score_identity(rows, {"config": identity}) == identity
    assert verified == [True] and rows[0]["score_snapshot_key"] == key
    rows[0]["socket_v1"] = "unrecorded.png"
    with pytest.raises(ValueError, match="missing from verified snapshot manifest"):
        analysis.verify_score_identity(rows, {"config": identity})
