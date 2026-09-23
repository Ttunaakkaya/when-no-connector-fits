"""Validate experiment records before applying frozen thresholds or scoring test images.

The historical records remain immutable. Offline replay checks their saved identities; live
scoring additionally compares the complete configuration against the installed runtime.
"""

from __future__ import annotations

import json
import math
import tomllib
from importlib.metadata import version
from pathlib import Path

from wncf import REPO_ROOT
from wncf.cache import file_sha, frozen_identity
from wncf.config import FROZEN, arm_config

BLOCKS = REPO_ROOT / "data/candidate_sets/blocks.csv"
PRIMARY_FREEZE = REPO_ROOT / "results/freeze/freeze_record.json"
TEST_PASS = REPO_ROOT / "results/test/test_pass_record.json"


class ProvenanceError(ValueError):
    """A run does not match the record that is supposed to authorize it."""


def read_record(path: Path) -> dict:
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProvenanceError(f"cannot read experiment record {path}: {error}") from error
    if not isinstance(record, dict):
        raise ProvenanceError("experiment record must be an object")
    return record


def _unit(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ProvenanceError(f"invalid calibration {name}: expected a finite value in [0, 1]")


def validate_record(record: dict, *, arm: str | None = None, blocks: Path = BLOCKS,
                    live: bool = True) -> dict:
    config = record.get("config")
    if not isinstance(config, dict):
        raise ProvenanceError("missing frozen configuration")
    identity_fields = ("model_id", "revision", "quant", "grouping", "chat", "prompt_id", "prompt_sha", "views", "versions")
    if any(key not in config for key in identity_fields):
        raise ProvenanceError("incomplete frozen configuration")
    if live:
        expected = {**arm_config(arm), **frozen_identity(arm), "views": list(FROZEN["views"])}
        for key, value in expected.items():
            if config.get(key) != value:
                raise ProvenanceError(f"frozen configuration mismatch: {key}")
        lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
        for name in ("torchvision", "numpy", "pillow"):
            pinned = {package["version"] for package in lock["package"] if package["name"] == name}
            if version(name) not in pinned:
                raise ProvenanceError(f"frozen configuration mismatch: installed {name} differs from uv.lock")
    if arm is not None and record.get("arm") != arm:
        raise ProvenanceError("calibration arm does not match the requested arm")
    paths = {"candidates_sha256": REPO_ROOT / "data/candidate_sets/candidates.csv",
             "compat_sha256": REPO_ROOT / "data/parts/compat.csv"}
    if arm is None:
        paths["blocks_sha256"] = blocks
    for key, path in paths.items():
        if record.get(key) != file_sha(path):
            raise ProvenanceError(f"frozen manifest mismatch: {key}")
    cal = record.get("calibration")
    if not isinstance(cal, dict) or cal.get("split") != "calib":
        raise ProvenanceError("missing calibration on the calibration split")
    n = cal.get("n_queries")
    families = cal.get("families")
    if (not isinstance(n, int) or isinstance(n, bool) or n <= 0 or not isinstance(families, dict)
            or not families or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in families.values())
            or sum(families.values()) != n):
        raise ProvenanceError("invalid calibration family/query counts")
    _unit(cal.get("coverage_target_present"), "coverage target")
    for method, keys in (("S1", ("tau",)), ("S2", ("tau", "delta"))):
        if not isinstance(cal.get(method), dict):
            raise ProvenanceError(f"missing calibration for {method}")
        for key in keys:
            _unit(cal[method].get(key), f"{method}.{key}")
    if cal.get("one_policy_across_tiers") is not True:
        raise ProvenanceError("calibration must freeze one policy across difficulty tiers")
    return record


def load_calibration(path: Path = PRIMARY_FREEZE, *, arm: str | None = None,
                     blocks: Path = BLOCKS, live: bool = True) -> dict:
    if arm is not None:
        # Historical arm records predate a blocks_sha field; their block membership is
        # certified by the primary record, and their own geometry hashes are checked below.
        validate_record(read_record(PRIMARY_FREEZE), blocks=blocks, live=live)
    return validate_record(read_record(path), arm=arm, blocks=blocks, live=live)


def thresholds(record: dict) -> dict:
    cal = record["calibration"]
    return {"B0": {}, "B1": {}, "U0": {}, "S1": {"tau": cal["S1"]["tau"]},
            "S2": {"tau": cal["S2"]["tau"], "delta": cal["S2"]["delta"]}}


def validate_test_pass(path: Path = TEST_PASS, *, live: bool = True) -> dict:
    primary = load_calibration(live=live)
    record = read_record(path)
    if record.get("freeze_record_sha256") != file_sha(PRIMARY_FREEZE) or record.get("blocks_sha256") != file_sha(BLOCKS):
        raise ProvenanceError("test-pass record does not match the frozen experiment")
    identity = record.get("identity", {})
    required = {"model_id", "revision", "quant", "grouping", "chat", "prompt_id", "prompt_sha", "views", "versions"}
    if (not isinstance(identity, dict) or not required <= identity.keys()
            or any(primary["config"].get(key) != value for key, value in identity.items())):
        raise ProvenanceError("test-pass configuration does not match the frozen experiment")
    if record.get("thresholds") != thresholds(primary):
        raise ProvenanceError("test-pass thresholds do not match the frozen experiment")
    return record
