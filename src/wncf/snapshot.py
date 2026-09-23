"""Verified replay of the committed pilot scores, independent of GPU/model/image availability."""

from __future__ import annotations


from wncf import REPO_ROOT
from wncf.cache import ScoreCache, cache_key, file_sha
from wncf.provenance import ProvenanceError, load_calibration, read_record

MANIFEST = REPO_ROOT / "results/scores/snapshot_manifest.json"
SCORES = REPO_ROOT / "results/scores/scores.jsonl"
IDENTITY_FIELDS = ("model_id", "revision", "quant", "grouping", "chat", "prompt_id", "prompt_sha", "views", "versions")


def verify_snapshot(*, images: bool = False) -> dict:
    manifest = read_record(MANIFEST)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ProvenanceError("invalid snapshot integrity manifest")
    for path, expected in manifest["files"].items():
        target = REPO_ROOT / path
        if not target.is_file() or file_sha(target) != expected:
            raise ProvenanceError(f"snapshot source mismatch: {path}")
    if images:
        for path, expected in manifest["images"].items():
            target = REPO_ROOT / path
            if not target.is_file() or file_sha(target) != expected:
                raise ProvenanceError(f"frozen image mismatch: {path}; use snapshot replay for archived scores")
    return manifest


def pair_scores(arm: str | None = None) -> tuple[dict, dict]:
    manifest = verify_snapshot()
    path = REPO_ROOT / (f"results/arms/{arm}/arm_calibration_record.json" if arm
                        else "results/freeze/freeze_record.json")
    record = load_calibration(path, arm=arm, live=False)
    identity = {key: record["config"][key] for key in IDENTITY_FIELDS}
    result = {}
    for row in ScoreCache(SCORES).rows.values():
        if row.get("arm") != arm:
            continue
        if any(row.get(key) != identity[key] for key in IDENTITY_FIELDS if key not in ("views", "versions")):
            raise ProvenanceError("snapshot score does not match its recorded configuration")
        if row["key"] != cache_key({**identity, "image_sha": row["image_sha"]}):
            raise ProvenanceError("snapshot score key does not match its recorded identity")
        pair = row["peg_id"], row["cand_id"]
        if pair in result and result[pair] != row:
            raise ProvenanceError(f"ambiguous snapshot score for {pair}")
        result[pair] = row
    return result, manifest
