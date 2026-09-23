"""Build evaluation queries (one per candidate block) from the blocks manifest and the score cache."""

from __future__ import annotations

import csv
from collections import defaultdict

from wncf import REPO_ROOT
from wncf.cache import ScoreCache, frozen_identity, pair_key
from wncf.config import arm_image_path
from wncf.metrics import Query
from wncf.rules import CandidateScore

BLOCKS = REPO_ROOT / "data" / "candidate_sets" / "blocks.csv"


class MissingScores(RuntimeError):
    pass


def load_queries(splits: set[str], arm: str | None = None, *, snapshot: bool = False) -> list[Query]:
    """Blocks in `splits`, each candidate's score looked up under the frozen identity (or one arm's)."""
    if snapshot:
        from wncf.snapshot import pair_scores
        from wncf.provenance import ProvenanceError

        saved_pairs, saved_manifest = pair_scores(arm)
    else:
        from wncf.snapshot import verify_snapshot
        from wncf.provenance import ProvenanceError

        saved_manifest = verify_snapshot()
        cache, base = ScoreCache(), frozen_identity(arm)
    blocks = defaultdict(list)
    with BLOCKS.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] in splits:
                blocks[r["block_id"]].append(r)
    queries, missing = [], set()
    for bid, rows in blocks.items():
        rows.sort(key=lambda r: int(r["position"]))
        scores = []
        for r in rows:
            paths = [arm_image_path(r[k], arm) for k in ("peg_v1", "peg_v2", "socket_v1", "socket_v2")]
            if snapshot:
                row = saved_pairs.get((r["peg_id"], r["cand_id"]))
                if row is not None:
                    expected_images = [saved_manifest["images"].get(path) for path in paths]
                    if row["image_sha"] != expected_images or row["split"] != r["split"]:
                        raise ProvenanceError(f"snapshot images/split do not match block {bid}")
            else:
                key, hashes = pair_key(base, paths)
                if hashes != [saved_manifest["images"].get(path) for path in paths]:
                    raise ProvenanceError(f"live images do not match the frozen benchmark for block {bid}")
                row = cache.get(key)
            if row is None:
                missing.add((r["peg_id"], r["cand_id"]))
                continue
            scores.append(CandidateScore(r["cand_id"], row["answer"], row["top_prob"], row["p_yes"],
                                         row["mass_yes"], row["mass_no"]))
        if len(scores) == len(rows):
            r0 = rows[0]
            queries.append(Query(bid, r0["peg_id"], r0["family"], r0["split"], r0["regime"],
                                 {r["cand_id"]: r["label"] for r in rows}, tuple(scores)))
    if missing:
        source = "the verified snapshot" if snapshot else "the live cache; run scripts/score.py first"
        raise MissingScores(f"{len(missing)} peg-candidate pairs are missing from {source}")
    return queries
