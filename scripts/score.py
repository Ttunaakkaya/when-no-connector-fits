"""Score the candidate blocks under the frozen configuration, into the append-only cache.

    uv run python scripts/score.py --split calib          # calibration blocks
    uv run python scripts/score.py --split dev calib      # several splits
    uv run python scripts/score.py --split test --allow-test   # refused until thresholds are frozen

Blocks are scored per peg-candidate pair: the four blocks of a peg reuse five candidates, so a peg
costs five calls, and each pair is scored once however many blocks contain it.

The test split is the single final pass. It is refused unless --allow-test is given AND
results/freeze/freeze_record.json exists and names the current blocks manifest, i.e. thresholds were
frozen on calibration data first.

Other pair lists (for example the aperture series) use the same columns and are passed with
--manifest; the split, test guard, cache and GPU etiquette apply unchanged.

GPU etiquette: the run will not start while more than GPU_START_MAX_MIB is in use, and it stops
(exit code 3, cache intact) if *other* processes' use grows by more than GPU_YIELD_OTHERS_MIB over
what it was at the start, meaning someone else has started. Its own allocation is subtracted, so a
large configuration such as int8 never mistakes itself for a competing job. Rerunning resumes from
the cache.
"""

import argparse
import csv
import json
import subprocess
import sys
import time

from PIL import Image

from wncf import REPO_ROOT
from wncf.cache import ScoreCache, file_sha, frozen_identity, pair_key
from wncf.config import ARM_OVERRIDES, ARM_SPLITS, arm_config, arm_image_path

BLOCKS = REPO_ROOT / "data" / "candidate_sets" / "blocks.csv"
FREEZE = REPO_ROOT / "results" / "freeze" / "freeze_record.json"
TEST_PASS = REPO_ROOT / "results" / "test" / "test_pass_record.json"
GPU_START_MAX_MIB = 4000
GPU_YIELD_OTHERS_MIB = 3000
CUDA_CONTEXT_MIB = 700  # our process's memory outside PyTorch's allocator (CUDA context, cuBLAS)
EXIT_GPU_BUSY = 3


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def pairs_for(splits: set[str], arm: str | None = None, manifest=BLOCKS) -> list[dict]:
    """Unique peg-candidate pairs in the requested splits, with image paths in prompt order."""
    seen = {}
    with manifest.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] in splits:
                seen.setdefault((r["peg_id"], r["cand_id"]), {
                    "split": r["split"], "peg_id": r["peg_id"], "cand_id": r["cand_id"],
                    "paths": [arm_image_path(r[k], arm) for k in ("peg_v1", "peg_v2", "socket_v1", "socket_v2")],
                })
    return list(seen.values())


def check_test_allowed(allow: bool) -> None:
    if not allow:
        sys.exit("refusing to score the test split without --allow-test")
    if not FREEZE.exists():
        sys.exit(f"refusing to score test: no freeze record at {FREEZE}")
    record = json.loads(FREEZE.read_text(encoding="utf-8"))
    if record.get("blocks_sha256") != file_sha(BLOCKS):
        sys.exit("refusing to score test: the freeze record was made for a different blocks manifest")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", nargs="+", required=True, choices=["dev", "calib", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--arm", choices=sorted(ARM_OVERRIDES), help="a predeclared sensitivity arm (calibration only)")
    ap.add_argument("--exploratory", action="store_true",
                    help="allow an arm outside calibration, only after the primary test pass; results are exploratory")
    ap.add_argument("--manifest", default=str(BLOCKS.relative_to(REPO_ROOT)),
                    help="pair list with the blocks.csv columns (default: the candidate blocks)")
    args = ap.parse_args()
    manifest = REPO_ROOT / args.manifest
    if args.arm and not set(args.split) <= ARM_SPLITS:
        if not args.exploratory:
            sys.exit(f"sensitivity arms run on {sorted(ARM_SPLITS)} only (use --exploratory after the test pass)")
        if not TEST_PASS.exists():
            sys.exit("exploratory arm runs outside calibration are allowed only after the primary test pass")
    if "test" in args.split:
        check_test_allowed(args.allow_test)

    cache = ScoreCache()
    cfg = arm_config(args.arm)
    base = frozen_identity(args.arm)
    todo = []
    pairs = pairs_for(set(args.split), args.arm, manifest)
    for p in pairs:
        p["key"], p["image_sha"] = pair_key(base, p["paths"])
        if p["key"] not in cache:
            todo.append(p)
    label = "+".join(args.split) + (f" ({args.arm} arm)" if args.arm else "") + \
        (f" from {args.manifest}" if manifest != BLOCKS else "")
    print(f"{len(pairs)} pairs in {label}: {len(pairs) - len(todo)} cached, {len(todo)} to score")
    if not todo:
        return

    baseline = gpu_used_mib()  # everything else on the GPU before we load (display, other apps)
    if baseline > GPU_START_MAX_MIB:
        print(f"GPU busy ({baseline} MiB in use); not starting")
        sys.exit(EXIT_GPU_BUSY)

    import torch

    from wncf.scorer_llava import LlavaScorer

    scorer = LlavaScorer(model=cfg["model"], quant=cfg["quant"], grouping=cfg["grouping"], chat=cfg["chat"])
    t0 = time.perf_counter()
    for i, p in enumerate(todo, 1):
        if i % 10 == 0:
            ours = torch.cuda.memory_reserved() // 2**20 + CUDA_CONTEXT_MIB
            others = gpu_used_mib() - ours
            if others > baseline + GPU_YIELD_OTHERS_MIB:
                print(f"another job is using the GPU ({others} MiB besides ours); stopping after {i - 1} calls "
                      f"(cache intact)")
                sys.exit(EXIT_GPU_BUSY)
        s = scorer.score([Image.open(REPO_ROOT / path).convert("RGB") for path in p["paths"]], cfg["prompt"])
        cache.add(p["key"], {
            "split": p["split"], "arm": args.arm, "exploratory": bool(args.exploratory),
            "peg_id": p["peg_id"], "cand_id": p["cand_id"],
            "image_sha": p["image_sha"],
            **{k: base[k] for k in ("model_id", "revision", "quant", "grouping", "chat", "prompt_id", "prompt_sha")},
            **s.as_dict(), "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        if i % 20 == 0 or i == len(todo):
            rate = (time.perf_counter() - t0) / i
            print(f"  {i}/{len(todo)} scored, {rate:.1f} s/call, ~{rate * (len(todo) - i) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
