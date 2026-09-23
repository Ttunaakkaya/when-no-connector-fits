"""Stage 5: build the matched candidate blocks, verify them and record the manifests.

    uv run python scripts/build_sets.py

Writes, under data/candidate_sets/:
  blocks.csv       one row per block-candidate, in presentation order, with image paths
  candidates.csv   every distinct candidate opening (WKT, kind, image paths and hashes)
  eligibility.csv  one row per peg: eligible or not, with the reason and its difficulty numbers
  blocks_meta.json seeds, declared rule constants, counts, exclusions and input hashes
and results/figures/stage5_difficulty.png plus results/figures/stage5_example_blocks.png.

Nothing here calls the model. Selection is geometric and deterministic; rerunning reproduces it.
"""

import csv
import hashlib
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from shapely import from_wkt

from wncf import REPO_ROOT
from wncf.geometry import label_pair
from wncf.parts import FAMILIES, build_catalog, wkt
from wncf.render import VIEWS, Renderer, socket_scene
from wncf.sets import (CANDIDATE_SEED, EASY_MAX_PAIR_SIM, HARD_MAX_MARGIN, MIN_DIFFICULTY_GAP, MIN_HARD_SIM, REGIMES,
                       SHRINK_LADDER, build_blocks, verify)
from wncf.splits import SPLIT_SEED, family_splits

OUT = REPO_ROOT / "data" / "candidate_sets"
RENDERS = REPO_ROOT / "data" / "renders"
FIGS = REPO_ROOT / "results" / "figures"


def load_labels():
    path = REPO_ROOT / "data" / "parts" / "compat.csv"
    with path.open(newline="", encoding="utf-8") as f:
        labels = {(r["peg_id"], r["socket_id"]): (r["label"], float(r["margin_mm"])) for r in csv.DictReader(f)}
    return labels, hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    parts = build_catalog()
    splits = family_splits(FAMILIES)
    labels, compat_sha = load_labels()

    blocks, log = build_blocks(parts, splits, labels)
    peg_bores = {p.part_id: p.bore for p in parts}
    pegs = {p.part_id: p for p in parts}

    # independent re-verification: recompute every candidate label with the live oracle
    mismatches = []
    for b in blocks:
        verify(b)
        for c in b.candidates:
            live = label_pair(pegs[b.peg_id].peg, c.bore)
            # compat.csv stores margins rounded to 4 decimals, so allow half of that last digit
            if live.label != c.label or abs(live.margin_mm - c.margin_mm) > 1e-4:
                mismatches.append((b.block_id, c.cand_id, c.label, live.label))
    if mismatches:
        raise AssertionError(f"oracle re-verification failed for {len(mismatches)}: {mismatches[:5]}")

    # render the constructed look-alikes in the frozen neutral style
    constructed = {c.cand_id: c for b in blocks for c in b.candidates if c.kind == "shrunk"}
    out_dir = RENDERS / "blocks"
    out_dir.mkdir(parents=True, exist_ok=True)
    if constructed:
        r = Renderer()
        for cand in constructed.values():
            for view in VIEWS:
                path = out_dir / f"{cand.cand_id}_socket_{view}.png"
                if not path.exists():
                    Image.fromarray(r.render(socket_scene(cand.bore), view)).save(path)
        r.close()

    def socket_paths(cand):
        base = out_dir if cand.kind == "shrunk" else RENDERS / "procedural"
        return {v: (base / f"{cand.cand_id}_socket_{v}.png").relative_to(REPO_ROOT).as_posix() for v in VIEWS}

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "blocks.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["block_id", "peg_id", "family", "split", "regime", "position", "cand_id", "kind", "role",
                "label", "margin_mm", "sim_to_mate", "is_mate", "peg_v1", "peg_v2", "socket_v1", "socket_v2"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for b in blocks:
            for i, c in enumerate(b.candidates):
                sp = socket_paths(c)
                w.writerow({
                    "block_id": b.block_id, "peg_id": b.peg_id, "family": b.family, "split": b.split,
                    "regime": b.regime, "position": i, "cand_id": c.cand_id, "kind": c.kind, "role": c.role,
                    "label": c.label, "margin_mm": round(c.margin_mm, 4), "sim_to_mate": round(c.sim_to_mate, 4),
                    "is_mate": int(i == b.mate_index),
                    "peg_v1": f"data/renders/procedural/{b.peg_id}_peg_v1.png",
                    "peg_v2": f"data/renders/procedural/{b.peg_id}_peg_v2.png",
                    "socket_v1": sp["v1"], "socket_v2": sp["v2"],
                })

    seen = {}
    for b in blocks:
        for c in b.candidates:
            seen.setdefault(c.cand_id, c)
    with (OUT / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cand_id", "kind", "bore_wkt", "socket_v1", "socket_v2", "sha256_v1", "sha256_v2"])
        w.writeheader()
        for cid, c in sorted(seen.items()):
            sp = socket_paths(c)
            w.writerow({"cand_id": cid, "kind": c.kind, "bore_wkt": wkt(c.bore),
                        **{f"socket_{v}": sp[v] for v in VIEWS},
                        **{f"sha256_{v}": hashlib.sha256((REPO_ROOT / sp[v]).read_bytes()).hexdigest()[:16] for v in VIEWS}})

    with (OUT / "eligibility.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)

    eligible = [r for r in log if r["eligible"]]
    meta = {
        "built": "2026-09-22", "split_seed": SPLIT_SEED, "candidate_seed": CANDIDATE_SEED,
        "rules": {"similarity": "max over quarter turns of IoU of openings, mm scale, own centroids",
                  "shrink_ladder_mm": SHRINK_LADDER, "hard_max_margin_mm": HARD_MAX_MARGIN,
                  "easy_max_pair_sim": EASY_MAX_PAIR_SIM, "min_hard_sim": MIN_HARD_SIM, "min_difficulty_gap": MIN_DIFFICULTY_GAP,
                  "excluded": "ambiguous pairs and loose-compatible sockets; candidates stay within the peg's split"},
        "regimes": list(REGIMES), "k": 3,
        "n_pegs_total": len(parts), "n_pegs_eligible": len(eligible), "n_blocks": len(blocks),
        "blocks_per_split": Counter(b.split for b in blocks), "eligible_per_split": Counter(r["split"] for r in eligible),
        "exclusions": Counter(r["reason"] for r in log if not r["eligible"]),
        "hard_kind": Counter(r["h_kind"] for r in eligible),
        "inputs": {"catalog_sha256": hashlib.sha256((REPO_ROOT / "data/parts/catalog.csv").read_bytes()).hexdigest()[:16],
                   "compat_sha256": compat_sha},
    }
    (OUT / "blocks_meta.json").write_text(json.dumps(meta, indent=2, default=dict), encoding="utf-8")

    difficulty_figure(eligible, log, FIGS / "stage5_difficulty.png")
    example_figure(blocks, FIGS / "stage5_example_blocks.png")
    report(meta, eligible, blocks)


def report(meta, eligible, blocks):
    print(f"{meta['n_pegs_eligible']}/{meta['n_pegs_total']} pegs eligible -> {len(blocks)} blocks "
          f"({len(REGIMES)} regimes x 3 candidates)")
    print("  eligible per split:", dict(meta["eligible_per_split"]), "| hard candidate kind:", dict(meta["hard_kind"]))
    for reason, n in meta["exclusions"].items():
        print(f"  excluded {n}: {reason}")
    h = np.array([r["h_sim"] for r in eligible], dtype=float)
    e = np.array([r["easy_max_sim"] for r in eligible], dtype=float)
    m = np.array([r["h_margin_mm"] for r in eligible], dtype=float)
    print(f"  similarity to mate: hard {h.min():.2f}-{h.max():.2f} (median {np.median(h):.2f}), "
          f"easiest-worst {e.min():.2f}-{e.max():.2f} (median {np.median(e):.2f})")
    print(f"  hard-candidate interference: {-m.max():.2f}-{-m.min():.2f} mm "
          f"({-m.max() / 0.0677:.1f}-{-m.min() / 0.0677:.1f} px at the frozen 0.0677 mm/px)")


def difficulty_figure(eligible, log, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    h = [r["h_sim"] for r in eligible]
    e = [r["easy_max_sim"] for r in eligible]
    ax[0].hist([h, e], bins=np.linspace(0, 1, 21), label=["hard candidate H", "hardest easy distractor"],
               color=["#b2182b", "#2166ac"])
    ax[0].set_xlabel("similarity to the true mate (IoU over quarter turns)")
    ax[0].set_ylabel("pegs")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Difficulty separation per peg", fontsize=10)
    ax[1].scatter([-r["h_margin_mm"] for r in eligible], h, s=18, c="#b2182b")
    ax[1].set_xlabel("interference of the hard candidate (mm)")
    ax[1].set_ylabel("similarity to the mate")
    ax[1].set_title("Hard candidates: how close, how wrong", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


def example_figure(blocks, path, thumb=150):
    pegs = sorted({b.peg_id for b in blocks})
    peg = pegs[len(pegs) // 2]
    chosen = [b for b in blocks if b.peg_id == peg]
    img = Image.new("RGB", (4 * thumb + 120, len(chosen) * thumb + 20), "white")
    d = ImageDraw.Draw(img)
    d.text((6, 4), f"{peg}: peg (v1) then the three candidates (v1), in presentation order", fill="black")
    for row, b in enumerate(chosen):
        y = 20 + row * thumb
        d.text((4, y + thumb // 2 - 10), b.regime, fill="black")
        d.text((4, y + thumb // 2 + 2), f"mate at {b.mate_index}" if b.mate_index is not None else "no mate",
               fill="#888888")
        img.paste(Image.open(REPO_ROOT / f"data/renders/procedural/{peg}_peg_v1.png").resize((thumb, thumb)), (120, y))
        for i, c in enumerate(b.candidates):
            base = "data/renders/blocks" if c.kind == "shrunk" else "data/renders/procedural"
            tile = Image.open(REPO_ROOT / f"{base}/{c.cand_id}_socket_v1.png").resize((thumb, thumb))
            img.paste(tile, (120 + (i + 1) * thumb, y))
            d.text((126 + (i + 1) * thumb, y + 4), f"{c.role} {c.label[:6]}", fill="#b2182b" if c.role == "H" else "#333333")
    img.save(path)


if __name__ == "__main__":
    main()
