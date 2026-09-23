"""Render the calibration blocks in the paper-imitating look for the appearance arm.

    uv run python scripts/render_arm.py

Same geometry, camera, views and file names as the primary renders; only colours and background
change (config.PAPER_LOOK). Covers every peg and candidate the calibration blocks use, including the
constructed look-alikes. Writes data/renders/arm_appearance/*.png and a manifest with hashes.
"""

import csv
import hashlib

import numpy as np
from PIL import Image
from shapely import from_wkt

from wncf import REPO_ROOT
from wncf.config import ARM_RENDERS, ARM_SPLITS, PAPER_LOOK
from wncf.parts import build_catalog
from wncf.render import VIEWS, Renderer, peg_scene, socket_scene

OUT = REPO_ROOT / ARM_RENDERS["appearance"]


def main():
    parts = {p.part_id: p for p in build_catalog()}
    bores = {r["cand_id"]: from_wkt(r["bore_wkt"])
             for r in csv.DictReader((REPO_ROOT / "data/candidate_sets/candidates.csv").open(encoding="utf-8"))}
    pegs, cands = set(), set()
    with (REPO_ROOT / "data/candidate_sets/blocks.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] in ARM_SPLITS:
                pegs.add(r["peg_id"])
                cands.add(r["cand_id"])

    OUT.mkdir(parents=True, exist_ok=True)
    r = Renderer(background=PAPER_LOOK["background"])
    bg = np.array([int(PAPER_LOOK["background"][i:i + 2], 16) for i in (1, 3, 5)])
    rows = []
    jobs = [(f"{p}_peg", peg_scene(parts[p].peg, PAPER_LOOK["peg"])) for p in sorted(pegs)]
    jobs += [(f"{c}_socket", socket_scene(bores[c], PAPER_LOOK["socket"], PAPER_LOOK["floor"])) for c in sorted(cands)]
    for stem, scene in jobs:
        for view in VIEWS:
            img = r.render(scene, view)
            border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(int)
            if np.abs(border - bg).max() > 3:
                raise AssertionError(f"{stem} {view}: part touches the frame edge")
            path = OUT / f"{stem}_{view}.png"
            Image.fromarray(img).save(path)
            rows.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    r.close()
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print(f"rendered {len(rows)} images for {len(pegs)} pegs and {len(cands)} candidates into {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
