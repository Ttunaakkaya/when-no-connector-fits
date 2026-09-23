"""Replica of the paper's 3D-printed matching experiment (Yajima et al., Table I, "3D print": 7/8).

    uv run python scripts/replica_3dprint.py

The paper's eight 3D-printed shapes (their Fig. 4a: cross, octagon, square, circle, stadium, trapezoid,
hexagon, rectangle) are rebuilt with our generator at similar proportions. Their protocol: each peg is
scored against all eight holes, holes answered Yes are ranked by the probability of that answer
(descending), then holes answered No (ascending), and a peg succeeds if its own hole ranks first.
Chance is 1 of 8 pegs.

Two looks, to separate image style from everything else:
- gray:  our benchmark style (grey parts, white background, dark cavity floor)
- paper: their photo's look (red pegs, light-green plate, dark background, black hole interior)
Model 7B NF4, paper prompt, qwen_1_5 wrapper; nested and flat image grouping.

This is a diagnostic reproduction, not part of the dev/calib/test benchmark. None of its parts or
images are benchmark items, but its shape templates overlap calib and test families, so any render
style adopted because of it is recorded as a development decision informed by those templates.
"""

import csv

import numpy as np
from PIL import Image, ImageDraw
from shapely.affinity import rotate

from wncf import REPO_ROOT
from wncf.geometry import label_pair
from wncf.parts import centred, circle, cross, nominal_bore, obround, rect, regular, square, trapezoid
from wncf.render import VIEWS, Renderer, peg_scene, socket_scene

OUT = REPO_ROOT / "results" / "replica"
RENDERS = REPO_ROOT / "data" / "renders" / "replica"

SHAPES = {  # sizes in mm, orientations as in their photo
    "cross": cross(22, 22, 8),
    "octagon": regular(8, 12),
    "square": square(18),
    "circle": circle(22),
    "stadium": rotate(obround(28, 13), 90, origin=(0, 0)),
    "trapezoid": rotate(trapezoid(12, 24, 18), -90, origin=(0, 0)),
    "hexagon": rotate(regular(6, 12), 30, origin=(0, 0)),
    "rectangle": rect(14, 24),
}
STYLES = {
    "gray": dict(background="#ffffff", peg="#9aa3ad", socket="#9aa3ad", floor="#2b2b2b"),
    "paper": dict(background="#141414", peg="#d8322e", socket="#8fdca4", floor="#0b0b0b"),
}


def render_all():
    pegs = {k: centred(g) for k, g in SHAPES.items()}
    bores = {k: nominal_bore(p) for k, p in pegs.items()}
    paths = {}
    for style, st in STYLES.items():
        r = Renderer(background=st["background"])
        for name in SHAPES:
            for view in VIEWS:
                for role, scene in (("peg", peg_scene(pegs[name], st["peg"])),
                                    ("socket", socket_scene(bores[name], st["socket"], st["floor"]))):
                    path = RENDERS / style / f"{name}_{role}_{view}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(r.render(scene, view)).save(path)
                    paths[style, name, role, view] = path
        r.close()
    return pegs, bores, paths


def paper_rank(cands):
    """Their ranking: Yes answers by descending answer probability, then No answers by ascending."""
    yes = sorted((c for c in cands if c["answer"] == "yes"), key=lambda c: -c["top_prob"])
    no = sorted((c for c in cands if c["answer"] != "yes"), key=lambda c: c["top_prob"])
    return [c["socket"] for c in yes + no]


def main():
    pegs, bores, paths = render_all()
    labels = {(p, s): label_pair(pegs[p], bores[s]).label for p in SHAPES for s in SHAPES}
    cross_fits = [(p, s) for (p, s), lab in labels.items() if p != s and lab == "compatible"]
    print(f"oracle: every own pair fits; {len(cross_fits)} other pairs also fit: {cross_fits}")
    for style in STYLES:
        sheet(style, paths)

    from wncf.scorer_llava import LlavaScorer

    s = LlavaScorer(model="7b", quant="nf4")
    rows = []
    for grouping in ("nested", "flat"):
        s.grouping = grouping
        for style in STYLES:
            for p in SHAPES:
                for q in SHAPES:
                    imgs = [Image.open(paths[style, p, "peg", v]).convert("RGB") for v in VIEWS] + \
                           [Image.open(paths[style, q, "socket", v]).convert("RGB") for v in VIEWS]
                    sc = s.score(imgs, "paper")
                    rows.append({"style": style, "grouping": grouping, "peg": p, "socket": q,
                                 "own": int(p == q), "oracle": labels[p, q], **sc.as_dict()})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "replica_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)


def summarise(rows):
    print(f"\n{'style':6s} {'grouping':8s} {'top-1 (paper rank)':>19s} {'top-1 (p_yes)':>14s} "
          f"{'own-hole rank':>14s} {'own beats other':>16s} {'yes rate':>9s}")
    for grouping in ("nested", "flat"):
        for style in STYLES:
            rs = [r for r in rows if r["style"] == style and r["grouping"] == grouping]
            top1 = top1_p = 0
            ranks, wins = [], []
            for p in SHAPES:
                cands = [dict(r, socket=r["socket"]) for r in rs if r["peg"] == p]
                order = paper_rank(cands)
                top1 += order[0] == p
                ranks.append(order.index(p) + 1)
                top1_p += max(cands, key=lambda c: c["p_yes"])["socket"] == p
                own = next(c["p_yes"] for c in cands if c["socket"] == p)
                wins += [own > c["p_yes"] for c in cands if c["socket"] != p]
            yes = np.mean([r["answer"] == "yes" for r in rs])
            print(f"{style:6s} {grouping:8s} {top1:>15d}/8 {top1_p:>11d}/8 {np.mean(ranks):>14.2f} "
                  f"{np.mean(wins):>16.2f} {yes:>9.2f}")
    print("chance: top-1 1/8, mean own-hole rank 4.5, own beats other 0.50;  paper: 7/8")


def sheet(style, paths, thumb=160):
    img = Image.new("RGB", (4 * thumb + 90, len(SHAPES) * thumb), "white")
    d = ImageDraw.Draw(img)
    for i, name in enumerate(SHAPES):
        d.text((6, i * thumb + thumb // 2), name, fill="black")
        for j, (role, view) in enumerate([("peg", "v1"), ("peg", "v2"), ("socket", "v1"), ("socket", "v2")]):
            tile = Image.open(paths[style, name, role, view]).resize((thumb, thumb), Image.LANCZOS)
            img.paste(tile, (90 + j * thumb, i * thumb))
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / f"replica_sheet_{style}.png")


if __name__ == "__main__":
    main()
