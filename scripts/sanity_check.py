"""Bounded development sanity check under the frozen configuration.

    uv run python scripts/sanity_check.py

Dev families only. Five groups, reported separately and never pooled:
  mate        each dev peg with its own socket                          (compatible)
  look_alike  each dev peg with the hard candidate its block uses       (incompatible)
  easy        each dev peg with the easy distractor E1 its block uses   (incompatible)
              a positive control added 22 September after the first run showed no separation between
              the other groups; the frozen configuration was not changed
  aperture    six dev pegs, bore inflated from +0.30 mm down to -1.20 mm relative to the peg
  rotation    dev pegs with their own bore turned 90 degrees            (compatible by contract)
              excludes bores with 90-degree symmetry

Its purpose is to confirm that inputs, labels and outputs are auditable and that invalid-response
handling works. No accuracy threshold gates progress, and no configuration search follows it.
Writes results/sanity/sanity_scores.csv and results/figures/sanity_aperture.png.

    uv run python scripts/sanity_check.py --prompt paper    # EXPLORATORY, after the primary test pass

With --prompt paper the identical design runs with only the prompt changed, as an exploratory paired
comparison; it writes to results/exploratory/sanity_paper_prompt/ and
results/figures/exploratory_paper_prompt_aperture.png and never overwrites the primary sanity run.
"""

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shapely.affinity import rotate

from wncf import REPO_ROOT
from wncf.geometry import label_pair
from wncf.parts import FAMILIES, build_catalog, nominal_bore
from wncf.render import VIEWS, Renderer, socket_scene
from wncf.rules import CandidateScore, invalid_reason
from wncf.splits import family_splits

OUT = REPO_ROOT / "results" / "sanity"
RENDERS = REPO_ROOT / "data" / "renders" / "sanity"
PROC = REPO_ROOT / "data" / "renders" / "procedural"
APERTURE_MM = (0.30, 0.10, 0.00, -0.10, -0.30, -0.60, -1.20)  # bore inflation relative to the peg
N_APERTURE_PEGS = 6
PROMPT = "controlled"  # frozen; overridden only by the exploratory --prompt option


def block_candidates(role):
    """The candidate in `role` that each dev peg's blocks use, from the stage-5 manifest."""
    path = REPO_ROOT / "data" / "candidate_sets" / "blocks.csv"
    from shapely import from_wkt

    bores = {r["cand_id"]: from_wkt(r["bore_wkt"])
             for r in csv.DictReader((REPO_ROOT / "data/candidate_sets/candidates.csv").open(encoding="utf-8"))}
    out = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r["role"] == role and r["split"] == "dev":
            out[r["peg_id"]] = (r["cand_id"], bores[r["cand_id"]], r["socket_v1"], r["socket_v2"])
    return out


def main():
    global PROMPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="controlled", choices=["controlled", "paper"])
    PROMPT = ap.parse_args().prompt
    out_dir, fig_path = OUT, REPO_ROOT / "results" / "figures" / "sanity_aperture.png"
    if PROMPT != "controlled":
        if not (REPO_ROOT / "results" / "test" / "test_pass_record.json").exists():
            raise SystemExit("the exploratory prompt variant runs only after the primary test pass")
        out_dir = REPO_ROOT / "results" / "exploratory" / f"sanity_{PROMPT}_prompt"
        fig_path = REPO_ROOT / "results" / "figures" / f"exploratory_{PROMPT}_prompt_aperture.png"

    parts = [p for p in build_catalog() if family_splits(FAMILIES)[p.family] == "dev"]
    hard, easy = block_candidates("H"), block_candidates("E1")
    RENDERS.mkdir(parents=True, exist_ok=True)
    renderer = Renderer()
    cases = []

    def socket_images(name, bore):
        paths = []
        for v in VIEWS:
            path = RENDERS / f"{name}_socket_{v}.png"
            if not path.exists():
                Image.fromarray(renderer.render(socket_scene(bore), v)).save(path)
            paths.append(path)
        return paths

    for p in parts:
        cases.append(dict(group="mate", peg_id=p.part_id, socket=p.part_id, detail="",
                          bore=p.bore, paths=[PROC / f"{p.part_id}_socket_{v}.png" for v in VIEWS]))
        for group, source in (("look_alike", hard), ("easy", easy)):
            if p.part_id in source:
                cid, bore, v1, v2 = source[p.part_id]
                cases.append(dict(group=group, peg_id=p.part_id, socket=cid, detail="",
                                  bore=bore, paths=[REPO_ROOT / v1, REPO_ROOT / v2]))
        turned = rotate(p.bore, 90, origin=(0, 0))
        if turned.symmetric_difference(p.bore).area > 1e-3:
            cases.append(dict(group="rotation", peg_id=p.part_id, socket=f"{p.part_id}_rot90", detail="90 deg",
                              bore=turned, paths=socket_images(f"{p.part_id}_rot90", turned)))

    for p in sorted(parts, key=lambda q: q.part_id)[:N_APERTURE_PEGS]:
        for a in APERTURE_MM:
            bore = nominal_bore(p.peg, a)
            name = f"{p.part_id}_ap{a:+.2f}"
            cases.append(dict(group="aperture", peg_id=p.part_id, socket=name, detail=f"{a:+.2f}",
                              bore=bore, paths=socket_images(name, bore)))
    renderer.close()

    pegs = {p.part_id: p for p in parts}
    for c in cases:
        lab = label_pair(pegs[c["peg_id"]].peg, c["bore"])
        c.update(label=lab.label, reason=lab.reason, margin_mm=round(lab.margin_mm, 4))

    from wncf.scorer_llava import LlavaScorer

    scorer = LlavaScorer(model="7b", quant="nf4", grouping="flat", chat="qwen_1_5")
    rows = []
    for i, c in enumerate(cases, 1):
        imgs = [Image.open(PROC / f"{c['peg_id']}_peg_{v}.png").convert("RGB") for v in VIEWS]
        imgs += [Image.open(p).convert("RGB") for p in c["paths"]]
        s = scorer.score(imgs, PROMPT)
        bad = invalid_reason([CandidateScore(c["socket"], s.answer, s.top_prob, s.p_yes, s.mass_yes, s.mass_no)])
        rows.append({k: c[k] for k in ("group", "peg_id", "socket", "detail", "label", "reason", "margin_mm")}
                    | s.as_dict() | {"invalid": bad or ""})
        if i % 20 == 0:
            print(f"  {i}/{len(cases)} scored")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sanity_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)
    aperture_figure(rows, fig_path)


def summarise(rows):
    print(f"\n{len(rows)} calls, configuration: 7B NF4 / flat / qwen_1_5 / '{PROMPT}' prompt / neutral renders")
    invalid = [r for r in rows if r["invalid"]]
    print(f"invalid responses: {len(invalid)}" + (f" e.g. {invalid[0]['invalid']}" if invalid else ""))
    print(f"\n{'group':11s} {'n':>3s} {'labels':28s} {'yes rate':>9s} {'mean p_yes':>11s} {'mean top_prob':>14s}")
    for group in ("mate", "look_alike", "easy", "rotation", "aperture"):
        rs = [r for r in rows if r["group"] == group]
        if not rs:
            continue
        labels = ", ".join(f"{k} {v}" for k, v in sorted(defaultdict(int, {
            l: sum(r["label"] == l for r in rs) for l in {r["label"] for r in rs}}).items()))
        print(f"{group:11s} {len(rs):3d} {labels:28s} {np.mean([r['answer'] == 'yes' for r in rs]):9.2f} "
              f"{np.mean([r['p_yes'] for r in rs]):11.3f} {np.mean([r['top_prob'] for r in rs]):14.3f}")

    ap = [r for r in rows if r["group"] == "aperture"]
    if ap:
        print(f"\naperture ladder (bore inflation relative to the peg; {len({r['peg_id'] for r in ap})} pegs)")
        print(f"{'offset mm':>10s} {'label':14s} {'n':>3s} {'yes rate':>9s} {'mean p_yes':>11s}")
        for a in APERTURE_MM:
            rs = [r for r in ap if r["detail"] == f"{a:+.2f}"]
            labs = {r["label"] for r in rs}
            print(f"{a:+10.2f} {'/'.join(sorted(labs)):14s} {len(rs):3d} "
                  f"{np.mean([r['answer'] == 'yes' for r in rs]):9.2f} {np.mean([r['p_yes'] for r in rs]):11.3f}")
    means = {g: np.mean([r["p_yes"] for r in rows if r["group"] == g]) for g in ("mate", "look_alike", "easy")}
    print(f"\nmean p_yes: mate {means['mate']:.3f} | look-alike {means['look_alike']:.3f} "
          f"(difference {means['mate'] - means['look_alike']:+.3f}) | easy distractor {means['easy']:.3f} "
          f"(difference {means['mate'] - means['easy']:+.3f})")
    wins = [next(r["p_yes"] for r in rows if r["group"] == "mate" and r["peg_id"] == peg)
            > next(r["p_yes"] for r in rows if r["group"] == g and r["peg_id"] == peg)
            for g in ("look_alike", "easy")
            for peg in {r["peg_id"] for r in rows if r["group"] == g}]
    print(f"mate scored above the paired distractor in {sum(wins)}/{len(wins)} comparisons (chance 50%)")
    print("Recorded as a diagnostic; no threshold here gates the core experiment.")


def aperture_figure(rows, path):
    ap = [r for r in rows if r["group"] == "aperture"]
    if not ap:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for peg in sorted({r["peg_id"] for r in ap}):
        rs = sorted((r for r in ap if r["peg_id"] == peg), key=lambda r: float(r["detail"]))
        ax.plot([float(r["detail"]) for r in rs], [r["p_yes"] for r in rs], marker="o", ms=4, label=peg, alpha=0.8)
    ax.axvline(0.05, color="#888", ls="--", lw=1)
    ax.text(0.06, ax.get_ylim()[1], " compatible →", fontsize=8, va="top", color="#555")
    ax.set_xlabel("bore inflation relative to the peg (mm); below about +0.05 the pair no longer fits")
    ax.set_ylabel("p_yes")
    title = "frozen configuration" if PROMPT == "controlled" else f"EXPLORATORY, '{PROMPT}' prompt"
    ax.set_title(f"Development aperture response, {title}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()
