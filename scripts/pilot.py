"""Stage 4: development pilot on dev families only (ellipse, dshape, cross, tube).

    uv run python scripts/pilot.py --quant nf4 --grouping nested flat --prompt paper controlled
    uv run python scripts/pilot.py --quant int8 --grouping <chosen> --prompt <chosen>

Purpose: check that inputs, outputs and labels are auditable and correct, and choose the image
grouping, prompt and precision on development data before anything is frozen.

Selection rule, declared 22 September before any pilot scoring:
1. Validity gate: a configuration is usable only if every greedy first token is a Yes/No variant
   (no model_failure) and P(yes)+P(no) >= 0.5 on every pair.
2. Defaults: the paper prompt, nested grouping and NF4 (faithful to the source and the cheapest).
   An alternative replaces a default only if it raises the dev AUROC of p_yes (compatible vs
   incompatible pairs) by at least 0.10, or if the default fails the validity gate.
3. Run the four NF4 configurations first, then int8 on the configuration that step chose.
The panel has about 50 pairs, so this detects gross effects only; it cannot rank close configurations.

Result of steps 1-3 (22 September): every NF4 configuration passed the validity gate, but answered Yes to
98-100% of pairs with dev AUROC 0.34-0.47. scripts/diagnose_vision.py then showed that the setup is sound
(the model names the shapes it is given) and that p_yes barely moves even for four blank images.

Probes, declared 22 September after that result and before running them (dev panel, nested grouping):
  P1 int8 precision, paper prompt           --quant int8 --prompt paper
  P2 shape-comparison question               --prompt shape
  P3 paper's single-view variant (v1 only)   --prompt paper_v1
  P4 checkpoint's own chat template          --chat hf --prompt paper
A probe counts as informative only if dev AUROC >= 0.75 and same-peg ranking >= 0.75 (the fraction of
(own or turned-own socket, incompatible socket) comparisons for one peg in which the own socket has the
higher p_yes). Any probe adopted for the frozen setup is recorded as a deviation from the paper.

Panel (labels from the stage-2 oracle, rebuilt deterministically from PANEL_SEED):
- nominal:   every dev peg with its own socket (compatible)
- rotated:   the peg's own bore turned 90 deg, for bores without 90-deg symmetry (compatible)
- clear_neg: each dev peg with one other dev socket at >= 2 mm interference (incompatible)
- hard_neg:  same-family incompatible dev pairs with < 2 mm interference (incompatible)
- shrunk:    four pegs' own bores shrunk to -0.3 and -1.0 mm per-side clearance (incompatible)
"""

import argparse
import csv
import hashlib
import random
import time

import numpy as np
from PIL import Image, ImageDraw
from shapely.affinity import rotate
from sklearn.metrics import roc_auc_score

from wncf import REPO_ROOT
from wncf.geometry import MARGIN_CAP, label_pair
from wncf.parts import build_catalog, nominal_bore
from wncf.render import VIEWS, Renderer, socket_scene
from wncf.scorer_llava import PROMPT_VIEWS
from wncf.splits import family_splits

PANEL_SEED = 20260924
RENDERS = REPO_ROOT / "data" / "renders"
OUT = REPO_ROOT / "results" / "pilot"
SCORES = OUT / "stage4_scores.csv"


def build_panel():
    parts = build_catalog()
    split = family_splits({p.family for p in parts})
    dev = [p for p in parts if split[p.family] == "dev"]
    rng = random.Random(PANEL_SEED)
    panel = []

    def add(category, peg, socket_name, bore, expect):
        lab = label_pair(peg.peg, bore)
        assert lab.label == expect, (category, peg.part_id, socket_name, lab)
        panel.append(dict(category=category, peg_id=peg.part_id, socket=socket_name, bore=bore,
                          label=lab.label, margin_mm=lab.margin_mm))

    for p in dev:
        add("nominal", p, p.part_id, p.bore, "compatible")
    for p in dev:
        turned = rotate(p.bore, 90, origin=(0, 0))
        if turned.symmetric_difference(p.bore).area > 1e-3:
            add("rotated", p, f"{p.part_id}_rot90", turned, "compatible")
    for p in dev:
        options = [q for q in dev if q is not p and label_pair(p.peg, q.bore).margin_mm <= -MARGIN_CAP]
        q = rng.choice(options)
        add("clear_neg", p, q.part_id, q.bore, "incompatible")
    hard = [(p, q) for p in dev for q in dev if p is not q and p.family == q.family
            and (lab := label_pair(p.peg, q.bore)).label == "incompatible" and lab.margin_mm > -MARGIN_CAP]
    for p, q in rng.sample(hard, min(12, len(hard))):
        add("hard_neg", p, q.part_id, q.bore, "incompatible")
    for p in rng.sample(dev, 4):
        for c in (-0.3, -1.0):
            add("shrunk", p, f"{p.part_id}_c{c:+.1f}", nominal_bore(p.peg, c), "incompatible")
    return panel


def images_for(row, renderer):
    """{(role, view): path}; extra sockets are rendered once into data/renders/pilot."""
    paths = {("peg", v): RENDERS / "procedural" / f"{row['peg_id']}_peg_{v}.png" for v in VIEWS}
    for v in VIEWS:
        if "_rot90" in row["socket"] or "_c" in row["socket"]:
            path = RENDERS / "pilot" / f"{row['socket']}_socket_{v}.png"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(renderer.render(socket_scene(row["bore"]), v)).save(path)
        else:
            path = RENDERS / "procedural" / f"{row['socket']}_socket_{v}.png"
        paths["socket", v] = path
    return paths


def prompt_images(row, prompt):
    """Peg views then the same socket views, in the order the prompt's image slots expect."""
    views = PROMPT_VIEWS[prompt]
    return [Image.open(row["paths"][role, v]).convert("RGB") for role in ("peg", "socket") for v in views]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="nf4", choices=["nf4", "int8"])
    ap.add_argument("--grouping", nargs="+", default=["nested", "flat"])
    ap.add_argument("--prompt", nargs="+", default=["paper", "controlled"])
    ap.add_argument("--chat", nargs="+", default=["qwen_1_5"], choices=["qwen_1_5", "hf"])
    args = ap.parse_args()

    panel = build_panel()
    renderer = Renderer()
    for row in panel:
        row["paths"] = images_for(row, renderer)
        row["images_sha"] = hashlib.sha256(b"".join(p.read_bytes() for p in row["paths"].values())).hexdigest()[:16]
    renderer.close()
    counts = {c: sum(r["category"] == c for r in panel) for c in dict.fromkeys(r["category"] for r in panel)}
    print(f"panel: {len(panel)} pairs  " + ", ".join(f"{k} {v}" for k, v in counts.items()))

    from wncf.scorer_llava import LlavaScorer

    scorer = LlavaScorer(model="7b", quant=args.quant, grouping=args.grouping[0])
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for grouping in args.grouping:
        scorer.grouping = grouping
        audit_inputs(scorer, panel[0], grouping)
        for chat in args.chat:
            scorer.chat = chat
            for prompt in args.prompt:
                scorer.score(prompt_images(panel[0], prompt), prompt)  # warm-up
                t0 = time.perf_counter()
                for r in panel:
                    s = scorer.score(prompt_images(r, prompt), prompt)
                    rows.append({"quant": args.quant, "grouping": grouping, "prompt": prompt, "chat": chat,
                                 **{k: r[k] for k in ("category", "peg_id", "socket", "label", "margin_mm", "images_sha")},
                                 **s.as_dict()})
                print(f"  scored {args.quant}/{grouping}/{prompt}/{chat} in {time.perf_counter() - t0:.0f}s")
    save(rows)
    summarise()


def audit_inputs(scorer, row, grouping):
    """Save the four images exactly as the processor hands them to the model, in prompt order."""
    imgs = prompt_images(row, "paper")
    batch = [imgs] if grouping == "nested" else imgs
    x = scorer.processor(images=batch, text=scorer.build_prompt("paper"), return_tensors="pt")
    pv = x["pixel_values"].float().numpy()  # (4, patches, 3, 384, 384); patch 0 is the whole image
    tiles = [((pv[i, 0].transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8) for i in range(4)]
    strip = Image.fromarray(np.hstack(tiles))
    d = ImageDraw.Draw(strip)
    for i, name in enumerate(["image 1: peg v1", "image 2: peg v2", "image 3: hole v1", "image 4: hole v2"]):
        d.text((i * 384 + 6, 6), name, fill="black")
    d.text((6, 364), f"{grouping}: pixel_values {tuple(pv.shape)}, {x['input_ids'].shape[1]} tokens", fill="black")
    strip.save(OUT / f"stage4_input_audit_{grouping}.png")


def config(r) -> tuple:
    return r["quant"], r["grouping"], r["prompt"], r.get("chat") or "qwen_1_5"  # rows before 'chat' existed


def save(new_rows):
    old = []
    if SCORES.exists():
        with SCORES.open(newline="", encoding="utf-8") as f:
            done = {config(r) for r in new_rows}
            old = [{**r, "chat": config(r)[3]} for r in csv.DictReader(f) if config(r) not in done]
    rows = old + new_rows
    with SCORES.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(new_rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarise():
    with SCORES.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    configs = sorted({config(r) for r in rows})
    cats = list(dict.fromkeys(r["category"] for r in rows))
    out = []
    for cfg in configs:
        rs = [r for r in rows if config(r) == cfg]
        y = np.array([r["label"] == "compatible" for r in rs])
        p = np.array([float(r["p_yes"]) for r in rs])
        ans = np.array([r["answer"] for r in rs])
        mass = np.array([float(r["mass_yes"]) + float(r["mass_no"]) for r in rs])
        rec = {
            "quant": cfg[0], "grouping": cfg[1], "prompt": cfg[2], "chat": cfg[3], "n": len(rs),
            "failures": int((ans == "other").sum()), "min_mass": round(mass.min(), 3),
            "auroc": round(roc_auc_score(y, p), 3), "rank": round(same_peg_ranking(rs), 3),
            "acc": round(((ans == "yes") == y).mean(), 3),
            "yes_rate": round((ans == "yes").mean(), 3),
            **{f"acc_{c}": round(np.mean([(r["answer"] == "yes") == (r["label"] == "compatible")
                                          for r in rs if r["category"] == c]), 2) for c in cats},
            "mean_p_compat": round(p[y].mean(), 3), "mean_p_incompat": round(p[~y].mean(), 3),
            "latency_s": round(np.mean([float(r["latency_s"]) for r in rs]), 2),
            "prep_s": round(np.mean([float(r["prep_s"]) for r in rs]), 2),
            "tokens": int(np.mean([int(r["n_input_tokens"]) for r in rs])),
            "peak_vram_gb": round(max(float(r["peak_vram_gb"]) for r in rs), 2),
        }
        out.append(rec)
    with (OUT / "stage4_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    keys = ["quant", "grouping", "prompt", "chat", "failures", "min_mass", "auroc", "rank", "acc", "yes_rate",
            *[f"acc_{c}" for c in cats], "latency_s", "prep_s", "tokens", "peak_vram_gb"]
    print("  ".join(f"{k:>10s}" for k in keys))
    for rec in out:
        print("  ".join(f"{str(rec[k]):>10s}" for k in keys))


def same_peg_ranking(rs) -> float:
    """Fraction of (compatible, incompatible) socket comparisons for the same peg won by the compatible one."""
    wins = total = 0
    for peg in {r["peg_id"] for r in rs}:
        pos = [float(r["p_yes"]) for r in rs if r["peg_id"] == peg and r["label"] == "compatible"]
        neg = [float(r["p_yes"]) for r in rs if r["peg_id"] == peg and r["label"] == "incompatible"]
        for a in pos:
            for b in neg:
                wins += (a > b) + 0.5 * (a == b)
                total += 1
    return wins / total


if __name__ == "__main__":
    main()
