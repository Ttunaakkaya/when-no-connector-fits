"""Run the image-only containment control on dev and calibration pairs and the dev aperture ladder.

    uv run python scripts/image_control.py

For every unique peg-candidate pair in the dev and calibration blocks, and every rung of the dev
aperture ladder, segment the processed top-view pixels and compare the pixel-derived decision with
the oracle label. Test pairs are not touched. Writes results/image_control/image_control.csv and
results/figures/image_control.png.
"""

import csv

import wncf  # noqa: F401  (must precede transformers: pins the HF cache to the project's copy)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoProcessor

from wncf import REPO_ROOT
from wncf.image_control import best_margin_px, stitched_tiles
from wncf.render import MM_PER_PX

OUT = REPO_ROOT / "results" / "image_control"


def pairs():
    rows = {}
    with (REPO_ROOT / "data/candidate_sets/blocks.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] in ("dev", "calib"):
                rows.setdefault((r["peg_id"], r["cand_id"]), {
                    "source": "blocks", "split": r["split"], "peg_id": r["peg_id"], "cand_id": r["cand_id"],
                    "role": r["role"], "label": r["label"], "oracle_margin_mm": float(r["margin_mm"]),
                    "peg_v1": r["peg_v1"], "socket_v1": r["socket_v1"]})
    sanity = REPO_ROOT / "results/sanity/sanity_scores.csv"
    if sanity.exists():
        with sanity.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["group"] == "aperture":
                    rows[(r["peg_id"], r["socket"])] = {
                        "source": "aperture", "split": "dev", "peg_id": r["peg_id"], "cand_id": r["socket"],
                        "role": r["detail"], "label": r["label"], "oracle_margin_mm": float(r["margin_mm"]),
                        "peg_v1": f"data/renders/procedural/{r['peg_id']}_peg_v1.png",
                        "socket_v1": f"data/renders/sanity/{r['socket']}_socket_v1.png"}
    return list(rows.values())


def main():
    proc = AutoProcessor.from_pretrained(wncf.MODELS["7b"][0], revision=wncf.MODELS["7b"][1], local_files_only=True)
    seen = {}

    def tiles(path):
        if path not in seen:
            seen[path] = stitched_tiles(proc, Image.open(REPO_ROOT / path).convert("RGB"))
        return seen[path]

    out = []
    for p in pairs():
        m_px, turn = best_margin_px(tiles(p["peg_v1"]), tiles(p["socket_v1"]))
        out.append({**{k: p[k] for k in ("source", "split", "peg_id", "cand_id", "role", "label", "oracle_margin_mm")},
                    "image_margin_px": round(m_px, 3), "image_margin_mm": round(m_px * MM_PER_PX, 4),
                    "image_turn": turn, "image_fits": int(m_px > 0)})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "image_control.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    report(out)
    figure(out, REPO_ROOT / "results" / "figures" / "image_control.png")


def report(out):
    decided = [r for r in out if r["label"] in ("compatible", "incompatible")]
    agree = [r for r in decided if r["image_fits"] == (r["label"] == "compatible")]
    print(f"{len(out)} pairs; {len(decided)} with a compatible/incompatible oracle label")
    print(f"pixel decision agrees with the oracle on {len(agree)}/{len(decided)} ({len(agree) / len(decided):.1%})")
    for role in ("M", "H", "E1", "E2", "E3"):
        rs = [r for r in decided if r["source"] == "blocks" and r["role"] == role]
        if rs:
            ok = sum(r["image_fits"] == (r["label"] == "compatible") for r in rs)
            print(f"  block role {role:2s}: {ok}/{len(rs)} agree")
    wrong = [r for r in decided if r not in agree]
    for r in wrong[:10]:
        print(f"  disagreement: {r['peg_id']} -> {r['cand_id']} oracle {r['label']} {r['oracle_margin_mm']:+.3f} mm, "
              f"image {r['image_margin_mm']:+.3f} mm")
    ap = [r for r in out if r["source"] == "aperture"]
    if ap:
        print("\naperture ladder (dev): oracle margin vs pixel margin, mean over pegs")
        for rung in sorted({r["role"] for r in ap}, key=float, reverse=True):
            rs = [r for r in ap if r["role"] == rung]
            print(f"  offset {float(rung):+.2f} mm: oracle {np.mean([r['oracle_margin_mm'] for r in rs]):+.3f} mm | "
                  f"image {np.mean([r['image_margin_mm'] for r in rs]):+.3f} mm | image says fits "
                  f"{sum(r['image_fits'] for r in rs)}/{len(rs)} | labels {sorted({r['label'] for r in rs})}")
    near = [r for r in decided if abs(r["oracle_margin_mm"]) < 0.5]
    err = np.array([r["image_margin_mm"] - r["oracle_margin_mm"] for r in near])
    print(f"\nwithin 0.5 mm of the boundary ({len(near)} pairs): pixel minus oracle margin "
          f"{err.mean():+.3f} mm mean, {np.abs(err).max():.3f} mm worst")


def figure(out, path):
    decided = [r for r in out if r["label"] in ("compatible", "incompatible")]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for label, colour in (("compatible", "#1b7837"), ("incompatible", "#b2182b")):
        rs = [r for r in decided if r["label"] == label]
        ax.scatter([r["oracle_margin_mm"] for r in rs], [r["image_margin_mm"] for r in rs], s=14, alpha=0.7,
                   color=colour, label=f"oracle {label}")
    lim = [-2.1, 0.5]
    ax.plot(lim, lim, color="#888", lw=1, ls="--")
    ax.axhline(0, color="#333", lw=0.8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlim(lim)
    ax.set_xlabel("oracle signed margin (mm, from the true polygons; clipped at -2)")
    ax.set_ylabel("pixel-derived signed margin (mm, from processed tiles)")
    ax.set_title("Image-only containment control, dev + calibration pairs", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()
