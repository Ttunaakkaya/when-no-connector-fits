"""Aperture series (master plan RQ3) on the calibration and test families, frozen configuration.

    uv run python scripts/aperture_series.py build
    uv run python scripts/score.py --split calib --manifest data/aperture/pairs.csv
    uv run python scripts/score.py --split test --allow-test --manifest data/aperture/pairs.csv
    uv run python scripts/aperture_series.py report

Each eligible peg is paired with its own opening at seven sizes, from 0.30 mm of per-side clearance
(the nominal socket) down to 1.20 mm of interference, with the exterior, camera and lighting fixed.
The ladder is the one the dev sanity check used, fixed before any calibration or test score existed,
and the thresholds come from the freeze record; nothing here is fitted. The summary statistics were
chosen when this script was written, after the core test pass, so they are descriptive.

The report shows the model's p_yes and its acceptance (answer Yes; p_yes at or above the frozen S1
threshold) against the oracle's measured margin, per split, and runs the image-only containment
control on the same images as the positive check that the size change is visible in the pixels.
"""

import csv
import json
import sys

import wncf  # noqa: F401  (must precede transformers: pins the HF cache to the project's copy)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from wncf import REPO_ROOT
from wncf.cache import ScoreCache, frozen_identity, pair_key
from wncf.geometry import label_pair
from wncf.parts import build_catalog, nominal_bore
from wncf.render import MM_PER_PX, VIEWS, Renderer, socket_scene

APERTURE_MM = (0.30, 0.10, 0.00, -0.10, -0.30, -0.60, -1.20)  # the dev sanity ladder
SPLITS = ("calib", "test")
MANIFEST = REPO_ROOT / "data" / "aperture" / "pairs.csv"
RENDERS = REPO_ROOT / "data" / "renders" / "aperture"
OUT = REPO_ROOT / "results" / "aperture"
FREEZE = REPO_ROOT / "results" / "freeze" / "freeze_record.json"
COLS = ["split", "family", "peg_id", "cand_id", "offset_mm", "label", "reason", "margin_mm",
        "peg_v1", "peg_v2", "socket_v1", "socket_v2"]


def eligible_pegs():
    with (REPO_ROOT / "data" / "candidate_sets" / "eligibility.csv").open(newline="", encoding="utf-8") as f:
        return {r["peg_id"]: r["split"] for r in csv.DictReader(f) if r["eligible"] == "1" and r["split"] in SPLITS}


def build():
    parts = {p.part_id: p for p in build_catalog()}
    pegs = eligible_pegs()
    RENDERS.mkdir(parents=True, exist_ok=True)
    r = Renderer()
    rows = []
    for peg_id, split in sorted(pegs.items()):
        part = parts[peg_id]
        for a in APERTURE_MM:
            bore = nominal_bore(part.peg, a)
            if a == 0.30:  # identical to the catalogue socket, so reuse its render and its cached score
                assert bore.equals(part.bore)
                cand = peg_id
                sock = {v: f"data/renders/procedural/{peg_id}_socket_{v}.png" for v in VIEWS}
            else:
                cand = f"{peg_id}_ap{a:+.2f}"
                sock = {}
                for v in VIEWS:
                    path = RENDERS / f"{cand}_socket_{v}.png"
                    if not path.exists():
                        Image.fromarray(r.render(socket_scene(bore), v)).save(path)
                    sock[v] = path.relative_to(REPO_ROOT).as_posix()
            lab = label_pair(part.peg, bore)
            rows.append({"split": split, "family": part.family, "peg_id": peg_id, "cand_id": cand,
                         "offset_mm": a, "label": lab.label, "reason": lab.reason,
                         "margin_mm": round(lab.margin_mm, 4),
                         "peg_v1": f"data/renders/procedural/{peg_id}_peg_v1.png",
                         "peg_v2": f"data/renders/procedural/{peg_id}_peg_v2.png",
                         "socket_v1": sock["v1"], "socket_v2": sock["v2"]})
    r.close()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    by = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
    print(f"{len(rows)} ladder pairs from {len(pegs)} pegs {by} -> {MANIFEST.relative_to(REPO_ROOT)}")


def report():
    tau = json.loads(FREEZE.read_text(encoding="utf-8"))["calibration"]["S1"]["tau"]
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cache, base = ScoreCache(), frozen_identity()
    missing = 0
    for r in rows:
        row = cache.get(pair_key(base, [r["peg_v1"], r["peg_v2"], r["socket_v1"], r["socket_v2"]])[0])
        if row is None:
            missing += 1
            continue
        r.update(p_yes=row["p_yes"], answer=row["answer"], top_prob=row["top_prob"])
    if missing:
        sys.exit(f"{missing} ladder pairs are not scored yet; run scripts/score.py with --manifest first")

    image_control(rows)
    for r in rows:
        r["offset_mm"], r["margin_mm"] = float(r["offset_mm"]), float(r["margin_mm"])
        r["accept_tau"] = int(r["p_yes"] >= tau)
        r["accept_yes"] = int(r["answer"] == "yes")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "aperture_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"frozen_s1_tau": tau, "ladder_mm": APERTURE_MM, "splits": {}}
    for split in SPLITS:
        rs = [r for r in rows if r["split"] == split]
        per_rung = []
        for a in APERTURE_MM:
            rr = [r for r in rs if r["offset_mm"] == a]
            per_rung.append({
                "offset_mm": a, "n": len(rr), "labels": {l: sum(r["label"] == l for r in rr) for l in {r["label"] for r in rr}},
                "mean_margin_mm": round(float(np.mean([r["margin_mm"] for r in rr])), 3),
                "mean_p_yes": round(float(np.mean([r["p_yes"] for r in rr])), 4),
                "accept_yes": round(float(np.mean([r["accept_yes"] for r in rr])), 3),
                "accept_tau": round(float(np.mean([r["accept_tau"] for r in rr])), 3),
                "image_says_fits": round(float(np.mean([r["image_fits"] for r in rr])), 3),
            })
        decided = [r for r in rs if r["label"] in ("compatible", "incompatible")]
        y = np.array([r["label"] == "compatible" for r in decided])
        rhos = []
        for peg in sorted({r["peg_id"] for r in rs}):
            pr = [r for r in rs if r["peg_id"] == peg]
            rho = spearmanr([r["offset_mm"] for r in pr], [r["p_yes"] for r in pr]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
        summary["splits"][split] = {
            "n_pairs": len(rs), "n_pegs": len({r["peg_id"] for r in rs}),
            "auroc_p_yes": round(float(roc_auc_score(y, [r["p_yes"] for r in decided])), 3),
            "accept_tau_compatible": round(float(np.mean([r["accept_tau"] for r in decided if r["label"] == "compatible"])), 3),
            "accept_tau_incompatible": round(float(np.mean([r["accept_tau"] for r in decided if r["label"] == "incompatible"])), 3),
            "image_control_agreement": round(float(np.mean([r["image_fits"] == (r["label"] == "compatible") for r in decided])), 3),
            "per_peg_spearman_offset_vs_p_yes": {"median": round(float(np.median(rhos)), 3),
                                                 "share_positive": round(float(np.mean([x > 0 for x in rhos])), 3),
                                                 "n": len(rhos)},
            "per_rung": per_rung,
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure(rows, tau)
    print_summary(summary)


def image_control(rows):
    """The same pixel-only containment check as scripts/image_control.py, on the ladder images."""
    from transformers import AutoProcessor

    from wncf.image_control import best_margin_px, stitched_tiles

    proc = AutoProcessor.from_pretrained(wncf.MODELS["7b"][0], revision=wncf.MODELS["7b"][1], local_files_only=True)
    seen = {}

    def tiles(path):
        if path not in seen:
            seen[path] = stitched_tiles(proc, Image.open(REPO_ROOT / path).convert("RGB"))
        return seen[path]

    for r in rows:
        m_px, _ = best_margin_px(tiles(r["peg_v1"]), tiles(r["socket_v1"]))
        r["image_margin_mm"] = round(m_px * MM_PER_PX, 4)
        r["image_fits"] = int(m_px > 0)


def figure(rows, tau):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, split in zip(axes[:2], SPLITS):
        rs = [r for r in rows if r["split"] == split]
        for peg in sorted({r["peg_id"] for r in rs}):
            pr = sorted((r for r in rs if r["peg_id"] == peg), key=lambda r: r["offset_mm"])
            ax.plot([r["offset_mm"] for r in pr], [r["p_yes"] for r in pr], color="#9aa3ad", lw=0.6, alpha=0.6)
        means = [np.mean([r["p_yes"] for r in rs if r["offset_mm"] == a]) for a in APERTURE_MM]
        ax.plot(APERTURE_MM, means, color="#b2182b", lw=2.2, marker="o", label="mean over pegs")
        ax.axhline(tau, color="#2166ac", ls="--", lw=1, label=f"frozen S1 tau {tau:.3f}")
        ax.axvspan(-0.05, 0.05, color="#f1a340", alpha=0.15, label="about the fit boundary")
        ax.set_title(f"{split}: model p_yes as the opening shrinks", fontsize=10)
        ax.set_xlabel("per-side bore offset relative to the peg (mm); negative = interference")
        ax.set_ylabel("p_yes")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    ax = axes[2]
    for split, style in (("calib", "-"), ("test", "--")):
        rs = [r for r in rows if r["split"] == split]
        ax.plot(APERTURE_MM, [np.mean([r["accept_tau"] for r in rs if r["offset_mm"] == a]) for a in APERTURE_MM],
                color="#b2182b", ls=style, marker="o", label=f"model accepts (p_yes >= tau), {split}")
        ax.plot(APERTURE_MM, [np.mean([r["image_fits"] for r in rs if r["offset_mm"] == a]) for a in APERTURE_MM],
                color="#1b7837", ls=style, marker="s", label=f"image-only control says fits, {split}")
    ax.set_title("Acceptance against the offset", fontsize=10)
    ax.set_xlabel("per-side bore offset (mm)")
    ax.set_ylabel("share of pegs")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle("Aperture series, frozen configuration (7B NF4, flat, controlled prompt, neutral renders)", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "results" / "figures" / "aperture_series.png", dpi=140)


def print_summary(summary):
    print(f"frozen S1 tau {summary['frozen_s1_tau']:.4f}")
    for split, s in summary["splits"].items():
        print(f"\n{split}: {s['n_pairs']} pairs from {s['n_pegs']} pegs | AUROC(p_yes, compatible vs incompatible) "
              f"{s['auroc_p_yes']} | accept at tau: compatible {s['accept_tau_compatible']}, incompatible "
              f"{s['accept_tau_incompatible']} | image control agrees {s['image_control_agreement']}")
        sp = s["per_peg_spearman_offset_vs_p_yes"]
        print(f"  per-peg Spearman(offset, p_yes): median {sp['median']}, positive for {sp['share_positive']:.0%} of {sp['n']} pegs")
        print(f"  {'offset':>7s} {'margin':>7s} {'labels':32s} {'p_yes':>7s} {'yes':>5s} {'>=tau':>6s} {'image':>6s}")
        for r in s["per_rung"]:
            labs = ", ".join(f"{k} {v}" for k, v in sorted(r["labels"].items()))
            print(f"  {r['offset_mm']:+7.2f} {r['mean_margin_mm']:+7.3f} {labs:32s} {r['mean_p_yes']:7.4f} "
                  f"{r['accept_yes']:5.2f} {r['accept_tau']:6.2f} {r['image_says_fits']:6.2f}")


if __name__ == "__main__":
    {"build": build, "report": report}[sys.argv[1] if len(sys.argv) > 1 else ""]()
