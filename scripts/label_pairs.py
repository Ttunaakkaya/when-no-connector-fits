"""Stage 2: label every peg-socket pair in the part catalog with the fit oracle.

    uv run python scripts/label_pairs.py

Writes data/parts/compat.csv (one row per ordered pair), data/parts/compat_meta.json (contract,
tolerances, catalog hash) and results/figures/stage2_labels.png.
"""

import csv
import hashlib
import json
import time
from collections import Counter
from importlib.metadata import version

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import ListedColormap

from wncf import REPO_ROOT
from wncf.geometry import BISECT_TOL, EPSILON, LOOSE, MARGIN_CAP, QUARTER_TURNS, RASTER_H, label_matrix
from wncf.parts import build_catalog

OUT = REPO_ROOT / "data" / "parts"
COLS = ["peg_id", "socket_id", "label", "reason", "margin_mm", "max_gap_mm", "best_rot", "fit_rots", "loose",
        "nominal", "same_family"]
CATEGORIES = [  # (name, colour) for the figure
    ("incompatible", "#e8e8e8"),
    ("compatible, snug", "#1b7837"),
    ("compatible, loose", "#a6dba0"),
    ("ambiguous: boundary", "#f1a340"),
    ("ambiguous: off-centre", "#998ec3"),
]


def main():
    parts = build_catalog()
    t0 = time.perf_counter()
    labels = label_matrix({p.part_id: p.peg for p in parts}, {p.part_id: p.bore for p in parts})
    elapsed = time.perf_counter() - t0
    family = {p.part_id: p.family for p in parts}

    with (OUT / "compat.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for (pid, sid), lab in labels.items():
            w.writerow({
                "peg_id": pid, "socket_id": sid, "label": lab.label, "reason": lab.reason,
                "margin_mm": round(lab.margin_mm, 4), "max_gap_mm": round(lab.max_gap_mm, 4), "best_rot": lab.best_rot,
                "fit_rots": " ".join(map(str, lab.fit_rots)), "loose": int(lab.loose),
                "nominal": int(pid == sid), "same_family": int(family[pid] == family[sid]),
            })

    meta = {
        "contract": "constant-profile rigid parts; peg centroid on bore centroid; quarter turns 0/90/180/270 "
                    "before straight insertion; off-centre fits are ambiguous so labels hold with or without a nudge",
        "epsilon_mm": EPSILON, "loose_mm": LOOSE, "margin_cap_mm": MARGIN_CAP, "bisect_tol_mm": BISECT_TOL,
        "shift_search_grid_mm": RASTER_H, "quarter_turns": QUARTER_TURNS,
        "catalog_sha256": hashlib.sha256((OUT / "catalog.csv").read_bytes()).hexdigest(),
        "versions": {"shapely": shapely.__version__, "geos": shapely.geos_version_string, "scipy": version("scipy")},
        "n_pairs": len(labels), "seconds": round(elapsed, 1),
    }
    (OUT / "compat_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    figure(parts, labels, REPO_ROOT / "results" / "figures" / "stage2_labels.png")
    summary(parts, labels, family, elapsed)


def category(lab) -> int:
    if lab.label == "compatible":
        return 2 if lab.loose else 1
    if lab.label == "ambiguous":
        return 3 if lab.reason == "boundary" else 4
    return 0


def figure(parts, labels, path):
    ids = [p.part_id for p in parts]
    grid = np.array([[category(labels[pid, sid]) for sid in ids] for pid in ids])
    fig, ax = plt.subplots(figsize=(12, 12.6))
    ax.imshow(grid, cmap=ListedColormap([c for _, c in CATEGORIES]), vmin=0, vmax=len(CATEGORIES) - 1, interpolation="nearest")
    edges = [i - 0.5 for i in range(1, len(parts)) if parts[i].family != parts[i - 1].family]
    for e in edges:
        ax.axhline(e, color="white", lw=1.2)
        ax.axvline(e, color="white", lw=1.2)
    ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=5)
    ax.set_yticks(range(len(ids)), ids, fontsize=5)
    ax.set_xlabel("socket")
    ax.set_ylabel("peg")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in CATEGORIES]
    ax.legend(handles, [n for n, _ in CATEGORIES], loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=5, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def summary(parts, labels, family, elapsed):
    c = Counter((lab.label, lab.reason) for lab in labels.values())
    print(f"{len(labels)} pairs labelled in {elapsed:.1f}s")
    for (lab, reason), n in sorted(c.items()):
        print(f"  {lab:12s} {reason:10s} {n}")
    nominal = [labels[p.part_id, p.part_id] for p in parts]
    print(f"nominal pairs compatible: {sum(l.label == 'compatible' for l in nominal)}/{len(nominal)}, "
          f"margin {min(l.margin_mm for l in nominal):.3f}-{max(l.margin_mm for l in nominal):.3f} mm, "
          f"largest gap {min(l.max_gap_mm for l in nominal):.3f}-{max(l.max_gap_mm for l in nominal):.3f} mm")
    cross = [(k, v) for k, v in labels.items() if k[0] != k[1] and v.label == "compatible"]
    snug = [(k, v) for k, v in cross if not v.loose]
    print(f"non-nominal compatible pairs: {len(cross)} ({len(cross) - len(snug)} loose, {len(snug)} snug)")
    for (pid, sid), v in snug:
        print(f"  snug cross-pair: peg {pid} -> socket {sid}  largest gap {v.max_gap_mm:.3f} turn {v.best_rot}")
    for kind, keep in (("any", lambda v: v.label == "compatible"), ("snug", lambda v: v.label == "compatible" and not v.loose)):
        per_peg = Counter({p.part_id: 0 for p in parts})
        per_peg.update(pid for (pid, _), v in labels.items() if keep(v))
        vals = sorted(per_peg.values())
        print(f"{kind} compatible sockets per peg: min {vals[0]}, median {vals[len(vals) // 2]}, "
              f"max {vals[-1]} ({max(per_peg, key=per_peg.get)})")
    mirrors = [(p, q) for p in parts for q in parts
               if p.family == q.family and p.part_id < q.part_id and q.params.get("mirror")
               and {k: v for k, v in q.params.items() if k != "mirror"} == p.params]
    for p, q in mirrors:
        print(f"  mirror pair {p.part_id} / {q.part_id}: {labels[p.part_id, q.part_id].label}")


if __name__ == "__main__":
    main()
