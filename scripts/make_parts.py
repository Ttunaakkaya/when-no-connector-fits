"""Stage 1: build the procedural part catalog, its meshes and a contact sheet of all profiles.

    uv run python scripts/make_parts.py

Writes data/parts/catalog.csv (the source of truth: profiles as WKT, in mm), data/parts/meshes/*.stl
(derived, regenerated on demand) and results/figures/stage1_profiles.png.
"""

import csv
import hashlib
import math
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

from wncf import REPO_ROOT
from wncf.parts import CLEARANCE, MAX_EXTENT, build_catalog, peg_mesh, socket_mesh, wkt

OUT = REPO_ROOT / "data" / "parts"
COLS = ["part_id", "family", "params", "peg_w_mm", "peg_h_mm", "peg_area_mm2", "clearance_mm", "peg_sha", "peg_wkt", "bore_wkt"]


def main():
    parts = build_catalog()
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)

    with (OUT / "catalog.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for p in parts:
            x0, y0, x1, y1 = p.peg.bounds
            peg_wkt = wkt(p.peg)
            w.writerow({
                "part_id": p.part_id, "family": p.family, "params": p.params_json,
                "peg_w_mm": round(x1 - x0, 3), "peg_h_mm": round(y1 - y0, 3), "peg_area_mm2": round(p.peg.area, 3),
                "clearance_mm": CLEARANCE, "peg_sha": hashlib.sha256(peg_wkt.encode()).hexdigest()[:12],
                "peg_wkt": peg_wkt, "bore_wkt": wkt(p.bore),
            })
            peg_mesh(p.peg).export(OUT / "meshes" / f"{p.part_id}_peg.stl")
            socket_mesh(p.bore).export(OUT / "meshes" / f"{p.part_id}_socket.stl")

    contact_sheet(parts, REPO_ROOT / "results" / "figures" / "stage1_profiles.png")
    fam = Counter(p.family for p in parts)
    print(f"{len(parts)} parts in {len(fam)} families: " + ", ".join(f"{k} {v}" for k, v in fam.items()))
    print(f"wrote {OUT.relative_to(REPO_ROOT)}/catalog.csv, {2 * len(parts)} meshes, results/figures/stage1_profiles.png")


def contact_sheet(parts, path):
    ncol = 9
    nrow = math.ceil(len(parts) / ncol)
    lim = MAX_EXTENT / 2 + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.75))
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, p in zip(axes.flat, parts):
        plot_polygon(p.bore, ax=ax, add_points=False, facecolor="none", edgecolor="#888", linewidth=0.6, linestyle="--")
        plot_polygon(p.peg, ax=ax, add_points=False, facecolor="#4a78b5", edgecolor="#1f3b63", linewidth=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(p.part_id + (" (m)" if p.params.get("mirror") else ""), fontsize=7)
    fig.suptitle(f"Stage 1 profiles: peg (blue) and nominal bore (dashed, +{CLEARANCE} mm per side). "
                 f"Every cell is the same {2 * lim:.0f} mm square.", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
