"""Stage 3: render every catalog part (peg and socket, views v1 and v2) and check each image.

    uv run python scripts/render_parts.py

Writes data/renders/procedural/<part>_<peg|socket>_<view>.png (derived, not committed),
data/renders/procedural/manifest.csv (paths, SHA-256, geometry of the camera), render_settings.json
and results/figures/stage3_contact_sheet.png (the first part of every family).

Checks on every image, so a wrong scale, a clipped part or an invisible opening stops the run:
- the one-pixel border is pure background (nothing clipped at the frame edge)
- v1 peg: the silhouette area matches the profile area (so the pixel size is what we record)
- v1 socket: the dark cavity-floor area matches the bore area (the opening is visible and to scale)
"""

import csv
import hashlib
import json
import time

import numpy as np
from PIL import Image, ImageDraw

from wncf import REPO_ROOT
from wncf.parts import build_catalog
from wncf.render import MM_PER_PX, VIEWS, Renderer, peg_scene, settings, settings_sha, socket_scene

OUT = REPO_ROOT / "data" / "renders" / "procedural"
AREA_TOL = 0.04  # relative; anti-aliased edges on small, long-perimeter shapes
FG_LUMA = 197  # halfway between the white background and the lit part (unbiased edge counting)
DARK_LUMA = 88  # halfway between the dark cavity floor and the lit block top


def luma(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32).mean(-1)


def check(img, role, view, part) -> dict:
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    if (border < 250).any():
        raise AssertionError(f"{part.part_id} {role} {view}: part touches the frame edge")
    out = {}
    if view == "v1":
        if role == "peg":
            measured, true = (luma(img) < FG_LUMA).sum() * MM_PER_PX**2, part.peg.area
        else:
            measured, true = (luma(img) < DARK_LUMA).sum() * MM_PER_PX**2, part.bore.area
        err = measured / true - 1
        if abs(err) > AREA_TOL:
            raise AssertionError(f"{part.part_id} {role} v1: pixel area {measured:.1f} vs {true:.1f} mm2")
        out["area_err"] = err
    return out


def main():
    parts = build_catalog()
    OUT.mkdir(parents=True, exist_ok=True)
    r = Renderer()
    rows, errs, t0 = [], [], time.perf_counter()
    images = {}
    for p in parts:
        for role, scene in (("peg", peg_scene(p.peg)), ("socket", socket_scene(p.bore))):
            for view in VIEWS:
                img = r.render(scene, view)
                errs.append(check(img, role, view, p).get("area_err"))
                path = OUT / f"{p.part_id}_{role}_{view}.png"
                Image.fromarray(img).save(path)
                images[p.part_id, role, view] = img
                rows.append({
                    "part_id": p.part_id, "family": p.family, "role": role, "view": view,
                    "path": path.relative_to(REPO_ROOT).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "image_px": img.shape[0], "mm_per_px": round(MM_PER_PX, 6), "tilt_deg": VIEWS[view],
                    "settings_sha": settings_sha(),
                })
    r.close()
    elapsed = time.perf_counter() - t0

    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "render_settings.json").write_text(json.dumps(settings(), indent=2), encoding="utf-8")

    first_of_family = [p for i, p in enumerate(parts) if i == 0 or p.family != parts[i - 1].family]
    contact_sheet(first_of_family, images, REPO_ROOT / "results" / "figures" / "stage3_contact_sheet.png")

    area = np.array([e for e in errs if e is not None])
    print(f"rendered {len(rows)} images in {elapsed:.1f}s ({elapsed / len(rows) * 1000:.0f} ms each), "
          f"{MM_PER_PX:.4f} mm/px, settings {settings_sha()}")
    print(f"v1 area check: {len(area)} images, error {area.min():+.2%} to {area.max():+.2%}; all borders clear")


def contact_sheet(parts, images, path, thumb=192, blocks=2):
    per_block = -(-len(parts) // blocks)
    label_w = 110
    sheet = Image.new("RGB", (blocks * (label_w + 4 * thumb), per_block * thumb + 24), "white")
    d = ImageDraw.Draw(sheet)
    for b in range(blocks):
        x0 = b * (label_w + 4 * thumb)
        for j, head in enumerate(["peg v1", "peg v2", "socket v1", "socket v2"]):
            d.text((x0 + label_w + j * thumb + 6, 6), head, fill="black")
    for i, p in enumerate(parts):
        b, row = divmod(i, per_block)
        x0, y0 = b * (label_w + 4 * thumb), 24 + row * thumb
        d.text((x0 + 6, y0 + thumb // 2 - 6), p.part_id, fill="black")
        for j, (role, view) in enumerate([("peg", "v1"), ("peg", "v2"), ("socket", "v1"), ("socket", "v2")]):
            tile = Image.fromarray(images[p.part_id, role, view]).resize((thumb, thumb), Image.LANCZOS)
            sheet.paste(tile, (x0 + label_w + j * thumb, y0))
            d.rectangle([x0 + label_w + j * thumb, y0, x0 + label_w + (j + 1) * thumb - 1, y0 + thumb - 1], outline="#dddddd")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


if __name__ == "__main__":
    main()
