"""Draw one crop box per photo by hand and write it to data/manifest.csv.

    uv run python scripts/crop.py            # every photo not yet in the manifest
    uv run python scripts/crop.py --session s2
    uv run python scripts/crop.py --redo s1/usba_socket_02_v1.jpg

In the window: drag a box, then ENTER or SPACE to accept, 'c' to skip this photo.
Ctrl+C in the terminal stops; the manifest is saved after every accepted box, so rerunning resumes.
"""

import argparse

import cv2

from wncf import REPO_ROOT
from wncf.data import RAW, parse_photo_name, read_manifest, write_manifest

MAX_W, MAX_H = 1400, 900  # display size; boxes are mapped back to full resolution
EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="only this session folder, e.g. s1")
    ap.add_argument("--redo", nargs="*", default=[], help="paths relative to data/raw to re-crop")
    args = ap.parse_args()

    rows = {r["path"]: r for r in read_manifest()}
    redo = {f"data/raw/{p}".replace("\\", "/") for p in args.redo}
    sessions = [RAW / args.session] if args.session else sorted(p for p in RAW.iterdir() if p.is_dir())
    photos = [p for s in sessions for p in sorted(s.iterdir()) if p.suffix.lower() in EXTS]

    todo = []
    for p in photos:
        rel = p.relative_to(REPO_ROOT).as_posix()
        parse_photo_name(p)  # fail early on a bad name, before any clicking
        if rel not in rows or rel in redo:
            todo.append((p, rel))
    print(f"{len(photos)} photos, {len(todo)} to crop")

    for i, (p, rel) in enumerate(todo, 1):
        img = cv2.imread(str(p))  # applies EXIF rotation, as wncf.data.open_upright does
        h, w = img.shape[:2]
        scale = min(1.0, MAX_W / w, MAX_H / h)
        shown = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
        title = f"[{i}/{len(todo)}] {rel}  (ENTER accept, c skip)"
        x, y, bw, bh = cv2.selectROI(title, shown, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        if bw == 0 or bh == 0:
            print(f"skipped {rel}")
            continue

        n = parse_photo_name(p)
        rows[rel] = {
            "object_id": n.object_id, "type": n.type, "role": n.role, "specimen": n.specimen,
            "session": p.parent.name, "view": n.view, "path": rel,
            "crop_x": round(x / scale), "crop_y": round(y / scale),
            "crop_w": round(bw / scale), "crop_h": round(bh / scale),
        }
        write_manifest(list(rows.values()))
        print(f"{rel}: box {rows[rel]['crop_w']}x{rows[rel]['crop_h']}")

    print(f"manifest: {len(rows)} rows")


if __name__ == "__main__":
    main()
