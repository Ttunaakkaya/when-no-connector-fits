"""One-time integrity record of the saved scores and model inputs (results/scores/snapshot_manifest.json).

Created after the original results, so it is not a preregistration. It refuses to replace an
existing manifest; never rerun it to get past a failed verification.
"""

import csv
import json

from wncf import REPO_ROOT
from wncf.cache import file_sha
from wncf.config import arm_image_path


def main():
    paths = ["results/scores/scores.jsonl", "results/freeze/freeze_record.json",
             "results/test/test_pass_record.json", "data/candidate_sets/blocks.csv",
             "data/candidate_sets/candidates.csv", "data/parts/compat.csv", "data/parts/catalog.csv",
             "data/parts/splits.csv", "data/aperture/pairs.csv", "results/aperture/aperture_scores.csv", "data/renders/procedural/manifest.csv",
             "data/renders/procedural/render_settings.json", "data/renders/arm_appearance/manifest.csv", "uv.lock"]
    paths += [f"results/arms/{arm}/arm_calibration_record.json" for arm in ("appearance", "prompt", "int8")]
    images = {}
    for manifest in ("data/candidate_sets/blocks.csv", "data/aperture/pairs.csv"):
        for row in csv.DictReader((REPO_ROOT / manifest).open(newline="")):
            for key in ("peg_v1", "peg_v2", "socket_v1", "socket_v2"):
                path = row[key]
                images[path] = file_sha(REPO_ROOT / path)
                if manifest.endswith("blocks.csv") and row["split"] == "calib":
                    appearance = arm_image_path(path, "appearance")
                    images[appearance] = file_sha(REPO_ROOT / appearance)
    output = REPO_ROOT / "results/scores/snapshot_manifest.json"
    if output.exists():
        raise SystemExit("snapshot manifest already exists; it must not be silently replaced")
    output.write_text(json.dumps({
        "schema_version": 1,
        "status": "Post-audit integrity record, created after the original results; not a preregistration",
        "files": {path: file_sha(REPO_ROOT / path) for path in paths},
        "images": images,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Recorded {len(paths)} source artifacts and {len(images)} input images in {output}")


if __name__ == "__main__":
    main()
