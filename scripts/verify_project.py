"""Verify the saved-score snapshot, run records, corrected reports and README links.

    uv run python scripts/verify_project.py

No model weights, live cache or rendered input images are required. Regression tests are run
separately with pytest. This command checks the recorded artifacts that the tests do not.
"""

import csv
import json
import math
import re

from wncf import REPO_ROOT
from wncf.cache import file_sha
from wncf.metrics import evaluate
from wncf.provenance import load_calibration, thresholds, validate_test_pass
from wncf.queries import load_queries
from wncf.snapshot import verify_snapshot


def check_hashes(mapping):
    for relative, expected in mapping.items():
        path = REPO_ROOT / relative
        if not path.is_file() or file_sha(path) != expected:
            raise AssertionError(f"artifact hash mismatch: {relative}")


def main():
    validation_output = REPO_ROOT / "results/corrected/validation.json"
    snapshot = verify_snapshot()
    validate_test_pass(live=False)

    analysis = json.loads((REPO_ROOT / "results/corrected/analysis_record.json").read_text())
    for key in ("input_sha256", "analysis_source_sha256", "outputs_sha256"):
        check_hashes(analysis[key])
    aperture = json.loads((REPO_ROOT / "results/corrected/aperture/analysis_meta.json").read_text())
    check_hashes(aperture["source_sha256"])
    assert aperture["new_model_calls"] == 0 and aperture["labels_changed"] == 0

    queries = load_queries({"test"}, snapshot=True)
    params = thresholds(load_calibration(live=False))
    rows = list(csv.DictReader((REPO_ROOT / "results/corrected/primary/test_original_thresholds.csv").open(newline="")))
    checked = 0
    for row in rows:
        if (row["families"], row["tier"], row["regime"]) != ("all", "all", "all"):
            continue
        expected = evaluate(queries, row["method"], **params[row["method"]])
        for field in ("n", "selected", "correct", "wrong", "correct_yield", "accepted_risk",
                      "present_coverage", "absent_false_accept", "top1_present"):
            saved = None if row[field] == "" else float(row[field])
            if expected[field] is None:
                assert saved is None, (row["method"], field)
            else:
                assert saved is not None and math.isclose(saved, expected[field], abs_tol=1e-12), (row["method"], field)
            checked += 1
    assert checked == 45
    for path in (REPO_ROOT / "results/corrected").rglob("*.csv"):
        if path.is_relative_to(REPO_ROOT / "results/corrected/aperture"):
            continue
        for row in csv.DictReader(path.open(newline="")):
            if row.get("method") == "B1":
                assert row.get("top1_present", "") == "", f"B1 ranking in {path}"
                assert row.get("top1_present_ci95", "") in ("", "[None, None]", "[null, null]"), path

    documents = [REPO_ROOT / "README.md"]
    link_count = 0
    for path in documents:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0].strip("<>")
            if target:
                resolved = (path.parent / target).resolve()
                # This command creates its validation report only after all checks pass.
                assert resolved == validation_output.resolve() or resolved.exists(), f"broken link in {path.name}: {target}"
                link_count += 1
    report = {
        "status": "passed",
        "snapshot_source_files_verified": len(snapshot["files"]),
        "corrected_outputs_verified": len(analysis["outputs_sha256"]),
        "primary_summary_numeric_checks": checked,
        "historical_numeric_ci_checks": sum(r["checks"] for r in analysis["historical_replay"]),
        "local_document_links_verified": link_count,
        "new_model_calls": 0,
    }
    validation_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
