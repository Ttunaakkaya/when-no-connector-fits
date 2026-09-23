"""Correct the historical estimators from saved scores, without rendering or running a VLM.

    uv run python scripts/reanalyze.py

All new files go to results/corrected. The main corrected result keeps the original frozen
thresholds and changes only reporting weights. A separate, explicitly post-hoc sensitivity fits
family-balanced thresholds on calibration families and applies them unchanged to test. Neither
is a new held-out experiment; neither replaces the historical freeze, tables or score snapshot.
"""

import argparse
import ast
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wncf import REPO_ROOT
from wncf.cache import file_sha
from wncf.metrics import (COVERAGE_TARGET, S2_DELTA_GRID, S2_TAU_GRID, Query, bootstrap_by_family,
                         calibrate_s1, calibrate_s2, evaluate, family_counts, risk_coverage)
from wncf.queries import load_queries
from wncf.rules import METHODS
from wncf.splits import EXPOSED_FAMILIES

OUT = REPO_ROOT / "results" / "corrected"
CI_METRICS = ("correct_yield", "accepted_risk", "absent_false_accept", "present_coverage", "top1_present")
BOOT_SEED = 20260924
WEIGHTING = "family"
WEIGHTING_NOTE = (
    "Each family contributes unit total weight, divided equally among its observed regimes and "
    "then among queries in each family/regime cell. Rates divide weighted count sums. "
    "Unprefixed counts are raw integers; weighted_* fields are weighted sums, not object counts."
)
LIMITATIONS = [
    "Corrective reanalysis after the historical test results were known; no new held-out evidence.",
    "original_thresholds keeps historical thresholds unchanged; posthoc_thresholds refits on calibration only.",
    "Calibration results and their fixed-threshold intervals are in-sample, not cross-validated.",
    "Intervals resample query families and retain draw multiplicities, conditional on the fixed shared distractor pool.",
    "Seven test families and diagnostic template exposure limit generalization; exposed/unexposed groups differ in shape.",
    "B1 is a unique-Yes decision rule: its top-1 value and interval are undefined.",
    "Test risk-coverage curves are descriptive sweeps, never a source of new operating points.",
]


def read_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def params_from_record(record: dict) -> dict:
    cal = record["calibration"]
    return {"B0": {}, "B1": {}, "U0": {}, "S1": {"tau": cal["S1"]["tau"]},
            "S2": {"tau": cal["S2"]["tau"], "delta": cal["S2"]["delta"]}}


def posthoc_params(queries: list[Query]) -> dict:
    if any(q.split != "calib" for q in queries):
        raise ValueError("post-hoc calibration must contain calibration queries only")
    tau = calibrate_s1(queries, weighting=WEIGHTING)
    tau2, delta = calibrate_s2(queries, tau, weighting=WEIGHTING)
    return {"B0": {}, "B1": {}, "U0": {}, "S1": {"tau": tau}, "S2": {"tau": tau2, "delta": delta}}


def groups(queries: list[Query]):
    yield "all", queries
    for label, exposed in (("exposed", True), ("unexposed", False)):
        subset = [q for q in queries if (q.family in EXPOSED_FAMILIES) == exposed]
        if subset:
            yield label, subset
    for family in sorted({q.family for q in queries}):
        yield f"family:{family}", [q for q in queries if q.family == family]


def slices(queries: list[Query]):
    yield "all", "all", queries
    for tier in ("easy", "hard"):
        subset = [q for q in queries if q.tier == tier]
        if subset:
            yield tier, "all", subset
    for regime in sorted({q.regime for q in queries}):
        yield "hard" if regime.endswith("hard") else "easy", regime, [q for q in queries if q.regime == regime]


def result_rows(queries: list[Query], params: dict, dataset: str, policy: str, boot_n: int) -> list[dict]:
    rows = []
    for method in METHODS:
        for group, grouped in groups(queries):
            for tier, regime, subset in slices(grouped):
                m = evaluate(subset, method, weighting=WEIGHTING, **params[method])
                # Individual-family intervals would resample one family and misleadingly collapse.
                cis = {f"{metric}_ci95": list(bootstrap_by_family(
                    subset, method, metric, n=boot_n, seed=BOOT_SEED, weighting=WEIGHTING, **params[method]))
                    for metric in CI_METRICS} if group == "all" else {}
                rows.append({"analysis": "corrective reanalysis after test inspection", "dataset": dataset,
                             "threshold_policy": policy, "families": group, "tier": tier, "regime": regime,
                             **m, **cis})
    return rows


def write_table(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def figure(queries: list[Query], params: dict, dataset: str, policy: str, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, tier in zip(axes, ("all", "easy", "hard")):
        subset = queries if tier == "all" else [q for q in queries if q.tier == tier]
        for method, color in (("S1", "#2166ac"), ("S2", "#b2182b")):
            kw = {"delta": params[method]["delta"]} if method == "S2" else {}
            points = [(c, r) for _, c, r, _ in risk_coverage(subset, method, weighting=WEIGHTING, **kw)
                      if r is not None]
            ax.plot([100 * c for c, _ in points], [100 * r for _, r in points], color=color,
                    lw=1.5, label=f"{method}: threshold swept")
            m = evaluate(subset, method, weighting=WEIGHTING, **params[method])
            if m["accepted_risk"] is not None:
                ax.scatter(100 * m["coverage"], 100 * m["accepted_risk"], color=color,
                           s=45, edgecolor="black", zorder=4)
        for method, marker in (("B0", "s"), ("U0", "^")):
            m = evaluate(subset, method, weighting=WEIGHTING)
            if m["accepted_risk"] is not None:
                ax.scatter(100 * m["coverage"], 100 * m["accepted_risk"], color="#444444",
                           marker=marker, s=40, label=method, zorder=3)
        ax.set_title(f"{tier} tiers" if tier == "all" else f"{tier} tier", fontsize=10)
        ax.set_xlabel("family-balanced selection coverage (%)")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("family-balanced accepted-choice risk (%)")
    axes[0].legend(fontsize=7, loc="lower right")
    threshold_label = "original frozen thresholds" if policy == "original_thresholds" else "post-hoc calibration thresholds"
    display_name = dataset.replace("_", " ").replace("exploratory prompt", "exploratory paper-prompt")
    fig.suptitle(f"Corrective reanalysis: {display_name}; {threshold_label}\n"
                 "Curves are descriptive; filled circles mark the stated operating points", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def verify_historical_table(queries: list[Query], params: dict, table: Path) -> dict:
    """Check the old pooled results before producing differently weighted results.

    The old B1 top-1 and interval are intentionally excluded: they were U0's ranking assigned to
    a rule that does not rank. New reports mark them undefined instead of perpetuating the error.
    """
    checks = 0
    group_map = dict(groups(queries))
    with table.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subset = group_map[row["families"]]
            if row["tier"] != "all":
                subset = [q for q in subset if q.tier == row["tier"]]
            m = evaluate(subset, row["method"], weighting="query", **params[row["method"]])
            for key, expected in row.items():
                if not expected or key in ("method", "families", "tier"):
                    continue
                if row["method"] == "B1" and key in ("top1_present", "top1_present_ci95"):
                    continue
                if key.endswith("_ci95"):
                    actual = bootstrap_by_family(subset, row["method"], key.removesuffix("_ci95"),
                                                 n=2000, seed=BOOT_SEED, weighting="query", **params[row["method"]])
                    expected = ast.literal_eval(expected)
                    if any((a is None) != (e is None) or
                           (a is not None and not math.isclose(a, e, rel_tol=0, abs_tol=1e-12))
                           for a, e in zip(actual, expected)):
                        raise ValueError(f"historical interval changed: {table}, {row['method']}, {key}")
                elif key in m:
                    if m[key] is None or not math.isclose(m[key], float(expected), rel_tol=0, abs_tol=1e-12):
                        raise ValueError(f"historical result changed: {table}, {row['method']}, {key}")
                else:
                    continue
                checks += 1
    return {"path": table.relative_to(REPO_ROOT).as_posix(), "checks": checks,
            "matches": True, "excluded_obsolete_metrics": ["B1 top1_present", "B1 top1_present_ci95"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=2000, help="family resamples for corrected intervals (default: 2000)")
    args = ap.parse_args()
    if args.bootstrap < 1:
        ap.error("--bootstrap must be positive")

    records = {"primary": REPO_ROOT / "results/freeze/freeze_record.json"}
    records.update({arm: REPO_ROOT / f"results/arms/{arm}/arm_calibration_record.json"
                    for arm in ("appearance", "int8", "prompt")})
    tables = {"primary_calibration": REPO_ROOT / "results/tables/calib_metrics.csv",
              "primary_test": REPO_ROOT / "results/test/test_metrics.csv",
              "exploratory_prompt_test": REPO_ROOT / "results/exploratory/prompt_test/test_metrics.csv"}
    tables.update({f"{arm}_calibration": REPO_ROOT / f"results/arms/{arm}/calib_metrics.csv"
                   for arm in ("appearance", "int8", "prompt")})
    inputs = set(records.values()) | set(tables.values()) | {
        REPO_ROOT / "results/scores/scores.jsonl", REPO_ROOT / "results/scores/snapshot_manifest.json",
        REPO_ROOT / "data/candidate_sets/blocks.csv",
        REPO_ROOT / "data/candidate_sets/candidates.csv", REPO_ROOT / "data/parts/compat.csv",
        REPO_ROOT / "results/test/test_pass_record.json", REPO_ROOT / "results/exploratory/prompt_test/record.json",
    }
    before = {p: file_sha(p) for p in sorted(inputs)}
    summary, replay, corrected_calibration, output_paths = [], [], {}, []
    for arm_name, source in records.items():
        arm = None if arm_name == "primary" else arm_name
        calibration = load_queries({"calib"}, arm, snapshot=True)
        original = params_from_record(read_record(source))
        refitted = posthoc_params(calibration)
        destination = OUT / ("primary" if arm is None else f"arms/{arm}")
        cal_record = {
            "status": "post-hoc family-balanced sensitivity, fitted on calibration only after test results were known",
            "historical_record": source.relative_to(REPO_ROOT).as_posix(), "historical_record_sha256": before[source],
            "weighting": WEIGHTING, "weighting_definition": WEIGHTING_NOTE,
            "n_queries": len(calibration), "families": family_counts(calibration),
            "coverage_target_present": COVERAGE_TARGET, "original_thresholds": original, "posthoc_thresholds": refitted,
            "S2_grid": {"tau": list(S2_TAU_GRID) + ["S1 tau"], "delta": list(S2_DELTA_GRID)},
            "tie_breaking": "minimize weighted risk; then higher weighted correct yield, higher tau, higher delta",
            "target_reached": {m: evaluate(calibration, m, weighting=WEIGHTING,
                                          **refitted[m])["present_coverage"] >= COVERAGE_TARGET for m in ("S1", "S2")},
        }
        destination.mkdir(parents=True, exist_ok=True)
        record_path = destination / "posthoc_calibration_record.json"
        record_path.write_text(json.dumps(cal_record, indent=2) + "\n", encoding="utf-8")
        output_paths.append(record_path)
        corrected_calibration[arm_name] = cal_record
        datasets = [("calibration", calibration, tables[f"{arm_name}_calibration"], destination)]
        if arm_name in ("primary", "prompt"):
            test = load_queries({"test"}, arm, snapshot=True)
            datasets.append(("test", test, tables["primary_test" if arm is None else "exploratory_prompt_test"],
                             OUT / "primary" if arm is None else OUT / "exploratory"))
        for split, queries, historical_table, folder in datasets:
            replay.append(verify_historical_table(queries, original, historical_table))
            dataset = f"{arm_name}_{split}" if arm is None or split == "calibration" else "exploratory_prompt_test"
            stem = "prompt_test" if arm == "prompt" and split == "test" else split
            for policy, params in (("original_thresholds", original), ("posthoc_thresholds", refitted)):
                rows = result_rows(queries, params, dataset, policy, args.bootstrap)
                path = folder / f"{stem}_{policy}.csv"
                write_table(rows, path)
                output_paths.append(path)
                summary.extend(r for r in rows if r["families"] == "all" and r["regime"] == "all")
                fig_path = OUT / "figures" / f"{dataset}_{policy}_risk_coverage.png"
                figure(queries, params, dataset, policy, fig_path)
                output_paths.append(fig_path)
            print(f"{dataset}: {len(queries)} cached queries; historical pooled table verified; corrected tables written")

    summary_path = OUT / "summary.csv"
    write_table(summary, summary_path)
    output_paths.append(summary_path)
    if any(file_sha(p) != sha for p, sha in before.items()):
        raise RuntimeError("an original input changed during reanalysis; do not use these outputs")
    analysis = {
        "status": "corrective reanalysis of historical scores after test inspection; not a new held-out experiment",
        "command": f"uv run python scripts/reanalyze.py --bootstrap {args.bootstrap}",
        "score_source": "audited results/scores/scores.jsonl snapshot; zero new model calls",
        "weighting": WEIGHTING, "weighting_definition": WEIGHTING_NOTE,
        "bootstrap": {"resamples": args.bootstrap, "seed": BOOT_SEED, "unit": "query family, with draw multiplicity"},
        "limitations": LIMITATIONS, "historical_replay": replay,
        "input_sha256": {p.relative_to(REPO_ROOT).as_posix(): sha for p, sha in before.items()},
        "analysis_source_sha256": {p: file_sha(REPO_ROOT / p) for p in (
            "src/wncf/metrics.py", "src/wncf/queries.py", "src/wncf/snapshot.py",
            "src/wncf/provenance.py", "scripts/reanalyze.py")},
        "outputs_sha256": {p.relative_to(REPO_ROOT).as_posix(): file_sha(p) for p in output_paths},
        "original_inputs_unchanged": True, "calibration_records": corrected_calibration,
    }
    (OUT / "analysis_record.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(output_paths)} artifacts plus analysis_record.json under results/corrected; no originals modified")


if __name__ == "__main__":
    main()
