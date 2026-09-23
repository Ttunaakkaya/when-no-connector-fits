"""Calibrate S1/S2 on the calibration blocks, freeze the thresholds, and report calibration metrics.

    uv run python scripts/calibrate.py
    uv run python scripts/calibrate.py --arm appearance    # a predeclared sensitivity arm

An arm is calibrated inside the arm and written under results/arms/<arm>/. It never writes the
primary freeze record, so it can never unlock or alter the test pass; its trigger line is descriptive.

Declared 23 September 2026, before any calibration block was scored:

  Appearance-arm trigger (protocol.md section 7, made operational here). The red/green appearance
  arm runs if, on the calibration blocks, neither ungated selector (B0 or U0) has a 95% family-
  bootstrap lower bound for mate-present/easy top-1 above 1/3, the chance rate with K = 3.

Writes results/freeze/freeze_record.json (thresholds, procedure, configuration and manifest hashes),
results/tables/calib_metrics.csv and results/figures/calib_risk_coverage.png. Test-split numbers are
never computed here. The calibration numbers for S1 and S2 are in-sample by construction.
"""

import argparse
import csv
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wncf import REPO_ROOT
from wncf.cache import file_sha, frozen_identity
from wncf.config import ARM_OVERRIDES, ARM_SPLITS, FROZEN, arm_config
from wncf.metrics import (COVERAGE_TARGET, S2_DELTA_GRID, S2_TAU_GRID, Query, bootstrap_by_family, calibrate_s1,
                          calibrate_s2, evaluate, evaluate_by_tier, family_counts, fmt, risk_coverage)
from wncf.queries import MissingScores
from wncf.queries import load_queries as _load_queries
from wncf.rules import METHODS
from wncf.splits import EXPOSED_FAMILIES

BLOCKS = REPO_ROOT / "data" / "candidate_sets" / "blocks.csv"
FREEZE = REPO_ROOT / "results" / "freeze" / "freeze_record.json"
CHANCE_TOP1 = 1 / 3
BOOT_N = 1000


def load_queries(splits: set[str], arm: str | None = None) -> list[Query]:
    try:
        return _load_queries(splits, arm)
    except MissingScores as e:
        sys.exit(str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARM_OVERRIDES))
    arm = ap.parse_args().arm
    arm_dir = REPO_ROOT / "results" / "arms" / arm if arm else None
    queries = load_queries(set(ARM_SPLITS) if arm else {"calib"}, arm)
    s1_tau = calibrate_s1(queries)
    s2_tau, s2_delta = calibrate_s2(queries, s1_tau)
    params = {"B0": {}, "B1": {}, "U0": {}, "S1": {"tau": s1_tau}, "S2": {"tau": s2_tau, "delta": s2_delta}}

    rows = []
    for method in METHODS:
        for group, subset in (("all", queries),
                              ("exposed", [q for q in queries if q.family in EXPOSED_FAMILIES]),
                              ("unexposed", [q for q in queries if q.family not in EXPOSED_FAMILIES])):
            if not subset:
                continue
            for tier, m in evaluate_by_tier(subset, method, **params[method]).items():
                rows.append({"method": method, "families": group, "tier": tier, **m})

    easy_present = [q for q in queries if q.regime == "present_easy"]
    trigger = {}
    for method in ("B0", "U0"):
        lo, hi = bootstrap_by_family(easy_present, method, "top1_present", n=BOOT_N)
        trigger[method] = {"top1": evaluate(easy_present, method)["top1_present"], "ci95": [lo, hi]}
    fires = not any(t["ci95"][0] is not None and t["ci95"][0] > CHANCE_TOP1 for t in trigger.values())

    cis = {}
    for method in METHODS:
        for metric in ("correct_yield", "accepted_risk", "absent_false_accept", "top1_present"):
            cis[method, metric] = bootstrap_by_family(queries, method, metric, n=BOOT_N, **params[method])

    record_path = arm_dir / "arm_calibration_record.json" if arm else FREEZE
    table_path = arm_dir / "calib_metrics.csv" if arm else REPO_ROOT / "results" / "tables" / "calib_metrics.csv"
    fig_path = arm_dir / "calib_risk_coverage.png" if arm else REPO_ROOT / "results" / "figures" / "calib_risk_coverage.png"
    write_freeze(queries, params, trigger, fires, record_path, arm)
    write_table(rows, table_path)
    figure(queries, s2_delta, params, fig_path, arm)
    report(queries, rows, params, trigger, fires, cis, record_path, arm)


def write_freeze(queries, params, trigger, fires, path, arm):
    reached = {m: (evaluate(queries, m, **params[m])["present_coverage"] or 0) >= COVERAGE_TARGET for m in ("S1", "S2")}
    record = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arm": arm, "primary": arm is None,
        "config": {**arm_config(arm), "views": list(FROZEN["views"]), **frozen_identity(arm),
                   **({"appearance": "paper look (config.PAPER_LOOK)"} if arm == "appearance" else {})},
        "candidates_sha256": file_sha(REPO_ROOT / "data" / "candidate_sets" / "candidates.csv"),
        "compat_sha256": file_sha(REPO_ROOT / "data" / "parts" / "compat.csv"),
        "calibration": {
            "split": "calib", "n_queries": len(queries), "families": family_counts(queries),
            "coverage_target_present": COVERAGE_TARGET,
            "S1": {"tau": params["S1"]["tau"], "rule": "highest observed breakpoint meeting the target"},
            "S2": {"tau": params["S2"]["tau"], "delta": params["S2"]["delta"],
                   "grid": {"tau": list(S2_TAU_GRID) + ["S1 tau"], "delta": list(S2_DELTA_GRID)},
                   "rule": "meet target, minimise accepted-choice risk; ties: higher yield, higher tau, higher delta"},
            "target_reached": reached,
            "one_policy_across_tiers": True,
        },
        "appearance_arm_trigger": {
            "definition": "fires if neither B0 nor U0 has a 95% family-bootstrap lower bound for "
                          "present/easy top-1 above 1/3 on calibration blocks",
            "results": trigger, "fires": fires,
            **({"note": "descriptive only inside an arm"} if arm else {}),
        },
    }
    if arm is None:
        # only the primary record carries the manifest hash that unlocks the test split
        record["blocks_sha256"] = file_sha(BLOCKS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def write_table(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def figure(queries, s2_delta, params, path, arm):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (name, subset) in zip(axes, (("all tiers", queries),
                                          ("easy tier", [q for q in queries if q.tier == "easy"]),
                                          ("hard tier", [q for q in queries if q.tier == "hard"]))):
        for method, kw, colour in (("S1", {}, "#2166ac"), ("S2", {"delta": s2_delta}, "#b2182b")):
            pts = [(c, r) for _, c, r, _ in risk_coverage(subset, method, **kw) if r is not None]
            ax.plot([c for c, _ in pts], [r for _, r in pts], color=colour, lw=1.4,
                    label=f"{method} (tau swept{', delta ' + str(s2_delta) if method == 'S2' else ''})")
            m = evaluate(subset, method, **params[method])
            if m["accepted_risk"] is not None:
                ax.scatter([m["coverage"]], [m["accepted_risk"]], color=colour, zorder=3, s=36, edgecolor="black")
        for method, marker in (("B0", "s"), ("U0", "^")):
            m = evaluate(subset, method)
            ax.scatter([m["coverage"]], [m["accepted_risk"]], marker=marker, color="#555", s=36, label=method)
        ax.set_title(f"Calibration, {name}", fontsize=10)
        ax.set_xlabel("selection coverage")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accepted-choice risk")
    axes[0].legend(fontsize=7)
    fig.suptitle((f"{arm.capitalize()} arm: " if arm else "") + "descriptive risk-coverage on calibration blocks; "
                 "filled circles are the calibrated operating points (in-sample)", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


def report(queries, rows, params, trigger, fires, cis, record_path, arm):
    fam = family_counts(queries)
    print(f"calibration{f' ({arm} arm)' if arm else ''}: {len(queries)} blocks from {len(fam)} families {fam}")
    print(f"frozen: S1 tau={params['S1']['tau']:.4f}; S2 tau={params['S2']['tau']:.4f}, delta={params['S2']['delta']}")
    head = ["method", "tier", "n", "select", "correct", "wrong", "defer", "fail", "cover", "risk", "yield", "absFA", "top1"]
    print("\n" + " ".join(f"{h:>7s}" for h in head))
    for r in rows:
        if r["families"] != "all":
            continue
        vals = [r["method"], r["tier"], f"{r['n']:.0f}", f"{r['selected']:.0f}", f"{r['correct']:.0f}",
                f"{r['wrong']:.0f}", f"{r['deferred']:.0f}", f"{r['failures']:.0f}", fmt(r["coverage"], 2),
                fmt(r["accepted_risk"], 2), fmt(r["correct_yield"], 2), fmt(r["absent_false_accept"], 2),
                fmt(r["top1_present"], 2)]
        print(" ".join(f"{v:>7s}" for v in vals))
    print("\n95% family-bootstrap intervals, all tiers:")
    for (method, metric), (lo, hi) in cis.items():
        print(f"  {method} {metric:20s} [{fmt(lo, 2)}, {fmt(hi, 2)}]")
    print("\nappearance-arm trigger: " + "; ".join(
        f"{m} present/easy top-1 {fmt(t['top1'], 2)} (95% [{fmt(t['ci95'][0], 2)}, {fmt(t['ci95'][1], 2)}])"
        for m, t in trigger.items()) + f" -> {'FIRES' if fires else 'does not fire'} (chance 0.33)")
    print(f"{'arm record' if arm else 'freeze record'}: {record_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
