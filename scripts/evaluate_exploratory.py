"""EXPLORATORY: evaluate a sensitivity arm on the test split, after the primary test pass.

    uv run python scripts/evaluate_exploratory.py --arm prompt

Added 23 September 2026 after the primary test pass (results/test/) was complete and known, so every
number here is exploratory, not confirmatory. The arm's S1/S2 thresholds come from that arm's own
calibration record (results/arms/<arm>/arm_calibration_record.json), fitted on calibration blocks
before any test score for the arm existed; nothing is fitted on test. scripts/evaluate.py, which
produced the primary result, is left byte-identical to the hash recorded before the test pass.

Writes results/exploratory/<arm>_test/test_metrics.csv and record.json, and
results/figures/exploratory_<arm>_test_risk_coverage.png.
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
from wncf.config import ARM_OVERRIDES
from wncf.metrics import bootstrap_by_family, evaluate, evaluate_by_tier, family_counts, fmt, risk_coverage
from wncf.queries import MissingScores, load_queries
from wncf.rules import METHODS
from wncf.splits import EXPOSED_FAMILIES

TEST_PASS = REPO_ROOT / "results" / "test" / "test_pass_record.json"
BOOT_N = 2000
CI_METRICS = ("correct_yield", "accepted_risk", "absent_false_accept", "present_coverage", "top1_present")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARM_OVERRIDES))
    arm = ap.parse_args().arm
    if not TEST_PASS.exists():
        sys.exit("the primary test pass must exist before any exploratory test evaluation")
    record_path = REPO_ROOT / "results" / "arms" / arm / "arm_calibration_record.json"
    cal = json.loads(record_path.read_text(encoding="utf-8"))["calibration"]
    params = {"B0": {}, "B1": {}, "U0": {}, "S1": {"tau": cal["S1"]["tau"]},
              "S2": {"tau": cal["S2"]["tau"], "delta": cal["S2"]["delta"]}}
    try:
        queries = load_queries({"test"}, arm)
    except MissingScores as e:
        sys.exit(str(e))

    rows = []
    for method in METHODS:
        for group, subset in (("all", queries),
                              ("exposed", [q for q in queries if q.family in EXPOSED_FAMILIES]),
                              ("unexposed", [q for q in queries if q.family not in EXPOSED_FAMILIES])):
            for tier, m in evaluate_by_tier(subset, method, **params[method]).items():
                tier_q = subset if tier == "all" else [q for q in subset if q.tier == tier]
                cis = {f"{k}_ci95": list(bootstrap_by_family(tier_q, method, k, n=BOOT_N, **params[method]))
                       for k in CI_METRICS} if group == "all" else {}
                rows.append({"method": method, "families": group, "tier": tier, **m, **cis})

    out = REPO_ROOT / "results" / "exploratory" / f"{arm}_test"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    (out / "record.json").write_text(json.dumps({
        "status": "EXPLORATORY: run after the primary test pass was complete and known",
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "arm": arm,
        "thresholds_from": str(record_path.relative_to(REPO_ROOT)), "thresholds": params,
        "arm_record_sha256": file_sha(record_path), "identity": frozen_identity(arm),
        "n_queries": len(queries), "families": family_counts(queries),
    }, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (name, subset) in zip(axes, (("all tiers", queries), ("easy tier", [q for q in queries if q.tier == "easy"]),
                                          ("hard tier", [q for q in queries if q.tier == "hard"]))):
        for method, colour in (("S1", "#2166ac"), ("S2", "#b2182b")):
            kw = {"delta": params["S2"]["delta"]} if method == "S2" else {}
            pts = [(c, r) for _, c, r, _ in risk_coverage(subset, method, **kw) if r is not None]
            ax.plot([c for c, _ in pts], [r for _, r in pts], color=colour, lw=1.4, label=f"{method} (tau swept)")
            m = evaluate(subset, method, **params[method])
            if m["accepted_risk"] is not None:
                ax.scatter([m["coverage"]], [m["accepted_risk"]], color=colour, s=40, edgecolor="black", zorder=3)
        for method, marker in (("B0", "s"), ("U0", "^")):
            m = evaluate(subset, method)
            ax.scatter([m["coverage"]], [m["accepted_risk"]], marker=marker, color="#555", s=40, label=method)
        ax.set_title(f"EXPLORATORY {arm} arm, test, {name}", fontsize=10)
        ax.set_xlabel("selection coverage")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accepted-choice risk")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Exploratory: the {arm} arm on test after the primary test pass; filled circles use the arm's "
                 "calibration-split thresholds", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "results" / "figures" / f"exploratory_{arm}_test_risk_coverage.png", dpi=140)

    print(f"EXPLORATORY {arm} arm on test: {len(queries)} blocks {family_counts(queries)}")
    print(f"thresholds from the arm's calibration: {params['S1']} / {params['S2']}")
    head = ["method", "families", "tier", "select", "correct", "wrong", "defer", "presCov", "risk", "yield", "absFA", "top1"]
    print(" ".join(f"{h:>8s}" for h in head))
    for r in rows:
        vals = [r["method"], r["families"], r["tier"], f"{r['selected']:.0f}", f"{r['correct']:.0f}", f"{r['wrong']:.0f}",
                f"{r['deferred']:.0f}", fmt(r["present_coverage"], 2), fmt(r["accepted_risk"], 2),
                fmt(r["correct_yield"], 2), fmt(r["absent_false_accept"], 2), fmt(r["top1_present"], 2)]
        print(" ".join(f"{v:>8s}" for v in vals))
    print("\n95% family-bootstrap intervals (all families):")
    for r in rows:
        if r["families"] == "all":
            print(f"  {r['method']:2s} {r['tier']:4s}  " + "  ".join(
                f"{k.removesuffix('_ci95')} [{fmt(r[k][0], 2)}, {fmt(r[k][1], 2)}]" for k in r if k.endswith("_ci95")))


if __name__ == "__main__":
    main()
