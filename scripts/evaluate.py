"""Evaluate the test split once, with the thresholds frozen on calibration data.

    uv run python scripts/evaluate.py

Written 23 September 2026 before any test block was scored. It never calibrates: S1 and S2 take their
thresholds from results/freeze/freeze_record.json, and it refuses to run if that record was made for a
different blocks manifest. Every method is reported per difficulty tier and separately for query
families the eight-shape reconstruction exposed and those it did not (decisions.md D12), with 95%
family-bootstrap intervals. Risk-coverage curves are descriptive: they sweep tau with S2's delta
frozen and are never used to choose an operating point.

Writes results/test/test_metrics.csv, results/test/test_pass_record.json and
results/figures/test_risk_coverage.png.
"""

import csv
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wncf import REPO_ROOT
from wncf.cache import file_sha, frozen_identity
from wncf.metrics import bootstrap_by_family, evaluate, evaluate_by_tier, family_counts, fmt, risk_coverage
from wncf.queries import BLOCKS, MissingScores, load_queries
from wncf.rules import METHODS
from wncf.splits import EXPOSED_FAMILIES

FREEZE = REPO_ROOT / "results" / "freeze" / "freeze_record.json"
OUT = REPO_ROOT / "results" / "test"
BOOT_N = 2000
CI_METRICS = ("correct_yield", "accepted_risk", "absent_false_accept", "present_coverage", "top1_present")


def frozen_params() -> dict:
    record = json.loads(FREEZE.read_text(encoding="utf-8"))
    if record.get("blocks_sha256") != file_sha(BLOCKS):
        sys.exit("the freeze record was made for a different blocks manifest; refusing to evaluate")
    cal = record["calibration"]
    return {"B0": {}, "B1": {}, "U0": {}, "S1": {"tau": cal["S1"]["tau"]},
            "S2": {"tau": cal["S2"]["tau"], "delta": cal["S2"]["delta"]}}


def main():
    params = frozen_params()
    try:
        queries = load_queries({"test"})
    except MissingScores as e:
        sys.exit(str(e))

    groups = (("all", queries),
              ("exposed", [q for q in queries if q.family in EXPOSED_FAMILIES]),
              ("unexposed", [q for q in queries if q.family not in EXPOSED_FAMILIES]))
    rows = []
    for method in METHODS:
        for group, subset in groups:
            if not subset:
                continue
            for tier, m in evaluate_by_tier(subset, method, **params[method]).items():
                tier_q = subset if tier == "all" else [q for q in subset if q.tier == tier]
                cis = {f"{k}_ci95": bootstrap_by_family(tier_q, method, k, n=BOOT_N, **params[method])
                       for k in CI_METRICS} if group == "all" else {}
                rows.append({"method": method, "families": group, "tier": tier, **m,
                             **{k: list(v) for k, v in cis.items()}})

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT / "test_pass_record.json").write_text(json.dumps({
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "freeze_record_sha256": file_sha(FREEZE), "blocks_sha256": file_sha(BLOCKS),
        "identity": frozen_identity(), "thresholds": params,
        "n_queries": len(queries), "families": family_counts(queries),
        "exposed_families": sorted(EXPOSED_FAMILIES & set(family_counts(queries))),
    }, indent=2), encoding="utf-8")
    figure(queries, params)
    report(queries, rows)


def figure(queries, params):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (name, subset) in zip(axes, (("all tiers", queries),
                                          ("easy tier", [q for q in queries if q.tier == "easy"]),
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
        ax.set_title(f"Test, {name}", fontsize=10)
        ax.set_xlabel("selection coverage")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accepted-choice risk")
    axes[0].legend(fontsize=7)
    fig.suptitle("Test split, descriptive risk-coverage; filled circles are the operating points frozen on "
                 "calibration data", fontsize=9)
    fig.tight_layout()
    path = REPO_ROOT / "results" / "figures" / "test_risk_coverage.png"
    fig.savefig(path, dpi=140)


def report(queries, rows):
    fam = family_counts(queries)
    print(f"test: {len(queries)} blocks from {len(fam)} families {fam}")
    head = ["method", "families", "tier", "n", "select", "correct", "wrong", "defer", "fail",
            "cover", "presCov", "risk", "yield", "absFA", "top1"]
    print(" ".join(f"{h:>8s}" for h in head))
    for r in rows:
        vals = [r["method"], r["families"], r["tier"], f"{r['n']:.0f}", f"{r['selected']:.0f}",
                f"{r['correct']:.0f}", f"{r['wrong']:.0f}", f"{r['deferred']:.0f}", f"{r['failures']:.0f}",
                fmt(r["coverage"], 2), fmt(r["present_coverage"], 2), fmt(r["accepted_risk"], 2),
                fmt(r["correct_yield"], 2), fmt(r["absent_false_accept"], 2), fmt(r["top1_present"], 2)]
        print(" ".join(f"{v:>8s}" for v in vals))
    print("\n95% family-bootstrap intervals (all families):")
    for r in rows:
        if r["families"] == "all":
            cis = "  ".join(f"{k.removesuffix('_ci95')} [{fmt(r[k][0], 2)}, {fmt(r[k][1], 2)}]"
                            for k in r if k.endswith("_ci95"))
            print(f"  {r['method']:2s} {r['tier']:4s}  {cis}")


if __name__ == "__main__":
    main()
