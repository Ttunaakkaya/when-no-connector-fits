"""Corrected, descriptive aperture analysis from the saved scores; no model calls or relabelling.

    uv run python scripts/analyze_aperture.py

The historical offset plots and results are preserved. New outputs under results/corrected/aperture
use measured margins, with the oracle's -2 mm cap treated as left censoring. This follow-up was
specified after seeing the original results; its bins and summaries are descriptive, not new
calibration choices. The threshold remains the original primary S1 threshold. Applied to one pair,
it is a threshold-response diagnostic, not the full three-candidate selection method.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from wncf import REPO_ROOT
from wncf.snapshot import verify_snapshot

SOURCE = REPO_ROOT / "results/aperture/aperture_scores.csv"
FREEZE = REPO_ROOT / "results/freeze/freeze_record.json"
SNAPSHOT = REPO_ROOT / "results/scores/scores.jsonl"
SNAPSHOT_MANIFEST = REPO_ROOT / "results/scores/snapshot_manifest.json"
OUT = REPO_ROOT / "results/corrected/aperture"
# This is the cap used to create the saved labels, not a new oracle setting.
SAVED_MARGIN_CAP_MM = 2.0
BOOTSTRAP_SEED = 20260923
BOOTSTRAP_REPS = 2000
SCORE_STEP_TOL = 1e-12
BIN_EDGES = (-2.0, -1.0, -0.5, -0.15, -0.05, 0.05, 0.15, math.inf)
BIN_LABELS = ("(-2,-1)", "[-1,-0.5)", "[-0.5,-0.15)", "[-0.15,-0.05)",
              "[-0.05,0.05)", "[0.05,0.15)", "[0.15,+inf)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_rows(raw_rows: list[dict], tau: float) -> list[dict]:
    """Parse saved observations, flag censored margins, and apply the unchanged inclusive gate."""
    rows = []
    if not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("the frozen threshold must be finite and between zero and one")
    for raw in raw_rows:
        r = dict(raw)
        for field in ("offset_mm", "margin_mm", "p_yes"):
            r[field] = float(r[field])
            if not math.isfinite(r[field]):
                raise ValueError(f"non-finite {field} for {r.get('cand_id')}")
        if not 0 <= r["p_yes"] <= 1:
            raise ValueError(f"invalid p_yes for {r.get('cand_id')}")
        if r["label"] not in ("compatible", "incompatible", "ambiguous"):
            raise ValueError(f"unknown saved label {r['label']}")
        if r["margin_mm"] < -SAVED_MARGIN_CAP_MM:
            raise ValueError("saved margins cannot be below the declared reporting cap")
        r["margin_left_censored"] = int(r["margin_mm"] == -SAVED_MARGIN_CAP_MM)
        r["margin_relation"] = "<=" if r["margin_left_censored"] else "="
        r["binary_label_eligible"] = int(r["label"] != "ambiguous")
        r["accept_tau"] = int(r["p_yes"] >= tau)
        r["accept_yes"] = int(r["answer"] == "yes")
        r["image_fits"] = int(r["image_fits"])
        rows.append(r)
    return rows


def verify_score_identity(rows: list[dict], freeze: dict) -> dict:
    """Verify source artifacts and match archived image-hash associations without requiring PNGs."""
    manifest = verify_snapshot()
    fields = ("model_id", "revision", "quant", "grouping", "chat", "prompt_id", "prompt_sha", "views", "versions")
    identity = {k: freeze["config"][k] for k in fields}
    snapshot = {}
    with SNAPSHOT.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                saved = json.loads(line)
                if saved["key"] in snapshot and saved != snapshot[saved["key"]]:
                    raise ValueError(f"conflicting score snapshot key {saved['key']}")
                snapshot[saved["key"]] = saved
    for r in rows:
        paths = [r[k] for k in ("peg_v1", "peg_v2", "socket_v1", "socket_v2")]
        if any(path not in manifest["images"] for path in paths):
            raise ValueError(f"image path missing from verified snapshot manifest for {r['peg_id']}/{r['cand_id']}")
        shas = [manifest["images"][p] for p in paths]
        key = hashlib.sha256(json.dumps({**identity, "image_sha": shas}, sort_keys=True).encode()).hexdigest()
        saved = snapshot.get(key)
        if saved is None:
            raise ValueError(f"no frozen-identity snapshot score for {r['peg_id']}/{r['cand_id']}")
        if float(saved["p_yes"]) != r["p_yes"] or saved["answer"] != r["answer"]:
            raise ValueError(f"CSV/snapshot score mismatch for {r['peg_id']}/{r['cand_id']}")
        r["score_snapshot_key"] = key
    return identity


def margin_bin(row: dict) -> str:
    if row["margin_left_censored"]:
        return "<=-2 (left censored)"
    for lo, hi, name in zip(BIN_EDGES, BIN_EDGES[1:], BIN_LABELS):
        if lo <= row["margin_mm"] < hi:
            return name
    raise ValueError(f"margin outside analysis bins: {row['margin_mm']}")


def summarize_bins(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["split"], margin_bin(r)].append(r)
    result = []
    for split in sorted({r["split"] for r in rows}):
        for name in ("<=-2 (left censored)", *BIN_LABELS):
            rs = grouped[split, name]
            if not rs:
                continue
            censored = bool(rs[0]["margin_left_censored"])
            scores = np.array([r["p_yes"] for r in rs])
            result.append({
                "split": split, "margin_bin_mm": name, "n": len(rs),
                "n_families": len({r["family"] for r in rs}), "n_pegs": len({r["peg_id"] for r in rs}),
                "n_compatible": sum(r["label"] == "compatible" for r in rs),
                "n_incompatible": sum(r["label"] == "incompatible" for r in rs),
                "n_ambiguous": sum(r["label"] == "ambiguous" for r in rs),
                "n_left_censored": sum(r["margin_left_censored"] for r in rs),
                "mean_margin_mm": None if censored else float(np.mean([r["margin_mm"] for r in rs])),
                "observed_min_margin_mm": None if censored else min(r["margin_mm"] for r in rs),
                "observed_max_margin_mm": None if censored else max(r["margin_mm"] for r in rs),
                "p_yes_min": float(scores.min()), "p_yes_q25": float(np.quantile(scores, .25)),
                "p_yes_median": float(np.median(scores)), "p_yes_q75": float(np.quantile(scores, .75)),
                "p_yes_max": float(scores.max()), "accepted_tau": sum(r["accept_tau"] for r in rs),
                "accept_tau_rate": float(np.mean([r["accept_tau"] for r in rs])),
                "image_fits": sum(r["image_fits"] for r in rs),
                "image_fit_rate": float(np.mean([r["image_fits"] for r in rs])),
            })
    return result


def summarize_peg(rows: list[dict]) -> dict:
    """Describe measured ladder steps; transitions bracket observed samples, never interpolate."""
    rs = sorted(rows, key=lambda r: r["offset_mm"])
    if len({r["offset_mm"] for r in rs}) != len(rs):
        raise ValueError("duplicate offset for one peg")
    rises, falls, equal, unresolved = 0, 0, 0, 0
    transitions = []
    for low, high in zip(rs, rs[1:]):
        # A clipped or equal margin does not establish a strictly increasing measured clearance.
        if low["margin_left_censored"] or high["margin_left_censored"] or high["margin_mm"] <= low["margin_mm"]:
            unresolved += 1
        else:
            delta = high["p_yes"] - low["p_yes"]
            rises += delta > SCORE_STEP_TOL
            falls += delta < -SCORE_STEP_TOL
            equal += abs(delta) <= SCORE_STEP_TOL
        if low["accept_tau"] != high["accept_tau"]:
            transitions.append({
                "from_offset_mm": low["offset_mm"], "to_offset_mm": high["offset_mm"],
                "from_margin_mm": low["margin_mm"], "from_margin_relation": low["margin_relation"],
                "to_margin_mm": high["margin_mm"], "to_margin_relation": high["margin_relation"],
                "direction_with_increasing_offset": "reject_to_accept" if high["accept_tau"] else "accept_to_reject",
            })
    states = {r["accept_tau"] for r in rs}
    status = ("always_accepts" if states == {1} else "always_rejects" if states == {0}
              else "one_observed_transition" if len(transitions) == 1 else "multiple_observed_transitions")
    return {
        "split": rs[0]["split"], "family": rs[0]["family"], "peg_id": rs[0]["peg_id"], "n": len(rs),
        "n_left_censored": sum(r["margin_left_censored"] for r in rs),
        "min_p_yes": min(r["p_yes"] for r in rs), "max_p_yes": max(r["p_yes"] for r in rs),
        "p_yes_range": max(r["p_yes"] for r in rs) - min(r["p_yes"] for r in rs),
        "resolved_score_rises": int(rises), "resolved_score_falls": int(falls), "resolved_score_ties": int(equal),
        "unresolved_margin_steps": unresolved,
        "nonmonotonic_on_resolved_steps": bool(rises and falls),
        "violates_nondecreasing_response": bool(falls),
        "tau_response": status, "n_observed_gate_transitions": len(transitions),
        "transitions": transitions,
    }


def rate_summary(rows: list[dict], label: str, value: str = "accept_tau") -> dict:
    """Raw counts and a ratio of family-weighted counts, bootstrapping entire families."""
    chosen = [r for r in rows if r["label"] == label]
    families = sorted({r["family"] for r in rows})
    counts = []
    for family in families:
        family_rows = [r for r in rows if r["family"] == family]
        selected = [r for r in family_rows if r["label"] == label]
        counts.append([sum(r[value] for r in selected) / len(family_rows), len(selected) / len(family_rows)])
    counts = np.asarray(counts, dtype=float)
    num, den = counts.sum(axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = counts[rng.integers(0, len(families), size=(BOOTSTRAP_REPS, len(families)))].sum(axis=1)
    finite = boot[:, 1] > 0
    intervals = np.quantile(boot[finite, 0] / boot[finite, 1], (.025, .975)).tolist() if finite.any() else None
    return {"accepted": sum(r[value] for r in chosen), "n": len(chosen),
            "raw_rate": sum(r[value] for r in chosen) / len(chosen) if chosen else None,
            "family_balanced_rate": float(num / den) if den > 0 else None,
            "family_bootstrap_95_percentile_interval": intervals,
            "bootstrap_replicates_with_defined_rate": int(finite.sum()), "n_families": len(families)}


def summarize(rows: list[dict], per_peg: list[dict], tau: float) -> dict:
    result = {
        "analysis": "descriptive correction; saved scores, unchanged labels and frozen single-pair threshold",
        "frozen_s1_tau": tau, "threshold_comparison": "p_yes >= tau",
        "unit": "one peg paired with one aperture rung; not a three-candidate selection query",
        "n_pairs": len(rows), "n_pegs": len(per_peg),
        "labels": dict(Counter(r["label"] for r in rows)),
        "n_binary_labelled": sum(r["binary_label_eligible"] for r in rows),
        "n_ambiguous": sum(not r["binary_label_eligible"] for r in rows),
        "ambiguity_reasons": dict(Counter(r["reason"] for r in rows if r["label"] == "ambiguous")),
        "n_left_censored": sum(r["margin_left_censored"] for r in rows),
        "censoring": "A saved margin of -2 mm means true margin <= -2 mm; do not treat it as exact or average it.",
        "binary_metrics": "Ambiguous rows are excluded; they remain visible in the score and gate-response plots.",
        "uncertainty": {
            "method": "Percentile bootstrap of entire query families; all pegs and rungs within a family travel together.",
            "weighting": "Each family has total weight one before accepted and label counts are divided; raw counts also reported.",
            "repetitions": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
            "limitations": "Descriptive intervals from seven families per split; repeated rungs are not independent trials. "
                          "Conditional on these templates, renders and the frozen threshold; no fresh confirmatory test.",
        },
        "transitions": "Observed adjacent ladder samples bracket threshold changes; no interpolated crossing point. "
                       "Score rises/falls use only strictly increasing, uncensored measured-margin steps; tolerance 1e-12.",
        "splits": {},
    }
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        ps = [p for p in per_peg if p["split"] == split]
        decided = [r for r in rs if r["binary_label_eligible"]]
        y = [r["label"] == "compatible" for r in decided]
        result["splits"][split] = {
            "n_pairs": len(rs), "n_pegs": len(ps), "n_families": len({r["family"] for r in rs}),
            "labels": dict(Counter(r["label"] for r in rs)),
            "n_left_censored": sum(r["margin_left_censored"] for r in rs),
            "compatible_retention_tau": rate_summary(rs, "compatible"),
            "incompatible_acceptance_tau": rate_summary(rs, "incompatible"),
            "compatible_yes_rate": rate_summary(rs, "compatible", "accept_yes"),
            "incompatible_yes_rate": rate_summary(rs, "incompatible", "accept_yes"),
            "pooled_auroc_p_yes_descriptive": float(roc_auc_score(y, [r["p_yes"] for r in decided])),
            "image_control": {"agree": sum(r["image_fits"] == (r["label"] == "compatible") for r in decided),
                              "n_binary_labelled": len(decided)},
            "tau_response_pegs": dict(Counter(p["tau_response"] for p in ps)),
            "pegs_nonmonotonic_on_resolved_steps": sum(p["nonmonotonic_on_resolved_steps"] for p in ps),
            "pegs_violating_nondecreasing_response": sum(p["violates_nondecreasing_response"] for p in ps),
            "unresolved_margin_steps": sum(p["unresolved_margin_steps"] for p in ps),
        }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def figure(rows: list[dict], bins: list[dict], tau: float, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex="col")
    colours = {"compatible": "#16804a", "incompatible": "#b32936", "ambiguous": "#97752c"}
    censor_x = -2.23  # separate display gutter, not an imputed measured margin
    score_limits = (min(r["p_yes"] for r in rows) - .005, max(r["p_yes"] for r in rows) + .005)
    for col, split in enumerate(("calib", "test")):
        rs = [r for r in rows if r["split"] == split]
        ax = axes[0, col]
        for peg in sorted({r["peg_id"] for r in rs}):
            pr = sorted((r for r in rs if r["peg_id"] == peg and not r["margin_left_censored"]), key=lambda r: r["margin_mm"])
            ax.plot([r["margin_mm"] for r in pr], [r["p_yes"] for r in pr], color="#b2b8bd", alpha=.35, lw=.65)
        for label, colour in colours.items():
            pr = [r for r in rs if r["label"] == label and not r["margin_left_censored"]]
            ax.scatter([r["margin_mm"] for r in pr], [r["p_yes"] for r in pr], s=14, alpha=.8, color=colour, label=label)
        pr = [r for r in rs if r["margin_left_censored"]]
        ax.scatter([censor_x] * len(pr), [r["p_yes"] for r in pr], marker="<", s=34,
                   color=colours["incompatible"], label="left censored")
        ax.axhline(tau, color="#2166ac", ls="--", lw=1, label=f"frozen tau {tau:.4f}")
        ax.set_title(f"{split}: {len(rs)} rungs, {len({r['peg_id'] for r in rs})} pegs, 7 families; {len(pr)} censored", fontsize=11)
        ax.set_ylabel("normalized Yes score")
        ax.set_ylim(score_limits)
        if col == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(.5, .942))
        ax = axes[1, col]
        bs = [b for b in bins if b["split"] == split]
        exact = [b for b in bs if not b["n_left_censored"]]
        xs = [b["mean_margin_mm"] for b in exact]
        # Binned points are descriptive rates, not a fitted response function or a new threshold.
        ax.plot(xs, [b["accept_tau_rate"] for b in exact], "o-", color="#2166ac", label="frozen score gate")
        ax.plot(xs, [b["image_fit_rate"] for b in exact], "s--", color="#16804a", label="image control says fits")
        for b in (b for b in bs if b["n_left_censored"]):
            ax.scatter([censor_x], [b["accept_tau_rate"]], marker="<", s=40, color="#2166ac")
            ax.scatter([censor_x], [b["image_fit_rate"]], marker="<", s=40, color="#16804a")
        ax.set_ylim(-.06, 1.19)
        ax.set_ylabel("fraction of rungs in measured-margin bin")
        ax.set_xlabel("measured geometric margin (mm); left gutter is censored")
        ax.legend(fontsize=8, loc="center left")
        for ax in axes[:, col]:
            ax.axvline(-2.1, color="#aaa", lw=.8, ls=":")
            ax.axvline(0, color="#aaa", lw=.6)
            ax.set_xlim(-2.36, .37)
            ax.set_xticks([censor_x, -2, -1.5, -1, -.5, 0, .3], ["≤−2", "−2", "−1.5", "−1", "−0.5", "0", "+0.3"])
            ax.grid(alpha=.2)
    fig.suptitle("Saved aperture experiment: response against measured margin", fontsize=14)
    fig.text(.5, .015, "399 rungs: 290 binary-labelled + 109 ambiguous; 12 margins left censored at −2 mm. "
             "Ambiguous rungs remain in plots, but not correctness metrics.\n"
             "Bottom panels pool observations within declared bins (counts in margin_bins.csv); repeated rungs are not independent. "
             "The single-pair gate uses the original frozen threshold; this analysis is descriptive.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .90))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def near_duplicate_diagnostic() -> list[dict]:
    """Illustrative overlap search across unchanged splits; 0.9 IoU is not a leakage definition."""
    from wncf.parts import build_catalog
    from wncf.sets import similarity
    from wncf.splits import family_splits

    parts = build_catalog()
    splits = family_splits(p.family for p in parts)
    rows = []
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            if splits[a.family] == splits[b.family]:
                continue
            sim = similarity(a.bore, b.bore)
            if sim >= .9:
                rows.append({"part_a": a.part_id, "family_a": a.family, "split_a": splits[a.family],
                             "part_b": b.part_id, "family_b": b.family, "split_b": splits[b.family],
                             "bore_iou_over_quarter_turns": sim})
    return sorted(rows, key=lambda r: -r["bore_iou_over_quarter_turns"])


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    tau = freeze["calibration"]["S1"]["tau"]
    with SOURCE.open(newline="", encoding="utf-8") as f:
        rows = prepare_rows(list(csv.DictReader(f)), tau)
    identity = verify_score_identity(rows, freeze)
    per_peg = [summarize_peg([r for r in rows if r["peg_id"] == peg]) for peg in sorted({r["peg_id"] for r in rows})]
    bins = summarize_bins(rows)
    summary = summarize(rows, per_peg, tau)
    near_duplicates = near_duplicate_diagnostic()
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "per_pair.csv", rows)
    write_csv(OUT / "per_peg.csv", per_peg)
    write_csv(OUT / "margin_bins.csv", bins)
    write_csv(OUT / "cross_split_similarity.csv", near_duplicates)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure(rows, bins, tau, OUT / "measured_margin.png")
    sources = [SOURCE, FREEZE, SNAPSHOT, SNAPSHOT_MANIFEST, REPO_ROOT / "data/aperture/pairs.csv",
               REPO_ROOT / "data/parts/catalog.csv", REPO_ROOT / "data/parts/splits.csv", Path(__file__)]
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_outputs_preserved": True, "new_model_calls": 0, "labels_changed": 0, "split_membership_changed": False,
        "frozen_at_recorded": freeze["frozen_at"], "score_identity": identity,
        "freeze_record_sha256": sha256(FREEZE), "snapshot_matches_verified": len(rows),
        "snapshot_manifest_sha256": sha256(SNAPSHOT_MANIFEST),
        "image_identity_check": "Archived image paths/hashes from the verified snapshot manifest reproduce score keys. "
                                "Rendered PNGs are not required or rehashed during this offline analysis.",
        "margin_bin_edges_mm": ["+inf" if math.isinf(x) else x for x in BIN_EDGES],
        "near_duplicate_diagnostic": "Post hoc examples with bore IoU >=0.9 across unchanged template splits; "
                                     "not an independence certificate or a reason to reshuffle the inspected test set.",
        "source_sha256": {p.relative_to(REPO_ROOT).as_posix(): sha256(p) for p in sources},
    }
    (OUT / "analysis_meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}: {len(rows)} pairs; {summary['n_binary_labelled']} binary-labelled, "
          f"{summary['n_ambiguous']} ambiguous, {summary['n_left_censored']} left censored; frozen tau={tau:.8f}")


if __name__ == "__main__":
    main()
