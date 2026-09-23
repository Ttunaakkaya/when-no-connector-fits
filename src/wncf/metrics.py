"""Stage 7-8: calibration and metrics (master plan sections 7 and 8).

One query is one candidate block. Outcomes are correct, wrong, defer or failure, and the reported
quantities are raw counts first:

  selection coverage        selected queries / all queries
  present coverage          selected / mate-present queries
  accepted-choice risk      incompatible selections / all selections, undefined when nothing is
                            selected (never reported as zero risk)
  correct-selection yield   compatible selections / mate-present queries
  missing-mate false accept any selection / mate-absent queries
  model-failure rate        queries with an invalid response / all queries
  top-1 (ungated)           mate-present queries whose top-ranked candidate is compatible, before
                            any rejection; invalid responses count as unsuccessful

Every quantity is also reported per difficulty tier, because a pooled number can hide a rule that
buys safety by declining nearly every hard query. Regime weights are applied to the underlying
counts, not by averaging ratios. Calibration uses calibration-split queries only, and each method
gets a single threshold policy used unchanged across tiers.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

from wncf.rules import CandidateScore, Decision, best_by_p_yes, decide, invalid_reason, rank_b0

COVERAGE_TARGET = 0.70  # declared design target: mate-present selection coverage on calibration
S2_TAU_GRID = tuple(round(0.1 * i, 2) for i in range(11))
S2_DELTA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


@dataclass(frozen=True)
class Query:
    block_id: str
    peg_id: str
    family: str
    split: str
    regime: str
    labels: dict[str, str]  # candidate id -> compatible | incompatible
    scores: tuple[CandidateScore, ...]

    @property
    def has_mate(self) -> bool:
        return any(v == "compatible" for v in self.labels.values())

    @property
    def tier(self) -> str:
        return "hard" if self.regime.endswith("hard") else "easy"


def outcome(q: Query, d: Decision) -> str:
    if d.status == "model_failure":
        return "failure"
    if d.status == "defer":
        return "defer"
    return "correct" if q.labels[d.selected] == "compatible" else "wrong"


def top1_correct(q: Query, method: str) -> bool:
    """Ungated ranking accuracy, before any rejection; invalid responses count as unsuccessful."""
    if not q.has_mate or invalid_reason(list(q.scores)) is not None:
        return False
    top = rank_b0(list(q.scores))[0] if method == "B0" else best_by_p_yes(list(q.scores)).cand_id
    return q.labels[top] == "compatible"


def evaluate(queries: list[Query], method: str, tau: float | None = None, delta: float | None = None,
             weights: dict[str, float] | None = None) -> dict:
    """Weighted counts and the rates derived from them, for one method over one set of queries."""
    regimes = sorted({q.regime for q in queries})
    weights = weights or {r: 1.0 for r in regimes}
    c = defaultdict(float)
    for q in queries:
        w = weights.get(q.regime, 1.0)
        d = decide(method, list(q.scores), tau=tau, delta=delta)
        res = outcome(q, d)
        c["n"] += w
        c[res] += w
        if res in ("correct", "wrong"):
            c["selected"] += w
        if q.has_mate:
            c["present"] += w
            c["present_selected"] += w if res in ("correct", "wrong") else 0
            c["present_correct"] += w if res == "correct" else 0
            c["top1"] += w if top1_correct(q, method) else 0
        else:
            c["absent"] += w
            c["absent_selected"] += w if res in ("correct", "wrong") else 0
    return {
        "method": method, "tau": tau, "delta": delta,
        "n": c["n"], "n_present": c["present"], "n_absent": c["absent"],
        "selected": c["selected"], "correct": c["correct"], "wrong": c["wrong"],
        "deferred": c["defer"], "failures": c["failure"],
        "coverage": _ratio(c["selected"], c["n"]),
        "present_coverage": _ratio(c["present_selected"], c["present"]),
        "accepted_risk": _ratio(c["wrong"], c["selected"]),  # None when nothing was selected
        "correct_yield": _ratio(c["present_correct"], c["present"]),
        "absent_false_accept": _ratio(c["absent_selected"], c["absent"]),
        "failure_rate": _ratio(c["failure"], c["n"]),
        "top1_present": _ratio(c["top1"], c["present"]),
    }


def evaluate_by_tier(queries: list[Query], method: str, **kw) -> dict[str, dict]:
    out = {"all": evaluate(queries, method, **kw)}
    for tier in ("easy", "hard"):
        subset = [q for q in queries if q.tier == tier]
        if subset:
            out[tier] = evaluate(subset, method, **kw)
    return out


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


# --- calibration (calibration split only) ---

def calibrate_s1(queries: list[Query], target: float = COVERAGE_TARGET) -> float:
    """Highest threshold among the observed score breakpoints that still meets the coverage target."""
    # coverage only grows as tau falls, so scanning from the highest breakpoint down, the first
    # threshold that meets the target is the highest one that does
    for tau in sorted({s.p_yes for q in queries for s in q.scores}, reverse=True):
        if (evaluate(queries, "S1", tau=tau)["present_coverage"] or 0) >= target:
            return tau
    return 0.0  # nothing meets the target: keep the loosest gate, and report that it failed


def calibrate_s2(queries: list[Query], s1_tau: float, target: float = COVERAGE_TARGET) -> tuple[float, float]:
    """Prescribed grid; among settings meeting the target, minimise accepted-choice risk."""
    best, best_key = None, None
    for tau in sorted(set(S2_TAU_GRID) | {s1_tau}):
        for delta in S2_DELTA_GRID:
            m = evaluate(queries, "S2", tau=tau, delta=delta)
            if (m["present_coverage"] or 0) < target:
                continue
            risk = m["accepted_risk"]
            if risk is None:
                continue
            key = (risk, -(m["correct_yield"] or 0), -tau, -delta)  # declared tie-breaks
            if best_key is None or key < best_key:
                best, best_key = (tau, delta), key
    return best if best is not None else (s1_tau, 0.0)


def risk_coverage(queries: list[Query], method: str = "S1", delta: float | None = None) -> list[tuple]:
    """Descriptive curve: sweep tau only, so the curve stays one-dimensional."""
    points = []
    for tau in sorted({s.p_yes for q in queries for s in q.scores} | {0.0, 1.0}):
        m = evaluate(queries, method, tau=tau, delta=delta)
        points.append((tau, m["coverage"], m["accepted_risk"], m["correct_yield"]))
    return points


# --- uncertainty ---

def bootstrap_by_family(queries: list[Query], method: str, metric: str, n: int = 2000, seed: int = 20260924,
                        **kw) -> tuple[float | None, float | None]:
    """Percentile interval, resampling whole query families with all their repeated blocks."""
    families = defaultdict(list)
    for q in queries:
        families[q.family].append(q)
    keys = list(families)
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        sample = [q for k in (rng.choice(keys) for _ in keys) for q in families[k]]
        value = evaluate(sample, method, **kw)[metric]
        if value is not None:
            draws.append(value)
    if len(draws) < n // 10:  # mostly undefined, e.g. a rule that selects almost nothing
        return None, None
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[min(int(0.975 * len(draws)), len(draws) - 1)]


def family_counts(queries: list[Query]) -> dict[str, int]:
    counts = defaultdict(int)
    for q in queries:
        counts[q.family] += 1
    return dict(sorted(counts.items()))


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "undefined"
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"
