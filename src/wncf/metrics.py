"""Stage 7-8: calibration and metrics.

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

The default estimator gives each family unit total weight, divides it equally among that family's
observed regimes, then among queries in each family/regime cell. Rates are ratios of these weighted
counts, never averages of family risks. Raw integer counts stay in their own fields; weighted sums
use the ``weighted_`` prefix. ``weighting="query"`` explicitly replays the historical pooled
estimator. Bootstrap samples retain family draw multiplicities. B1 has no ranking metric.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from wncf.rules import CandidateScore, Decision, best_by_p_yes, decide, invalid_reason, rank_b0

COVERAGE_TARGET = 0.70  # declared design target: mate-present selection coverage on calibration
S2_TAU_GRID = tuple(round(0.1 * i, 2) for i in range(11))
S2_DELTA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)

COUNT_FIELDS = ("n", "n_present", "n_absent", "selected", "correct", "wrong", "deferred", "failures",
                "present_selected", "absent_selected", "top1_correct")
RATIO_COUNTS = {
    "coverage": ("selected", "n"),
    "present_coverage": ("present_selected", "n_present"),
    "accepted_risk": ("wrong", "selected"),
    "correct_yield": ("correct", "n_present"),
    "absent_false_accept": ("absent_selected", "n_absent"),
    "failure_rate": ("failures", "n"),
    "top1_present": ("top1_correct", "n_present"),
}


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


def top1_correct(q: Query, method: str) -> bool | None:
    """Ungated ranking accuracy, before any rejection; invalid responses count as unsuccessful."""
    if method == "B1":
        return None  # unique-Yes is a decision rule, not a candidate ranking
    if not q.has_mate or invalid_reason(list(q.scores)) is not None:
        return False
    top = rank_b0(list(q.scores))[0] if method == "B0" else best_by_p_yes(list(q.scores)).cand_id
    return q.labels[top] == "compatible"


def _query_weights(queries: list[Query], weighting: str,
                   weights: dict[str, float] | None = None) -> list[float]:
    if weighting not in ("family", "query"):
        raise ValueError("weighting must be 'family' or 'query'")
    weights = weights or {}
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("regime weights must be finite and non-negative")
    cells = Counter((q.family, q.regime) for q in queries)
    regimes_per_family = Counter(f for f, _ in cells)
    return [weights.get(q.regime, 1.0) /
            (regimes_per_family[q.family] * cells[q.family, q.regime] if weighting == "family" else 1)
            for q in queries]


def evaluate(queries: list[Query], method: str, tau: float | None = None, delta: float | None = None,
             weights: dict[str, float] | None = None, weighting: str = "family") -> dict:
    """Raw counts, separately named weighted sums, and rates from the weighted sums.

    In the complete benchmark every family has all four regimes. On a restricted subset (for
    example easy or present only), equal mass is assigned to the regimes observed in each family.
    Optional ``weights`` multiplies regime contributions, allowing a stated prevalence analysis.
    """
    raw = dict.fromkeys(COUNT_FIELDS, 0)
    weighted = {k: [] for k in COUNT_FIELDS}
    for q, w in zip(queries, _query_weights(queries, weighting, weights)):
        d = decide(method, list(q.scores), tau=tau, delta=delta)
        res = outcome(q, d)
        keys = ["n", {"defer": "deferred", "failure": "failures"}.get(res, res)]
        if res in ("correct", "wrong"):
            keys.append("selected")
        if q.has_mate:
            keys.append("n_present")
            if res in ("correct", "wrong"):
                keys.append("present_selected")
            if top1_correct(q, method):
                keys.append("top1_correct")
        else:
            keys.append("n_absent")
            if res in ("correct", "wrong"):
                keys.append("absent_selected")
        for key in keys:
            raw[key] += 1
            weighted[key].append(w)
    c = {k: math.fsum(v) for k, v in weighted.items()}
    ratios = {k: _ratio(c[num], c[den]) for k, (num, den) in RATIO_COUNTS.items()}
    if method == "B1":
        raw["top1_correct"] = c["top1_correct"] = ratios["top1_present"] = None
    return {
        "method": method, "tau": tau, "delta": delta, "weighting": weighting,
        "n_families": len({q.family for q in queries}), **raw,
        **{f"weighted_{k}": v for k, v in c.items()}, **ratios,
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

def calibrate_s1(queries: list[Query], target: float = COVERAGE_TARGET, weighting: str = "family",
                 weights: dict[str, float] | None = None) -> float:
    """Highest threshold among the observed score breakpoints that still meets the coverage target."""
    # coverage only grows as tau falls, so scanning from the highest breakpoint down, the first
    # threshold that meets the target is the highest one that does
    for tau in sorted(_score_breakpoints(queries), reverse=True):
        if (evaluate(queries, "S1", tau=tau, weighting=weighting, weights=weights)["present_coverage"] or 0) >= target:
            return tau
    return 0.0  # nothing meets the target: keep the loosest gate, and report that it failed


def calibrate_s2(queries: list[Query], s1_tau: float, target: float = COVERAGE_TARGET,
                 weighting: str = "family", weights: dict[str, float] | None = None) -> tuple[float, float]:
    """Prescribed grid; among settings meeting the target, minimise accepted-choice risk."""
    best, best_key = None, None
    for tau in sorted(set(S2_TAU_GRID) | {s1_tau}):
        for delta in S2_DELTA_GRID:
            m = evaluate(queries, "S2", tau=tau, delta=delta, weighting=weighting, weights=weights)
            if (m["present_coverage"] or 0) < target:
                continue
            risk = m["accepted_risk"]
            if risk is None:
                continue
            key = (risk, -(m["correct_yield"] or 0), -tau, -delta)  # declared tie-breaks
            if best_key is None or key < best_key:
                best, best_key = (tau, delta), key
    return best if best is not None else (s1_tau, 0.0)


def _score_breakpoints(queries: list[Query]) -> set[float]:
    # A failed query cannot define a usable score gate (in particular, NaN is never a threshold).
    return {s.p_yes for q in queries if invalid_reason(list(q.scores)) is None for s in q.scores}


def risk_coverage(queries: list[Query], method: str = "S1", delta: float | None = None,
                  weighting: str = "family", weights: dict[str, float] | None = None) -> list[tuple]:
    """Descriptive curve: sweep tau only, so the curve stays one-dimensional."""
    points = []
    for tau in sorted(_score_breakpoints(queries) | {0.0, 1.0}):
        m = evaluate(queries, method, tau=tau, delta=delta, weighting=weighting, weights=weights)
        points.append((tau, m["coverage"], m["accepted_risk"], m["correct_yield"]))
    return points


# --- uncertainty ---

def bootstrap_by_family(queries: list[Query], method: str, metric: str, n: int = 2000, seed: int = 20260924,
                        weighting: str = "family", weights: dict[str, float] | None = None,
                        **kw) -> tuple[float | None, float | None]:
    """Percentile interval for the declared estimator, resampling entire query families.

    Preaggregate each family's weighted numerator/denominator once. Each sampled copy contributes
    again, even when the same original family is drawn repeatedly; regrouping a flattened sample
    by its original ids before balancing would incorrectly erase this multiplicity. Shared socket
    pools are held fixed, so these intervals remain conditional on the observed distractor pool.
    """
    if n < 1:
        raise ValueError("bootstrap n must be positive")
    if metric not in RATIO_COUNTS:
        raise ValueError(f"unsupported bootstrap metric {metric!r}")
    if not queries or (method == "B1" and metric == "top1_present"):
        return None, None
    families = defaultdict(list)
    for q in queries:
        families[q.family].append(q)
    num, den = RATIO_COUNTS[metric]
    counts = [evaluate(qs, method, weighting=weighting, weights=weights, **kw) for qs in families.values()]
    contributions = [(c[f"weighted_{num}"], c[f"weighted_{den}"]) for c in counts]
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        sample = [rng.choice(contributions) for _ in contributions]
        value = _ratio(math.fsum(c[0] for c in sample), math.fsum(c[1] for c in sample))
        if value is not None:
            draws.append(value)
    if len(draws) < max(1, n // 10):  # mostly undefined, e.g. a rule that selects almost nothing
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
