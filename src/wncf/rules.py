"""Stage 6: selection and rejection rules.

All rules read the same cached per-candidate scores and differ only in how they pick a candidate and
whether they may decline:

  B0  the paper's ranking: Yes answers first by descending probability of the emitted token, then No
      answers by ascending probability; select the first. Forced choice, conditional on validity.
  B1  the paper's No Probability ablation: select only if exactly one candidate answered Yes.
  S1  select the highest normalized-score candidate only if that score reaches tau.
  S2  the same candidate, requiring tau and a top-minus-second margin of delta.
  U0  diagnostic comparator: the highest normalized-score candidate, always selected. B0 and S1/S2
      rank by different quantities, so U0 separates the change of selector from the rejection itself.

Validity (master section 7): a candidate response needs finite non-negative Yes/No masses summing to
more than 0 and at most 1 + 1e-6, a finite normalized score, an emitted-token probability in [0, 1],
and an emitted token in the pinned Yes/No sets. If any candidate in a query fails, the whole query is
`model_failure` and every rule defers; invalid output is never silently read as "No".

Ties break on the candidate id, so a rule's output never depends on dictionary or file order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

METHODS = ("B0", "B1", "S1", "S2", "U0")
MASS_SUM_MAX = 1 + 1e-6


@dataclass(frozen=True)
class CandidateScore:
    cand_id: str
    answer: str  # "yes" | "no" | "other"
    top_prob: float  # probability of the emitted token: the paper's p(o_m)
    p_yes: float  # mass_yes / (mass_yes + mass_no)
    mass_yes: float
    mass_no: float


@dataclass(frozen=True)
class Decision:
    method: str
    status: str  # "select" | "defer" | "model_failure"
    selected: str | None  # candidate id when status == "select"


def invalid_reason(scores: list[CandidateScore]) -> str | None:
    """Why this query's responses are unusable, or None if all candidates are valid."""
    for s in scores:
        mass_sum = s.mass_yes + s.mass_no
        if not all(math.isfinite(v) for v in (s.mass_yes, s.mass_no, s.p_yes, s.top_prob)):
            return f"{s.cand_id}: non-finite value"
        if s.mass_yes < 0 or s.mass_no < 0:
            return f"{s.cand_id}: negative answer mass"
        if not 0 < mass_sum <= MASS_SUM_MAX:
            return f"{s.cand_id}: answer mass {mass_sum:.6f} outside (0, {MASS_SUM_MAX}]"
        if not 0 <= s.top_prob <= 1:
            return f"{s.cand_id}: emitted-token probability {s.top_prob}"
        if s.answer not in ("yes", "no"):
            return f"{s.cand_id}: emitted token is not a Yes/No variant ({s.answer})"
    return None


def rank_b0(scores: list[CandidateScore]) -> list[str]:
    """The paper's ordering; ties break on candidate id."""
    yes = sorted((s for s in scores if s.answer == "yes"), key=lambda s: (-s.top_prob, s.cand_id))
    no = sorted((s for s in scores if s.answer != "yes"), key=lambda s: (s.top_prob, s.cand_id))
    return [s.cand_id for s in yes + no]


def best_by_p_yes(scores: list[CandidateScore]) -> CandidateScore:
    return min(scores, key=lambda s: (-s.p_yes, s.cand_id))


def margin_p_yes(scores: list[CandidateScore]) -> float:
    """Top minus second normalized score; 0 when a single candidate is offered."""
    ordered = sorted((s.p_yes for s in scores), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else 0.0


def decide(method: str, scores: list[CandidateScore], tau: float | None = None,
           delta: float | None = None) -> Decision:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if not scores:
        raise ValueError("no candidates")
    if invalid_reason(scores) is not None:
        return Decision(method, "model_failure", None)

    if method == "B0":
        return Decision(method, "select", rank_b0(scores)[0])
    if method == "B1":
        yes = [s for s in scores if s.answer == "yes"]
        return Decision(method, "select", yes[0].cand_id) if len(yes) == 1 else Decision(method, "defer", None)

    best = best_by_p_yes(scores)
    if method == "U0":
        return Decision(method, "select", best.cand_id)
    if tau is None:
        raise ValueError(f"{method} needs a calibrated tau")
    if best.p_yes < tau:
        return Decision(method, "defer", None)
    if method == "S2":
        if delta is None:
            raise ValueError("S2 needs a calibrated delta")
        if margin_p_yes(scores) < delta:
            return Decision(method, "defer", None)
    return Decision(method, "select", best.cand_id)
