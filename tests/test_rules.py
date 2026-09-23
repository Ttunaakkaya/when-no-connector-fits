"""Stage 6 rules on hand-built scores: no model, no files."""

import math

import pytest

from wncf.rules import CandidateScore, decide, invalid_reason, margin_p_yes, rank_b0

def cs(cand_id, answer, top_prob, p_yes, mass_yes=None, mass_no=None):
    mass_yes = p_yes if mass_yes is None else mass_yes
    mass_no = (1 - p_yes) if mass_no is None else mass_no
    return CandidateScore(cand_id, answer, top_prob, p_yes, mass_yes, mass_no)


def test_b0_puts_yes_before_no_and_orders_each_group_by_the_emitted_probability():
    scores = [cs("a", "no", 0.90, 0.10), cs("b", "yes", 0.60, 0.60),
              cs("c", "yes", 0.80, 0.70), cs("d", "no", 0.55, 0.45)]
    assert rank_b0(scores) == ["c", "b", "d", "a"]  # yes desc, then no asc
    assert decide("B0", scores).selected == "c"


def test_b0_ties_break_on_candidate_id():
    scores = [cs("z", "yes", 0.7, 0.7), cs("a", "yes", 0.7, 0.7)]
    assert rank_b0(scores) == ["a", "z"]
    assert decide("B0", list(reversed(scores))).selected == "a"


def test_b1_selects_only_a_unique_yes():
    one = [cs("a", "yes", 0.7, 0.7), cs("b", "no", 0.6, 0.3), cs("c", "no", 0.6, 0.3)]
    assert decide("B1", one) == decide("B1", one)
    assert (decide("B1", one).status, decide("B1", one).selected) == ("select", "a")
    two = [cs("a", "yes", 0.7, 0.7), cs("b", "yes", 0.6, 0.6), cs("c", "no", 0.6, 0.3)]
    assert decide("B1", two).status == "defer"
    none = [cs("a", "no", 0.7, 0.3), cs("b", "no", 0.6, 0.2)]
    assert decide("B1", none).status == "defer"


def test_s1_needs_the_threshold_and_keeps_the_top_scoring_candidate():
    scores = [cs("a", "yes", 0.7, 0.62), cs("b", "yes", 0.6, 0.55), cs("c", "no", 0.6, 0.20)]
    assert decide("S1", scores, tau=0.60).selected == "a"
    assert decide("S1", scores, tau=0.65).status == "defer"
    with pytest.raises(ValueError):
        decide("S1", scores)


def test_s2_also_needs_the_margin():
    close = [cs("a", "yes", 0.7, 0.62), cs("b", "yes", 0.6, 0.60)]
    assert decide("S2", close, tau=0.5, delta=0.01).selected == "a"
    assert decide("S2", close, tau=0.5, delta=0.05).status == "defer"
    assert margin_p_yes(close) == pytest.approx(0.02)


def test_u0_always_selects_the_top_normalized_score():
    scores = [cs("a", "no", 0.9, 0.30), cs("b", "no", 0.8, 0.49)]
    d = decide("U0", scores)
    assert (d.status, d.selected) == ("select", "b")  # selects even when every answer was No


def test_b0_and_the_normalized_selector_can_disagree():
    # the emitted-token probability and the summed-spelling score need not order candidates alike
    scores = [cs("a", "yes", 0.52, 0.61, mass_yes=0.61, mass_no=0.39),
              cs("b", "yes", 0.58, 0.55, mass_yes=0.55, mass_no=0.45)]
    assert decide("B0", scores).selected == "b"
    assert decide("U0", scores).selected == "a"


@pytest.mark.parametrize("bad", [
    cs("x", "other", 0.5, 0.5),
    cs("x", "yes", 1.4, 0.5),
    cs("x", "yes", 0.5, float("nan"), mass_yes=float("nan"), mass_no=0.5),
    cs("x", "yes", 0.5, 0.5, mass_yes=0.8, mass_no=0.4),  # masses sum above 1
    cs("x", "yes", 0.5, 0.5, mass_yes=0.0, mass_no=0.0),  # no answer mass at all
    cs("x", "yes", 0.5, 0.5, mass_yes=-0.1, mass_no=0.6),
])
def test_one_invalid_candidate_fails_the_whole_query_for_every_method(bad):
    scores = [cs("a", "yes", 0.7, 0.7), bad]
    assert invalid_reason(scores) is not None
    for method in ("B0", "B1", "S1", "S2", "U0"):
        d = decide(method, scores, tau=0.5, delta=0.0)
        assert (d.status, d.selected) == ("model_failure", None)


def test_valid_scores_report_no_reason():
    assert invalid_reason([cs("a", "yes", 0.7, 0.7), cs("b", "no", 0.6, 0.3)]) is None
    assert invalid_reason([cs("a", "yes", 1.0, 1.0, mass_yes=1.0, mass_no=0.0)]) is None
    assert math.isfinite(margin_p_yes([cs("a", "yes", 0.7, 0.7)]))
