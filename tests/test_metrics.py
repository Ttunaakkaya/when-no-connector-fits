"""Stage 7-8 metrics and calibration on hand-built queries."""

import random
import math

import pytest

from wncf.metrics import (COVERAGE_TARGET, Query, bootstrap_by_family, calibrate_s1, calibrate_s2,
                          evaluate, evaluate_by_tier, outcome, risk_coverage, top1_correct)
from wncf.rules import CandidateScore, decide


def sc(cid, p_yes, answer="yes", top_prob=None):
    return CandidateScore(cid, answer, p_yes if top_prob is None else top_prob, p_yes, p_yes, 1 - p_yes)


def query(block, regime, mate_score=None, other_scores=(0.4, 0.3), family="fam", split="calib", **kw):
    labels, scores = {}, []
    if mate_score is not None:
        labels["M"] = "compatible"
        scores.append(sc("M", mate_score))
    for i, s in enumerate(other_scores, 1):
        labels[f"X{i}"] = "incompatible"
        scores.append(sc(f"X{i}", s, **kw))
    return Query(block, block.split("_")[0], family, split, regime, labels, tuple(scores))


PRESENT = query("p1_present_easy", "present_easy", mate_score=0.9)
PRESENT_LOSS = query("p2_present_hard", "present_hard", mate_score=0.5, other_scores=(0.8, 0.3))
ABSENT = query("p3_absent_easy", "absent_easy", other_scores=(0.7, 0.6, 0.5))


def test_outcomes_cover_correct_wrong_defer_and_failure():
    assert outcome(PRESENT, decide("U0", list(PRESENT.scores))) == "correct"
    assert outcome(PRESENT_LOSS, decide("U0", list(PRESENT_LOSS.scores))) == "wrong"
    assert outcome(ABSENT, decide("S1", list(ABSENT.scores), tau=0.95)) == "defer"
    broken = Query("b", "b", "fam", "calib", "present_easy", {"M": "compatible"},
                   (CandidateScore("M", "other", 0.5, 0.5, 0.5, 0.5),))
    assert outcome(broken, decide("B0", list(broken.scores))) == "failure"


def test_counts_and_rates_on_a_small_set():
    m = evaluate([PRESENT, PRESENT_LOSS, ABSENT], "U0")
    assert (m["n"], m["n_present"], m["n_absent"]) == (3, 2, 1)
    assert (m["correct"], m["wrong"], m["selected"]) == (1, 2, 3)  # U0 always selects
    assert m["coverage"] == 1.0
    assert m["accepted_risk"] == pytest.approx(2 / 3)
    assert m["correct_yield"] == pytest.approx(0.5)
    assert m["absent_false_accept"] == 1.0
    assert m["top1_present"] == pytest.approx(0.5)


def test_accepted_risk_is_undefined_rather_than_zero_when_nothing_is_selected():
    m = evaluate([PRESENT, ABSENT], "S1", tau=1.01)
    assert m["selected"] == 0
    assert m["accepted_risk"] is None
    assert m["coverage"] == 0.0


def test_failures_are_counted_separately_and_never_read_as_no():
    broken = Query("b", "b", "fam", "calib", "absent_easy", {"X1": "incompatible"},
                   (CandidateScore("X1", "yes", 0.5, float("nan"), float("nan"), 0.5),))
    m = evaluate([broken], "B0")
    assert (m["failures"], m["selected"], m["failure_rate"]) == (1, 0, 1.0)
    assert m["accepted_risk"] is None


def test_tiers_are_reported_separately():
    out = evaluate_by_tier([PRESENT, PRESENT_LOSS, ABSENT], "U0")
    assert set(out) == {"all", "easy", "hard"}
    assert out["easy"]["n"] == 2 and out["hard"]["n"] == 1
    assert out["hard"]["accepted_risk"] == 1.0  # the only hard query is the one U0 gets wrong


def test_regime_weights_apply_to_counts_not_to_averaged_ratios():
    qs = [PRESENT, PRESENT, ABSENT]  # easy present over-represented
    equal = evaluate(qs, "U0", weighting="query", weights={"present_easy": 0.5, "absent_easy": 1.0})
    assert equal["n"] == 3  # raw counts are never replaced by weighted sums
    assert equal["weighted_n"] == pytest.approx(2.0)
    assert equal["accepted_risk"] == pytest.approx(0.5)  # 1 wrong of 2 weighted selections


def test_top1_ignores_rejection_and_follows_the_method_selector():
    swapped = query("s_present_easy", "present_easy", mate_score=0.4, other_scores=(0.6,))
    assert top1_correct(swapped, "U0") is False
    # B0 ranks by the emitted-token probability, which can disagree with p_yes
    q = Query("q", "q", "fam", "calib", "present_easy", {"M": "compatible", "X": "incompatible"},
              (sc("M", 0.4, top_prob=0.9), sc("X", 0.6, top_prob=0.2)))
    assert top1_correct(q, "B0") is True and top1_correct(q, "U0") is False


def test_s1_calibration_picks_the_highest_threshold_meeting_the_coverage_target():
    present = [query(f"p{i}_present_easy", "present_easy", mate_score=m, other_scores=(0.2,))
               for i, m in enumerate([0.9, 0.8, 0.7, 0.6, 0.5])]
    tau = calibrate_s1(present, target=0.6)
    assert tau == pytest.approx(0.7)  # 3 of 5 present queries still selected
    assert (evaluate(present, "S1", tau=tau)["present_coverage"] or 0) >= 0.6


def test_s1_calibration_reports_the_loosest_gate_when_the_target_is_unreachable():
    # coverage counts selections, right or wrong, so only invalid responses can keep it below target
    broken = Query("b_present_easy", "b", "fam", "calib", "present_easy",
                   {"M": "compatible", "X": "incompatible"},
                   (CandidateScore("M", "other", 0.5, 0.6, 0.6, 0.4), sc("X", 0.3)))
    assert (evaluate([broken], "S1", tau=0.0)["present_coverage"] or 0) == 0.0
    assert calibrate_s1([broken], target=0.99) == 0.0


def test_s2_calibration_stays_on_the_declared_grid_and_meets_the_target():
    qs = [query(f"p{i}_present_easy", "present_easy", mate_score=0.9, other_scores=(0.2,)) for i in range(4)]
    qs += [query(f"a{i}_absent_easy", "absent_easy", other_scores=(0.85, 0.84)) for i in range(4)]
    tau, delta = calibrate_s2(qs, s1_tau=0.5, target=COVERAGE_TARGET)
    assert delta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
    m = evaluate(qs, "S2", tau=tau, delta=delta)
    assert (m["present_coverage"] or 0) >= COVERAGE_TARGET
    assert m["absent_false_accept"] == 0.0  # the margin gate rejects the near-tied absent sets


def test_risk_coverage_curve_is_monotone_in_coverage():
    qs = [PRESENT, PRESENT_LOSS, ABSENT]
    pts = risk_coverage(qs, "S1")
    covs = [c for _, c, _, _ in pts]
    assert covs == sorted(covs, reverse=True)  # rising tau cannot raise coverage
    assert all(r is None or 0 <= r <= 1 for _, _, r, _ in pts)


def test_bootstrap_resamples_whole_families():
    qs = [query(f"p{i}_present_easy", "present_easy", mate_score=0.9, family=f"f{i % 3}") for i in range(9)]
    lo, hi = bootstrap_by_family(qs, "U0", "correct_yield", n=200)
    assert lo == 1.0 and hi == 1.0  # every query is correct, so every resample is too
    mixed = qs + [query("w_present_hard", "present_hard", mate_score=0.1, other_scores=(0.9,), family="f3")]
    lo2, hi2 = bootstrap_by_family(mixed, "U0", "correct_yield", n=400)
    assert lo2 < 1.0 <= hi2


def test_family_balancing_does_not_let_repeated_geometry_dominate():
    good = query("g", "present_easy", mate_score=0.9, family="large")
    bad = query("b", "present_easy", mate_score=0.1, family="small")
    qs = [good] * 9 + [bad]
    balanced = evaluate(qs, "U0")
    assert (balanced["n"], balanced["correct"], balanced["wrong"]) == (10, 9, 1)
    assert (balanced["weighted_n"], balanced["weighted_correct"], balanced["weighted_wrong"]) == pytest.approx((2, 1, 1))
    assert balanced["correct_yield"] == pytest.approx(0.5)
    assert evaluate(qs, "U0", weighting="query")["correct_yield"] == pytest.approx(0.9)


def test_regimes_are_balanced_within_each_family_before_taking_ratios():
    qs = [PRESENT] * 9 + [ABSENT]
    m = evaluate(qs, "U0")
    assert m["n_present"] == 9 and m["n_absent"] == 1
    assert m["weighted_n_present"] == pytest.approx(0.5)
    assert m["weighted_n_absent"] == pytest.approx(0.5)
    assert m["accepted_risk"] == pytest.approx(0.5)


def test_family_risk_is_ratio_of_weighted_counts_not_average_of_family_risks():
    qs = [query("g", "present_easy", mate_score=0.9, family="good")]
    qs += [query("w", "present_easy", mate_score=0.1, other_scores=(0.9,), family="mixed"),
           query("d", "present_easy", mate_score=0.1, other_scores=(0.2,), family="mixed")]
    m = evaluate(qs, "S1", tau=0.8)
    # Good family: 1/1 selected, no errors. Mixed family: 1/2 selected, one error.
    assert m["accepted_risk"] == pytest.approx(0.5 / 1.5)
    assert m["coverage"] == pytest.approx(1.5 / 2)
    assert (m["selected"], m["wrong"]) == (2, 1)


def test_b1_has_no_ranking_metric_even_when_a_unique_yes_is_selected():
    q = query("b1", "present_easy", mate_score=0.9, answer="no")
    assert evaluate([q], "B1")["selected"] == 1
    assert top1_correct(q, "B1") is None
    assert evaluate([q], "B1")["top1_present"] is None
    assert bootstrap_by_family([q], "B1", "top1_present", n=30) == (None, None)


def test_calibration_uses_the_same_explicit_weighting_as_evaluation():
    qs = [query("g", "present_easy", mate_score=0.9, family="large")] * 9
    qs += [query("s", "present_easy", mate_score=0.6, family="small")]
    assert calibrate_s1(qs) == pytest.approx(0.6)
    assert calibrate_s1(qs, weighting="query") == pytest.approx(0.9)
    assert calibrate_s2(qs, 0.6)[0] <= 0.6


def test_bootstrap_retains_draw_multiplicity_with_family_balancing():
    # Independently calculate the percentile distribution for 12 unequal families. If duplicate
    # family ids are collapsed when assigning weights, the middle quantiles change.
    qs = []
    rates = []
    for i in range(12):
        rate = (i % 4) / 3
        rates.append(rate)
        for j in range(3):
            qs += [query(f"{i}-{j}", "present_easy", mate_score=0.9 if j < i % 4 else 0.1,
                         family=str(i))] * (i + 1)
    rng = random.Random(94)
    draws = sorted(sum(rng.choice(rates) for _ in rates) / len(rates) for _ in range(800))
    expected = (draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))])
    assert bootstrap_by_family(qs, "U0", "correct_yield", n=800, seed=94) == pytest.approx(expected)


def test_query_bootstrap_replays_pooled_estimator():
    good = query("g", "present_easy", mate_score=0.9, family="good")
    bad = query("b", "present_easy", mate_score=0.1, family="bad")
    qs = [good] * 9 + [bad]
    rng = random.Random(42)
    draws = []
    for _ in range(400):
        picked = [rng.choice(("good", "bad")) for _ in range(2)]
        count = picked.count("good")
        draws.append(9 * count / (9 * count + 2 - count))
    draws.sort()
    expected = (draws[10], draws[390])
    assert bootstrap_by_family(qs, "U0", "correct_yield", n=400, seed=42, weighting="query") == expected


def test_invalid_scores_cannot_define_nonfinite_calibration_or_curve_thresholds():
    invalid = Query("bad", "bad", "bad", "calib", "absent_easy", {"X": "incompatible"},
                    (CandidateScore("X", "yes", 0.5, float("nan"), float("nan"), 0.5),))
    assert calibrate_s1([invalid, PRESENT]) == pytest.approx(0.9)
    assert all(math.isfinite(tau) for tau, *_ in risk_coverage([invalid, PRESENT]))


def test_bootstrap_of_a_rule_that_never_selects_is_undefined_even_with_few_draws():
    assert bootstrap_by_family([PRESENT], "S1", "accepted_risk", n=5, tau=1.01) == (None, None)
