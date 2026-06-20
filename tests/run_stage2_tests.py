"""
Stdlib-only tests for Stage 2 Judge & Refiner (no pytest/GPU required).
Run: python3 tests/run_stage2_tests.py
Validates the formula-driven decision logic from docs/QUANTITATIVE_*.md.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stage2.judge import (  # noqa: E402
    JudgeConfig, Judge, RubricScores,
    expected_score, hard_gate, weighted_score, platt,
    expected_calibration_error, krippendorff_alpha_interval,
    samples_to_prob_dist, evidence_grounding_score,
)
from src.stage2.refiner import (  # noqa: E402
    RefinerConfig, Refiner, violates_non_regression,
    refine_gain, refine_success_rate, build_external_feedback,
)

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ---- J2: expected_score (probability-weighted) ----
print("J2 expected_score:")
check("certain 4 -> 4.0", approx(expected_score({4: 1.0}), 4.0))
check("split 3/5 -> 4.0", approx(expected_score({3: 0.5, 5: 0.5}), 4.0))
check("renormalizes", approx(expected_score({4: 2.0}), 4.0))
try:
    expected_score({6: 1.0}); check("reject v=6", False)
except ValueError:
    check("reject v=6", True)

# ---- J3: hard gate + weighted score ----
print("J3 aggregation:")
gate = {"r1": 4.0, "r2": 4.0, "r5": 3.0}
check("gate pass", hard_gate({"r1": 4, "r2": 5, "r5": 3}, gate) is True)
check("gate fail r2", hard_gate({"r1": 4, "r2": 3.9, "r5": 3}, gate) is False)
w = {"r1": .25, "r2": .25, "r3": .2, "r4": .1, "r5": .1, "r6": .1}
norm_perfect = {k: 1.0 for k in w}
check("weighted all-1 -> 1.0", approx(weighted_score(norm_perfect, w), 1.0))

# ---- J4: platt + ECE ----
print("J4 calibration:")
check("platt identity midpoint", approx(platt(0.0, 1.0, 0.0), 0.5))
check("platt monotone", platt(0.9, 2.0, 0.0) > platt(0.1, 2.0, 0.0))
perfect_cal = [(0.05, 0), (0.95, 1)]
check("ECE perfect ~0", expected_calibration_error(perfect_cal, 10) < 0.1)
bad_cal = [(0.9, 0), (0.9, 0)]  # conf .9 but always wrong
check("ECE bad high", expected_calibration_error(bad_cal, 10) > 0.5)

# ---- J7: Krippendorff alpha ----
print("J7 agreement:")
agree = [[4.0, 4.0], [3.0, 3.0], [5.0, 5.0]]
a_hi = krippendorff_alpha_interval(agree)
check("alpha perfect agree ~1", a_hi is not None and approx(a_hi, 1.0, 1e-6))
disagree = [[1.0, 5.0], [5.0, 1.0], [1.0, 5.0]]
a_lo = krippendorff_alpha_interval(disagree)
check("alpha strong disagree < 0.5", a_lo is not None and a_lo < 0.5)
check("alpha single judge -> None", krippendorff_alpha_interval([[4.0]]) is None)

# ---- RubricScores ----
print("RubricScores:")
rs = RubricScores({"r1": 5, "r2": 5, "r3": 1, "r4": 5, "r5": 5, "r6": 5})
check("lowest_dim = r3", rs.lowest_dim() == "r3")
check("normalize 5 -> 1.0", approx(rs.normalized()["r1"], 1.0))
check("normalize 1 -> 0.0", approx(rs.normalized()["r3"], 0.0))

# ---- Judge.evaluate (injected deterministic scorer) ----
print("Judge.evaluate:")


def make_scorer(score_map):
    def fn(qa, ctx, dim):
        return {score_map[dim]: 1.0}
    return fn


cfg = JudgeConfig()
good = {"r1": 5, "r2": 5, "r3": 4, "r4": 4, "r5": 4, "r6": 5}
j_good = Judge(cfg, make_scorer(good)).evaluate({"qa_id": "g", "question": "x"}, "ctx")
check("good QA accepts", j_good.decision == "accept")
check("good gate passed", j_good.hard_gate_passed is True)

bad_corr = {"r1": 5, "r2": 2, "r3": 4, "r4": 4, "r5": 4, "r6": 5}
j_bad = Judge(cfg, make_scorer(bad_corr)).evaluate({"qa_id": "b", "question": "x"}, "ctx")
check("wrong-answer QA rejects (gate)", j_bad.decision == "reject")
check("bad gate failed", j_bad.hard_gate_passed is False)

mid = {"r1": 4, "r2": 4, "r3": 3, "r4": 3, "r5": 3, "r6": 3}
j_mid = Judge(cfg, make_scorer(mid)).evaluate({"qa_id": "m", "question": "x"}, "ctx")
check("mid QA -> refine", j_mid.decision == "refine")

# ---- Refiner: non-regression ----
print("Refiner non-regression:")
prev = {"r1": 1.0, "r2": 0.8, "r5": 0.6}
worse = {"r1": 1.0, "r2": 0.5, "r5": 0.9}  # r2 drops 0.3 > 0.1
check("blocks r2 regression", violates_non_regression(prev, worse, 0.10) is True)
ok_new = {"r1": 1.0, "r2": 0.78, "r5": 0.9}
check("allows small dip", violates_non_regression(prev, ok_new, 0.10) is False)


# ---- Refiner.run: monotone-best + stopping ----
print("Refiner.run:")


class FakeJ:
    def __init__(self, s, low="r5"):
        self.S = s
        self.lowest_dim = low
        self.rubric_scores = {"r1": 5, "r2": 5, "r3": 4, "r4": 4, "r5": 3, "r6": 4}


def make_refine_improving():
    state = {"s": 0.6}

    def refine_fn(qa, dim, reason, ctx):
        return dict(qa)

    def judge_fn(qa, ctx):
        state["s"] += 0.15
        return FakeJ(min(state["s"], 0.95))
    return refine_fn, judge_fn


rf, jf = make_refine_improving()
res = Refiner(RefinerConfig(), rf, jf).run(
    {"qa_id": "r", "question": "x"}, "ctx", FakeJ(0.6))
check("improving reaches accept", res.final_decision == "accept")
check("trajectory monotone-ish", res.s_trajectory[-1] >= res.s_trajectory[0])
check("stop reason accept", res.stop_reason == "reached_tau_accept")


# refine that never helps -> reject, but never worse than start
def make_refine_flat():
    def refine_fn(qa, dim, reason, ctx):
        return dict(qa)

    def judge_fn(qa, ctx):
        return FakeJ(0.61)  # barely moves
    return refine_fn, judge_fn


rf2, jf2 = make_refine_flat()
res2 = Refiner(RefinerConfig(), rf2, jf2).run(
    {"qa_id": "r2", "question": "x"}, "ctx", FakeJ(0.60))
check("flat refine rejects", res2.final_decision == "reject")
check("best never below start", max(res2.s_trajectory) >= res2.s_trajectory[0])
check("stops on diminishing returns", res2.stop_reason == "diminishing_returns")

# ---- batch metrics ----
print("Batch metrics:")
check("refine_gain", approx(refine_gain([0.5, 0.6], [0.8, 0.7]), 0.2))
check("success_rate", approx(
    refine_success_rate([True, True, False], [True, False, True]), 1/3))

# ---- VERDI: samples_to_prob_dist (self-consistency, NOT logprobs) ----
print("VERDI samples_to_prob_dist:")
pd = samples_to_prob_dist([4, 4, 4, 5, 3])
check("freq P(4)=0.6", approx(pd[4], 0.6))
check("freq P(5)=0.2", approx(pd[5], 0.2))
check("probs sum to 1", approx(sum(pd.values()), 1.0))
check("feeds expected_score", approx(
    expected_score(samples_to_prob_dist([4, 4, 4, 4, 4])), 4.0))
try:
    samples_to_prob_dist([]); check("reject empty samples", False)
except ValueError:
    check("reject empty samples", True)
try:
    samples_to_prob_dist([6]); check("reject sample=6", False)
except ValueError:
    check("reject sample=6", True)

# ---- VERDI: evidence_grounding_score (r5 legal validity) ----
print("VERDI evidence_grounding_score:")
src = "điều 5 khoản 2 luật doanh nghiệp quy định về vốn điều lệ công ty"
check("grounded citation -> 1.0", approx(
    evidence_grounding_score(["điều 5 khoản 2"], src), 1.0))
check("hallucinated citation -> 0.0", approx(
    evidence_grounding_score(["điều 99 khoản 88 nghị định bịa"], src), 0.0))
check("no quotes -> vacuously 1.0", approx(
    evidence_grounding_score([], src), 1.0))
check("length-weighted partial", 0.0 < evidence_grounding_score(
    ["điều 5 khoản 2", "điều 99 khoản 88 bịa đặt hoàn toàn"], src) < 1.0)

# ---- Self-Correction Illusion: external-judge framing ----
print("External-judge framing (2606.05976):")
fb = build_external_feedback("r5", 2.0)
check("framed as external judge", "Giám khảo độc lập" in fb)
check("not self-reflection", "bạn" in fb and "không phải bạn" in fb)
check("names the weak dim", "pháp lý" in fb)
check("includes score", "2.0/5" in fb)
fb_noscore = build_external_feedback("r3")
check("works without score", "Bloom" in fb_noscore and "Giám khảo" in fb_noscore)

print(f"\n{'='*40}\nStage 2: {_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
