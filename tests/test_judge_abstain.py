"""Tests for risk-calibrated abstention in the BloomDepth U7 Judge.

Verifies the manuscript abstention rule: an item is admitted only when the
gate passes, S >= tau_accept, judges agree, AND calibrated critical-dimension
uncertainty <= gamma. Otherwise the judge abstains -> human, never a silent
accept under an uncalibrated judge call.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage2.judge import (
    Judge, JudgeConfig,
    consistency_uncertainty, calibrate_uncertainty,
)


def _confident_dist(rating: int):
    # All m samples agree on `rating` -> uncertainty 0.
    return {rating: 1.0}


def _split_dist(r_a: int, r_b: int):
    # Half the samples each -> high uncertainty.
    return {r_a: 0.5, r_b: 0.5}


def test_uncertainty_helpers():
    assert consistency_uncertainty({5: 1.0}) == 0.0
    assert abs(consistency_uncertainty({4: 0.5, 5: 0.5}) - 0.5) < 1e-9
    # identity at T=1
    assert abs(calibrate_uncertainty(0.5, 1.0) - 0.5) < 1e-6
    # T>1 inflates a low uncertainty toward 0.5 (less confident)
    assert calibrate_uncertainty(0.2, 3.0) > 0.2


def test_confident_high_scores_accept():
    cfg = JudgeConfig(tau_accept=0.80, gamma_uncertainty=0.40)

    def score_fn(qa, ctx, dim):
        return _confident_dist(5)  # perfect, fully agreed

    j = Judge(cfg, score_fn, judge_models=["m1"])
    res = j.evaluate({"qa_id": "q1", "question": "x"}, "ctx")
    assert res.decision == "accept"
    assert res.abstained is False
    assert res.max_critical_uncertainty == 0.0


def test_high_uncertainty_forces_abstain():
    """Scores clear thresholds but critical dims are split -> abstain, not accept."""
    cfg = JudgeConfig(tau_accept=0.50, gamma_uncertainty=0.30)

    def score_fn(qa, ctx, dim):
        # critical dims (r1,r2,r5) get a 4/5 split (passes hard gate >=4/>=3,
        # high score) but uncertainty 0.5 > gamma 0.30 -> abstain.
        if dim in ("r1", "r2", "r5"):
            return _split_dist(4, 5)
        return _confident_dist(5)

    j = Judge(cfg, score_fn, judge_models=["m1"])
    res = j.evaluate({"qa_id": "q2", "question": "x"}, "ctx")
    assert res.max_critical_uncertainty > 0.30
    assert res.abstained is True
    assert res.decision == "abstain"  # never silently accepted


def test_gate_failure_still_rejects():
    cfg = JudgeConfig(tau_accept=0.80, gamma_uncertainty=0.40)

    def score_fn(qa, ctx, dim):
        return _confident_dist(1)  # fails hard gate

    j = Judge(cfg, score_fn, judge_models=["m1"])
    res = j.evaluate({"qa_id": "q3", "question": "x"}, "ctx")
    assert res.decision == "reject"


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = fail = 0
    for fn in fns:
        try:
            fn(); ok += 1; print("PASS", fn.__name__)
        except Exception as e:
            fail += 1; print("FAIL", fn.__name__, "->", repr(e))
    print(f"== {ok} passed, {fail} failed ==")
