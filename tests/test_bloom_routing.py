"""Tests for confidence-aware Bloom routing (manuscript Eq. bloomroute)."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bloom_classifier import classify_bloom_llm_sc, route_bloom_level


def _engine_from(seq):
    """Return an engine that yields the given responses in cycle."""
    box = {"i": 0}
    def engine(prompt):
        v = seq[box["i"] % len(seq)]
        box["i"] += 1
        return v
    return engine


def test_confident_vote_keeps_level():
    eng = _engine_from(["Analyze"] * 5)
    level, unc = classify_bloom_llm_sc("So sánh A và B", eng, m=5)
    assert level == "Analyze"
    assert unc == 0.0
    out = route_bloom_level("So sánh A và B", eng, m=5, gamma_bloom=0.40)
    assert out["escalate"] is False
    assert out["status"] == "auto"


def test_split_on_boundary_escalates():
    # 3 Apply / 2 Analyze -> modal Apply, uncertainty 0.4 > gamma 0.30, on boundary
    eng = _engine_from(["Apply", "Analyze", "Apply", "Analyze", "Apply"])
    out = route_bloom_level("Tình huống áp dụng điều X", eng, m=5, gamma_bloom=0.30)
    assert out["bloom_level"] == "Apply"
    assert out["uncertainty"] > 0.30
    assert out["escalate"] is True
    assert out["status"] == "review"


def test_split_off_boundary_does_not_escalate():
    # High uncertainty but modal level is Remember (not Apply/Analyze) -> keep.
    eng = _engine_from(["Remember", "Understand", "Remember", "Understand", "Remember"])
    out = route_bloom_level("Theo Điều 5 là gì", eng, m=5, gamma_bloom=0.30)
    assert out["bloom_level"] == "Remember"
    assert out["escalate"] is False


def test_no_engine_falls_back():
    out = route_bloom_level("So sánh hai khái niệm", llm_engine=None)
    assert out["uncertainty"] == 0.0
    assert out["escalate"] is False


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = fail = 0
    for fn in fns:
        try:
            fn(); ok += 1; print("PASS", fn.__name__)
        except Exception as e:
            fail += 1; print("FAIL", fn.__name__, "->", repr(e))
    print(f"== {ok} passed, {fail} failed ==")
