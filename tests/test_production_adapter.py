"""Stdlib tests for production_adapter (GPU-free). Mock critic/generator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stage2.production_adapter import (
    build_rubric_prompt, parse_rubric_scores, make_sampling_judge,
    make_refine_fn, _samples_to_dist,
)
from stage2.judge import Judge, JudgeConfig
from stage2.refiner import Refiner, RefinerConfig, build_external_feedback

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  PASS  {name}")
    else:
        failed += 1; print(f"  FAIL  {name}")

QA = {"qa_id": "c1_analyze_0", "bloom_level": "Analyze",
      "question": "Phân tích vì sao hợp đồng vô hiệu?",
      "choices": {"A": "x", "B": "y", "C": "z", "D": "w"}, "answer": "A"}
CTX = "Điều 122 Bộ luật Dân sự quy định điều kiện có hiệu lực của giao dịch."

# 1. prompt builder
p = build_rubric_prompt(QA, CTX, "Analyze")
check("prompt has all 6 dim tags hint", all(d in p for d in ("r1","r6")) and "1..5" in p)

# 2. parser — well-formed
ok = parse_rubric_scores("<r1>5</r1><r2>4</r2><r3>3</r3><r4>4</r4><r5>2</r5><r6>4</r6>")
check("parse well-formed", ok == {"r1":5,"r2":4,"r3":3,"r4":4,"r5":2,"r6":4})
# parser — tolerant fallback
ok2 = parse_rubric_scores("r1: 4 r2 = 5 r3:3 r4:4 r5:5 r6:4")
check("parse tolerant", ok2 == {"r1":4,"r2":5,"r3":3,"r4":4,"r5":5,"r6":4})
# parser — incomplete -> None
check("parse incomplete None", parse_rubric_scores("<r1>5</r1>") is None)

# 3. dist
check("dist freq", _samples_to_dist([5,5,5,4]) == {5:0.75, 4:0.25})

# 4. sampling judge: m samples of ONE prompt -> per-dim cache; count calls
calls = {"n": 0}
def fake_critic(prompts, temperature):
    calls["n"] += len(prompts)
    # critic consistently rates r5 low (legal), others high
    return ["<r1>5</r1><r2>4</r2><r3>4</r3><r4>5</r4><r5>2</r5><r6>4</r6>"] * len(prompts)

score_fn = make_sampling_judge(fake_critic, m=5, temperature=0.7)
judge = Judge(JudgeConfig(), score_fn, judge_models=["gemma-3-4b"])
res = judge.evaluate(QA, CTX)
# CRITICAL: 6 per-dim score_fn calls must collapse to m=5 LLM calls, not 30
check("cost: m=5 LLM calls not 30", calls["n"] == 5)
check("judge flags r5 lowest", res.lowest_dim == "r5")
check("judge rejects (r5=2 fails hard gate r5>=3)", res.decision == "reject")

# 5. refine_fn wiring
def fake_build_refine(qa, crit, bloom, ctx):
    return f"REFINE[{bloom}] issues={crit['issues'][:20]}"
def fake_parse_qa(raw):
    return [{"question": "Câu hỏi đã sửa", "choices": {"A":"a","B":"b","C":"c","D":"d"}, "answer":"A"}]
def fake_gen(prompts, temperature):
    return ["dummy"] * len(prompts)

refine_fn = make_refine_fn(fake_gen, fake_build_refine, fake_parse_qa)
fb = build_external_feedback("r5", 2.0)
new_qa = refine_fn(QA, "r5", fb, CTX)
check("refine preserves qa_id", new_qa.get("qa_id") == "c1_analyze_0")
check("refine external framing", "Giám khảo độc lập" in fb)

print(f"\nproduction_adapter: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
