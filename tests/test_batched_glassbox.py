"""Test for batched_glassbox runner (GPU-free, mock engines).

Runs under pytest (via test_batched_glassbox) and as a standalone script
(python tests/test_batched_glassbox.py). All sub-checks use mock critic/gen
engines so the runner economics are verified without a GPU.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stage2.batched_glassbox import (
    batch_score_critic, split_by_decision, run_glassbox_loops,
)
from stage2.judge import JudgeConfig

passed = failed = 0
failures: list[str] = []


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        failures.append(name)
        print(f"  FAIL  {name}")


def mk(qid, ans="A"):
    return ({"qa_id": qid, "question": f"Q {qid}", "bloom_level": "Apply",
             "choices": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": ans},
            {"chunk_id": qid.split("__")[0], "text": "Điều 1 Luật X quy định...", "source_doc": "d"},
            "Apply")


def _make_engines():
    """Return (FakeCritic, FakeGen, load_count) sharing one load counter."""
    load_count = {"critic": 0, "gen": 0}

    class FakeCritic:
        def __init__(self):
            load_count["critic"] += 1

        def generate_batch(self, prompts):
            out = []
            for p in prompts:
                if "Q c1" in p:  # c1 strong
                    out.append("<r1>5</r1><r2>5</r2><r3>5</r3><r4>5</r4><r5>5</r5><r6>5</r6>")
                else:            # c2 weak legal r5=2 -> reject
                    out.append("<r1>5</r1><r2>4</r2><r3>4</r3><r4>5</r4><r5>2</r5><r6>4</r6>")
            return out

        def unload(self):
            pass

    class FakeGen:
        def __init__(self):
            load_count["gen"] += 1

        def generate_batch(self, prompts):
            return ["dummy"] * len(prompts)

        def unload(self):
            pass

    return FakeCritic, FakeGen, load_count


def _run_checks():
    items = [mk("c1__apply__0"), mk("c2__apply__0")]
    FakeCritic, FakeGen, load_count = _make_engines()

    # batch_score_critic cost: 2 items * m=5 = 10 prompts
    seen = {"n": 0}

    def cg(prompts):
        seen["n"] += len(prompts)
        return FakeCritic().generate_batch(prompts)

    cache = batch_score_critic(items, cg, m=5)
    check("score cost = items*m (10)", seen["n"] == 10)
    check("cache has per-dim dist", set(cache[0].keys()) == {"r1", "r2", "r3", "r4", "r5", "r6"})

    acc, refz, rej = split_by_decision(items, cache, JudgeConfig())
    check("c1 accepted", any(it[0].get("qa_id") == "c1__apply__0" for it, _ in acc))
    check("c2 rejected (r5=2 hard gate)", any(it[0].get("qa_id") == "c2__apply__0" for it, _ in rej))

    # full runner: c1 accept loop1, c2 reject -> finalized only converged accepts
    def bld_refine(qa, crit, bloom, ctx):
        return "refine"

    def parse_qa(raw):
        return [{"question": "Q sửa", "choices": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "A"}]

    FakeCritic2, FakeGen2, load_count2 = _make_engines()
    fin = run_glassbox_loops(
        [mk("c1__apply__0"), mk("c2__apply__0")],
        critic_factory=FakeCritic2, generator_factory=FakeGen2,
        build_refine_prompt=bld_refine, parse_qa=parse_qa,
        judge_config=JudgeConfig(), max_loops=3, m_samples=5,
    )
    check("finalized includes c1 converged",
          any(p["qa_id"] == "c1__apply__0" and p["converged"] for p in fin))
    check("each finalized has judge provenance", all(p.get("judge") is not None for p in fin))
    # c1 accepts loop1 -> no refine zone -> break after loop1: 1 critic load, 0 gen
    check("model-swap economy: 1 critic load", load_count2["critic"] == 1)
    check("no generator load when nothing to refine", load_count2["gen"] == 0)


def test_batched_glassbox():
    """Pytest entry: run all checks, fail if any sub-check failed."""
    global passed, failed
    passed = failed = 0
    failures.clear()
    _run_checks()
    print(f"\nbatched_glassbox: {passed} passed, {failed} failed")
    assert failed == 0, f"sub-checks failed: {failures}"


if __name__ == "__main__":
    _run_checks()
    print(f"\nbatched_glassbox: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
