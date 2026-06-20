"""
Stage 2 / Production adapter — glass-box Judge/Refiner ↔ batched vLLM engine
============================================================================
Bridges the GPU-free, unit-tested glass-box layer (`stage2.judge.Judge`,
`stage2.refiner.Refiner`) to the real batched generate/critique/refine engine
used by `run_batched_adaptive`.

Cost design (IMPORTANT — answers the 30x blow-up):
  `Judge.evaluate` calls `score_fn(qa, context, dim)` once PER rubric dim
  (6 dims). Naively sampling m=5 per dim => 30 LLM calls / question.
  Instead, this adapter issues ONE rubric prompt that scores ALL 6 dims at
  once, sampled m times => m calls / question (m=5 default, configurable).
  The per-dim `score_fn` then just reads the cached m-sample distribution.

VERDI (arXiv:2605.11334): P(v) per dim comes from FREQUENCY over m samples,
never from token logprobs.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from stage2.judge import RUBRIC_DIMS, RUBRIC_LABELS

# Map our fixed rubric keys r1..r6 to the legal-domain meaning the critic sees.
_RUBRIC_PROMPT_DIMS: dict[str, str] = {
    "r1": "groundedness — câu hỏi và đáp án bám sát ngữ cảnh được cung cấp",
    "r2": "answer_correctness — đáp án đúng được đánh dấu là chính xác",
    "r3": "bloom_alignment — câu hỏi đúng cấp độ tư duy Bloom yêu cầu",
    "r4": "answerability — trả lời được chỉ từ ngữ cảnh, không mơ hồ",
    "r5": "legal_validity — trích dẫn/khái niệm pháp lý chính xác, không bịa điều luật",
    "r6": "linguistic_quality — tiếng Việt rõ ràng, đúng thuật ngữ",
}


def build_rubric_prompt(qa: dict[str, Any], context_text: str, bloom: str) -> str:
    """One prompt that asks the critic to score ALL 6 dims on a 1..5 scale.

    Output contract is strict XML so the parser is deterministic:
      <r1>..</r1> ... <r6>..</r6> each an integer 1..5.
    """
    q = str(qa.get("question", "")).strip()
    choices = qa.get("choices", qa.get("options", {}))
    if isinstance(choices, dict):
        ch = "\n".join(f"{k}. {v}" for k, v in choices.items())
    elif isinstance(choices, (list, tuple)):
        ch = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    else:
        ch = str(choices)
    ans = str(qa.get("answer", "")).strip()
    dims_desc = "\n".join(f"- {k}: {v}" for k, v in _RUBRIC_PROMPT_DIMS.items())
    return (
        "Bạn là giám khảo độc lập, chấm chất lượng một câu hỏi trắc nghiệm pháp "
        "lý tự sinh. Chấm TỪNG tiêu chí theo thang nguyên 1..5 (1 rất kém, 5 xuất "
        "sắc). KHÔNG giải thích, chỉ xuất đúng định dạng XML.\n\n"
        f"NGỮ CẢNH:\n{context_text[:2000]}\n\n"
        f"CẤP BLOOM YÊU CẦU: {bloom}\n\n"
        f"CÂU HỎI:\n{q}\nĐÁP ÁN:\n{ch}\nĐÁP ÁN ĐÚNG: {ans}\n\n"
        f"SÁU TIÊU CHÍ:\n{dims_desc}\n\n"
        "XUẤT (mỗi tiêu chí một số nguyên 1..5):\n"
        "<r1>?</r1><r2>?</r2><r3>?</r3><r4>?</r4><r5>?</r5><r6>?</r6>"
    )


def parse_rubric_scores(text: str) -> Optional[dict[str, int]]:
    """Parse <r1..r6> integers 1..5 from one critic sample. None if incomplete."""
    if not text:
        return None
    out: dict[str, int] = {}
    for dim in RUBRIC_DIMS:
        m = re.search(rf"<{dim}>\s*([1-5])\s*</{dim}>", text)
        if not m:
            # tolerant fallback: "r1: 4" or "r1 = 4"
            m = re.search(rf"{dim}\s*[:=]\s*([1-5])", text, re.IGNORECASE)
        if not m:
            return None
        out[dim] = int(m.group(1))
    return out


# --------------------------------------------------------------------------
# Sampling judge: m samples of ONE rubric prompt -> per-dim P(v) cache.
# Matches Judge's injected score_fn signature: (qa, context, dim) -> {v: p}.
# --------------------------------------------------------------------------

def _samples_to_dist(samples: list[int]) -> dict[int, float]:
    """Frequency P(v)=count(v)/m over m samples (VERDI, not logprobs)."""
    m = len(samples)
    counts: dict[int, float] = {}
    for v in samples:
        counts[v] = counts.get(v, 0.0) + 1.0
    return {v: c / m for v, c in counts.items()}


def make_sampling_judge(
    critic_generate: Callable[[list[str], float], list[str]],
    m: int = 5,
    temperature: float = 0.7,
    fallback_rating: int = 3,
) -> Callable[[dict, str, str], dict[int, float]]:
    """Build a `score_fn(qa, context, dim) -> {rating: prob}` for `Judge`.

    KEY COST TRICK: the FIRST call for a given (qa_id, context) issues m critic
    samples of ONE 6-dim rubric prompt and caches the resulting per-dim
    distributions. The 6 per-dim score_fn calls Judge makes then just read the
    cache => m LLM calls/question, NOT 6*m.

    `critic_generate(prompts, temperature) -> list[str]` is the real batched
    vLLM call (one string out per prompt in).
    """
    cache: dict[str, dict[str, dict[int, float]]] = {}

    def _key(qa: dict, context: str) -> str:
        return f"{qa.get('qa_id', qa.get('id', id(qa)))}|{hash(context) & 0xffffffff}"

    def _populate(qa: dict, context: str) -> dict[str, dict[int, float]]:
        bloom = str(qa.get("bloom_level", qa.get("bloom", "")))
        prompt = build_rubric_prompt(qa, context, bloom)
        # m samples of the SAME prompt at T>0 -> genuine self-consistency.
        raws = critic_generate([prompt] * m, temperature)
        per_dim_samples: dict[str, list[int]] = {d: [] for d in RUBRIC_DIMS}
        for raw in raws:
            parsed = parse_rubric_scores(raw)
            if parsed is None:
                continue
            for d in RUBRIC_DIMS:
                per_dim_samples[d].append(parsed[d])
        dists: dict[str, dict[int, float]] = {}
        for d in RUBRIC_DIMS:
            s = per_dim_samples[d]
            dists[d] = _samples_to_dist(s) if s else {fallback_rating: 1.0}
        return dists

    def score_fn(qa: dict, context: str, dim: str) -> dict[int, float]:
        k = _key(qa, context)
        if k not in cache:
            cache[k] = _populate(qa, context)
        return cache[k][dim]

    return score_fn


# --------------------------------------------------------------------------
# Refine function: matches Refiner's refine_fn(qa, target_dim, reason, context).
# --------------------------------------------------------------------------

def make_refine_fn(
    generator_generate: Callable[[list[str], float], list[str]],
    build_refine_prompt: Callable[..., str],
    parse_qa: Callable[[str], list[dict]],
    temperature: float = 0.7,
) -> Callable[[dict, str, str, str], dict]:
    """Build `refine_fn(qa, target_dim, reason, context) -> new_qa`.

    `reason` is the EXTERNAL-judge framed feedback string produced by
    `refiner.build_external_feedback` (arXiv:2606.05976). It is injected as the
    critique 'issues'/'suggestions' so the generator treats it as an external
    verdict to act on, targeting only `target_dim`.

    Reuses the existing `_build_refine_prompt` and `parse_qa_xml` from
    iterative_qag so prompt format / parsing stay identical to production.
    """
    def refine_fn(qa: dict, target_dim: str, reason: str, context: str) -> dict:
        bloom = str(qa.get("bloom_level", qa.get("bloom", "")))
        critique_like = {"issues": reason, "suggestions": reason}
        prompt = build_refine_prompt(qa, critique_like, bloom, context)
        outs = generator_generate([prompt], temperature)
        raw = outs[0] if outs else ""
        parsed = parse_qa(raw)
        new_qa = parsed[0] if parsed else dict(qa)
        # Preserve identity / routing fields across the rewrite.
        for keep in ("qa_id", "chunk_id", "bloom_level", "bloom",
                     "source_doc", "eligible_bloom_levels"):
            if keep in qa and keep not in new_qa:
                new_qa[keep] = qa[keep]
        return new_qa

    return refine_fn
