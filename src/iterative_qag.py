"""
Iterative QA Generation with Self-Refinement Loop
===================================================
Who:    Generator (Qwen2.5-7B) + Critic (Gemma-2-2b)
Where:  BloomDepth/src/iterative_qag.py
How:    Implements the Self-Refine loop (Madaan et al., NeurIPS 2023)
        adapted for MCQA generation across 6 Bloom taxonomy levels.

Pipeline: GENERATE → CRITIQUE → REFINE → CRITIQUE → ... → ACCEPT/MAX_LOOPS

Input:  Multimodal contexts from TQA_Pipeline Stage 2.
Output: Refined QA pairs with refinement traces (loop count, critique logs).

Reference: Madaan, A. et al. (2023). Self-Refine: Iterative Refinement
           with Self-Feedback. NeurIPS 2023. arXiv:2303.17651
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("bloom_depth.iterative_qag")


# ─────────────────────────────────────────────
# Prompt Templates — 6-Level Bloom Taxonomy
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert Vietnamese Civil Law examiner and professor.
Your task is to create high-quality exam questions from legal textbook content.
You MUST respond using predefined XML tags. Do NOT use markdown code blocks or JSON.
"""

# Extended Bloom definitions for 6-level generation
BLOOM_DEFINITIONS = {
    "Remember": (
        "Factual recall — who, what, when, where. Direct retrieval from text. "
        "Ví dụ: 'Theo Điều X, ai có thẩm quyền...?'"
    ),
    "Understand": (
        "Explain, summarize, compare concepts. Demonstrate comprehension. "
        "Ví dụ: 'Hãy giải thích sự khác biệt giữa...'"
    ),
    "Apply": (
        "Use legal knowledge in a new scenario or case study. "
        "Ví dụ: 'Trong tình huống A gây thiệt hại cho B, áp dụng Điều X thì...'"
    ),
    "Analyze": (
        "Break down complex information, compare and contrast legal provisions, "
        "identify relationships between different articles or legal concepts. "
        "Ví dụ: 'So sánh phạm vi điều chỉnh của Điều X và Điều Y, nêu điểm giống và khác nhau.'"
    ),
    "Evaluate": (
        "Judge the validity or reasonableness of a legal decision, argumentation, "
        "or interpretation. Provide evidence-based justification. "
        "Ví dụ: 'Đánh giá tính hợp lý của quyết định xử phạt trong tình huống sau...'"
    ),
    "Create": (
        "Design a legal solution, draft a clause, or propose a compliance workflow "
        "based on given legal principles. Requires synthesis of multiple articles. "
        "Ví dụ: 'Soạn thảo điều khoản hợp đồng đảm bảo tuân thủ quy định tại...'"
    ),
}


QA_GENERATION_TEMPLATE = """\
Based on the following legal textbook content, generate {n_questions} question(s) \
at the Bloom's Taxonomy level: **{bloom_level}**.

## Bloom's Taxonomy Level Definition:
**{bloom_level}**: {bloom_definition}

## Context:
{context_text}

{visual_context}

## Requirements:
1. Each question MUST be answerable from the given context.
2. Provide exactly 4 candidate answers (A, B, C, D) — one correct.
3. Structure the rationale using **Legal Syllogism**:
   - **Major Premise**: The general legal rule/article.
   - **Minor Premise**: The specific facts of the question scenario.
   - **Conclusion**: The logical deduction from the premises.
4. Questions and answers must be in Vietnamese.
5. Do NOT use English or Chinese in any field.
6. The cognitive level MUST match **{bloom_level}** — not higher, not lower.
7. For {bloom_level} level, the question should: {bloom_requirement}

## Output Format (strict XML tags):
<qa_pair>
<question>Câu hỏi thi ...</question>
<candidate_answers>
A. ...
B. ...
C. ...
D. ...
</candidate_answers>
<ground_truth>A. ...</ground_truth>
<legal_rationale>Đại tiền đề: ... Tiểu tiền đề: ... Kết luận: ...</legal_rationale>
</qa_pair>

Respond with ONLY the XML tags. No explanations before or after.
"""

# Bloom-level specific requirements for generation focus
BLOOM_REQUIREMENTS = {
    "Remember": "yêu cầu nhớ lại trực tiếp một sự kiện, định nghĩa, hoặc quy định cụ thể.",
    "Understand": "yêu cầu giải thích, diễn giải, hoặc tóm tắt nội dung pháp luật.",
    "Apply": "đưa ra tình huống thực tế mới và hỏi cách áp dụng quy định.",
    "Analyze": "yêu cầu so sánh, phân biệt, hoặc tìm mối quan hệ giữa các khái niệm pháp lý.",
    "Evaluate": "yêu cầu đánh giá, phản biện, hoặc nhận xét tính hợp lý của một quyết định pháp lý.",
    "Create": "yêu cầu đề xuất giải pháp, soạn thảo văn bản, hoặc thiết kế quy trình tuân thủ.",
}


# ─────────────────────────────────────────────
# Critique Prompt Template
# ─────────────────────────────────────────────
CRITIQUE_SYSTEM_PROMPT = """\
You are a strict, impartial QA quality evaluator for Vietnamese Civil Law exam questions.
You MUST evaluate the QA pair on ALL 5 dimensions below. Be specific about problems found.
Respond using strict XML tags.
"""

CRITIQUE_TEMPLATE = """\
Evaluate the following automatically-generated QA pair for a **{bloom_level}** level question.

## Original Context:
{context_text}

## Generated QA Pair:
**Question:** {question}
**Candidates:**
{candidates}
**Ground Truth:** {ground_truth}
**Legal Rationale:** {legal_rationale}

## Evaluation Dimensions (Pass=1, Fail=0):

1. **bloom_alignment**: Does the question ACTUALLY test **{bloom_level}** cognitive level? 
   (Not too easy for the level, not too hard)
2. **factual_grounding**: Is the answer fully supported by the context? No hallucinated facts?
3. **distractor_quality**: Are the 3 wrong answers plausible but clearly wrong? 
   Not obviously absurd, not ambiguously correct?
4. **question_clarity**: Is the question unambiguous and well-formed in Vietnamese?
5. **legal_accuracy**: Is the legal reasoning (syllogism) logically correct?

## Output Format:
<critique>
<bloom_alignment>1 or 0</bloom_alignment>
<factual_grounding>1 or 0</factual_grounding>
<distractor_quality>1 or 0</distractor_quality>
<question_clarity>1 or 0</question_clarity>
<legal_accuracy>1 or 0</legal_accuracy>
<issues>Mô tả cụ thể các vấn đề tìm thấy (nếu có). Nếu tất cả đạt, ghi "Không có vấn đề."</issues>
<suggestions>Gợi ý cải thiện cụ thể (nếu có). Nếu tất cả đạt, ghi "Không cần cải thiện."</suggestions>
</critique>
"""


# ─────────────────────────────────────────────
# Refinement Prompt Template
# ─────────────────────────────────────────────
REFINE_TEMPLATE = """\
You previously generated a QA pair that has quality issues. 
Please FIX the issues below and regenerate an IMPROVED version.

## Original Context:
{context_text}

## Your Previous QA Pair:
**Question:** {question}
**Candidates:**
{candidates}
**Ground Truth:** {ground_truth}
**Legal Rationale:** {legal_rationale}

## Critic's Feedback:
**Issues found:** {issues}
**Suggestions:** {suggestions}

## Target Bloom Level: **{bloom_level}**

## Instructions:
1. Fix ALL issues identified by the critic.
2. Follow the critic's suggestions.
3. Keep the same context and Bloom level.
4. Output the CORRECTED QA pair in XML format.

<qa_pair>
<question>Câu hỏi đã sửa...</question>
<candidate_answers>
A. ...
B. ...
C. ...
D. ...
</candidate_answers>
<ground_truth>A. ...</ground_truth>
<legal_rationale>Đại tiền đề: ... Tiểu tiền đề: ... Kết luận: ...</legal_rationale>
</qa_pair>

Respond with ONLY the XML tags.
"""


# ─────────────────────────────────────────────
# Refinement Trace (Logging)
# ─────────────────────────────────────────────
@dataclass
class RefinementTrace:
    """Records the full refinement history for one QA pair."""

    qa_id: str = ""
    bloom_level: str = ""
    chunk_id: str = ""
    total_loops: int = 0
    converged: bool = False

    # Per-loop records
    loop_records: list[dict[str, Any]] = field(default_factory=list)

    def add_loop(
        self,
        loop_num: int,
        qa_pair: dict[str, Any],
        critique: dict[str, Any] | None,
        passed: bool,
    ) -> None:
        self.loop_records.append({
            "loop": loop_num,
            "qa_snapshot": {
                "question": qa_pair.get("question", ""),
                "ground_truth": qa_pair.get("ground_truth", ""),
            },
            "critique": critique,
            "passed": passed,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "qa_id": self.qa_id,
            "bloom_level": self.bloom_level,
            "chunk_id": self.chunk_id,
            "total_loops": self.total_loops,
            "converged": self.converged,
            "loop_records": self.loop_records,
        }


# ─────────────────────────────────────────────
# Critique Parser
# ─────────────────────────────────────────────
import re

CRITIQUE_DIMENSIONS = (
    "bloom_alignment",
    "factual_grounding",
    "distractor_quality",
    "question_clarity",
    "legal_accuracy",
)


def parse_critique_xml(text: str) -> dict[str, Any] | None:
    """Parse critique scores from XML output.

    Returns dict with dimension scores + issues + suggestions,
    or None if parsing fails.
    """
    scores: dict[str, Any] = {}

    for dim in CRITIQUE_DIMENSIONS:
        match = re.search(
            rf"<{dim}>([\s\S]*?)</{dim}>", text, re.IGNORECASE
        )
        if match:
            val = match.group(1).strip()
            scores[dim] = 1.0 if "1" in val else 0.0
        else:
            return None  # Missing dimension → parse failure

    # Extract issues and suggestions
    issues_match = re.search(r"<issues>([\s\S]*?)</issues>", text, re.IGNORECASE)
    suggestions_match = re.search(r"<suggestions>([\s\S]*?)</suggestions>", text, re.IGNORECASE)

    scores["issues"] = issues_match.group(1).strip() if issues_match else ""
    scores["suggestions"] = suggestions_match.group(1).strip() if suggestions_match else ""

    # Check if all pass
    scores["all_pass"] = all(scores[d] == 1.0 for d in CRITIQUE_DIMENSIONS)

    return scores


def critique_passes(critique: dict[str, Any], threshold: float = 1.0) -> bool:
    """Check if a critique result passes the convergence threshold.

    Args:
        critique: Parsed critique dict with dimension scores.
        threshold: Fraction of dimensions that must pass (default: 1.0 = all).

    Returns:
        True if enough dimensions pass.
    """
    if not critique:
        return False

    pass_count = sum(1 for d in CRITIQUE_DIMENSIONS if critique.get(d, 0) == 1.0)
    return (pass_count / len(CRITIQUE_DIMENSIONS)) >= threshold


# ─────────────────────────────────────────────
# QA XML Parser (reused from TQA_Pipeline)
# ─────────────────────────────────────────────
def parse_qa_xml(text: str) -> list[dict[str, Any]]:
    """Extract QA pairs from LLM XML output via Regex.

    Reuses the same logic as TQA_Pipeline/src/03_qag_generator.py::_parse_qa_xml
    to maintain compatibility.
    """
    pairs: list[dict[str, Any]] = []

    blocks = re.findall(r"<qa_pair>(.*?)</qa_pair>", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        blocks = [text]

    for block in blocks:
        q_match = re.search(r"<question>(.*?)</question>", block, re.DOTALL | re.IGNORECASE)
        ca_match = re.search(r"<candidate_answers>(.*?)</candidate_answers>", block, re.DOTALL | re.IGNORECASE)
        gt_match = re.search(r"<ground_truth>(.*?)</ground_truth>", block, re.DOTALL | re.IGNORECASE)
        lr_match = re.search(r"<legal_rationale>(.*?)</legal_rationale>", block, re.DOTALL | re.IGNORECASE)

        if q_match and ca_match and gt_match and lr_match:
            ca_text = ca_match.group(1).strip()
            lines = [ln.strip() for ln in ca_text.split("\n") if ln.strip()]
            candidates = [ln for ln in lines if re.match(r"^[A-E][\.\)]\s*", ln)]
            if len(candidates) < 2:
                candidates = lines

            pairs.append({
                "question": q_match.group(1).strip(),
                "candidate_answers": [c.strip() for c in candidates],
                "ground_truth": gt_match.group(1).strip(),
                "legal_rationale": lr_match.group(1).strip(),
            })

    return pairs


# ─────────────────────────────────────────────
# Self-Refine Loop Engine
# ─────────────────────────────────────────────
class IterativeQAGenerator:
    """Self-Refine loop for QA generation.

    Architecture:
        Generator (Qwen2.5-7B) → generates/refines QA pairs
        Critic    (Gemma-2-2b)  → evaluates QA quality (cross-family anti-bias)

    The generator and critic are DIFFERENT model families to avoid
    self-reinforcement bias (established in Paper 1).

    Args:
        generator_engine: Callable that takes a prompt string and returns generated text.
        critic_engine: Callable that takes a prompt string and returns critique text.
        max_loops: Maximum refinement iterations per QA pair.
        adaptive: If True, stop early when critic approves.
        convergence_threshold: Fraction of critique dimensions that must pass.
    """

    def __init__(
        self,
        generator_engine: Any = None,
        critic_engine: Any = None,
        max_loops: int = 3,
        adaptive: bool = True,
        convergence_threshold: float = 1.0,
    ) -> None:
        self.generator_engine = generator_engine
        self.critic_engine = critic_engine
        self.max_loops = max_loops
        self.adaptive = adaptive
        self.convergence_threshold = convergence_threshold

    def _generate_initial(
        self,
        context: dict[str, Any],
        bloom_level: str,
        n_questions: int = 1,
    ) -> str:
        """Build generation prompt and call generator engine."""
        visual_ctx = ""
        if context.get("visual_descriptions"):
            descs = [d.get("summary", "") for d in context["visual_descriptions"]]
            visual_ctx = "## Visual Information:\n" + "\n".join(
                f"- Image: {d}" for d in descs if d
            )

        prompt = QA_GENERATION_TEMPLATE.format(
            n_questions=n_questions,
            bloom_level=bloom_level,
            bloom_definition=BLOOM_DEFINITIONS.get(bloom_level, ""),
            context_text=context.get("text", "")[:2000],
            visual_context=visual_ctx,
            bloom_requirement=BLOOM_REQUIREMENTS.get(bloom_level, ""),
        )

        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        return self.generator_engine(full_prompt)

    def _critique(
        self,
        qa_pair: dict[str, Any],
        context: dict[str, Any],
        bloom_level: str,
    ) -> dict[str, Any] | None:
        """Build critique prompt and call critic engine."""
        candidates_str = "\n".join(qa_pair.get("candidate_answers", []))

        prompt = CRITIQUE_TEMPLATE.format(
            bloom_level=bloom_level,
            context_text=context.get("text", "")[:1500],
            question=qa_pair.get("question", ""),
            candidates=candidates_str,
            ground_truth=qa_pair.get("ground_truth", ""),
            legal_rationale=qa_pair.get("legal_rationale", ""),
        )

        full_prompt = f"{CRITIQUE_SYSTEM_PROMPT}\n\n{prompt}"
        response = self.critic_engine(full_prompt)
        return parse_critique_xml(response)

    def _refine(
        self,
        qa_pair: dict[str, Any],
        critique: dict[str, Any],
        context: dict[str, Any],
        bloom_level: str,
    ) -> str:
        """Build refinement prompt and call generator engine."""
        candidates_str = "\n".join(qa_pair.get("candidate_answers", []))

        prompt = REFINE_TEMPLATE.format(
            context_text=context.get("text", "")[:1500],
            question=qa_pair.get("question", ""),
            candidates=candidates_str,
            ground_truth=qa_pair.get("ground_truth", ""),
            legal_rationale=qa_pair.get("legal_rationale", ""),
            issues=critique.get("issues", ""),
            suggestions=critique.get("suggestions", ""),
            bloom_level=bloom_level,
        )

        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        return self.generator_engine(full_prompt)

    def generate_with_refinement(
        self,
        context: dict[str, Any],
        bloom_level: str,
        n_questions: int = 1,
    ) -> tuple[list[dict[str, Any]], RefinementTrace]:
        """Run the full Self-Refine loop for one context × bloom level.

        Returns:
            Tuple of (refined_qa_pairs, refinement_trace)
        """
        chunk_id = context.get("chunk_id", "unknown")
        trace = RefinementTrace(
            bloom_level=bloom_level,
            chunk_id=chunk_id,
        )

        # ── Step 1: Initial Generation ──
        logger.info("  [Loop 0] GENERATE: %s × %s", chunk_id, bloom_level)
        raw_output = self._generate_initial(context, bloom_level, n_questions)
        qa_pairs = parse_qa_xml(raw_output)

        if not qa_pairs:
            logger.warning("  Generation produced no parseable QA pairs for %s", chunk_id)
            trace.total_loops = 0
            return [], trace

        # Process each QA pair through the loop
        refined_pairs: list[dict[str, Any]] = []
        for qi, qa in enumerate(qa_pairs):
            qa_id = f"{chunk_id}_{bloom_level.lower()}_{qi}"
            trace.qa_id = qa_id
            current_qa = qa

            for loop_num in range(1, self.max_loops + 1):
                # ── Step 2: Critique ──
                logger.info("  [Loop %d] CRITIQUE: %s", loop_num, qa_id)
                critique = self._critique(current_qa, context, bloom_level)

                if critique is None:
                    logger.warning("  Critique parse failed at loop %d for %s", loop_num, qa_id)
                    trace.add_loop(loop_num, current_qa, None, passed=False)
                    break

                passed = critique_passes(critique, self.convergence_threshold)
                trace.add_loop(loop_num, current_qa, critique, passed=passed)

                if passed and self.adaptive:
                    logger.info("  [Loop %d] CONVERGED: %s (all dimensions pass)", loop_num, qa_id)
                    trace.converged = True
                    trace.total_loops = loop_num
                    break

                # ── Step 3: Refine ──
                logger.info("  [Loop %d] REFINE: %s (issues: %s)", loop_num, qa_id, critique.get("issues", ""))
                refined_output = self._refine(current_qa, critique, context, bloom_level)
                refined_parsed = parse_qa_xml(refined_output)

                if refined_parsed:
                    current_qa = refined_parsed[0]
                else:
                    logger.warning("  Refinement parse failed at loop %d, keeping previous version", loop_num)

                trace.total_loops = loop_num

            # Add metadata — MUST include context for Phase B prompt building
            current_qa["refinement_loops"] = trace.total_loops
            current_qa["converged"] = trace.converged
            current_qa["qa_id"] = qa_id
            current_qa["chunk_id"] = chunk_id
            current_qa["context_text"] = context.get("text", "")
            refined_pairs.append(current_qa)

        return refined_pairs, trace


# ─────────────────────────────────────────────
# Ablation Mode Runner
# ─────────────────────────────────────────────
def run_ablation(
    mode: str,
    generator_engine: Any,
    critic_engine: Any,
    contexts: list[dict[str, Any]],
    bloom_levels: tuple[str, ...],
    n_questions: int = 1,
) -> tuple[list[dict[str, Any]], list[RefinementTrace]]:
    """Run QA generation with a specific ablation mode.

    Modes:
        single_pass: No refinement (baseline, equivalent to Paper 1)
        fixed_2:     Fixed 2 refinement loops
        fixed_3:     Fixed 3 refinement loops
        adaptive:    Adaptive — stop when critic approves, max 3 loops

    Returns:
        Tuple of (all_qa_pairs, all_traces)
    """
    mode_config = {
        "single_pass": {"max_loops": 0, "adaptive": False},
        "fixed_2": {"max_loops": 2, "adaptive": False},
        "fixed_3": {"max_loops": 3, "adaptive": False},
        "adaptive": {"max_loops": 3, "adaptive": True},
    }

    if mode not in mode_config:
        raise ValueError(f"Unknown ablation mode: {mode}. Choose from: {list(mode_config.keys())}")

    cfg = mode_config[mode]
    logger.info("Running ablation mode: %s (max_loops=%d, adaptive=%s)", mode, cfg["max_loops"], cfg["adaptive"])

    all_pairs: list[dict[str, Any]] = []
    all_traces: list[RefinementTrace] = []

    if mode == "single_pass":
        # Batched generation: build all prompts upfront, one GPU call
        gen = IterativeQAGenerator(
            generator_engine=generator_engine,
            critic_engine=critic_engine,
            max_loops=0,
            adaptive=False,
        )

        # Build all prompts first
        prompt_jobs = []  # (context, bloom_level)
        prompts = []
        for ctx in contexts:
            for bloom in bloom_levels:
                visual_ctx = ""
                if ctx.get("visual_descriptions"):
                    descs = [d.get("summary", "") for d in ctx["visual_descriptions"]]
                    visual_ctx = "## Visual Information:\n" + "\n".join(
                        f"- Image: {d}" for d in descs if d
                    )
                prompt = QA_GENERATION_TEMPLATE.format(
                    n_questions=n_questions,
                    bloom_level=bloom,
                    bloom_definition=BLOOM_DEFINITIONS.get(bloom, ""),
                    context_text=ctx.get("text", "")[:2000],
                    visual_context=visual_ctx,
                    bloom_requirement=BLOOM_REQUIREMENTS.get(bloom, ""),
                )
                prompts.append(f"{SYSTEM_PROMPT}\n\n{prompt}")
                prompt_jobs.append((ctx, bloom))

        # Single batch generation call (vLLM continuous batching)
        logger.info("  single_pass: batching %d prompts (%d contexts × %d blooms)",
                     len(prompts), len(contexts), len(bloom_levels))
        try:
            outputs = generator_engine(prompts)
            if not isinstance(outputs, list):
                outputs = [outputs]
        except TypeError:
            # Fallback: sequential if engine doesn't support batch
            outputs = [generator_engine(p) for p in prompts]

        # Parse results
        for (ctx, bloom), raw in zip(prompt_jobs, outputs):
            chunk_id = ctx.get("chunk_id", "unknown")
            pairs = parse_qa_xml(raw)
            trace = RefinementTrace(
                bloom_level=bloom,
                chunk_id=chunk_id,
                total_loops=0,
                converged=True,
            )
            for qi, p in enumerate(pairs):
                p["refinement_loops"] = 0
                p["converged"] = True
                p["bloom_level"] = bloom
                p["qa_id"] = f"{chunk_id}_{bloom.lower()}_{qi}"
                p["chunk_id"] = chunk_id
                p["context_text"] = ctx.get("text", "")
            all_pairs.extend(pairs)
            all_traces.append(trace)
    else:
        gen = IterativeQAGenerator(
            generator_engine=generator_engine,
            critic_engine=critic_engine,
            max_loops=cfg["max_loops"],
            adaptive=cfg["adaptive"],
        )
        for ctx in contexts:
            for bloom in bloom_levels:
                pairs, trace = gen.generate_with_refinement(ctx, bloom, n_questions)
                for p in pairs:
                    p["bloom_level"] = bloom
                    # context_text, qa_id, chunk_id already set in generate_with_refinement
                all_pairs.extend(pairs)
                all_traces.append(trace)

    logger.info(
        "Ablation '%s' complete: %d QA pairs generated, avg loops=%.1f",
        mode,
        len(all_pairs),
        sum(t.total_loops for t in all_traces) / max(len(all_traces), 1),
    )

    return all_pairs, all_traces
