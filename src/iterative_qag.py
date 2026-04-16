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
Below is a specific legal text. Your task is to generate {n_questions} high-quality Vietnamese law examination questions based ONLY on this content.

## Context:
{context_text}

{visual_context}

## Bloom's Taxonomy Level Architecture:
**Target Level**: {bloom_level}
**Definition**: {bloom_definition}

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
8. **STRICTLY PROHIBITED** — Do NOT generate questions about:
   - Book/textbook authors, editors (chủ biên), compilers, or publishing information.
   - Structural metadata: chapter numbers, page numbers, table of contents, foreword/preface content.
   - Institutional names (university, faculty) unless they appear as parties in a legal scenario.
   Focus ONLY on the **substantive legal content**: legal rules, principles, definitions, case applications.

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
You are a senior Vietnamese law professor with over 20 years of experience in Civil Law, \
Criminal Law, and Administrative Law, combined with deep expertise in educational assessment \
for national-level Vietnamese legal examinations (bar exams, civil service exams, university finals).

Your role is to act as a rigorous, impartial quality judge for automatically-generated \
multiple-choice exam questions. You are NOT a general AI assistant — you are a legal expert \
who deeply understands Vietnamese legal doctrine, Bloom's Taxonomy in legal pedagogy, \
and the standards expected of high-quality Vietnamese law examination questions.

Evaluation principles:
- Ground every judgment in the provided context. Do NOT approve facts not found in the context.
- Reject immediately (score 0) any question that asks about metadata: book authors, editors, \
  publishers, chapter numbers, or textbook structural content. This is not substantive legal content.
- Distractors (wrong answers) must be legally plausible — not obviously absurd to a law student.
- Legal Syllogism must be logically sound: Major Premise (general rule) → Minor Premise (facts) → Conclusion.
- The Bloom level must be genuine — a "Create" question that merely asks for recall is wrong.

You MUST evaluate ALL 5 dimensions. Be specific: cite the exact phrase causing the problem.
Respond using strict XML tags only. No text before or after the XML.
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


# ─────────────────────────────────────────────
# PLAN 2: Batched Adaptive Pipeline
# ─────────────────────────────────────────────
def _build_gen_prompt(ctx: dict[str, Any], bloom: str, n_questions: int) -> str:
    """Build a single generation prompt for (context, bloom) pair."""
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
    return f"{SYSTEM_PROMPT}\n\n{prompt}"


def _build_critique_prompt(qa: dict[str, Any], bloom: str, context_text: str) -> str:
    """Build a critique prompt for an existing QA pair."""
    candidates_str = "\n".join(f"  {c}" for c in qa.get("candidate_answers", []))
    prompt = CRITIQUE_TEMPLATE.format(
        bloom_level=bloom,
        context_text=context_text[:2000],
        question=qa.get("question", ""),
        candidates=candidates_str,
        ground_truth=qa.get("ground_truth", ""),
        legal_rationale=qa.get("legal_rationale", ""),
    )
    return f"{CRITIQUE_SYSTEM_PROMPT}\n\n{prompt}"


def _build_refine_prompt(qa: dict[str, Any], critique: dict[str, Any], bloom: str, context_text: str) -> str:
    """Build a refinement prompt given a QA pair and its critique."""
    candidates_str = "\n".join(f"  {c}" for c in qa.get("candidate_answers", []))
    return f"{SYSTEM_PROMPT}\n\n" + REFINE_TEMPLATE.format(
        context_text=context_text[:2000],
        question=qa.get("question", ""),
        candidates=candidates_str,
        ground_truth=qa.get("ground_truth", ""),
        legal_rationale=qa.get("legal_rationale", ""),
        issues=critique.get("issues", "Không rõ."),
        suggestions=critique.get("suggestions", "Không rõ."),
        bloom_level=bloom,
    )


def run_batched_adaptive(
    generator_factory: Any,             # Callable[[], engine] — creates Qwen3-8B engine
    critic_factory: Any,                # Callable[[], engine] — creates Gemma-3-4b engine
    contexts: list[dict[str, Any]],
    bloom_levels: tuple[str, ...],
    n_questions: int = 1,
    max_loops: int = 3,
    checkpoint_dir: Path | None = None,
    sync_callback: Any | None = None,
) -> list[dict[str, Any]]:
    """Batched Adaptive QA Generation Pipeline.

    Pass grouping strategy (optimal GPU utilization):
        Each pass loads ONE model at 90% VRAM (full KV cache),
        runs ALL pending prompts as one giant batch, then unloads.
        Only 6 model swaps total for ALL 47k chunks.

        Pass 1: Load G  → generate_batch(all N)       → Unload G
        Loop (max_loops):
          Pass 2: Load C  → critique_batch(N)         → Unload C  
          Pass 3: Load G  → refine_batch(~30% failed) → Unload G
        Final: accept remaining

    Checkpoint strategy:
        - After Pass 1: raw QA saved to pending.json + Drive sync
        - After each Loop: validated pairs saved to qa_pairs.json + Drive sync
        - Resume: loads qa_pairs.json (validated) OR pending.json (pre-critique)

    Args:
        generator_factory: Callable returning a loaded generator engine.
        critic_factory:    Callable returning a loaded critic engine.
        contexts:         List of chunk records with 'text', 'chunk_id', etc.
        bloom_levels:     Tuple of Bloom taxonomy level names.
        n_questions:      Questions per (chunk x bloom) pair.
        max_loops:        Maximum critique-refine iterations.
        checkpoint_dir:   Directory to save/resume progress.
        sync_callback:    Optional callable() that uploads checkpoint_dir to Drive.

    Returns:
        List of finalized QA pair dicts.
    """
    import json as _json

    all_pairs: list[dict[str, Any]] = []
    done_keys: set[str] = set()

    # ── Resume from checkpoint ──────────────────────────────────
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cp_file = checkpoint_dir / "qa_pairs.json"
        if cp_file.exists():
            try:
                with open(cp_file, "r", encoding="utf-8") as f:
                    existing = _json.load(f)
                all_pairs.extend(existing)
                done_keys = {f"{p.get('chunk_id')}_{p.get('bloom_level')}" for p in existing}
                logger.info("[Resume] Loaded %d completed QA pairs. Skipping done jobs.", len(existing))
            except Exception as e:
                logger.warning("[Resume] Failed loading checkpoint: %s", e)

    def _save_checkpoint() -> None:
        """Write qa_pairs.json locally then push to Drive immediately."""
        if checkpoint_dir:
            with open(checkpoint_dir / "qa_pairs.json", "w", encoding="utf-8") as f:
                _json.dump(all_pairs, f, ensure_ascii=False, indent=2)
            if sync_callback:
                try:
                    sync_callback()
                except Exception as e:
                    logger.warning("[Checkpoint] Drive sync failed (local copy safe): %s", e)

    # ── Build pending job list ──────────────────────────────────
    # Each job = (context_dict, bloom_level_str)
    jobs = [
        (ctx, bloom)
        for ctx in contexts
        for bloom in bloom_levels
        if f"{ctx.get('chunk_id')}_{bloom}" not in done_keys
    ]
    logger.info(
        "[BatchedAdaptive] %d pending jobs (%d skipped from checkpoint).",
        len(jobs), len(done_keys),
    )
    if not jobs:
        logger.info("[BatchedAdaptive] All jobs already done. Returning checkpoint data.")
        return all_pairs

    # ── Resume: check pending.json (pre-critique raw cache) ────
    pending_file = checkpoint_dir / "pending.json" if checkpoint_dir else None
    pending_cache: list[tuple[dict[str, Any] | None, dict[str, Any], str]] | None = None
    if pending_file and pending_file.exists() and done_keys == set():
        try:
            with open(pending_file, "r", encoding="utf-8") as f:
                raw_pending = _json.load(f)
            # Reconstruct qa_batch from persisted pending records
            pending_cache = [
                (r.get("qa"), {"chunk_id": r["chunk_id"], "text": r["context_text"],
                               "source_doc": r.get("source_doc", "")}, r["bloom"])
                for r in raw_pending
            ]
            logger.info("[Resume] Loaded %d pending QA from pass-1 cache (skipping re-generate).", len(pending_cache))
        except Exception as e:
            logger.warning("[Resume] Could not load pending.json: %s", e)
            pending_cache = None

    # ── Pass 1: Batch Generate ALL pending (or restore from cache) ─
    qa_batch = []
    start_idx = 0
    if pending_cache is not None:
        qa_batch = pending_cache
        start_idx = len(qa_batch)
        if start_idx == len(jobs):
            logger.info("[BatchedAdaptive] Pass 1 completely skipped — using %d cached QA pairs.", start_idx)
        elif start_idx < len(jobs):
            logger.info("[BatchedAdaptive] Pass 1 partially skipped — loaded %d/%d cached QA pairs. Resuming the rest...", start_idx, len(jobs))
    
    if start_idx < len(jobs):
        remaining_jobs = jobs[start_idx:]
        logger.info("[BatchedAdaptive] Pass 1 — Load Generator, batch generate %d remaining prompts...", len(remaining_jobs))
        generator = generator_factory()          # Load Qwen3-8B at 90% VRAM
        
        # SUB_BATCH=5000 is the proven sweet spot:
        # - generate() is blocking, checkpoint runs AFTER each sub-batch returns
        # - Warm-up overhead: ~2 min / 2.5 hr batch = ~1.3% (acceptable)
        # - Safe interrupt window: between sub-batches (~every 2.5 hours)
        # - Max data loss if crash: one sub-batch = ~2.5 hours of work
        # (AsyncLLMEngine would give zero overhead but requires full refactor)
        SUB_BATCH = 5000
        for i in range(0, len(remaining_jobs), SUB_BATCH):
            batch_jobs = remaining_jobs[i : i + SUB_BATCH]
            batch_prompts = [_build_gen_prompt(ctx, bloom, n_questions) for ctx, bloom in batch_jobs]
            
            logger.info("  -> Processing sub-batch %d to %d (out of %d remaining)...", i + 1, min(i + SUB_BATCH, len(remaining_jobs)), len(remaining_jobs))
            out = generator.generate_batch(batch_prompts)
            if not isinstance(out, list): out = [out]
            
            # Parse XML and append to global qa_batch
            new_count = 0
            for raw, (ctx, bloom) in zip(out, batch_jobs):
                parsed_list = parse_qa_xml(raw)
                qa = parsed_list[0] if parsed_list else None
                qa_batch.append((qa, ctx, bloom))
                new_count += 1

                # ── Incremental Checkpoint: Save every 100 NEW items ──
                if new_count % 100 == 0 and checkpoint_dir is not None:
                    pending_file = checkpoint_dir / "pending.json"
                    try:
                        persisted = [
                            {"chunk_id": c.get("chunk_id", "unknown"), "context_text": c.get("text", ""),
                             "source_doc": c.get("source_doc", ""), "bloom": b, "qa": q}
                            for q, c, b in qa_batch
                        ]
                        with open(pending_file, "w", encoding="utf-8") as f:
                            _json.dump(persisted, f, ensure_ascii=False, indent=2)

                        # Sync to Drive every 500 NEW items
                        if new_count % 500 == 0 and sync_callback:
                            sync_callback()
                            msg = f"[BatchedAdaptive] 💾 Drive Synced at {len(qa_batch)} total records (new: {new_count})"
                            logger.info(msg)
                            print(msg, flush=True)
                    except Exception as e:
                        logger.error("Incremental save failed: %s", e)
            
        generator.unload()                       # Free VRAM before next model
        del generator

        logger.info("[BatchedAdaptive] Pass 1 complete. %d/%d valid QA pairs parsed.", sum(1 for q, _, _ in qa_batch if q), len(jobs))

    # ── Loop: Batch Critique → Batch Refine failed ─────────────
    for loop_idx in range(max_loops):
        active = [(qa, ctx, bloom) for qa, ctx, bloom in qa_batch if qa is not None]
        logger.info(
            "[BatchedAdaptive] Pass Critique (loop %d/%d) — %d active QA pairs...",
            loop_idx + 1, max_loops, len(active),
        )
        if not active:
            break

        # Batch Critique — load Critic at full VRAM, run, unload
        logger.info("[BatchedAdaptive] Loop %d — Load Critic, critique %d pairs...", loop_idx + 1, len(active))
        critic = critic_factory()                # Load Gemma-3-4b at 90% VRAM
        crit_prompts = [
            _build_critique_prompt(qa, bloom, ctx.get("text", ""))
            for qa, ctx, bloom in active
        ]
        
        SUB_BATCH = 20000 # Critic uses less compute, can handle larger sub-batch
        crit_outputs = []
        for i in range(0, len(crit_prompts), SUB_BATCH):
            batch_slice = crit_prompts[i : i + SUB_BATCH]
            logger.info("  -> Critic sub-batch %d to %d...", i + 1, min(i + SUB_BATCH, len(crit_prompts)))
            out = critic.generate_batch(batch_slice)
            if not isinstance(out, list): out = [out]
            crit_outputs.extend(out)
            
        critic.unload()                          # Free VRAM
        del critic

        # Separate passed vs failed
        passed_batch: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        failed_batch: list[tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]] = []

        for (qa, ctx, bloom), crit_raw in zip(active, crit_outputs):
            critique = parse_critique_xml(crit_raw) if crit_raw else None
            if critique and critique.get("all_pass"):
                passed_batch.append((qa, ctx, bloom))
            else:
                failed_batch.append((qa, ctx, bloom, critique or {}))

        # Accept passed items immediately
        for qa, ctx, bloom in passed_batch:
            chunk_id = ctx.get("chunk_id", "unknown")
            qa.update({
                "bloom_level": bloom,
                "qa_id": f"{chunk_id}_{bloom.lower()}_0",
                "chunk_id": chunk_id,
                "source_doc": ctx.get("source_doc", ""),
                "context_text": ctx.get("text", ""),
                "refinement_loops": loop_idx + 1,
                "converged": True,
            })
            all_pairs.append(qa)

        logger.info(
            "[BatchedAdaptive] Loop %d: %d passed ✅, %d failed → refining...",
            loop_idx + 1, len(passed_batch), len(failed_batch),
        )

        # On last loop: accept all remaining failed as-is
        if loop_idx == max_loops - 1 or not failed_batch:
            for qa, ctx, bloom, _ in failed_batch:
                if qa is None:
                    continue
                chunk_id = ctx.get("chunk_id", "unknown")
                qa.update({
                    "bloom_level": bloom,
                    "qa_id": f"{chunk_id}_{bloom.lower()}_0",
                    "chunk_id": chunk_id,
                    "source_doc": ctx.get("source_doc", ""),
                    "context_text": ctx.get("text", ""),
                    "refinement_loops": max_loops,
                    "converged": False,
                })
                all_pairs.append(qa)
            _save_checkpoint()   # saves local + syncs Drive
            logger.info("[BatchedAdaptive] 💾 Loop %d complete — checkpoint pushed to Drive.", loop_idx + 1)
            # Clean up pending.json now that validated pairs are saved
            if pending_file and pending_file.exists():
                pending_file.unlink(missing_ok=True)
            break

        # Batch Refine — load Generator at full VRAM, run, unload
        logger.info("[BatchedAdaptive] Loop %d — Load Generator, refine %d failed pairs...", loop_idx + 1, len(failed_batch))
        generator = generator_factory()          # Load Qwen3-8B at 90% VRAM
        refine_prompts = [
            _build_refine_prompt(qa, crit, bloom, ctx.get("text", ""))
            for qa, ctx, bloom, crit in failed_batch
        ]
        
        SUB_BATCH = 15000
        refine_outputs = []
        for i in range(0, len(refine_prompts), SUB_BATCH):
            batch_slice = refine_prompts[i : i + SUB_BATCH]
            logger.info("  -> Refine sub-batch %d to %d...", i + 1, min(i + SUB_BATCH, len(refine_prompts)))
            out = generator.generate_batch(batch_slice)
            if not isinstance(out, list): out = [out]
            refine_outputs.extend(out)
            
        generator.unload()                       # Free VRAM
        del generator

        # Replace qa_batch for next loop iteration (only active = previously failed, now refined)
        qa_batch = []
        for (_, ctx, bloom, _), raw in zip(failed_batch, refine_outputs):
            parsed_list = parse_qa_xml(raw)
            qa = parsed_list[0] if parsed_list else None
            qa_batch.append((qa, ctx, bloom))

        # After each refine pass: checkpoint + Drive sync
        _save_checkpoint()
        logger.info("[BatchedAdaptive] 💾 Loop %d refine done — checkpoint pushed to Drive. (%d pairs saved)",
                     loop_idx + 1, len(all_pairs))

    logger.info(
        "[BatchedAdaptive] Complete: %d QA pairs total (%.1f%% converged).",
        len(all_pairs),
        100 * sum(1 for p in all_pairs if p.get("converged")) / max(len(all_pairs), 1),
    )
    return all_pairs



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
    checkpoint_dir: Path | None = None,
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
    all_traces: list[Any] = []
    done_keys: set[str] = set()

    # Load checkpoint if exists
    import json
    if checkpoint_dir:
        qa_file = checkpoint_dir / "qa_pairs.json"
        traces_file = checkpoint_dir / "traces.json"
        if qa_file.exists():
            try:
                with open(qa_file, "r", encoding="utf-8") as f:
                    old_pairs = json.load(f)
                    all_pairs.extend(old_pairs)
                    done_keys = {f"{p.get('chunk_id')}_{p.get('bloom_level')}" for p in old_pairs}
                logger.info("  [Resume] Loaded %d completed QA pairs from checkpoint.", len(old_pairs))
            except Exception as e:
                logger.warning("  [Resume] Failed to load qa_pairs checkpoint: %s", e)
        if traces_file.exists():
            try:
                with open(traces_file, "r", encoding="utf-8") as f:
                    all_traces.extend(json.load(f))
            except Exception: pass

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
        # Collect (context, bloom_level) jobs, skipping done ones
        jobs = []
        for ctx in contexts:
            for bloom in bloom_levels:
                key = f"{ctx.get('chunk_id')}_{bloom}"
                if key not in done_keys:
                    jobs.append((ctx, bloom))

        total = len(jobs)
        logger.info("  found %d pending jobs (skipped %d done).", total, len(contexts)*len(bloom_levels) - total)

        for idx, (ctx, bloom) in enumerate(jobs, 1):
            pairs, trace = gen.generate_with_refinement(ctx, bloom, n_questions)
            for p in pairs:
                p["bloom_level"] = bloom
            all_pairs.extend(pairs)
            all_traces.append(trace.to_dict())
            
            # Progress logging
            if idx % max(1, total // 20) == 0:
                logger.info("  [%s] Progress: %d/%d contexts×blooms processed", mode, idx, total)

            # Auto-save checkpoint every 1000 items
            if checkpoint_dir and (idx % 1000 == 0 or idx == total):
                with open(checkpoint_dir / "qa_pairs.json", "w", encoding="utf-8") as f:
                    json.dump(all_pairs, f, ensure_ascii=False, indent=2)
                with open(checkpoint_dir / "traces.json", "w", encoding="utf-8") as f:
                    json.dump(all_traces, f, ensure_ascii=False, indent=2)
                logger.info("  💾 [Checkpoint] Auto-saved %d pairs at job %d.", len(all_pairs), idx)

    logger.info(
        "Ablation '%s' complete: %d QA pairs generated, avg loops=%.1f",
        mode,
        len(all_pairs),
        sum(t.total_loops for t in all_traces) / max(len(all_traces), 1),
    )

    return all_pairs, all_traces
