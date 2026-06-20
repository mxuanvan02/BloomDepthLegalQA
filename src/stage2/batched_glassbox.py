"""
Stage 2 / Batched glass-box runner
==================================
Wires the GPU-free glass-box decision core (`stage2.judge.Judge`,
`stage2.refiner.Refiner`) into the batched, model-swap-economical engine used by
production, WITHOUT collapsing to one-LLM-call-per-item.

Design (preserves the 6-model-swap economy of run_batched_adaptive):
  Phase G  : load Generator once -> generate ALL -> unload.
  Phase J  : load Critic once -> score ALL (m samples each, ONE big batch)
             -> unload -> run Judge.evaluate per item reading ONLY the cache
             (pure CPU, zero LLM calls in the per-item loop).
  Phase R  : load Generator once -> refine the refine-zone items (one batch,
             external-judge framing on lowest_dim) -> unload -> re-judge next loop.

Per question: m critic calls (m=5 default), NOT 6*m, NOT 1-call-per-item-sync.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from stage2.judge import (
    Judge, JudgeConfig, RubricScores, RUBRIC_DIMS,
    expected_score,
)
from stage2.refiner import build_external_feedback
from stage2.production_adapter import (
    build_rubric_prompt, parse_rubric_scores, _samples_to_dist,
)

logger = logging.getLogger(__name__)


def batch_score_critic(
    items: list[tuple[dict, dict, str]],
    critic_generate: Callable[[list[str]], list[str]],
    m: int = 5,
    fallback_rating: int = 3,
) -> dict[int, dict[str, dict[int, float]]]:
    """Score every item with m self-consistency samples in ONE batched call.

    `items`: list of (qa, ctx_dict, bloom). `critic_generate(prompts)->texts`.
    Returns: index -> {dim -> {rating: prob}} distribution cache.

    Cost: len(items)*m critic prompts in a single batch (the engine sub-batches
    internally). This is the m-per-question cost, computed once per loop.
    """
    prompts: list[str] = []
    owner: list[int] = []  # which item each prompt belongs to
    for idx, (qa, ctx, bloom) in enumerate(items):
        rubric_prompt = build_rubric_prompt(qa, ctx.get("text", ""), bloom)
        for _ in range(m):
            prompts.append(rubric_prompt)
            owner.append(idx)

    raws = critic_generate(prompts)
    if not isinstance(raws, list):
        raws = [raws]

    # Collect per-item per-dim samples.
    samples: dict[int, dict[str, list[int]]] = {
        i: {d: [] for d in RUBRIC_DIMS} for i in range(len(items))
    }
    for raw, idx in zip(raws, owner):
        parsed = parse_rubric_scores(raw)
        if parsed is None:
            continue
        for d in RUBRIC_DIMS:
            samples[idx][d].append(parsed[d])

    cache: dict[int, dict[str, dict[int, float]]] = {}
    for i in range(len(items)):
        dists: dict[str, dict[int, float]] = {}
        for d in RUBRIC_DIMS:
            s = samples[i][d]
            dists[d] = _samples_to_dist(s) if s else {fallback_rating: 1.0}
        cache[i] = dists
    return cache


def make_cache_judge(
    cache: dict[int, dict[str, dict[int, float]]],
    config: JudgeConfig,
) -> Callable[[int, dict, str], Any]:
    """Return `judge_one(idx, qa, ctx_text) -> JudgeResult` that reads ONLY the
    precomputed distribution cache — zero LLM calls in the per-item loop.

    The injected score_fn closes over the current item's idx so Judge's 6
    per-dim calls all hit the same cached distributions.
    """
    def judge_one(idx: int, qa: dict, ctx_text: str):
        def score_fn(_qa, _ctx, dim):
            return cache[idx][dim]
        judge = Judge(config, score_fn, judge_models=["gemma-3-4b"])
        return judge.evaluate(qa, ctx_text)

    return judge_one


def split_by_decision(
    items: list[tuple[dict, dict, str]],
    cache: dict[int, dict[str, dict[int, float]]],
    config: JudgeConfig,
) -> tuple[list, list, list]:
    """Run the cache-backed Judge over all items (CPU only) and split into
    (accepted, refine_zone, rejected). Each entry carries (item, judge_result).
    """
    judge_one = make_cache_judge(cache, config)
    accepted, refine_zone, rejected = [], [], []
    for idx, (qa, ctx, bloom) in enumerate(items):
        jr = judge_one(idx, qa, ctx.get("text", ""))
        rec = ((qa, ctx, bloom), jr)
        if jr.decision == "accept":
            accepted.append(rec)
        elif jr.decision == "refine":
            refine_zone.append(rec)
        else:
            rejected.append(rec)
    return accepted, refine_zone, rejected


def batch_refine(
    refine_zone: list,
    generator_generate: Callable[[list[str]], list[str]],
    build_refine_prompt: Callable[..., str],
    parse_qa: Callable[[str], list[dict]],
) -> list[tuple[dict, dict, str]]:
    """Refine all refine-zone items in ONE batched Generator call.

    For each item, target the Judge's lowest_dim with external-judge framing
    (arXiv:2606.05976). Returns refreshed (qa, ctx, bloom) items for re-judging.
    """
    prompts: list[str] = []
    meta: list[tuple[dict, dict, str]] = []
    for (qa, ctx, bloom), jr in refine_zone:
        target = jr.lowest_dim
        score = jr.rubric_scores.get(target)
        reason = build_external_feedback(target, score)
        critique_like = {"issues": reason, "suggestions": reason}
        prompts.append(build_refine_prompt(qa, critique_like, bloom, ctx.get("text", "")))
        meta.append((qa, ctx, bloom))

    outs = generator_generate(prompts) if prompts else []
    if not isinstance(outs, list):
        outs = [outs]

    refreshed: list[tuple[dict, dict, str]] = []
    for (qa, ctx, bloom), raw in zip(meta, outs):
        parsed = parse_qa(raw)
        new_qa = parsed[0] if parsed else dict(qa)
        for keep in ("qa_id", "chunk_id", "bloom_level", "source_doc",
                     "eligible_bloom_levels"):
            if keep in qa and keep not in new_qa:
                new_qa[keep] = qa[keep]
        refreshed.append((new_qa, ctx, bloom))
    return refreshed


def finalize(qa: dict, ctx: dict, bloom: str, jr: Any, loops: int,
             converged: bool) -> dict:
    """Attach decision provenance + identity to an accepted/finalized QA pair."""
    chunk_id = ctx.get("chunk_id", "unknown")
    qa.update({
        "bloom_level": bloom,
        "qa_id": qa.get("qa_id", f"{chunk_id}__{bloom.lower()}__0"),
        "chunk_id": chunk_id,
        "source_doc": ctx.get("source_doc", ""),
        "context_text": ctx.get("text", ""),
        "refinement_loops": loops,
        "converged": converged,
        "judge": jr.to_dict() if hasattr(jr, "to_dict") else None,
    })
    return qa


def run_glassbox_loops(
    qa_batch: list[tuple[dict, dict, str]],
    critic_factory: Callable[[], Any],
    generator_factory: Callable[[], Any],
    build_refine_prompt: Callable[..., str],
    parse_qa: Callable[[str], list[dict]],
    judge_config: JudgeConfig,
    max_loops: int = 3,
    m_samples: int = 5,
    save_checkpoint: Callable[[list], None] | None = None,
) -> list[dict]:
    """Glass-box critique→refine loops over an already-generated qa_batch.

    Preserves model-swap economy: each loop = ONE critic load (score all) +
    ONE generator load (refine zone). Judge decisions run on CPU from cache.

    Returns finalized QA pairs with full judge provenance attached.
    """
    finalized: list[dict] = []
    # Drop failed-parse generations up front.
    active = [(qa, ctx, bloom) for qa, ctx, bloom in qa_batch if qa is not None]

    for loop_idx in range(max_loops):
        if not active:
            break
        last_loop = (loop_idx == max_loops - 1)

        # ── Phase J: ONE critic load, score all m-sampled, then unload ──
        critic = critic_factory()
        def critic_gen(prompts: list[str]) -> list[str]:
            out = critic.generate_batch(prompts)
            return out if isinstance(out, list) else [out]
        logger.info("[GlassBox] Loop %d/%d — critic scoring %d items (m=%d)...",
                    loop_idx + 1, max_loops, len(active), m_samples)
        cache = batch_score_critic(active, critic_gen, m=m_samples)
        critic.unload(); del critic

        # ── Judge decisions on CPU (zero LLM calls) ──
        accepted, refine_zone, rejected = split_by_decision(
            active, cache, judge_config)
        logger.info("[GlassBox] Loop %d: %d accept, %d refine, %d reject.",
                    loop_idx + 1, len(accepted), len(refine_zone), len(rejected))

        for (qa, ctx, bloom), jr in accepted:
            finalized.append(finalize(qa, ctx, bloom, jr, loop_idx + 1, True))

        # On last loop, keep best-effort refine-zone (non-converged), drop rejects.
        if last_loop or not refine_zone:
            for (qa, ctx, bloom), jr in refine_zone:
                finalized.append(finalize(qa, ctx, bloom, jr, max_loops, False))
            if save_checkpoint:
                save_checkpoint(finalized)
            break

        # ── Phase R: ONE generator load, refine the zone, then unload ──
        generator = generator_factory()
        def gen_gen(prompts: list[str]) -> list[str]:
            out = generator.generate_batch(prompts)
            return out if isinstance(out, list) else [out]
        active = batch_refine(refine_zone, gen_gen, build_refine_prompt, parse_qa)
        generator.unload(); del generator

        if save_checkpoint:
            save_checkpoint(finalized)

    conv = sum(1 for p in finalized if p.get("converged"))
    logger.info("[GlassBox] Done: %d finalized (%.1f%% converged).",
                len(finalized), 100 * conv / max(len(finalized), 1))
    return finalized
