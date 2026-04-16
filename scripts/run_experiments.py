"""
Experiment orchestrator — runs all 3 phases with real GPU engines.

Phases:
    A: Iterative Refinement Ablation (RQ1)
    B: Inference Depth Benchmark (RQ2 & RQ3)
    C: Statistical Analysis

Usage:
    python scripts/run_experiments.py --phase a --limit 10
    python scripts/run_experiments.py --phase b --model Qwen/Qwen3-8B-Instruct-AWQ
    python scripts/run_experiments.py --phase c
    python scripts/run_experiments.py --phase all --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any


def _graceful_sigterm(signum: int, frame: object) -> None:  # noqa: ARG001
    """Convert SIGTERM (from Colab 'Stop' button) into KeyboardInterrupt
    so that all try/except KeyboardInterrupt blocks trigger correctly."""
    logger_root = logging.getLogger("bloom_depth")
    logger_root.warning("SIGTERM received — converting to KeyboardInterrupt for graceful shutdown.")
    raise KeyboardInterrupt("SIGTERM")


signal.signal(signal.SIGTERM, _graceful_sigterm)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CFG, BLOOM_LEVELS_6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("bloom_depth.experiments")


def _hf_login() -> None:
    """Authenticate HuggingFace Hub using HF_TOKEN env var.

    Called once at startup so vLLM (and any HF library spawned downstream)
    can pull gated model weights without interactive prompts.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        logger.warning(
            "HF_TOKEN env var not set. Downloads of gated models will fail.\n"
            "  → Set it in the notebook cell BEFORE calling timed_run:\n"
            "       import os; os.environ['HF_TOKEN'] = 'hf_xxx...'"
        )
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        logger.info("✅ HuggingFace Hub authenticated (token: hf_...%s)", token[-4:])
    except Exception as exc:
        logger.error("HF Hub login failed: %s", exc)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_bloom(records: list[dict]) -> dict[str, list[dict]]:
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r.get("bloom_level", "Unknown")].append(r)
    return dict(groups)


# ─────────────────────────────────────────────
# Model Engine Creation (simultaneous load for small models)
# ─────────────────────────────────────────────
def create_model_engine(model_name: str, task: str = "generate") -> Any:
    """Create a ModelEngine with task-specific parameters.

    Plan 1+2: Qwen3-8B (~5GB) + Gemma-3-4b (~4GB) = ~9GB total.
    Both fit simultaneously on L4 24GB — no more sequential load/unload.
    """
    from src.model_engine import create_engine

    temperature = 0.7 if task == "generate" else 0.1
    max_tokens  = 1024 if task == "generate" else 512

    return create_engine(
        model_name=model_name,
        backend="vllm",
        max_new_tokens=max_tokens,
        temperature=temperature,
        gpu_memory_utilization=0.44,   # 44% each → both fit within 88% total (L4 safe margin)
    )


def _preflight_check(path: Path, label: str) -> None:
    """Abort early with a clear error if a required file is missing."""
    if not path.exists():
        logger.error(
            "[Preflight FAIL] %s not found: %s\n"
            "  → Make sure the previous stage completed successfully before running this phase.",
            label, path,
        )
        import sys
        sys.exit(1)
    logger.info("[Preflight OK] %s: %s", label, path)


# ─────────────────────────────────────────────
# Drive Sync Helper
# ─────────────────────────────────────────────
from src.drive_sync import get_drive_sync


# ─────────────────────────────────────────────
# Phase A: Final Dataset Generation (Batched Adaptive)
# ─────────────────────────────────────────────
def run_phase_a(
    contexts: list[dict],
    output_dir: Path,
    drive_sync: Any,
    limit: int | None = None,
) -> dict[str, Any]:
    """Phase A: Generate final dataset using Batched Adaptive pipeline.

    Plan 1+2 Architecture:
    - Generator: Qwen3-8B-AWQ  (~5GB VRAM)
    - Critic:    Gemma-3-4b-it (~4GB VRAM)
    - Both loaded SIMULTANEOUSLY on L4 (~9GB total, safe margin).
    - Batched Adaptive: all N jobs processed per pass, not sequentially.
    - Auto-checkpoint every refinement loop pass.
    - Resume-safe: re-running skips already completed chunks.
    """
    from src.iterative_qag import run_batched_adaptive

    if limit:
        contexts = contexts[:limit]
        logger.info("[Limit] Using %d contexts.", limit)

    mode = "adaptive"
    mode_dir = output_dir / "refinement" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    if drive_sync and drive_sync.is_completed("refinement", mode):
        logger.info("Skipping %s (already completed per Drive checkpoint).", mode)
        return {}

    logger.info(
        "Phase A — Batched Adaptive (%d contexts × %d Bloom = %d jobs)",
        len(contexts), len(BLOOM_LEVELS_6), len(contexts) * len(BLOOM_LEVELS_6),
    )

    # Load Generator + Critic SIMULTANEOUSLY (Plan 1: both fit on L4)
    logger.info("Loading Generator: %s", CFG.generator.model_name)
    generator = create_model_engine(CFG.generator.model_name, task="generate")
    logger.info("Loading Critic: %s", CFG.critic.model_name)
    critic = create_model_engine(CFG.critic.model_name, task="critique")

    t0 = time.perf_counter()
    try:
        qa_pairs = run_batched_adaptive(
            generator_engine=generator.generate_batch,
            critic_engine=critic.generate_batch,
            contexts=contexts,
            bloom_levels=BLOOM_LEVELS_6,
            n_questions=CFG.qag.questions_per_level,
            max_loops=CFG.refinement.max_loops,
            checkpoint_dir=mode_dir,
        )
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t0
        logger.warning("⚠️  Interrupted after %.0fs. Checkpoint on disk is safe to resume.", elapsed)
        try:
            generator.unload()
            critic.unload()
        except Exception:  # noqa: BLE001
            pass
        if drive_sync:
            drive_sync.sync_dir(mode_dir, f"refinement/{mode}")
            logger.info("💾 Partial results synced to Drive.")
        raise

    elapsed = time.perf_counter() - t0
    generator.unload()
    critic.unload()

    # Definitive final save
    with open(mode_dir / "qa_pairs.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    converged = sum(1 for p in qa_pairs if p.get("converged"))
    summary = {
        "mode": mode,
        "n_pairs": len(qa_pairs),
        "converged": converged,
        "convergence_rate": round(converged / max(len(qa_pairs), 1), 4),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(mode_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if drive_sync:
        drive_sync.sync_dir(mode_dir, f"refinement/{mode}")
        drive_sync.mark_completed("refinement", mode, summary)

    logger.info(
        "✅ Phase A done: %d QA pairs | %.0f%% converged | %.0fs",
        len(qa_pairs), summary["convergence_rate"] * 100, elapsed,
    )
    return summary




# ─────────────────────────────────────────────
# Phase B: Inference Depth Benchmark (RQ2 & RQ3)
# ─────────────────────────────────────────────
def run_phase_b(
    dataset: dict[str, list[dict]],
    model_name: str,
    output_dir: Path,
    drive_sync: Any,
    exemplar_bank: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Benchmark one model across all strategies × Bloom × conditions.

    Uses batched inference for high GPU utilization.
    Checkpoints per (model × bloom_level × strategy) to Drive.
    """
    from src.depth_benchmark import DepthBenchmark, PromptBuilder, save_benchmark_results

    # Check full model resume
    model_slug = model_name.replace("/", "_")
    if drive_sync and drive_sync.is_completed("benchmark", model_slug):
        logger.info("Skipping %s (already completed)", model_name)
        return []

    logger.info("Phase B — Loading: %s", model_name)
    engine = create_model_engine(model_name, task="benchmark")

    prompt_builder = PromptBuilder(
        exemplar_bank=exemplar_bank,
        few_shot_k=CFG.depth_benchmark.few_shot_k,
    )

    benchmark = DepthBenchmark(
        model_engine=engine.generate,
        model_name=model_name,
        prompt_builder=prompt_builder,
        sc_num_paths=CFG.depth_benchmark.sc_num_paths,
        sc_temperature=CFG.depth_benchmark.sc_temperature,
    )

    t0 = time.perf_counter()
    results = benchmark.run_full_benchmark(
        dataset=dataset,
        strategies=CFG.depth_benchmark.strategies,
        conditions=CFG.depth_benchmark.conditions,
    )
    elapsed = time.perf_counter() - t0

    # Save
    result_path = output_dir / "benchmark" / f"{model_slug}.json"
    save_benchmark_results(results, result_path)

    # Unload to free VRAM for next model
    engine.unload()

    # Checkpoint
    if drive_sync:
        drive_sync.sync_file(result_path, f"benchmark/{model_slug}.json")
        drive_sync.mark_completed("benchmark", model_slug, {
            "n_results": len(results), "elapsed": round(elapsed, 1),
        })

    logger.info("  → %d cells in %.0fs", len(results), elapsed)
    return [r.to_dict() for r in results]


# ─────────────────────────────────────────────
# Phase C: Statistical Analysis
# ─────────────────────────────────────────────
def run_phase_c(output_dir: Path, condition: str = "with_context") -> dict[str, Any]:
    """Aggregate benchmark results → ANOVA + visualizations."""
    from src.analysis import generate_analysis_report

    logger.info("Phase C — Statistical Analysis")

    benchmark_dir = output_dir / "benchmark"
    if not benchmark_dir.exists() or not any(benchmark_dir.glob("*.json")):
        logger.error(
            "[Preflight FAIL] No benchmark JSON files found in %s.\n"
            "  → Run Phase B first to generate benchmark results.",
            benchmark_dir,
        )
        sys.exit(1)

    all_results = []
    for f in sorted(benchmark_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            all_results.extend(json.load(fh))

    logger.info("[Preflight OK] Phase C: loaded %d benchmark cells from %d model files",
                len(all_results), len(list(benchmark_dir.glob("*.json"))))

    traces = []
    for f in sorted((output_dir / "refinement").glob("*/traces.json")):
        with open(f, encoding="utf-8") as fh:
            traces.extend(json.load(fh))

    report = generate_analysis_report(
        benchmark_results=all_results,
        refinement_traces=traces if traces else None,
        output_dir=CFG.paths.artifacts,
        condition=condition,
    )

    logger.info("  ANOVA interaction p=%.6f",
                report.get("anova", {}).get("interaction_effect", {}).get("p_value", float("nan")))
    return report


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BloomDepth experiment orchestrator")
    parser.add_argument("--phase", choices=["a", "b", "c", "all"], default="all")
    parser.add_argument("--dataset", type=Path, default=None, help="corpus_validated.jsonl path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=None, help="Single model to benchmark")
    parser.add_argument("--condition", choices=["none_context", "with_context", "both"], default="with_context")
    parser.add_argument("--no-drive", action="store_true", help="Disable Drive sync")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Authenticate HF Hub first — must happen before any model download
    _hf_login()

    dataset_path = args.dataset or CFG.paths.root / "data" / "processed" / "corpus_validated.jsonl"
    output_dir = CFG.paths.results

    logger.info("Phase: %s | Dataset: %s | Model: %s",
                args.phase, dataset_path, args.model or "all")

    if args.dry_run:
        logger.info("[DRY RUN] Config validated. Exiting.")
        return

    _preflight_check(dataset_path, "corpus_validated.jsonl")

    records = load_jsonl(dataset_path)
    logger.info("Loaded %d records", len(records))

    # Drive sync
    drive_sync = None
    if not args.no_drive:
        try:
            drive_sync = get_drive_sync()
        except Exception as e:
            logger.warning("Drive sync unavailable: %s (continuing without)", e)

    total_start = time.perf_counter()

    # Phase A
    if args.phase in ("a", "all"):
        seen = set()
        contexts = []
        for r in records:
            cid = r.get("chunk_id", r.get("qa_id", ""))
            if cid not in seen:
                seen.add(cid)
                contexts.append({"chunk_id": cid, "text": r.get("text", "")})
        run_phase_a(contexts, output_dir, drive_sync, args.limit)

    # Phase B
    if args.phase in ("b", "all"):
        # Phase B needs the QA pairs generated by Phase A ("adaptive" mode preferred).
        # The mode directory name matches the ablation mode string exactly.
        refinement_dir = CFG.paths.results / "refinement"
        qa_dataset_path = refinement_dir / "adaptive" / "qa_pairs.json"

        if not qa_dataset_path.exists():
            # Fallback: use any available mode's output (deterministic: sorted first)
            candidate_paths = sorted(refinement_dir.glob("*/qa_pairs.json"))
            if candidate_paths:
                qa_dataset_path = candidate_paths[0]
                logger.warning(
                    "'adaptive' QA pairs not found; falling back to: %s", qa_dataset_path
                )
            else:
                logger.error(
                    "[Preflight FAIL] No Phase A qa_pairs.json found under %s.\n"
                    "  → Run Phase A first.",
                    refinement_dir,
                )
                sys.exit(1)

        logger.info("[Preflight OK] Phase B QA dataset: %s", qa_dataset_path)
        with open(qa_dataset_path, "r", encoding="utf-8") as f:
            qa_records = json.load(f)

        # Schema validation — fail fast if Phase A output is malformed
        required_fields = {"question", "candidate_answers", "ground_truth", "bloom_level", "context_text"}
        if qa_records:
            sample = qa_records[0]
            missing = required_fields - set(sample.keys())
            if missing:
                logger.error(
                    "Phase A QA pairs are missing required fields: %s. "
                    "Re-run Phase A to regenerate with correct schema.",
                    missing,
                )
                sys.exit(1)
            logger.info("Schema OK: %d QA pairs, %d bloom levels",
                        len(qa_records), len(set(r.get("bloom_level") for r in qa_records)))

        dataset_by_bloom = group_by_bloom(qa_records)
        models = [args.model] if args.model else list(CFG.depth_benchmark.benchmark_models)
        for model_name in models:
            run_phase_b(dataset_by_bloom, model_name, output_dir, drive_sync)

    # Phase C
    if args.phase in ("c", "all"):
        run_phase_c(output_dir, args.condition)

    logger.info("Done in %.0fs", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
