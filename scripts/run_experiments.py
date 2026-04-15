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
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CFG, BLOOM_LEVELS_6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("bloom_depth.experiments")


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
# Model Engine Creation (L4 sequential loading)
# ─────────────────────────────────────────────
def create_model_engine(model_name: str, task: str = "generate") -> Any:
    """Create a ModelEngine, fitting within L4 24GB constraints.

    On L4, only one large model fits at a time.
    Generator and Critic are loaded/unloaded sequentially.
    """
    from src.model_engine import create_engine

    temperature = 0.7 if task == "generate" else 0.1
    max_tokens = 1024 if task == "generate" else 768

    return create_engine(
        model_name=model_name,
        backend="vllm",
        max_new_tokens=max_tokens,
        temperature=temperature,
        gpu_memory_utilization=CFG.colab.vllm_gpu_memory_utilization,
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
# Phase A: Iterative Refinement Ablation (RQ1)
# ─────────────────────────────────────────────
def run_phase_a(
    contexts: list[dict],
    output_dir: Path,
    drive_sync: Any,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run Self-Refine ablation: single_pass / fixed_2 / fixed_3 / adaptive."""
    from src.iterative_qag import run_ablation

    if limit:
        contexts = contexts[:limit]

    all_results: dict[str, Any] = {}

    for mode in CFG.refinement.ablation_modes:
        # Check resume
        if drive_sync and drive_sync.is_completed("refinement", mode):
            logger.info("Skipping %s (already completed)", mode)
            continue

        logger.info("Phase A — Ablation: %s (%d contexts)", mode, len(contexts))

        # Load generator
        generator = create_model_engine(CFG.generator.model_name, task="generate")

        # Critic: use separate model if VRAM allows, otherwise self-critique.
        # Self-critique is valid (Madaan et al. 2023 uses same model for both).
        # Cross-family critic eliminates self-reinforcement bias but requires
        # sequential load/unload on L4 (only 1 model fits at a time).
        #
        # IMPORTANT: run_ablation passes generator_engine to BOTH roles:
        #   - single_pass mode: calls generator_engine(list_of_prompts) → MUST be .generate_batch
        #   - other modes:      calls generator_engine(single_prompt)   → .generate is fine
        # To satisfy both callers, pass generate_batch (it accepts both str and list).
        critic_engine = generator.generate  # single-prompt, always OK

        t0 = time.perf_counter()
        qa_pairs, traces = run_ablation(
            mode=mode,
            generator_engine=generator.generate_batch,  # FIX: generate_batch accepts list AND str
            critic_engine=critic_engine,
            contexts=contexts,
            bloom_levels=BLOOM_LEVELS_6,
            n_questions=CFG.qag.questions_per_level,
        )
        elapsed = time.perf_counter() - t0

        # Unload generator to free VRAM
        generator.unload()

        # Save results
        mode_dir = output_dir / "refinement" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        with open(mode_dir / "qa_pairs.json", "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

        traces_data = [t.to_dict() for t in traces]
        with open(mode_dir / "traces.json", "w", encoding="utf-8") as f:
            json.dump(traces_data, f, ensure_ascii=False, indent=2)

        all_results[mode] = {
            "n_pairs": len(qa_pairs),
            "avg_loops": sum(t.total_loops for t in traces) / max(len(traces), 1),
            "convergence_rate": sum(1 for t in traces if t.converged) / max(len(traces), 1),
            "elapsed_seconds": round(elapsed, 1),
        }

        # Checkpoint to Drive
        if drive_sync:
            drive_sync.sync_dir(mode_dir, f"refinement/{mode}")
            drive_sync.mark_completed("refinement", mode, all_results[mode])

        logger.info("  → %d QA pairs, avg %.1f loops, %.0f%% converged, %.0fs",
                     len(qa_pairs), all_results[mode]["avg_loops"],
                     all_results[mode]["convergence_rate"] * 100, elapsed)

    with open(output_dir / "refinement" / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results


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
