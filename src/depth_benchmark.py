"""
Inference-Time Compute Depth Benchmark
=======================================
Who:    4 benchmark models × 4 strategies × 6 Bloom levels × 2 conditions
Where:  BloomDepth/src/depth_benchmark.py
How:    Tests how much "reasoning depth" each model needs per Bloom level
        by comparing Standard / Few-shot / CoT / Self-Consistency strategies.

Core Hypothesis (H₁):
    The performance gap between Standard and CoT strategies is proportional
    to Bloom cognitive level — higher levels benefit MORE from deeper reasoning.

Reference: Snell et al. (2024). Scaling LLM Test-Time Compute Optimally
           can be More Effective than Scaling Model Parameters. arXiv:2408.03314
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("bloom_depth.depth_benchmark")

InferenceStrategy = Literal["standard", "few_shot", "cot", "self_consistency"]


# ─────────────────────────────────────────────
# Prompt Templates per Strategy
# ─────────────────────────────────────────────

STANDARD_PROMPT = """\
{context_block}
Câu hỏi: {question}

{candidates}

Chọn đáp án đúng nhất. CHỈ trả lời bằng MỘT chữ cái (A, B, C, hoặc D).
Đáp án:"""

FEW_SHOT_PROMPT = """\
Dưới đây là một số ví dụ về câu hỏi trắc nghiệm pháp luật và cách trả lời:

{exemplars}

Bây giờ, hãy trả lời câu hỏi sau:

{context_block}
Câu hỏi: {question}

{candidates}

CHỈ trả lời bằng MỘT chữ cái (A, B, C, hoặc D).
Đáp án:"""

COT_PROMPT = """\
{context_block}
Câu hỏi: {question}

{candidates}

Hãy suy luận từng bước trước khi chọn đáp án:
1. Xác định quy định pháp luật liên quan.
2. Phân tích từng đáp án.
3. Loại trừ các đáp án sai.
4. Chọn đáp án đúng nhất.

Suy luận:"""

SC_PROMPT = COT_PROMPT  # Self-consistency uses the same CoT prompt, sampled N times


# ─────────────────────────────────────────────
# Benchmark Result Schema
# ─────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    """Result of one benchmark trial: model × strategy × bloom × condition."""

    model_name: str = ""
    strategy: str = ""
    bloom_level: str = ""
    condition: str = ""  # "none_context" or "with_context"

    total_questions: int = 0
    correct: int = 0
    accuracy: float = 0.0

    # Per-question details (optional, for detailed analysis)
    predictions: list[dict[str, Any]] = field(default_factory=list)

    # Self-consistency specific
    sc_agreement_rate: float = 0.0  # How often majority vote agrees

    def compute_accuracy(self) -> None:
        """Compute accuracy from correct/total."""
        self.accuracy = self.correct / max(self.total_questions, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "strategy": self.strategy,
            "bloom_level": self.bloom_level,
            "condition": self.condition,
            "total_questions": self.total_questions,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "sc_agreement_rate": round(self.sc_agreement_rate, 4),
        }


# ─────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────
class PromptBuilder:
    """Builds prompts for each inference strategy.

    Args:
        exemplar_bank: Dict mapping Bloom level → list of exemplar QA pairs
                       (used for few-shot strategy only).
        few_shot_k: Number of exemplars per prompt.
    """

    def __init__(
        self,
        exemplar_bank: dict[str, list[dict[str, Any]]] | None = None,
        few_shot_k: int = 3,
    ) -> None:
        self.exemplar_bank = exemplar_bank or {}
        self.few_shot_k = few_shot_k

    def build(
        self,
        qa: dict[str, Any],
        strategy: InferenceStrategy,
        condition: str = "with_context",
    ) -> str:
        """Build a prompt for a given QA pair and strategy.

        Args:
            qa: QA record from the dataset.
            strategy: One of "standard", "few_shot", "cot", "self_consistency".
            condition: "none_context" (no context) or "with_context" (full context).

        Returns:
            Formatted prompt string.
        """
        # Context block
        if condition == "with_context":
            ctx = qa.get("context_text", qa.get("context_payload", {}).get("text", ""))
            context_block = f"Ngữ cảnh pháp lý:\n{ctx[:2000]}"
        else:
            context_block = ""

        # Candidates
        cands = qa.get("candidate_answers", [])
        candidates_str = "\n".join(cands) if cands else ""

        question = qa.get("question_content", qa.get("question", ""))

        if strategy == "standard":
            return STANDARD_PROMPT.format(
                context_block=context_block,
                question=question,
                candidates=candidates_str,
            )

        elif strategy == "few_shot":
            bloom = qa.get("bloom_level", "Remember")
            exemplars = self._build_exemplars(bloom)
            return FEW_SHOT_PROMPT.format(
                exemplars=exemplars,
                context_block=context_block,
                question=question,
                candidates=candidates_str,
            )

        elif strategy in ("cot", "self_consistency"):
            return COT_PROMPT.format(
                context_block=context_block,
                question=question,
                candidates=candidates_str,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _build_exemplars(self, bloom_level: str) -> str:
        """Build few-shot exemplars string from exemplar bank."""
        pool = self.exemplar_bank.get(bloom_level, [])
        if not pool:
            # Fallback to any available exemplars
            pool = [ex for exs in self.exemplar_bank.values() for ex in exs]

        selected = random.sample(pool, min(self.few_shot_k, len(pool)))

        parts = []
        for i, ex in enumerate(selected, 1):
            parts.append(
                f"Ví dụ {i}:\n"
                f"Câu hỏi: {ex.get('question', ex.get('question_content', ''))}\n"
                f"{chr(10).join(ex.get('candidate_answers', []))}\n"
                f"Đáp án: {ex.get('ground_truth', '')[:1]}\n"
            )

        return "\n".join(parts)


# ─────────────────────────────────────────────
# Answer Extractor
# ─────────────────────────────────────────────
import re

ANSWER_PATTERN = re.compile(r"\b([A-D])\b")


def extract_answer(text: str, strategy: InferenceStrategy = "standard") -> str | None:
    """Extract the predicted answer letter from model output.

    For standard/few-shot: expects just a letter.
    For CoT/SC: expects reasoning followed by final answer.

    Returns:
        Single letter (A/B/C/D) or None if unparseable.
    """
    text = text.strip()

    if strategy in ("standard", "few_shot"):
        # Direct answer — first letter found
        match = ANSWER_PATTERN.search(text[:10])
        return match.group(1) if match else None

    else:
        # CoT — look for the LAST letter mention (after reasoning)
        # Common patterns: "Đáp án: B", "chọn A", "→ C"
        # Search from end of text
        matches = list(ANSWER_PATTERN.finditer(text))
        if matches:
            return matches[-1].group(1)
        return None


def majority_vote(answers: list[str | None]) -> tuple[str | None, float]:
    """Compute majority vote from multiple answer predictions.

    Returns:
        Tuple of (winning_answer, agreement_rate)
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0.0

    counter = Counter(valid)
    winner, count = counter.most_common(1)[0]
    agreement = count / len(valid)
    return winner, agreement


# ─────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────
class DepthBenchmark:
    """Inference depth benchmark with batched generation for high GPU utilization.

    The engine callable supports both single and batch modes:
        - engine(prompt: str, **kw) → str
        - engine([prompt1, prompt2, ...], **kw) → [str, str, ...]
    """

    def __init__(
        self,
        model_engine: Any = None,
        model_name: str = "",
        prompt_builder: PromptBuilder | None = None,
        sc_num_paths: int = 10,
        sc_temperature: float = 0.7,
    ) -> None:
        self.model_engine = model_engine
        self.model_name = model_name
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.sc_num_paths = sc_num_paths
        self.sc_temperature = sc_temperature

    def _call_engine(self, prompts: list[str], **kwargs) -> list[str]:
        """Call engine with batch support. Falls back to sequential if needed."""
        try:
            result = self.model_engine(prompts, **kwargs)
            if isinstance(result, list):
                return result
        except TypeError:
            pass
        # Fallback: sequential
        return [self.model_engine(p, **kwargs) for p in prompts]

    def run_single_strategy(
        self,
        qa_pairs: list[dict[str, Any]],
        strategy: InferenceStrategy,
        condition: str = "with_context",
    ) -> BenchmarkResult:
        """Batched benchmark for one strategy × condition × Bloom level."""
        bloom_level = qa_pairs[0].get("bloom_level", "Unknown") if qa_pairs else "Unknown"

        result = BenchmarkResult(
            model_name=self.model_name,
            strategy=strategy,
            bloom_level=bloom_level,
            condition=condition,
            total_questions=len(qa_pairs),
        )

        # Extract ground truths
        gt_answers = []
        for qa in qa_pairs:
            gt_raw = qa.get("ground_truth", "")
            gt_match = ANSWER_PATTERN.search(gt_raw[:5])
            gt_answers.append(gt_match.group(1) if gt_match else None)

        if strategy == "self_consistency":
            # Batch all N paths × all QAs in one call
            prompts = [self.prompt_builder.build(qa, strategy, condition) for qa in qa_pairs]
            all_sc_prompts = prompts * self.sc_num_paths  # [q1,q2,...qN] × sc_paths
            all_outputs = self._call_engine(all_sc_prompts, temperature=self.sc_temperature)

            # Reshape: [sc_paths][n_questions] and majority vote per question
            n = len(qa_pairs)
            for i, qa in enumerate(qa_pairs):
                path_answers = [extract_answer(all_outputs[j * n + i], "cot")
                                for j in range(self.sc_num_paths)]
                predicted, agreement = majority_vote(path_answers)
                result.sc_agreement_rate += agreement

                is_correct = predicted == gt_answers[i] if gt_answers[i] else False
                if is_correct:
                    result.correct += 1
                result.predictions.append({
                    "qa_id": qa.get("qa_id", ""),
                    "predicted": predicted, "ground_truth": gt_answers[i], "correct": is_correct,
                })
            result.sc_agreement_rate /= max(len(qa_pairs), 1)
        else:
            # Standard / few-shot / CoT: build all prompts, one batch call
            prompts = [self.prompt_builder.build(qa, strategy, condition) for qa in qa_pairs]
            outputs = self._call_engine(prompts)

            for i, (qa, output) in enumerate(zip(qa_pairs, outputs)):
                predicted = extract_answer(output, strategy)
                is_correct = predicted == gt_answers[i] if gt_answers[i] else False
                if is_correct:
                    result.correct += 1
                result.predictions.append({
                    "qa_id": qa.get("qa_id", ""),
                    "predicted": predicted, "ground_truth": gt_answers[i], "correct": is_correct,
                })

        result.compute_accuracy()
        return result

    def run_full_benchmark(
        self,
        dataset: dict[str, list[dict[str, Any]]],
        strategies: tuple[InferenceStrategy, ...] = ("standard", "few_shot", "cot", "self_consistency"),
        conditions: tuple[str, ...] = ("none_context", "with_context"),
    ) -> list[BenchmarkResult]:
        """Run the complete benchmark matrix.

        Args:
            dataset: Dict mapping bloom_level → list of QA pairs.
            strategies: Tuple of strategies to benchmark.
            conditions: Tuple of context conditions.

        Returns:
            List of BenchmarkResult objects.
        """
        all_results: list[BenchmarkResult] = []

        for bloom_level, qa_pairs in dataset.items():
            for strategy in strategies:
                for condition in conditions:
                    logger.info(
                        "  Benchmarking: model=%s, strategy=%s, bloom=%s, condition=%s, n=%d",
                        self.model_name, strategy, bloom_level, condition, len(qa_pairs),
                    )

                    t0 = time.perf_counter()
                    result = self.run_single_strategy(qa_pairs, strategy, condition)
                    elapsed = time.perf_counter() - t0

                    logger.info(
                        "    → Accuracy: %.1f%% (%d/%d) in %.1fs",
                        result.accuracy * 100,
                        result.correct,
                        result.total_questions,
                        elapsed,
                    )

                    all_results.append(result)

        return all_results


# ─────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────
def save_benchmark_results(
    results: list[BenchmarkResult],
    output_path: Path,
) -> None:
    """Save benchmark results as JSON."""
    data = [r.to_dict() for r in results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d benchmark results to %s", len(results), output_path)


def load_benchmark_results(input_path: Path) -> list[BenchmarkResult]:
    """Load benchmark results from JSON."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for d in data:
        r = BenchmarkResult(
            model_name=d["model_name"],
            strategy=d["strategy"],
            bloom_level=d["bloom_level"],
            condition=d["condition"],
            total_questions=d["total_questions"],
            correct=d["correct"],
            accuracy=d["accuracy"],
            sc_agreement_rate=d.get("sc_agreement_rate", 0.0),
        )
        results.append(r)

    return results
