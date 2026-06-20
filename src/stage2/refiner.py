"""
Stage 2 / U8 — Refiner (glass-box, formula-driven)
==================================================
Targeted, constrained refinement of QA pairs that the Judge routes to "refine".
Refine is treated as bounded optimization: raise S(q) without regressing any
important rubric dimension, with mathematical stopping criteria and a
monotone-best (never-worse-than-original) termination guarantee.

SOTA grounding (see docs/research_wiki.md + research_wiki_fulltext_findings.md):
- Iterative self-feedback refinement: Self-Refine (Madaan et al., 2023,
  arXiv:2303.17651). This module adds quantitative stopping + non-regression
  + argmax-best selection, which the original method leaves unspecified.
- Decompose + step-confidence + argmin-target: SSR (Shi et al., 2025,
  arXiv:2511.10621) — refine the single weakest dimension per round.
- EXTERNAL-SOURCE FRAMING: The Self-Correction Illusion (Chen et al., 2026,
  arXiv:2606.05976). A model corrects a byte-identical error 23-93 pp MORE
  often when the error is presented as an EXTERNAL signal vs inside its own
  <think>. => refine feedback MUST be framed as coming from an external judge,
  never as the generator's own self-reflection. See `build_external_feedback`.

Model-agnostic: the rewrite step is injected as `refine_fn`, and scoring is
delegated to a Judge-like callable, so the control logic is GPU-free and
unit-testable. See docs/QUANTITATIVE_REFINER.md for the spec this implements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class RefinerConfig:
    """Explicit stopping/regression parameters (R2–R3). Defaults are INITIAL;
    tune R_max and epsilon from RefineGain/cost analysis (R4–R5)."""

    tau_accept: float = 0.80         # R3(1) — accept threshold on S
    r_max: int = 3                   # R3(3) — hard loop cap
    epsilon_min: float = 0.02        # R3(2) — diminishing-returns delta
    stall_patience: int = 2          # consecutive sub-epsilon rounds → stop
    delta_reg: float = 0.10          # R2/R3(4) — max allowed per-dim regression


@dataclass
class RefineResult:
    """Traceable refinement record (R6 output contract)."""

    qa_id: str
    rounds: int
    s_trajectory: list[float]
    best_round: int
    stop_reason: str
    final_decision: str               # "accept" | "reject"
    regression_blocked: int
    targeted_dims: list[str]
    final_qa: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "qa_id": self.qa_id,
            "rounds": self.rounds,
            "S_trajectory": [round(s, 4) for s in self.s_trajectory],
            "best_round": self.best_round,
            "stop_reason": self.stop_reason,
            "final_decision": self.final_decision,
            "regression_blocked": self.regression_blocked,
            "targeted_dims": self.targeted_dims,
        }


# Human-readable rubric dimension labels for external-judge feedback framing.
_DIM_LABELS: dict[str, str] = {
    "r1": "tính bám ngữ cảnh (groundedness)",
    "r2": "tính đúng của đáp án (answer correctness)",
    "r3": "đúng cấp Bloom (bloom alignment)",
    "r4": "khả năng trả lời được (answerability)",
    "r5": "hiệu lực pháp lý / trích dẫn luật (legal validity)",
    "r6": "chất lượng ngôn ngữ (linguistic quality)",
}


def build_external_feedback(target_dim: str, score_1to5: Optional[float] = None) -> str:
    """Frame refine feedback as an EXTERNAL judge signal, not self-reflection.

    The Self-Correction Illusion (arXiv:2606.05976) shows a model corrects a
    byte-identical error 23-93 pp more often when it is presented as coming
    from an external source rather than the model's own <think>. We therefore
    phrase the target as a verdict from a separate judge model ("Giám khảo độc
    lập đánh giá ...") so the generator treats it as an addressable external
    claim to act on, not its own thought to defend.
    """
    label = _DIM_LABELS.get(target_dim, target_dim)
    score_clause = (
        f" (điểm hiện tại {score_1to5:.1f}/5)" if score_1to5 is not None else ""
    )
    return (
        f"Giám khảo độc lập (mô hình đánh giá riêng, không phải bạn) kết luận: "
        f"chiều {label} của câu hỏi-đáp án này còn yếu{score_clause}. "
        f"Hãy chỉnh sửa để nâng riêng chiều này, giữ nguyên các chiều khác."
    )


def violates_non_regression(
    prev_norm: dict[str, float],
    new_norm: dict[str, float],
    delta_reg: float,
) -> bool:
    """R2 — reject a refine if ANY dimension drops more than delta_reg.

    Pareto-not-worse constraint on the important dims. r1 (groundedness) is
    treated like any other dim here; the gate in Judge already hard-blocks
    low r1, and this prevents refine from silently degrading it.
    """
    for k in new_norm:
        if prev_norm.get(k, 0.0) - new_norm[k] > delta_reg:
            return True
    return False


class Refiner:
    """U8 Refiner loop.

    Dependencies are injected:
    - refine_fn(qa, target_dim, reason, context) -> new_qa (dict)
    - judge_fn(qa, context) -> JudgeResult-like with .S (float),
      .lowest_dim (str), and .rubric_scores (dict[str,float] on 1..5 scale).
    """

    def __init__(
        self,
        config: RefinerConfig,
        refine_fn: Callable[[dict, str, str, str], dict],
        judge_fn: Callable[[dict, str], object],
    ) -> None:
        self.cfg = config
        self.refine_fn = refine_fn
        self.judge_fn = judge_fn

    @staticmethod
    def _norm(rubric_scores: dict[str, float]) -> dict[str, float]:
        """Convert 1..5 rubric scores to normalized [0,1] for regression check."""
        return {k: (v - 1.0) / 4.0 for k, v in rubric_scores.items()}

    def run(self, qa0: dict, context: str, initial_judge: object) -> RefineResult:
        """Execute the refine loop. `initial_judge` is the Judge result on qa0
        (S0, lowest_dim) so we don't re-score the original."""
        qa_id = str(qa0.get("qa_id", qa0.get("id", "unknown")))
        cfg = self.cfg

        best_qa = qa0
        best_j = initial_judge
        s0 = float(getattr(initial_judge, "S"))
        trajectory: list[float] = [s0]
        targeted: list[str] = []
        regression_blocked = 0
        best_round = 0
        stall = 0
        stop_reason = "max_rounds"

        # Early accept (shouldn't normally enter refine, but guard anyway).
        if s0 >= cfg.tau_accept:
            return RefineResult(
                qa_id, 0, trajectory, 0, "reached_tau_accept",
                "accept", 0, [], best_qa if isinstance(best_qa, dict) else {},
            )

        cur_qa = qa0
        cur_j = initial_judge
        for t in range(1, cfg.r_max + 1):
            target_dim = str(getattr(cur_j, "lowest_dim"))
            # External-judge framing (arXiv:2606.05976), not self-reflection.
            cur_rubric = getattr(cur_j, "rubric_scores", {})
            dim_score = cur_rubric.get(target_dim) if isinstance(cur_rubric, dict) else None
            reason = build_external_feedback(target_dim, dim_score)
            targeted.append(target_dim)

            new_qa = self.refine_fn(cur_qa, target_dim, reason, context)
            new_j = self.judge_fn(new_qa, context)

            prev_norm = self._norm(getattr(cur_j, "rubric_scores"))
            new_norm = self._norm(getattr(new_j, "rubric_scores"))

            # R2 — non-regression Pareto check; reject bad refine, keep cur.
            if violates_non_regression(prev_norm, new_norm, cfg.delta_reg):
                regression_blocked += 1
                stop_reason = "regression_blocked"
                break

            new_s = float(getattr(new_j, "S"))
            trajectory.append(new_s)

            # Track best-so-far (monotone-best guarantee, R4).
            if new_s > float(getattr(best_j, "S")):
                best_qa, best_j, best_round = new_qa, new_j, t

            delta = new_s - float(getattr(cur_j, "S"))
            cur_qa, cur_j = new_qa, new_j

            # R3(1) — reached accept.
            if new_s >= cfg.tau_accept:
                stop_reason = "reached_tau_accept"
                break
            # R3(4) — regression in score → stop, rollback to best.
            if delta < -cfg.delta_reg:
                stop_reason = "score_regression"
                break
            # R3(2) — diminishing returns.
            if delta < cfg.epsilon_min:
                stall += 1
                if stall >= cfg.stall_patience:
                    stop_reason = "diminishing_returns"
                    break
            else:
                stall = 0

        # R3 final decision on best-so-far (argmax_t S_t).
        best_s = float(getattr(best_j, "S"))
        decision = "accept" if best_s >= cfg.tau_accept else "reject"

        return RefineResult(
            qa_id=qa_id,
            rounds=len(trajectory) - 1,
            s_trajectory=trajectory,
            best_round=best_round,
            stop_reason=stop_reason,
            final_decision=decision,
            regression_blocked=regression_blocked,
            targeted_dims=targeted,
            final_qa=best_qa if isinstance(best_qa, dict) else {},
        )


def refine_gain(s_before: list[float], s_after: list[float]) -> float:
    """R4 — RefineGain = mean_q (S(q*) − S(q_0)) over a batch."""
    if not s_before or len(s_before) != len(s_after):
        raise ValueError("mismatched batch lengths")
    return sum(a - b for a, b in zip(s_after, s_before)) / len(s_before)


def refine_success_rate(
    rejected_before: list[bool], accepted_after: list[bool]
) -> float:
    """R4 — fraction of (reject→accept) flips among refine candidates."""
    if not rejected_before:
        return 0.0
    flips = sum(
        1 for rej, acc in zip(rejected_before, accepted_after) if rej and acc
    )
    return flips / len(rejected_before)
