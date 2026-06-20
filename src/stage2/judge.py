"""
Stage 2 / U7 — Judge / Critic (glass-box, formula-driven)
=========================================================
Maps each QA pair to a rubric vector r ∈ [0,1]^6, aggregates via a two-tier
scheme (hard gate + weighted score), calibrates the scalar score to an
empirical P_correct, and emits a fully traceable decision.

SOTA grounding (see docs/research_wiki.md + research_wiki_fulltext_findings.md):
- Probability-weighted scoring  : G-Eval (Liu et al., 2023, arXiv:2303.16634)
- Rubric anchored absolute grade : Prometheus 2 (Kim et al., 2024, arXiv:2405.01535)
- Bias taxonomy + mitigations    : Zheng et al., 2023 (arXiv:2306.05685)
- Calibration                    : Platt (1999); ECE (Guo et al., 2017, arXiv:1706.04599)
- Inter-judge agreement          : Krippendorff's alpha (Krippendorff, 2004)
- Confidence from rationale NOT  : VERDI (Qi et al., 2026, arXiv:2605.11334) shows
  token logprobs                   token logprobs SATURATE (>0.999 on ~99% of JSON
                                   outputs) and are ANTI-CALIBRATED on Qwen
                                   (AUROC 0.32-0.49). => P(v) MUST come from
                                   self-consistency sampling, NOT logprobs.

This module is model-agnostic: the actual LLM call is injected as a callable
(`score_fn`) so the math/decision layer is unit-testable without a GPU.

MANDATE (VERDI 2605.11334): `score_fn` must derive P(v) by sampling the judge
m times at T>0 and counting rating frequencies (`samples_to_prob_dist`), or from
structural rationale signals (`evidence_grounding_score`). It must NOT read raw
token logprobs of the rating token — those are saturated/anti-calibrated under
structured output and break ECE calibration downstream.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# Rubric dimension keys, fixed order r1..r6 (see QUANTITATIVE_JUDGE.md J1).
RUBRIC_DIMS: tuple[str, ...] = ("r1", "r2", "r3", "r4", "r5", "r6")
RUBRIC_LABELS: dict[str, str] = {
    "r1": "groundedness",
    "r2": "answer_correctness",
    "r3": "bloom_alignment",
    "r4": "answerability",
    "r5": "legal_validity",
    "r6": "linguistic_quality",
}


@dataclass
class JudgeConfig:
    """All thresholds explicit and traceable. Defaults are INITIAL values;
    they MUST be re-calibrated on a gold set (see QUANTITATIVE_JUDGE.md J5)."""

    # J3 — two-tier aggregation
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "r1": 0.25, "r2": 0.25, "r3": 0.20, "r4": 0.10, "r5": 0.10, "r6": 0.10,
        }
    )
    # Hard gate minimums on the RAW 1..5 scale (necessary conditions).
    hard_gate: dict[str, float] = field(
        default_factory=lambda: {"r1": 4.0, "r2": 4.0, "r5": 3.0}
    )
    # J5 — decision thresholds on calibrated/normalized S in [0,1].
    tau_accept: float = 0.80
    tau_refine: float = 0.50
    # J4 — Platt calibration params (a*S + b). Defaults = identity-ish.
    platt_a: float = 1.0
    platt_b: float = 0.0
    use_calibration: bool = False  # off until gold set fits (a, b)
    # J6 — verbosity regularizer.
    length_lambda: float = 0.0
    length_target: int = 40  # L* tokens; 0 disables
    # J7 — multi-judge disagreement flag.
    delta_disagree: float = 0.30
    # Risk-calibrated abstention (manuscript Eq. abstain): an item is admitted
    # only when critical-dimension uncertainty is below `gamma_uncertainty`
    # AFTER temperature calibration; otherwise the judge abstains -> human.
    gamma_uncertainty: float = 0.40
    calib_temperature: float = 1.0  # T_k fit on held-out HCA; 1.0 = identity
    critical_dims: tuple[str, ...] = ("r1", "r2", "r5")

    def validate(self) -> None:
        s = sum(self.weights.values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {s:.6f}")
        if not (0.0 <= self.tau_refine <= self.tau_accept <= 1.0):
            raise ValueError("require 0 <= tau_refine <= tau_accept <= 1")


@dataclass
class RubricScores:
    """Probability-weighted scores s_i ∈ [1,5] per dimension (J2)."""

    scores: dict[str, float]

    def normalized(self) -> dict[str, float]:
        """r_i = (s_i - 1) / 4 ∈ [0,1] (J1)."""
        return {k: (self.scores[k] - 1.0) / 4.0 for k in RUBRIC_DIMS}

    def lowest_dim(self) -> str:
        """argmin_i r_i — the dimension Refiner should target (R1)."""
        norm = self.normalized()
        return min(RUBRIC_DIMS, key=lambda k: norm[k])


@dataclass
class JudgeResult:
    """Fully traceable decision record (J8 output contract)."""

    qa_id: str
    rubric_scores: dict[str, float]
    hard_gate_passed: bool
    S: float
    p_correct: float
    decision: str           # "accept" | "refine" | "reject"
    lowest_dim: str
    median_S: float
    score_range: float
    krippendorff_alpha: Optional[float]
    disagreement_flag: bool
    judge_models: list[str]
    max_critical_uncertainty: float = 0.0
    abstained: bool = False
    judge_version: str = "rubric-v1.0"

    def to_dict(self) -> dict:
        return {
            "qa_id": self.qa_id,
            "rubric_scores": {k: round(v, 3) for k, v in self.rubric_scores.items()},
            "hard_gate_passed": self.hard_gate_passed,
            "S": round(self.S, 4),
            "p_correct": round(self.p_correct, 4),
            "decision": self.decision,
            "lowest_dim": self.lowest_dim,
            "judges": {
                "J": len(self.judge_models),
                "median_S": round(self.median_S, 4),
                "range": round(self.score_range, 4),
                "krippendorff_alpha": (
                    round(self.krippendorff_alpha, 4)
                    if self.krippendorff_alpha is not None else None
                ),
            },
            "disagreement_flag": self.disagreement_flag,
            "max_critical_uncertainty": round(self.max_critical_uncertainty, 4),
            "abstained": self.abstained,
            "judge_models": self.judge_models,
            "judge_version": self.judge_version,
        }


# --------------------------------------------------------------------------
# Pure helper functions (unit-testable without any LLM)
# --------------------------------------------------------------------------

def expected_score(prob_dist: dict[int, float]) -> float:
    """J2 — probability-weighted score: s_i = Σ_v v · P(v).

    `prob_dist` maps integer rating v∈{1..5} → probability. Probabilities are
    renormalized defensively. Raises if empty or values out of range.
    """
    if not prob_dist:
        raise ValueError("prob_dist is empty")
    total = sum(prob_dist.values())
    if total <= 0:
        raise ValueError("prob_dist sums to <= 0")
    s = 0.0
    for v, p in prob_dist.items():
        if not (1 <= v <= 5):
            raise ValueError(f"rating {v} outside 1..5")
        s += v * (p / total)
    return s


def samples_to_prob_dist(samples: Sequence[int]) -> dict[int, float]:
    """J2 source — build P(v) from m SELF-CONSISTENCY samples, NOT logprobs.

    The judge is sampled m times at T>0; each sample is an integer rating
    v∈{1..5}. The empirical frequency P(v)=count(v)/m is the rating
    distribution fed to `expected_score`.

    Rationale (VERDI, arXiv:2605.11334): token logprobs saturate (>0.999 on
    ~99% of structured-output samples) and are anti-calibrated on Qwen
    (AUROC 0.32-0.49), so they carry no usable variance. Frequencies over
    independent samples recover a genuine, calibratable distribution.

    Raises if empty or any sample is outside 1..5.
    """
    if not samples:
        raise ValueError("no samples for prob dist")
    m = len(samples)
    counts: dict[int, float] = {}
    for v in samples:
        if not (1 <= v <= 5):
            raise ValueError(f"sample rating {v} outside 1..5")
        counts[v] = counts.get(v, 0.0) + 1.0
    return {v: c / m for v, c in counts.items()}


def evidence_grounding_score(
    quoted_spans: Sequence[str],
    source_text: str,
    overlap_threshold: float = 0.80,
) -> float:
    """Structural signal for r5 (legal validity), after VERDI's EGS.

    Fraction of quoted spans that are verifiable in `source_text`, weighted by
    span length. A span is "grounded" if its token-overlap ratio with any
    window of the source meets `overlap_threshold` (0.80 per VERDI). This turns
    legal-citation validity from a qualitative judge feeling into a measurable
    quantity: hallucinated article/clause citations -> low EGS.

    Returns 1.0 when there are no quoted spans (nothing to ground -> vacuously
    grounded), matching VERDI's "nothing to verify -> pass" early exit.
    """
    spans = [s for s in quoted_spans if s and s.strip()]
    if not spans:
        return 1.0
    src_tokens = source_text.lower().split()
    src_set = set(src_tokens)
    total_w = 0.0
    grounded_w = 0.0
    for span in spans:
        toks = span.lower().split()
        if not toks:
            continue
        w = float(len(toks))
        total_w += w
        overlap = sum(1 for t in toks if t in src_set) / len(toks)
        if overlap >= overlap_threshold:
            grounded_w += w
    if total_w <= 0:
        return 1.0
    return grounded_w / total_w


def hard_gate(scores: dict[str, float], gate: dict[str, float]) -> bool:
    """J3 tier-1 — G = Π_i 1[s_i >= g_i] over gated dims. AND semantics."""
    return all(scores.get(dim, 0.0) >= thr for dim, thr in gate.items())


def weighted_score(norm: dict[str, float], weights: dict[str, float]) -> float:
    """J3 tier-2 — S = Σ_i w_i · r_i over normalized r_i ∈ [0,1]."""
    return sum(weights[k] * norm[k] for k in RUBRIC_DIMS)


def platt(s: float, a: float, b: float) -> float:
    """J4 — P_correct = σ(a·S + b)."""
    return 1.0 / (1.0 + math.exp(-(a * s + b)))


def consistency_uncertainty(prob_dist: dict[int, float]) -> float:
    """Self-consistency uncertainty u = 1 - max_v P(v) (manuscript Eq.).

    1 when ratings are maximally split, 0 when all m samples agree. This is the
    per-dimension dispersion that the abstention rule thresholds.
    """
    if not prob_dist:
        return 1.0
    total = sum(prob_dist.values()) or 1.0
    return 1.0 - max(p / total for p in prob_dist.values())


def calibrate_uncertainty(u: float, temperature: float) -> float:
    """Temperature-scale a raw uncertainty into a calibrated u~ in [0,1].

    Self-consistency reduces variance but not systematic over-confidence
    (Tian et al. 2025, arXiv:2508.06225); T_k > 1 inflates under-dispersed
    uncertainty so stated confidence matches empirical accuracy on a held-out
    set. T = 1 is identity. Implemented in logit space and clamped.
    """
    if temperature <= 0:
        return max(0.0, min(1.0, u))
    eps = 1e-6
    u = min(1.0 - eps, max(eps, u))
    logit = math.log(u / (1.0 - u))
    scaled = 1.0 / (1.0 + math.exp(-(logit / temperature)))
    return max(0.0, min(1.0, scaled))


def expected_calibration_error(
    pairs: Sequence[tuple[float, int]], n_bins: int = 10
) -> float:
    """J4 — ECE = Σ_m (|B_m|/N) · |acc(B_m) − conf(B_m)|.

    `pairs` = list of (confidence∈[0,1], label∈{0,1}). Equal-width bins.
    """
    if not pairs:
        raise ValueError("no pairs for ECE")
    n = len(pairs)
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for conf, label in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, label))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf_mean = sum(c for c, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / n) * abs(acc - conf_mean)
    return ece


def krippendorff_alpha_interval(
    ratings: Sequence[Sequence[float]],
) -> Optional[float]:
    """J7 — Krippendorff's alpha for interval data, α = 1 − D_o/D_e.

    `ratings`: list of items, each a list of J judge scores (same length).
    Uses squared difference as the interval distance metric. Returns None if
    fewer than 2 judges or zero expected disagreement (degenerate).
    """
    if not ratings:
        return None
    j = len(ratings[0])
    if j < 2 or any(len(r) != j for r in ratings):
        return None
    # Observed disagreement: mean within-item pairwise squared diff.
    obs_terms: list[float] = []
    all_vals: list[float] = []
    for r in ratings:
        all_vals.extend(r)
        pair_sum = 0.0
        for a in range(j):
            for b in range(a + 1, j):
                pair_sum += (r[a] - r[b]) ** 2
        obs_terms.append(pair_sum / (j * (j - 1) / 2))
    d_observed = sum(obs_terms) / len(obs_terms)
    # Expected disagreement: variance across all values (interval metric).
    if len(all_vals) < 2:
        return None
    mean_all = sum(all_vals) / len(all_vals)
    d_expected = 2.0 * sum((v - mean_all) ** 2 for v in all_vals) / len(all_vals)
    if d_expected <= 0:
        return None
    return 1.0 - (d_observed / d_expected)


class Judge:
    """U7 Judge orchestrator. The LLM rubric scorer is injected as `score_fn`,
    a callable (qa, context, dim) -> dict[int,float] returning P(v|·) so this
    class stays GPU-free and unit-testable."""

    def __init__(
        self,
        config: JudgeConfig,
        score_fn: Callable[[dict, str, str], dict[int, float]],
        judge_models: Optional[list[str]] = None,
    ) -> None:
        config.validate()
        self.cfg = config
        self.score_fn = score_fn
        self.judge_models = judge_models or ["gemma-3-4b"]

    def _score_once(self, qa: dict, context: str) -> tuple[RubricScores, dict[str, float]]:
        """One judge pass: probability-weighted score + per-dim uncertainty (J2).

        Returns (RubricScores, {dim: calibrated_uncertainty}). Uncertainty is
        derived from the self-consistency rating distribution, not logprobs.
        """
        scores: dict[str, float] = {}
        unc: dict[str, float] = {}
        for dim in RUBRIC_DIMS:
            dist = self.score_fn(qa, context, dim)
            scores[dim] = expected_score(dist)
            unc[dim] = calibrate_uncertainty(
                consistency_uncertainty(dist), self.cfg.calib_temperature
            )
        return RubricScores(scores), unc

    def _scalar(self, rs: RubricScores, token_len: int) -> tuple[bool, float]:
        """Two-tier aggregation + optional verbosity penalty (J3/J6)."""
        gate_ok = hard_gate(rs.scores, self.cfg.hard_gate)
        if not gate_ok:
            return False, 0.0
        s = weighted_score(rs.normalized(), self.cfg.weights)
        if self.cfg.length_lambda > 0 and self.cfg.length_target > 0:
            over = max(0, token_len - self.cfg.length_target)
            s = s - self.cfg.length_lambda * (over / self.cfg.length_target)
        return True, max(0.0, min(1.0, s))

    def evaluate(self, qa: dict, context: str) -> JudgeResult:
        """Full multi-judge evaluation producing a traceable JudgeResult."""
        qa_id = str(qa.get("qa_id", qa.get("id", "unknown")))
        token_len = len(str(qa.get("question", "")).split())

        # Multi-judge: run score_fn per model-seed (here, J independent passes).
        scored: list[tuple[RubricScores, dict[str, float]]] = [
            self._score_once(qa, context) for _ in self.judge_models
        ]
        per_judge: list[RubricScores] = [s for s, _ in scored]
        per_unc: list[dict[str, float]] = [u for _, u in scored]
        s_values: list[float] = []
        gate_flags: list[bool] = []
        for rs in per_judge:
            ok, s = self._scalar(rs, token_len)
            gate_flags.append(ok)
            s_values.append(s)

        median_s = statistics.median(s_values)
        score_range = max(s_values) - min(s_values)
        gate_passed = all(gate_flags)

        # Critical-dimension calibrated uncertainty, averaged across judges.
        crit = self.cfg.critical_dims
        max_crit_unc = max(
            (sum(u[d] for u in per_unc) / len(per_unc)) for d in crit
        ) if per_unc and crit else 0.0

        # Inter-judge agreement on the raw rubric matrix (J7).
        alpha = None
        if len(self.judge_models) >= 2:
            matrix = [
                [pj.scores[dim] for pj in per_judge] for dim in RUBRIC_DIMS
            ]
            alpha = krippendorff_alpha_interval(matrix)

        # Calibrate the consensus score (J4).
        if self.cfg.use_calibration:
            p_correct = platt(median_s, self.cfg.platt_a, self.cfg.platt_b)
        else:
            p_correct = median_s

        disagreement = score_range > self.cfg.delta_disagree
        abstain = max_crit_unc > self.cfg.gamma_uncertainty

        # Decision (J5). An item is admitted only when the gate passes, the
        # score clears tau_accept, judges agree, AND critical-dimension
        # calibrated uncertainty is below gamma. Otherwise it abstains/refines
        # to human review rather than being silently accepted (manuscript Eq.).
        if not gate_passed or median_s < self.cfg.tau_refine:
            decision = "reject"
        elif median_s >= self.cfg.tau_accept and not disagreement and not abstain:
            decision = "accept"
        elif abstain:
            decision = "abstain"
        else:
            decision = "refine"

        # Representative rubric = median judge by its scalar S.
        rep_idx = s_values.index(statistics.median_low(s_values))
        rep_rubric = per_judge[rep_idx]

        return JudgeResult(
            qa_id=qa_id,
            rubric_scores=rep_rubric.scores,
            hard_gate_passed=gate_passed,
            S=median_s,
            p_correct=p_correct,
            decision=decision,
            lowest_dim=rep_rubric.lowest_dim(),
            median_S=median_s,
            score_range=score_range,
            krippendorff_alpha=alpha,
            disagreement_flag=disagreement,
            judge_models=list(self.judge_models),
            max_critical_uncertainty=round(max_crit_unc, 4),
            abstained=abstain,
        )
