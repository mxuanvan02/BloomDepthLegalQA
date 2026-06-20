"""
Stage 2 — QA Generation, Judge, and Refine
===========================================
Glass-box QAG quality control. Every accept/refine/reject decision is
formula-driven and traceable (see docs/QUANTITATIVE_JUDGE.md and
docs/QUANTITATIVE_REFINER.md).

Modules:
- judge:   U7 Judge/Critic — rubric vector, probability-weighted scoring,
           hard gate, calibration, anti-bias, multi-judge agreement.
- refiner: U8 Refiner — targeted edit, non-regression, stopping criteria,
           monotone-best termination.
"""

from __future__ import annotations

from .judge import (
    Judge,
    JudgeConfig,
    RubricScores,
    JudgeResult,
    samples_to_prob_dist,
    evidence_grounding_score,
)
from .refiner import (
    Refiner,
    RefinerConfig,
    RefineResult,
    build_external_feedback,
)

__all__ = [
    "Judge",
    "JudgeConfig",
    "RubricScores",
    "JudgeResult",
    "samples_to_prob_dist",
    "evidence_grounding_score",
    "Refiner",
    "RefinerConfig",
    "RefineResult",
    "build_external_feedback",
]
