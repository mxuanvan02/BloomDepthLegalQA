"""
Glass-Box Quantitative Bloom Rubric for MCQ items
==================================================
Who:    Phase A QAG audit (deterministic pre-filter) + Phase C covariate.
Where:  BloomDepth/src/bloom_rubric.py
How:    Scores ONE multiple-choice (A/B/C/D) item against a fixed set of
        measurable signals, sums them into a continuous "cognitive demand"
        score, then maps that score to a Bloom level (1..6) via data-derived
        thresholds. Every decision ships its full per-signal breakdown.

Why this exists (CONTRACT.md spirit + user's glass-box requirement):
    The legacy `bloom_classifier.py` is QUALITATIVE: first-keyword-match-wins.
    A reviewer (and the project owner) rejects that as opaque and brittle.
    This module replaces qualitative judgement with a transparent, additive
    score: each signal is 0/1 (or a bounded float), the weights are explicit,
    the threshold mapping is explicit, and `explain()` returns the audit trail.

Design rules:
    - Deterministic: identical input -> identical output (no RNG, no model).
    - Offline: needs only the item's own text (question, choices, answer).
    - Traceable: returns score, per-signal contributions, and chosen level.
    - Bounded MCQ caveat: Evaluate/Create in MCQ form = "recognise the strong
      argument / correct design", NOT free creation. Encoded honestly so the
      paper does not overclaim (CONTRACT.md §2, §7 high-Bloom risk).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# Ordinal Bloom scale (the quantitative target).
BLOOM_ORDER: tuple[str, ...] = (
    "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create",
)
BLOOM_TO_INT: dict[str, int] = {name: i + 1 for i, name in enumerate(BLOOM_ORDER)}
INT_TO_BLOOM: dict[int, str] = {i + 1: name for i, name in enumerate(BLOOM_ORDER)}


def _norm(text: str) -> str:
    """NFC-normalise + lowercase; keeps Vietnamese diacritics intact."""
    return unicodedata.normalize("NFC", text or "").lower()


# ─────────────────────────────────────────────
# Signal group A — surface verb / phrasing cues
# ─────────────────────────────────────────────
# These are DIRECTIONAL cues, not a hard classifier. Each cue family adds a
# bounded amount of cognitive-demand evidence. A question may trigger several
# families; the score is additive so mixed signals are handled gracefully
# instead of the brittle first-match-wins of the legacy classifier.
_CUE_PATTERNS: dict[str, list[str]] = {
    # Higher-order construction cues (MCQ form = "recognise best design").
    "create": [
        r"đề\s+xuất", r"thiết\s+kế", r"soạn\s+thảo", r"xây\s+dựng\s+(?:phương|quy|giải)",
        r"phương\s+án\s+nào.*phù\s+hợp", r"giải\s+pháp\s+nào.*(?:tốt|phù\s+hợp)",
    ],
    "evaluate": [
        r"đánh\s+giá", r"nhận\s+xét", r"hợp\s+lý\s+(?:nhất|hơn)", r"phản\s+biện",
        r"lập\s+luận\s+nào.*(?:đúng|mạnh|thuyết\s+phục)", r"phù\s+hợp\s+nhất",
        r"đúng\s+(?:đắn\s+)?nhất", r"biện\s+minh", r"phê\s+phán",
    ],
    "analyze": [
        r"so\s+sánh", r"phân\s+biệt", r"phân\s+tích", r"mối\s+(?:quan\s+hệ|liên\s+hệ)",
        r"điểm\s+(?:giống|khác)", r"nguyên\s+nhân", r"hệ\s+quả", r"vì\s+sao",
        r"tại\s+sao", r"khác\s+(?:nhau|biệt)",
    ],
    "apply": [
        r"trong\s+tình\s+huống", r"áp\s+dụng", r"giải\s+quyết\s+(?:vụ|tình|trường)",
        r"(?:nếu|khi)\b.*\bthì\b", r"xử\s+lý\s+(?:như|thế|ra\s+sao)",
        r"\b[A-D]\b\s+(?:muốn|đã|cần|ký|mua|bán|thuê)",
    ],
    "understand": [
        r"giải\s+thích", r"nêu\s+ý\s+nghĩa", r"tóm\s+tắt", r"diễn\s+giải",
        r"trình\s+bày", r"mô\s+tả", r"có\s+nghĩa\s+là", r"hiểu\s+(?:như\s+)?thế\s+nào",
    ],
    "remember": [
        r"là\s+gì", r"bao\s+gồm", r"gồm\s+(?:có|những)", r"liệt\s+kê",
        r"nêu\s+(?:tên|các)", r"theo\s+(?:điều|khoản|luật)", r"khi\s+nào",
        r"ai\s+có\s+thẩm\s+quyền", r"ở\s+đâu",
    ],
}
_COMPILED_CUES: dict[str, list[re.Pattern]] = {
    fam: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in pats]
    for fam, pats in _CUE_PATTERNS.items()
}

# Directional weight each cue family contributes to the cognitive-demand
# score when it fires. Higher-order families push the score up; lower-order
# families pull it down. Tunable, but explicit and auditable.
_CUE_WEIGHT: dict[str, float] = {
    "create": 2.4,
    "evaluate": 1.9,
    "analyze": 1.3,
    "apply": 0.8,
    "understand": 0.2,
    "remember": -0.6,
}

_TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_CHOICE_PREFIX_RE = re.compile(r"^\s*[A-Da-d][\.\)\:]\s*")


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall(_norm(text))


def _get_question(qa: dict[str, Any]) -> str:
    return qa.get("question_content", qa.get("question", "")) or ""


def _get_choices(qa: dict[str, Any]) -> list[str]:
    cands = qa.get("candidate_answers", qa.get("choices", [])) or []
    return [_CHOICE_PREFIX_RE.sub("", c).strip() for c in cands if isinstance(c, str)]


def _get_context(qa: dict[str, Any]) -> str:
    return qa.get("context_text", "") or ""


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _cue_signals(question: str) -> dict[str, float]:
    """For each cue family, 1.0 if any pattern fires, else 0.0."""
    out: dict[str, float] = {}
    for fam, pats in _COMPILED_CUES.items():
        out[fam] = 1.0 if any(p.search(question) for p in pats) else 0.0
    return out


def _distractor_difficulty(choices: list[str]) -> float:
    """Mean pairwise lexical CONTENT overlap among options (0..1).

    High overlap => options are near-misses => harder to eliminate by surface
    cues => evidence of higher cognitive demand (the RIGHT kind of difficulty).

    Guard against the short-answer false positive: options like "2 năm",
    "3 năm", "5 năm" share the filler token "năm" and would otherwise score a
    spurious 1.0 overlap. We (a) drop a small set of generic filler tokens, and
    (b) damp the signal when the options carry too few content tokens to make a
    near-miss judgement meaningful.
    """
    _FILLER = {"năm", "tháng", "ngày", "đồng", "người", "là", "của", "và", "các", "có"}
    tok_sets = []
    content_token_counts = []
    for c in choices:
        if not c:
            continue
        toks = [t for t in _tok(c) if t not in _FILLER]
        tok_sets.append(set(toks))
        content_token_counts.append(len(toks))
    if len(tok_sets) < 2:
        return 0.0
    pairs = [
        _jaccard(tok_sets[i], tok_sets[j])
        for i in range(len(tok_sets))
        for j in range(i + 1, len(tok_sets))
    ]
    raw = sum(pairs) / len(pairs) if pairs else 0.0
    # Damping: if the mean content length per option is < 3 tokens, the options
    # are too short (numeric/date/short-noun) to support a near-miss claim.
    mean_content = sum(content_token_counts) / len(content_token_counts)
    if mean_content < 3.0:
        raw *= mean_content / 3.0
    return raw


def _answer_verbatim_in_context(qa: dict[str, Any]) -> float:
    """1.0 if the correct option text appears (near) verbatim in the context.

    Verbatim recall => low cognitive demand (Remember). This pulls the score
    DOWN, counteracting a misleading high-order verb cue.
    """
    ctx = _norm(_get_context(qa))
    if not ctx:
        return 0.0
    choices = _get_choices(qa)
    gt = (qa.get("ground_truth", "") or "")[:3].upper()
    m = re.search(r"[A-D]", gt)
    if not m:
        return 0.0
    idx = ord(m.group(0)) - ord("A")
    if not (0 <= idx < len(choices)):
        return 0.0
    ans = _norm(choices[idx])
    ans_core = " ".join(_tok(ans))
    if len(ans_core) < 6:
        return 0.0
    return 1.0 if ans_core and ans_core in ctx else 0.0


@dataclass
class RubricResult:
    """Full audit trail for one MCQ item's Bloom scoring."""

    bloom_level: str                       # mapped label, e.g. "Analyze"
    bloom_int: int                         # 1..6
    demand_score: float                    # continuous cognitive-demand score
    signals: dict[str, float] = field(default_factory=dict)   # raw signal values
    contributions: dict[str, float] = field(default_factory=dict)  # weighted parts
    notes: list[str] = field(default_factory=list)            # human-readable trail

    def to_dict(self) -> dict[str, Any]:
        return {
            "bloom_level": self.bloom_level,
            "bloom_int": self.bloom_int,
            "demand_score": round(self.demand_score, 4),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
            "notes": self.notes,
        }


# Demand-score cut points -> Bloom level. Derived to be monotone and
# overridable from a calibrated config. Lower bound inclusive.
DEFAULT_THRESHOLDS: list[tuple[float, str]] = [
    (-99.0, "Remember"),
    (0.30, "Understand"),
    (1.10, "Apply"),
    (1.80, "Analyze"),
    (2.60, "Evaluate"),
    (3.40, "Create"),
]


def _map_score_to_level(score: float, thresholds=DEFAULT_THRESHOLDS) -> str:
    level = thresholds[0][1]
    for lo, name in thresholds:
        if score >= lo:
            level = name
    return level


def score_item(qa: dict[str, Any], thresholds=DEFAULT_THRESHOLDS) -> RubricResult:
    """Quantitatively score one MCQ item -> Bloom level, with full trail.

    The score is an explicit weighted sum of measurable signals:
        demand = Σ cue_weight[f]·cue[f]
               + 1.2·distractor_difficulty
               - 1.0·answer_verbatim_in_context
    Then mapped to a Bloom level via DEFAULT_THRESHOLDS. No model, no RNG.
    """
    question = _get_question(qa)
    choices = _get_choices(qa)

    cues = _cue_signals(question)
    distractor = _distractor_difficulty(choices)
    verbatim = _answer_verbatim_in_context(qa)

    contributions: dict[str, float] = {}
    for fam, fired in cues.items():
        if fired:
            contributions[f"cue_{fam}"] = _CUE_WEIGHT[fam] * fired
    contributions["distractor_difficulty"] = 1.2 * distractor
    contributions["answer_verbatim_in_context"] = -1.0 * verbatim

    demand = sum(contributions.values())
    level = _map_score_to_level(demand, thresholds)

    notes: list[str] = []
    fired_fams = [f for f, v in cues.items() if v]
    notes.append(f"cue families fired: {fired_fams or ['none']}")
    if verbatim:
        notes.append("correct answer is verbatim in context -> recall pressure (Remember)")
    if distractor >= 0.25:
        notes.append(f"near-miss distractors (overlap={distractor:.2f}) -> genuine difficulty")
    notes.append(f"demand={demand:.3f} -> {level}")

    signals: dict[str, float] = dict(cues)
    signals["distractor_difficulty"] = distractor
    signals["answer_verbatim_in_context"] = verbatim

    return RubricResult(
        bloom_level=level,
        bloom_int=BLOOM_TO_INT[level],
        demand_score=demand,
        signals=signals,
        contributions=contributions,
        notes=notes,
    )


def classify_bloom_quant(qa: dict[str, Any]) -> str:
    """Drop-in quantitative replacement returning just the level string."""
    return score_item(qa).bloom_level


if __name__ == "__main__":
    import json

    _demo = [
        {
            "question": "Theo Điều 429 Bộ luật Dân sự, thời hiệu khởi kiện hợp đồng là bao nhiêu năm?",
            "candidate_answers": ["A. 2 năm", "B. 3 năm", "C. 5 năm", "D. 10 năm"],
            "ground_truth": "B",
            "context_text": "thời hiệu khởi kiện hợp đồng là 3 năm",
        },
        {
            "question": "So sánh điểm khác nhau giữa hợp đồng vô hiệu tuyệt đối và vô hiệu tương đối?",
            "candidate_answers": [
                "A. Hợp đồng vô hiệu tuyệt đối có thể khắc phục, tương đối thì không",
                "B. Vô hiệu tuyệt đối vi phạm điều cấm, tương đối do ý chí chủ thể",
                "C. Cả hai đều do tòa án tuyên và không thể khắc phục",
                "D. Vô hiệu tương đối vi phạm điều cấm của luật",
            ],
            "ground_truth": "B",
            "context_text": "",
        },
        {
            "question": "Lập luận nào sau đây hợp lý nhất để bảo vệ quyền lợi bên mua ngay tình?",
            "candidate_answers": [
                "A. Bên mua luôn được bảo vệ vô điều kiện",
                "B. Bên mua ngay tình được bảo vệ nếu đã đăng ký theo quy định",
                "C. Bên mua không bao giờ được bảo vệ",
                "D. Bên mua chỉ được bảo vệ khi bên bán đồng ý",
            ],
            "ground_truth": "B",
            "context_text": "",
        },
    ]
    for d in _demo:
        r = score_item(d)
        print(json.dumps({"q": d["question"][:50], **r.to_dict()}, ensure_ascii=False))
