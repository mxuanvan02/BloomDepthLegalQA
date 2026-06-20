"""
Surface-Difficulty Feature Extractor (Confound Control for RQ3)
================================================================
Who:    Phase C analysis (ANCOVA covariates), Phase B side-artifact builder.
Where:  BloomDepth/src/surface_features.py
How:    For every QA item, computes deterministic, GPU-free "surface difficulty"
        features so that the Bloom-level effect on accuracy can be tested while
        CONTROLLING for trivial confounds.

Why this exists (CONTRACT.md §1, RQ3):
    A reviewer will object: "Higher-Bloom questions are harder simply because
    they are LONGER, use RARER words, or have MORE-PLAUSIBLE distractors — not
    because of cognitive depth." To survive that objection we must show the
    Bloom effect SURVIVES after regressing out these surface features. This
    module produces those covariates.

Three feature families (each a confound the reviewer might invoke):
    1. Length          — # chars / tokens in question + choices.
    2. Lexical rarity   — mean / max inverse-document-frequency (IDF) of the
                          question tokens, with IDF estimated from the corpus
                          itself (no external resource, fully reproducible).
    3. Distractor load  — how hard the wrong options are to eliminate by
                          surface cues alone (length-gap, lexical overlap).

Design rules (CONTRACT.md spirit):
    - Deterministic: identical input → identical output (no RNG, no model).
    - Offline: needs only the QA text already saved in Phase A's qa_pairs.json,
      so it can be (re)built any time WITHOUT re-running the GPU benchmark.
    - Provenance-friendly: output keyed by qa_id for a clean Phase C join.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable

# ─────────────────────────────────────────────
# Tokenisation (Vietnamese-aware, deterministic)
# ─────────────────────────────────────────────
# We deliberately use a simple unicode word-splitter rather than a heavy
# segmenter. For COVARIATE purposes (relative rarity / length), a consistent
# whitespace+punctuation tokeniser is sufficient and fully reproducible.
_TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

# Letter-prefix stripper: choices look like "A. nội dung", "B) nội dung".
_CHOICE_PREFIX_RE = re.compile(r"^\s*[A-Da-d][\.\)\:]\s*")


def normalize(text: str) -> str:
    """NFC-normalise and lowercase (keeps Vietnamese diacritics intact)."""
    return unicodedata.normalize("NFC", text or "").lower()


def tokenize(text: str) -> list[str]:
    """Split into lowercase unicode word tokens (diacritics preserved)."""
    return _TOKEN_RE.findall(normalize(text))


def strip_choice_prefix(choice: str) -> str:
    """Remove a leading 'A.'/'B)'/'C:' option label so the prefix letter does
    not pollute length / lexical features."""
    return _CHOICE_PREFIX_RE.sub("", choice or "").strip()


# ─────────────────────────────────────────────
# QA field accessors (schema-tolerant)
# ─────────────────────────────────────────────
def get_question(qa: dict[str, Any]) -> str:
    return qa.get("question_content", qa.get("question", "")) or ""


def get_choices(qa: dict[str, Any]) -> list[str]:
    cands = qa.get("candidate_answers", []) or []
    return [strip_choice_prefix(c) for c in cands if isinstance(c, str)]


def get_context(qa: dict[str, Any]) -> str:
    return qa.get("context_text", qa.get("context_payload", {}).get("text", "")) or ""


def _ground_truth_letter(qa: dict[str, Any]) -> str | None:
    gt = qa.get("ground_truth", "") or ""
    m = re.search(r"\b([A-Da-d])\b", gt[:5])
    return m.group(1).upper() if m else None


# ─────────────────────────────────────────────
# IDF estimation from the corpus itself
# ─────────────────────────────────────────────
def build_idf(qa_pairs: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Estimate inverse-document-frequency from the QA corpus.

    Each QA item (question + its choices) is treated as one document. Rare
    words → high IDF. Smoothed so unseen words at scoring time get a defined,
    high value (treated as maximally rare).

    Returns:
        token → idf weight. Use idf_lookup() for safe defaults.
    """
    df: Counter[str] = Counter()
    n_docs = 0
    for qa in qa_pairs:
        n_docs += 1
        doc_tokens = set(tokenize(get_question(qa)))
        for ch in get_choices(qa):
            doc_tokens.update(tokenize(ch))
        df.update(doc_tokens)

    n_docs = max(n_docs, 1)
    # Smoothed idf: log((N + 1) / (df + 1)) + 1  → always positive, monotone.
    return {tok: math.log((n_docs + 1) / (cnt + 1)) + 1.0 for tok, cnt in df.items()}


def _idf_default(idf: dict[str, float]) -> float:
    """IDF assigned to a token never seen in the corpus (maximally rare)."""
    if not idf:
        return 1.0
    return max(idf.values())


# ─────────────────────────────────────────────
# Per-item feature extraction
# ─────────────────────────────────────────────
def _safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _safe_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _safe_mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def extract_surface_features(
    qa: dict[str, Any],
    idf: dict[str, float],
    include_context: bool = True,
) -> dict[str, float]:
    """Compute the surface-difficulty feature vector for one QA item.

    Args:
        qa: QA record (needs question, candidate_answers, ground_truth).
        idf: token→idf map from build_idf() over the SAME corpus.
        include_context: also report context length (a known confound for the
            with_context condition).

    Returns:
        Flat dict of float features (NaN-free, JSON-serialisable).
    """
    q_text = get_question(qa)
    q_tokens = tokenize(q_text)
    choices = get_choices(qa)
    choice_tokens = [tokenize(c) for c in choices]
    # Char-length per choice is the ROBUST length signal: token-based length
    # collapses to 0 for numeric/date answers ("01/01/2016", "Điều 429"),
    # which are common at low Bloom levels. Using non-whitespace char length
    # keeps the length confound measurable for every option type.
    choice_char_lens = [float(len(c.replace(" ", ""))) for c in choices]

    default_idf = _idf_default(idf)
    q_idfs = [idf.get(t, default_idf) for t in q_tokens]

    # ── Family 1: Length ────────────────────────────────────────────
    feats: dict[str, float] = {
        "q_char_len": float(len(q_text)),
        "q_token_len": float(len(q_tokens)),
        "n_choices": float(len(choices)),
        "mean_choice_char_len": _safe_mean(choice_char_lens),
        "total_choice_char_len": float(sum(choice_char_lens)),
        "mean_choice_token_len": _safe_mean([float(len(t)) for t in choice_tokens]),
        "total_choice_token_len": float(sum(len(t) for t in choice_tokens)),
    }

    # ── Family 2: Lexical rarity ────────────────────────────────────
    feats["mean_q_idf"] = _safe_mean(q_idfs)
    feats["max_q_idf"] = max(q_idfs) if q_idfs else 0.0
    # Fraction of question tokens that are "rare" (idf above corpus median).
    if idf:
        med = sorted(idf.values())[len(idf) // 2]
        feats["frac_rare_q_tokens"] = (
            sum(1 for v in q_idfs if v >= med) / len(q_idfs) if q_idfs else 0.0
        )
    else:
        feats["frac_rare_q_tokens"] = 0.0

    # ── Family 3: Distractor load ───────────────────────────────────
    # Length spread across options: a large gap lets a model pick by length
    # alone (a surface shortcut, NOT cognition).
    choice_lens = [float(len(t)) for t in choice_tokens]
    feats["choice_len_std"] = _safe_std(choice_lens)

    gt_letter = _ground_truth_letter(qa)
    gt_idx = (ord(gt_letter) - ord("A")) if gt_letter else None
    if gt_idx is not None and 0 <= gt_idx < len(choice_tokens):
        correct_len = float(len(choice_tokens[gt_idx]))
        distractor_lens = [
            float(len(t)) for j, t in enumerate(choice_tokens) if j != gt_idx
        ]
        feats["correct_vs_distractor_len_gap"] = abs(
            correct_len - _safe_mean(distractor_lens)
        )
        # Mean lexical overlap between the correct option and each distractor.
        # High overlap → distractors are "near-misses" → genuinely hard to
        # eliminate by surface cues (raises difficulty for the RIGHT reason).
        correct_set = set(choice_tokens[gt_idx])
        overlaps = [
            _jaccard(correct_set, set(t))
            for j, t in enumerate(choice_tokens)
            if j != gt_idx
        ]
        feats["mean_correct_distractor_overlap"] = _safe_mean(overlaps)
    else:
        feats["correct_vs_distractor_len_gap"] = 0.0
        feats["mean_correct_distractor_overlap"] = 0.0

    # Mean pairwise overlap among ALL options (cohesion of the option set).
    pair_overlaps: list[float] = []
    for i in range(len(choice_tokens)):
        for j in range(i + 1, len(choice_tokens)):
            pair_overlaps.append(_jaccard(set(choice_tokens[i]), set(choice_tokens[j])))
    feats["mean_pairwise_choice_overlap"] = _safe_mean(pair_overlaps)

    # ── Optional: context length (with_context confound) ────────────
    if include_context:
        ctx_tokens = tokenize(get_context(qa))
        feats["context_token_len"] = float(len(ctx_tokens))

    return feats


# ─────────────────────────────────────────────
# Batch helper: build idf once, attach to all rows
# ─────────────────────────────────────────────
def build_surface_feature_table(
    qa_pairs: list[dict[str, Any]],
    include_context: bool = True,
) -> list[dict[str, Any]]:
    """Build a qa_id-keyed surface-feature table for the whole dataset.

    IDF is estimated ONCE over the full corpus so rarity is comparable across
    Bloom levels (the comparison RQ3 cares about).

    Returns:
        List of {qa_id, bloom_level, surface: {...}} rows — ready to dump as
        surface_features.jsonl and join in Phase C.
    """
    idf = build_idf(qa_pairs)
    table: list[dict[str, Any]] = []
    for qa in qa_pairs:
        table.append(
            {
                "qa_id": qa.get("qa_id", ""),
                "bloom_level": qa.get("bloom_level", "Unknown"),
                "surface": extract_surface_features(qa, idf, include_context),
            }
        )
    return table
