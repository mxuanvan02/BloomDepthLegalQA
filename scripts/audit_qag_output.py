#!/usr/bin/env python3
"""Audit BloomDepth Phase-A QAG output for educational textbook QA use.

This audit is designed for the Option A claim:
Vietnamese legal textbook QA for university education.

It checks generated QA pairs before they are used for benchmarking or paper claims:
- schema validity;
- converged vs non-converged status;
- answer option format and answer-label distribution;
- Bloom heuristic agreement;
- duplicate questions;
- metadata/front-matter leakage;
- risky statutory/current-law phrasing;
- rationale presence and syllogism markers;
- high-level Bloom collapse risks.

It is a deterministic pre-filter, not final human/legal validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Glass-box quantitative Bloom rubric (replaces the legacy qualitative
# first-match-wins heuristic). Falls back gracefully if the module is missing
# so this audit never hard-crashes on an incomplete checkout.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from bloom_rubric import score_item as _quant_score_item  # type: ignore
    _HAS_QUANT = True
except Exception:  # pragma: no cover - defensive fallback
    _quant_score_item = None  # type: ignore
    _HAS_QUANT = False

BLOOM_LEVELS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")
ANSWER_RE = re.compile(r"^\s*([A-D])(?:[\.)]|\s|$)", re.I)
OPTION_RE = re.compile(r"^\s*([A-D])(?:[\.)])\s+.+", re.I)
VIETNAMESE_HINT_RE = re.compile(r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.I)
METADATA_RE = re.compile(
    r"(tác\s*giả|chủ\s*biên|biên\s*soạn|nhà\s*xuất\s*bản|mục\s*lục|lời\s*nói\s*đầu|"
    r"lời\s*giới\s*thiệu|trang\s+\d+|chương\s+\d+|giáo\s*trình\s+này)",
    re.I,
)
STATUTORY_RISK_RE = re.compile(
    r"(theo\s+pháp\s+luật\s+hiện\s+hành|theo\s+quy\s+định\s+hiện\s+hành|"
    r"theo\s+Điều\s+\d+|căn\s+cứ\s+Điều\s+\d+|Bộ\s+luật|Nghị\s+định|Thông\s+tư)",
    re.I,
)
SYLLOGISM_RE = re.compile(r"(Đại\s+tiền\s+đề|Tiểu\s+tiền\s+đề|Kết\s+luận)", re.I)

BLOOM_PATTERNS: list[tuple[str, list[str]]] = [
    ("Create", [r"soạn\s+thảo", r"đề\s+xuất", r"thiết\s+kế", r"xây\s+dựng", r"hãy\s+viết", r"hãy\s+lập", r"giải\s+pháp"]),
    ("Evaluate", [r"đánh\s+giá", r"nhận\s+xét", r"có\s+hợp\s+lý", r"phản\s+biện", r"có\s+đúng\s+không", r"lập\s+luận"]),
    ("Analyze", [r"so\s+sánh", r"phân\s+biệt", r"phân\s+tích", r"mối\s+quan\s+hệ", r"điểm\s+(?:giống|khác)", r"nguyên\s+nhân", r"hệ\s+quả"]),
    ("Apply", [r"trong\s+tình\s+huống", r"áp\s+dụng", r"giải\s+quyết\s+(?:vụ|tình)", r"nếu\s+.*?thì", r"xử\s+lý"]),
    ("Understand", [r"giải\s+thích", r"nêu\s+ý\s+nghĩa", r"tóm\s+tắt", r"diễn\s+giải", r"trình\s+bày", r"mô\s+tả"]),
    ("Remember", [r"là\s+gì", r"bao\s+gồm", r"gồm\s+có", r"liệt\s+kê", r"nêu\s+(?:tên|các)", r"quy\s+định\s+(?:nào|gì)"]),
]
COMPILED_BLOOM = [(lvl, [re.compile(p, re.I | re.U) for p in pats]) for lvl, pats in BLOOM_PATTERNS]


def load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    data = json.load(path.open(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "qa_pairs" in data:
        return data["qa_pairs"]
    raise ValueError(f"Unsupported QAG output schema: {path}")


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def norm_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def classify_bloom_heuristic(question: str) -> str:
    for level, patterns in COMPILED_BLOOM:
        for pattern in patterns:
            if pattern.search(question):
                return level
    return "Remember"


def audit_one(row: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    flags: List[str] = []
    q = str(row.get("question") or row.get("question_content") or "")
    cands = row.get("candidate_answers") or []
    gt = str(row.get("ground_truth") or "")
    rationale = str(row.get("legal_rationale") or "")
    bloom = str(row.get("bloom_level") or "")
    ctx = str(row.get("context_text") or "")

    if not q.strip():
        flags.append("missing_question")
    if not VIETNAMESE_HINT_RE.search(q + rationale):
        flags.append("low_vietnamese_signal")
    if not isinstance(cands, list) or len(cands) != 4:
        flags.append("not_exactly_four_candidates")
    else:
        labels = []
        for c in cands:
            m = OPTION_RE.match(str(c))
            labels.append(m.group(1).upper() if m else None)
        if labels != ["A", "B", "C", "D"]:
            flags.append("candidate_labels_invalid")
    gt_match = ANSWER_RE.match(gt)
    gt_label = gt_match.group(1).upper() if gt_match else None
    if gt_label is None:
        flags.append("ground_truth_label_invalid")
    if not rationale.strip():
        flags.append("missing_rationale")
    if rationale and len(SYLLOGISM_RE.findall(rationale)) < 2:
        flags.append("weak_syllogism_markers")
    for required in ("chunk_id", "qa_id", "bloom_level", "context_text"):
        if not row.get(required):
            flags.append(f"missing_{required}")
    if bloom not in BLOOM_LEVELS:
        flags.append("invalid_bloom_level")

    pred_bloom = classify_bloom_heuristic(q)
    # Quantitative glass-box prediction (primary). The heuristic above is kept
    # only as a cheap cross-check / fallback signal.
    quant_bloom = None
    quant_score = None
    quant_breakdown = None
    if _HAS_QUANT and _quant_score_item is not None:
        try:
            qr = _quant_score_item(row)
            quant_bloom = qr.bloom_level
            quant_score = round(qr.demand_score, 4)
            quant_breakdown = qr.contributions
        except Exception:
            quant_bloom = None
    # The authoritative predicted Bloom is the quantitative one when available.
    pred_primary = quant_bloom or pred_bloom
    if bloom in BLOOM_LEVELS and pred_primary != bloom:
        flags.append("bloom_label_mismatch")
    if bloom in {"Evaluate", "Create"} and pred_primary in {"Remember", "Understand", "Apply"}:
        flags.append("high_bloom_collapse_risk")
    if METADATA_RE.search(q + "\n" + rationale):
        flags.append("metadata_leakage")
    if STATUTORY_RISK_RE.search(q + "\n" + rationale) and not STATUTORY_RISK_RE.search(ctx):
        flags.append("external_statutory_phrasing_risk")
    if len(q) < 25:
        flags.append("question_too_short")
    if len(q) > 800:
        flags.append("question_too_long")

    return flags, {
        "ground_truth_label": gt_label,
        "predicted_bloom_heuristic": pred_bloom,
        "predicted_bloom_quant": quant_bloom,
        "bloom_demand_score": quant_score,
        "bloom_score_breakdown": quant_breakdown,
        "question_hash": norm_hash(q) if q else None,
        "question_len": len(q),
        "rationale_len": len(rationale),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("research/results/refinement/adaptive/qa_pairs.json"))
    ap.add_argument("--report", type=Path, default=Path("research/results/audit/qag_output_audit_report.json"))
    ap.add_argument("--flagged", type=Path, default=Path("research/results/audit/qag_output_flagged_examples.json"))
    ap.add_argument("--accepted", type=Path, default=Path("research/results/audit/qag_output_accepted_converged.jsonl"))
    args = ap.parse_args()

    rows = load_records(args.input)
    audited = []
    flag_counts: Counter[str] = Counter()
    by_bloom: Dict[str, Counter[str]] = defaultdict(Counter)
    answer_counts: Counter[str] = Counter()
    qhash_counts: Counter[str] = Counter()

    for row in rows:
        flags, metrics = audit_one(row)
        qh = metrics.get("question_hash")
        if qh:
            qhash_counts[qh] += 1
        answer_counts.update([metrics.get("ground_truth_label") or "INVALID"])
        flag_counts.update(flags)
        by_bloom[str(row.get("bloom_level") or "Unknown")].update(flags or ["PASS_RULES"])
        audited.append({"row": row, "flags": flags, "metrics": metrics})

    duplicate_hashes = {h for h, c in qhash_counts.items() if c > 1}
    for a in audited:
        if a["metrics"].get("question_hash") in duplicate_hashes:
            a["flags"].append("duplicate_question_text")
            flag_counts.update(["duplicate_question_text"])
            by_bloom[str(a["row"].get("bloom_level") or "Unknown")].update(["duplicate_question_text"])

    total = len(audited)
    schema_flags = {
        "missing_question", "not_exactly_four_candidates", "candidate_labels_invalid",
        "ground_truth_label_invalid", "missing_rationale", "missing_chunk_id",
        "missing_qa_id", "missing_bloom_level", "missing_context_text", "invalid_bloom_level",
    }
    schema_valid = [a for a in audited if not schema_flags.intersection(a["flags"])]
    converged = [a for a in audited if a["row"].get("converged") is True]
    accepted = [a for a in audited if a["row"].get("converged") is True and not a["flags"]]
    qlens = [a["metrics"]["question_len"] for a in audited]

    report = {
        "input": str(args.input),
        "total_qa": total,
        "schema_valid": len(schema_valid),
        "schema_valid_rate": round(len(schema_valid) / max(total, 1), 6),
        "converged": len(converged),
        "convergence_rate": round(len(converged) / max(total, 1), 6),
        "accepted_converged_rule_pass": len(accepted),
        "accepted_converged_rule_pass_rate": round(len(accepted) / max(total, 1), 6),
        "flag_counts": dict(flag_counts.most_common()),
        "by_bloom": {k: dict(v.most_common()) for k, v in by_bloom.items()},
        "answer_label_distribution": dict(answer_counts),
        "duplicate_question_hashes": len(duplicate_hashes),
        "question_length": {
            "min": min(qlens) if qlens else None,
            "median": statistics.median(qlens) if qlens else None,
            "mean": statistics.mean(qlens) if qlens else None,
            "max": max(qlens) if qlens else None,
        },
        "pilot_acceptance_thresholds": {
            "schema_valid_rate": ">= 0.95",
            "convergence_rate": ">= 0.50 preferred; lower requires prompt review",
            "metadata_leakage": "<= 0.02",
            "ground_truth_label_invalid": "0",
            "duplicate_question_text": "<= 0.05 of total",
            "high_bloom_collapse_risk": "inspect manually; Create/Evaluate are critical",
        },
        "decision_note": "This deterministic audit is necessary but not sufficient; accepted QA should still receive sampled manual/legal-education review before paper claims.",
    }

    flagged_examples = [
        {
            "qa_id": a["row"].get("qa_id"),
            "chunk_id": a["row"].get("chunk_id"),
            "bloom_level": a["row"].get("bloom_level"),
            "converged": a["row"].get("converged"),
            "flags": a["flags"],
            "question": a["row"].get("question"),
            "ground_truth": a["row"].get("ground_truth"),
        }
        for a in audited if a["flags"]
    ][:500]

    dump_json(args.report, report)
    dump_json(args.flagged, flagged_examples)
    args.accepted.parent.mkdir(parents=True, exist_ok=True)
    with args.accepted.open("w", encoding="utf-8") as f:
        for a in accepted:
            f.write(json.dumps(a["row"], ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
