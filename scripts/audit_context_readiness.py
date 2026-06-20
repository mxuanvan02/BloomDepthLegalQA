#!/usr/bin/env python3
"""Context readiness audit and Bloom routing for BloomDepth.

This script is deliberately deterministic and dependency-free. It gates noisy
Vietnamese legal textbook contexts before any expensive QAG/refinement run.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

VI_CHARS = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ")
VI_COMMON_ASCII = {
    "phap luat", "nha nuoc", "quyen", "nghia vu", "trach nhiem", "hanh vi",
    "quan he", "xa hoi", "con nguoi", "cong dan", "hinh su", "dan su",
    "hanh chinh", "to tung", "so huu", "thuc hien", "quy dinh", "van ban",
    "co quan", "to chuc", "ca nhan", "hop dong", "tai san", "lao dong",
    "hon nhan", "gia dinh", "quoc te", "hien phap", "chinh tri", "kinh te",
}
LEGAL_ANCHOR_RE = re.compile(r"\b(Điều|Khoản|Điểm|Luật|Bộ luật|Hiến pháp|Nghị định|Thông tư|Nghị quyết|Quyết định)\b", re.I)
ARTICLE_RE = re.compile(r"\bĐiều\s+\d+", re.I)
LAYOUT_RE = re.compile(r"(\|[- :|]{5,}\||_{4,}|={4,}|\.{5,}|\bTrang\s+\d+\b)", re.I)
FRONT_RE = re.compile(r"(mục lục|nhà xuất bản|tái bản|lời nói đầu|lời mở đầu|danh mục tài liệu|tài liệu tham khảo|chủ biên|biên soạn)", re.I)
EXERCISE_RE = re.compile(r"(câu hỏi\s+(ôn tập|thảo luận)|bài tập|câu hỏi nhận định|hướng dẫn học tập|ôn tập chương)", re.I)
OCR_RE = re.compile(r"(�|\x00|[{}<>]{2,}|[\u2500-\u257f])")
YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")
SPACED_OCR_RE = re.compile(r"\b[A-Za-zÀ-ỹ]{1,2}\s+[A-Za-zÀ-ỹ]{1,3}\s+[A-Za-zÀ-ỹ]{1,2}\b")
SPACED_OCR_MIN_HITS = 25

BLOOM_PATTERNS = {
    "Remember": [r"\blà\b", r"khái niệm", r"định nghĩa", r"bao gồm", r"gồm", r"đặc điểm", r"phân loại", r"nguyên tắc"],
    "Understand": [r"ý nghĩa", r"bản chất", r"vai trò", r"giải thích", r"thể hiện", r"cho thấy", r"được hiểu", r"mục đích"],
    "Apply": [r"điều kiện", r"trường hợp", r"khi", r"nếu", r"áp dụng", r"hậu quả", r"xử lý", r"thực hiện", r"căn cứ"],
    "Analyze": [r"so sánh", r"phân biệt", r"mối quan hệ", r"nguyên nhân", r"hệ quả", r"cấu thành", r"yếu tố", r"tương quan", r"khác nhau"],
    "Evaluate": [r"đánh giá", r"nhận xét", r"hợp lý", r"hạn chế", r"ưu điểm", r"nhược điểm", r"tranh luận", r"quan điểm", r"phê phán", r"cần thiết"],
    "Create": [r"đề xuất", r"xây dựng", r"thiết kế", r"soạn thảo", r"giải pháp", r"quy trình", r"mô hình", r"kiến nghị", r"hoàn thiện"],
}

WEIGHTS = {
    "vietnamese_text_health": 0.25,
    "ocr_cleanliness": 0.20,
    "sentence_coherence": 0.15,
    "legal_educational_value": 0.15,
    "bloom_affordance": 0.10,
    "provenance_safety": 0.10,
    "uniqueness": 0.05,
}


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                obj = json.loads(line)
                obj.setdefault("_audit_line_no", line_no)
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def text_metrics(text: str) -> dict[str, Any]:
    chars = len(text)
    letters = re.findall(r"[A-Za-zÀ-ỹ]+", text)
    diacritics = sum(1 for c in text if c in VI_CHARS)
    ascii_hits = sum(text.lower().count(p) for p in VI_COMMON_ASCII)
    punctuation = sum(text.count(x) for x in ".,;:!?…")
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    short_lines = sum(1 for ln in text.splitlines() if 0 < len(ln.strip()) < 25)
    line_count = max(1, len(text.splitlines()))
    weird = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,;:!?()[]{}\"'/-–—%+&=<>|_*\n" or c in VI_CHARS))
    spaced_ocr_hits = len(SPACED_OCR_RE.findall(text))
    return {
        "char_count": chars,
        "word_count": len(letters),
        "short_token_ratio": sum(1 for w in letters if len(w) <= 2) / max(1, len(letters)),
        "diacritic_ratio": diacritics / max(1, chars),
        "ascii_legal_phrase_hits": ascii_hits,
        "punctuation_density": punctuation / max(1, chars),
        "sentence_count": len(sentences),
        "median_sentence_chars": median([len(s) for s in sentences]) if sentences else 0,
        "short_line_ratio": short_lines / line_count,
        "weird_char_ratio": weird / max(1, chars),
        "has_layout_pattern": bool(LAYOUT_RE.search(text)),
        "has_front_matter": bool(FRONT_RE.search(text)),
        "has_exercise_marker": bool(EXERCISE_RE.search(text)),
        "has_ocr_pattern": bool(OCR_RE.search(text)),
        "spaced_ocr_hits": spaced_ocr_hits,
        "has_spaced_ocr_pattern": spaced_ocr_hits >= SPACED_OCR_MIN_HITS,
        "has_legal_anchor": bool(LEGAL_ANCHOR_RE.search(text)),
        "has_article_anchor": bool(ARTICLE_RE.search(text)),
        "has_year": bool(YEAR_RE.search(text)),
    }


def score_context(text: str, obj: dict[str, Any], dup_count: int) -> tuple[dict[str, float], list[str], list[str]]:
    m = text_metrics(text)
    risks: list[str] = []
    review: list[str] = []
    if m["char_count"] < 800:
        risks.append("too_short_for_rich_qag")
    if m["char_count"] > 4200:
        review.append("long_context_may_need_split")
    if m["diacritic_ratio"] < 0.08 or m["ascii_legal_phrase_hits"] >= 2:
        risks.append("possible_residual_missing_diacritics")
    if m["has_ocr_pattern"] or m["weird_char_ratio"] > 0.01:
        risks.append("ocr_or_encoding_noise")
    if m["has_spaced_ocr_pattern"] or m["short_token_ratio"] > 0.34:
        risks.append("spaced_ocr_word_fragmentation")
    if m["has_layout_pattern"] or m["short_line_ratio"] > 0.25:
        risks.append("layout_or_table_noise")
    if m["has_front_matter"]:
        risks.append("front_matter_or_bibliographic")
    if m["has_exercise_marker"]:
        review.append("exercise_or_review_marker")
    if m["has_legal_anchor"] or m["has_article_anchor"]:
        review.append("legal_anchor_needs_provenance_check")
    if dup_count > 1:
        risks.append("duplicate_text_hash")

    vietnamese = clamp((m["diacritic_ratio"] - 0.06) / 0.12) * (0.75 if m["ascii_legal_phrase_hits"] else 1.0)
    ocr = 1.0
    if m["has_ocr_pattern"]:
        ocr -= 0.4
    if m["has_spaced_ocr_pattern"] or m["short_token_ratio"] > 0.34:
        ocr -= 0.35
    if m["weird_char_ratio"] > 0.004:
        ocr -= 0.25
    if m["has_layout_pattern"]:
        ocr -= 0.25
    coherence = clamp(1 - abs(m["median_sentence_chars"] - 120) / 220)
    if m["sentence_count"] < 4:
        coherence *= 0.7
    edu = 0.55
    if 1200 <= m["char_count"] <= 3200:
        edu += 0.18
    if m["has_front_matter"]:
        edu -= 0.45
    if m["has_exercise_marker"]:
        edu -= 0.2
    if m["has_year"]:
        edu += 0.04
    bloom_scores = bloom_affordance_scores(text)
    bloom_affordance = max(bloom_scores.values()) if bloom_scores else 0.0
    provenance = 1.0
    if m["has_legal_anchor"]:
        provenance -= 0.25
    if not (obj.get("source_doc") and obj.get("chunk_id") and obj.get("content_hash")):
        provenance -= 0.35
    unique = 1.0 if dup_count <= 1 else 0.2
    scores = {
        "vietnamese_text_health": clamp(vietnamese),
        "ocr_cleanliness": clamp(ocr),
        "sentence_coherence": clamp(coherence),
        "legal_educational_value": clamp(edu),
        "bloom_affordance": clamp(bloom_affordance),
        "provenance_safety": clamp(provenance),
        "uniqueness": clamp(unique),
    }
    return scores, risks, review


def bloom_affordance_scores(text: str) -> dict[str, float]:
    low = text.lower()
    scores = {}
    for level, pats in BLOOM_PATTERNS.items():
        hits = sum(1 for pat in pats if re.search(pat, low, re.I))
        scores[level] = clamp(hits / 3.0)
    if len(text) < 1200:
        scores["Evaluate"] *= 0.6
        scores["Create"] *= 0.5
    return scores


def route_bloom(text: str, quality: float, risks: list[str]) -> tuple[list[str], dict[str, str], dict[str, float]]:
    scores = bloom_affordance_scores(text)
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    hard_risk = any(r in risks for r in ["possible_residual_missing_diacritics", "ocr_or_encoding_noise", "layout_or_table_noise", "front_matter_or_bibliographic"])
    for level in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
        threshold = 0.32 if level in {"Remember", "Understand", "Apply"} else 0.45
        if quality < 0.65 or hard_risk:
            excluded[level] = "context_quality_or_noise_risk"
        elif scores[level] >= threshold:
            eligible.append(level)
        else:
            excluded[level] = "insufficient_level_specific_affordance"
    if quality >= 0.78 and not eligible:
        # Safe fallback: good expository textbook chunks can support lower Bloom.
        eligible = ["Remember", "Understand"]
        excluded = {k: v for k, v in excluded.items() if k not in eligible}
    return eligible, excluded, scores


def tier_of(quality: float, risks: list[str], review: list[str], eligible: list[str]) -> str:
    critical = {"possible_residual_missing_diacritics", "ocr_or_encoding_noise", "spaced_ocr_word_fragmentation", "layout_or_table_noise", "front_matter_or_bibliographic", "duplicate_text_hash"}
    if any(r in critical for r in risks) or quality < 0.65 or not eligible:
        return "reject"
    if quality >= 0.85 and not review and len(eligible) >= 2:
        return "gold_candidate"
    return "silver_candidate"


def bar_svg(path: Path, counts: dict[str, int], title: str, width: int = 900, height: int = 420) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    maxv = max(vals) if vals else 1
    margin_l, margin_b, margin_t = 110, 80, 50
    plot_w = width - margin_l - 30
    plot_h = height - margin_t - margin_b
    bar_w = plot_w / max(1, len(labels)) * 0.68
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#fbfaf7"/>',
             f'<text x="{width/2}" y="28" text-anchor="middle" font-family="serif" font-size="22" font-weight="700">{html.escape(title)}</text>',
             f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{width-20}" y2="{margin_t+plot_h}" stroke="#222"/>']
    for i, (lab, val) in enumerate(zip(labels, vals)):
        x = margin_l + i * (plot_w / max(1, len(labels))) + (plot_w / max(1, len(labels)) - bar_w) / 2
        h = plot_h * val / maxv
        y = margin_t + plot_h - h
        color = ["#245c4f", "#d28c3c", "#7a4e2d", "#3b6f9e", "#8b3a3a", "#6a5a99"][i % 6]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13">{val}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{height-48}" text-anchor="middle" font-family="sans-serif" font-size="12" transform="rotate(-25 {x+bar_w/2:.1f},{height-48})">{html.escape(lab)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(path: Path, report: dict[str, Any], refs: list[str]) -> None:
    lines = [
        "# BloomDepth Context Readiness and Bloom Routing Report",
        "",
        "## Executive Summary",
        f"- Input contexts: **{report['total_contexts']}**.",
        f"- Gold candidates: **{report['tier_counts'].get('gold_candidate', 0)}**.",
        f"- Silver candidates: **{report['tier_counts'].get('silver_candidate', 0)}**.",
        f"- Rejected / repair-first contexts: **{report['tier_counts'].get('reject', 0)}**.",
        f"- Estimated QAG jobs after Bloom routing: **{report['estimated_qag_jobs']}**, versus naive 6-level generation: **{report['naive_qag_jobs']}**.",
        f"- Estimated job reduction: **{report['job_reduction_pct']:.2f}%**.",
        "",
        "## Algorithm",
        "1. Deterministic text-health audit: diacritics, ASCII legal phrase leakage, OCR/layout artifacts, sentence integrity, metadata/front-matter leakage, legal-anchor risk, and provenance completeness.",
        "2. Weighted readiness scoring: Vietnamese text health, OCR cleanliness, sentence coherence, educational value, Bloom affordance, provenance safety, and uniqueness.",
        "3. Bloom suitability routing: each context is assigned only to levels it can support, rather than generating six questions per context.",
        "4. Tiering: gold candidates are directly eligible for QAG pilot; silver candidates are usable after light review; rejects require repair or exclusion.",
        "",
        "## Tier Counts",
        "",
        "| Tier | Count |",
        "|---|---:|",
    ]
    for tier, count in report["tier_counts"].items():
        lines.append(f"| {tier} | {count} |")
    lines += ["", "## Top Risk Flags", "", "| Risk flag | Count |", "|---|---:|"]
    for flag, count in report["risk_counts"].items():
        lines.append(f"| `{flag}` | {count} |")
    lines += ["", "## Bloom-Level Routed Job Counts", "", "| Bloom level | Candidate jobs |", "|---|---:|"]
    for level, count in report["bloom_job_counts"].items():
        lines.append(f"| {level} | {count} |")
    lines += [
        "",
        "## Recommended Next Steps",
        "1. Use `gold_contexts.jsonl` for the first QAG pilot; do not run QAG on rejected contexts.",
        "2. Sample-check at least 30 gold, 30 silver, and 30 rejected contexts to calibrate thresholds before scaling.",
        "3. Run QAG only on routed Bloom levels, then post-filter QA by convergence, schema validity, grounding, single-best-answer validity, distractor quality, and Bloom alignment.",
        "4. Keep OCR/diacritic repair and official-source validation as separate streams; do not merge them into the benchmark until their provenance is validated.",
        "",
        "## References",
        "",
    ]
    lines += [f"{i+1}. {ref}" for i, ref in enumerate(refs)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/interim/gate_v2/paper_qag_contexts.jsonl")
    ap.add_argument("--out-dir", default="data/interim/gate_v2/readiness")
    ap.add_argument("--report-dir", default="research/results/audit")
    ap.add_argument("--paper-report", default="research/reports/context_readiness_deep_research_report.md")
    args = ap.parse_args()
    input_path = Path(args.input)
    rows = read_jsonl(input_path)
    hashes = Counter((r.get("content_hash") or r.get("text", "")) for r in rows)
    audited = []
    tier_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    risk_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    bloom_counts: Counter[str] = Counter()
    for r in rows:
        text = r.get("text", "")
        dup_count = hashes.get(r.get("content_hash") or text, 1)
        scores, risks, review = score_context(text, r, dup_count)
        quality = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
        eligible, excluded, bloom_scores = route_bloom(text, quality, risks)
        tier = tier_of(quality, risks, review, eligible)
        audit = {
            "readiness_quality_score": round(quality, 4),
            "readiness_component_scores": {k: round(v, 4) for k, v in scores.items()},
            "risk_flags": risks,
            "review_flags": review,
            "bloom_affordance_scores": {k: round(v, 4) for k, v in bloom_scores.items()},
            "eligible_bloom_levels": eligible,
            "excluded_bloom_levels": excluded,
            "readiness_tier": tier,
        }
        out = {**r, "readiness_audit": audit}
        audited.append(out)
        tier_rows[tier].append(out)
        risk_counts.update(risks)
        review_counts.update(review)
        bloom_counts.update(eligible)
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "context_readiness_audit.jsonl", audited)
    write_jsonl(out_dir / "gold_contexts.jsonl", tier_rows.get("gold_candidate", []))
    write_jsonl(out_dir / "silver_contexts.jsonl", tier_rows.get("silver_candidate", []))
    write_jsonl(out_dir / "reject_contexts.jsonl", tier_rows.get("reject", []))
    with (out_dir / "context_readiness_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "tier", "quality_score", "eligible_bloom_levels", "risk_flags", "review_flags", "source_doc"])
        for r in audited:
            a = r["readiness_audit"]
            w.writerow([r.get("chunk_id"), a["readiness_tier"], a["readiness_quality_score"], ";".join(a["eligible_bloom_levels"]), ";".join(a["risk_flags"]), ";".join(a["review_flags"]), r.get("source_doc")])
    tier_counts = Counter(r["readiness_audit"]["readiness_tier"] for r in audited)
    report = {
        "input": str(input_path),
        "total_contexts": len(audited),
        "tier_counts": dict(tier_counts),
        "risk_counts": dict(risk_counts.most_common()),
        "review_counts": dict(review_counts.most_common()),
        "bloom_job_counts": dict((lvl, bloom_counts.get(lvl, 0)) for lvl in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]),
        "estimated_qag_jobs": sum(bloom_counts.values()),
        "naive_qag_jobs": len(audited) * 6,
        "job_reduction_pct": 100 * (1 - sum(bloom_counts.values()) / max(1, len(audited) * 6)),
        "score_summary": {
            "mean": round(mean([r["readiness_audit"]["readiness_quality_score"] for r in audited]), 4),
            "median": round(median([r["readiness_audit"]["readiness_quality_score"] for r in audited]), 4),
        },
    }
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "context_readiness_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_dir = Path("research/artifacts/context_readiness")
    bar_svg(plot_dir / "tier_counts.svg", report["tier_counts"], "Context Readiness Tiers")
    bar_svg(plot_dir / "bloom_job_counts.svg", report["bloom_job_counts"], "Bloom-Routed Candidate Jobs")
    refs = REFERENCES
    write_report(Path(args.paper_report), report, refs)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


REFERENCES = [
    "Bloom, B. S. (1956). Taxonomy of Educational Objectives: The Classification of Educational Goals.",
    "Anderson, L. W., & Krathwohl, D. R. (2001). A Taxonomy for Learning, Teaching, and Assessing.",
    "Madaan, A. et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. NeurIPS.",
    "Snell, C. et al. (2024). Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.",
    "Wang, X. et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR.",
    "Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS.",
    "Kojima, T. et al. (2022). Large Language Models are Zero-Shot Reasoners. NeurIPS.",
    "Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems.",
    "Zelikman, E. et al. (2022). STaR: Bootstrapping Reasoning With Reasoning. NeurIPS.",
    "Saunders, W. et al. (2022). Self-critiquing models for assisting human evaluators.",
    "Bai, Y. et al. (2022). Constitutional AI: Harmlessness from AI Feedback.",
    "Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. NeurIPS.",
    "Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS Datasets and Benchmarks.",
    "Kadavath, S. et al. (2022). Language Models (Mostly) Know What They Know.",
    "Ribeiro, M. T. et al. (2020). Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL.",
    "Raji, I. D. et al. (2021). AI and the Everything in the Whole Wide World Benchmark. NeurIPS Datasets and Benchmarks.",
    "Gebru, T. et al. (2021). Datasheets for Datasets. CACM.",
    "Bender, E. M., & Friedman, B. (2018). Data Statements for NLP. TACL.",
    "Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT*.",
    "Rogers, A. (2021). Changing the World by Changing the Data. ACL.",
    "Northcutt, C. G. et al. (2021). Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks. NeurIPS.",
    "Paullada, A. et al. (2021). Data and its (dis)contents: A survey of dataset development and use in machine learning research. Patterns.",
    "Hendrycks, D. et al. (2021). Measuring Massive Multitask Language Understanding. ICLR.",
    "Chalkidis, I. et al. (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. ACL.",
    "Zhong, H. et al. (2020). How Does NLP Benefit Legal System: A Summary of Legal Artificial Intelligence. ACL.",
    "Henderson, P. et al. (2022). Pile of Law: Learning Responsible Data Filtering from the Law and a 256GB Open-Source Legal Dataset. NeurIPS.",
    "Nguyen, D. Q. et al. (2020). PhoBERT: Pre-trained language models for Vietnamese. Findings of EMNLP.",
    "Doan, L. et al. (2021). VLSP shared tasks and Vietnamese NLP resources: lessons for corpus construction.",
    "Smith, R. (2007). An Overview of the Tesseract OCR Engine. ICDAR.",
    "Mori, S., Suen, C. Y., & Yamamoto, K. (1992). Historical review of OCR research and development. Proceedings of the IEEE.",
]

if __name__ == "__main__":
    raise SystemExit(main())
