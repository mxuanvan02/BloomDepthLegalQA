#!/usr/bin/env python3
"""Build BloomDepth chunk gate v2.

This script partitions extracted chunks into four mutually exclusive layers:

1. ready_textbook_contexts.jsonl
2. needs_diacritic_ocr_repair.jsonl
3. needs_state_source_check.jsonl
4. excluded_from_qag.jsonl

Policy:
- Exclusion has highest priority for front matter, bibliography, review-only,
  very short, layout/image-only-like chunks.
- Diacritic/OCR repair is prioritized before state-source check because legal
  anchor extraction on non-diacritic OCR text is unreliable.
- State-source check is required for chunks with explicit legal document anchors.
- Ready contexts are textbook/theory chunks without strong repair/source-check
  requirements.

The script does not mutate the original chunk file and does not rewrite legal
content.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

VI_DIACRITICS = set(
    "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ĂÂĐÊÔƠƯÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)

LEGAL_DOC_RE = re.compile(
    r"\b(Hiến\s+pháp|Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư|Nghị\s+quyết|Pháp\s+lệnh|Quyết\s+định)\b",
    re.IGNORECASE,
)
LEGAL_ANCHOR_RE = re.compile(
    r"\b(Điều|Khoản|Điểm|Chương|Mục|quyền|nghĩa\s+vụ|trách\s+nhiệm|xử\s+phạt|hợp\s+đồng|"
    r"Quốc\s+hội|Chính\s+phủ|Tòa\s+án|Toà\s+án|Viện\s+kiểm\s+sát)\b",
    re.IGNORECASE,
)
ARTICLE_RE = re.compile(r"\bĐiều\s+(\d+[a-zA-Z]?)\b", re.IGNORECASE)
DOC_NO_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ-]+(?:-[A-ZĐ]+)*\b")
ASCII_LEGAL_RE = re.compile(
    r"\b(luat|bo luat|hien phap|nghi dinh|thong tu|nghi quyet|phap lenh|quyet dinh|"
    r"dieu|khoan|diem|quyen|nghia vu|trach nhiem|xu phat|quy dinh|toa an|vien kiem sat|"
    r"hop dong|tai san|thuong mai|hinh su|dan su|hanh chinh|nha nuoc|phap luat)\b",
    re.IGNORECASE,
)
NON_CONTENT_RE = re.compile(
    r"\b(mục\s*lục|muc\s*luc|danh\s*mục\s*tài\s*liệu|danh\s*muc\s*tai\s*lieu|"
    r"tài\s*liệu\s*tham\s*khảo|tai\s*lieu\s*tham\s*khao|tập\s*thể\s*tác\s*giả|tap\s*the\s*tac\s*gia|"
    r"lời\s*nói\s*đầu|loi\s*noi\s*dau|lời\s*giới\s*thiệu|loi\s*gioi\s*thieu|"
    r"nhà\s*xuất\s*bản|nha\s*xuat\s*ban|lưu\s*hành\s*nội\s*bộ|luu\s*hanh\s*noi\s*bo)\b",
    re.IGNORECASE,
)
REVIEW_Q_RE = re.compile(
    r"\b(câu\s*hỏi\s*(hướng\s*dẫn|ôn\s*tập|thảo\s*luận)|cau\s*hoi\s*(huong\s*dan|on\s*tap|thao\s*luan)|bài\s*tập|bai\s*tap)\b",
    re.IGNORECASE,
)
TABLE_RE = re.compile(r"\|[-: ]{3,}\||(?:\|[^\n]*){4,}")
OCR_GARBAGE_RE = re.compile(r"(�|□|_{3,}|\.{6,}|-{12,}|[ЛΩ]|\b[A-Z]{4,}\d[A-Z0-9]{3,}\b)")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                obj.setdefault("_line_no", line_no)
                yield obj


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def word_count(text: str) -> int:
    return len(re.findall(r"[\wÀ-ỹ]+", text, re.UNICODE))


def diacritic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in VI_DIACRITICS for c in letters) / len(letters)


def extract_source_anchors(text: str) -> Dict[str, Any]:
    law_doc_mentions = []
    for m in LEGAL_DOC_RE.finditer(text):
        start = max(0, m.start() - 10)
        end = min(len(text), m.end() + 90)
        phrase = re.sub(r"\s+", " ", text[start:end]).strip(" .,:;\n")
        if phrase and phrase not in law_doc_mentions:
            law_doc_mentions.append(phrase)
    return {
        "law_doc_mentions": law_doc_mentions[:10],
        "articles": sorted(set(ARTICLE_RE.findall(text)), key=lambda x: (len(x), x))[:20],
        "document_numbers": sorted(set(DOC_NO_RE.findall(text)))[:20],
        "legal_anchor_hits": len(LEGAL_ANCHOR_RE.findall(text)) + len(LEGAL_DOC_RE.findall(text)),
        "ascii_legal_hits": len(ASCII_LEGAL_RE.findall(text)),
    }


def classify(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    text = str(row.get("text") or "")
    wc = word_count(text)
    dr = diacritic_ratio(text)
    anchors = extract_source_anchors(text)
    first_700 = text[:700]
    flags: List[str] = []

    if wc < 60:
        flags.append("too_short")
    if NON_CONTENT_RE.search(first_700):
        flags.append("front_matter_or_bibliographic")
    if REVIEW_Q_RE.search(first_700):
        flags.append("review_questions_or_exercises")
    if TABLE_RE.search(text):
        flags.append("table_or_layout_heavy")
    if OCR_GARBAGE_RE.search(text):
        flags.append("ocr_garbage_tokens")
    if dr < 0.035 and anchors["ascii_legal_hits"] >= 2:
        flags.append("possible_missing_diacritics")
    if not (LEGAL_ANCHOR_RE.search(text) or LEGAL_DOC_RE.search(text) or ASCII_LEGAL_RE.search(text)):
        flags.append("no_legal_anchor")
    has_state_anchor = bool(anchors["law_doc_mentions"] or anchors["articles"] or anchors["document_numbers"])
    if has_state_anchor:
        flags.append("needs_state_source_check")
    if anchors["articles"] and not anchors["law_doc_mentions"] and not anchors["document_numbers"]:
        flags.append("article_without_law_identity")

    # Mutually exclusive decision.
    if "front_matter_or_bibliographic" in flags or ("too_short" in flags and "no_legal_anchor" in flags):
        bucket = "excluded_from_qag"
    elif "review_questions_or_exercises" in flags and anchors["legal_anchor_hits"] < 3:
        bucket = "excluded_from_qag"
    elif "possible_missing_diacritics" in flags or "ocr_garbage_tokens" in flags or "table_or_layout_heavy" in flags:
        bucket = "needs_diacritic_ocr_repair"
    elif "article_without_law_identity" in flags or has_state_anchor:
        bucket = "needs_state_source_check"
    else:
        bucket = "ready_textbook_contexts"

    meta = {
        "gate_v2": {
            "bucket": bucket,
            "flags": flags,
            "word_count": wc,
            "char_count": len(text),
            "diacritic_ratio": round(dr, 5),
            "source_anchors": anchors,
            "policy_note": "Do not use non-ready buckets for QAG until repaired, verified, or manually approved.",
        }
    }
    return bucket, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", type=Path, default=Path("data/interim/extracted_chunks.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/interim/gate_v2"))
    ap.add_argument("--report", type=Path, default=Path("research/results/audit/chunk_gate_v2_report.json"))
    args = ap.parse_args()

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "ready_textbook_contexts": [],
        "needs_diacritic_ocr_repair": [],
        "needs_state_source_check": [],
        "excluded_from_qag": [],
    }
    flag_counts: Counter[str] = Counter()
    source_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    domain_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    category_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in read_jsonl(args.chunks):
        bucket, meta = classify(row)
        enriched = dict(row)
        enriched.update(meta)
        buckets[bucket].append(enriched)
        flag_counts.update(meta["gate_v2"]["flags"])
        source_counts[str(row.get("source_path") or row.get("source_doc") or "unknown")][bucket] += 1
        domain_counts[str(row.get("legal_domain") or "unknown")][bucket] += 1
        category_counts[str(row.get("source_category") or "unknown")][bucket] += 1
        if len(examples[bucket]) < 20:
            examples[bucket].append({
                "chunk_id": row.get("chunk_id"),
                "source_path": row.get("source_path"),
                "chunk_index": row.get("chunk_index"),
                "flags": meta["gate_v2"]["flags"],
                "diacritic_ratio": meta["gate_v2"]["diacritic_ratio"],
                "preview": re.sub(r"\s+", " ", str(row.get("text") or "")[:400]).strip(),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for bucket, rows in buckets.items():
        path = args.out_dir / f"{bucket}.jsonl"
        write_jsonl(path, rows)
        output_paths[bucket] = str(path)

    total = sum(len(v) for v in buckets.values())
    report = {
        "input_chunks": total,
        "bucket_counts": {k: len(v) for k, v in buckets.items()},
        "bucket_ratios": {k: round(len(v) / total, 6) if total else 0 for k, v in buckets.items()},
        "flag_counts": dict(flag_counts),
        "outputs": output_paths,
        "by_source_top": {
            src: dict(c) for src, c in sorted(source_counts.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:30]
        },
        "by_domain": {d: dict(c) for d, c in domain_counts.items()},
        "by_source_category": {categ: dict(c) for categ, c in category_counts.items()},
        "examples": dict(examples),
        "policy": {
            "ready_textbook_contexts": "May feed QAG after sample QA audit.",
            "needs_diacritic_ocr_repair": "Do not feed QAG until OCR/diacritic restoration or re-extraction is done.",
            "needs_state_source_check": "Do not use for statutory QA until official state-source verification/provenance is recorded.",
            "excluded_from_qag": "Do not feed QAG.",
        },
    }
    dump_json(args.report, report)

    csv_path = args.report.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bucket", "count", "ratio", "output"])
        writer.writeheader()
        for bucket, count in report["bucket_counts"].items():
            writer.writerow({"bucket": bucket, "count": count, "ratio": report["bucket_ratios"][bucket], "output": output_paths[bucket]})

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
