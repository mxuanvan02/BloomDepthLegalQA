#!/usr/bin/env python3
"""Audit BloomDepth ready contexts for residual leakage.

This is a second-pass QA gate over data/interim/gate_v2/ready_textbook_contexts.jsonl.
It does not modify the input. It reports contexts that slipped through gate_v2
but still look risky for QAG: degraded diacritics, cover/front-matter, review
questions, tables, and ambiguous article anchors.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

VI_DIACRITICS = set(
    "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ĂÂĐÊÔƠƯÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)
ASCII_WORD_RE = re.compile(r"\b(luat|phap luat|dieu|khoan|quyen|nghia vu|nha nuoc|hinh su|dan su|hanh chinh)\b", re.I)
NONCONTENT_RE = re.compile(
    r"(mục\s*lục|muc\s*luc|tài\s*liệu\s*tham\s*khảo|tai\s*lieu\s*tham\s*khao|"
    r"nhà\s*xuất\s*bản|nha\s*xuat\s*ban|lời\s*nói\s*đầu|loi\s*noi\s*dau|"
    r"lời\s*giới\s*thiệu|loi\s*gioi\s*thieu|chủ\s*biên|chu\s*bien|biên\s*soạn|bien\s*soan)",
    re.I,
)
REVIEW_RE = re.compile(r"(câu\s*hỏi|cau\s*hoi|bài\s*tập|bai\s*tap|thảo\s*luận|thao\s*luan)", re.I)
TABLE_RE = re.compile(r"\|[-: ]{3,}\||(?:\|[^\n]*){4,}")
ARTICLE_RE = re.compile(r"\bĐiều\s+\d+", re.I)
LAW_ID_RE = re.compile(r"\b(Luật|Bộ\s+luật|Hiến\s+pháp|Nghị\s+định|Thông\s+tư|Nghị\s+quyết|Pháp\s+lệnh)\b", re.I)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def diacritic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in VI_DIACRITICS for c in letters) / len(letters)


def audit_row(row: Dict[str, Any]) -> Dict[str, Any]:
    text = str(row.get("text") or "")
    dr = diacritic_ratio(text)
    ascii_hits = len(ASCII_WORD_RE.findall(text))
    flags = []
    if dr < 0.08 and ascii_hits >= 1:
        flags.append("residual_low_diacritics")
    if dr < 0.12 and "<!-- image -->" in text:
        flags.append("image_or_cover_low_diacritics")
    if NONCONTENT_RE.search(text[:900]):
        flags.append("residual_front_matter")
    if REVIEW_RE.search(text[:900]):
        flags.append("residual_review_or_exercise")
    if TABLE_RE.search(text):
        flags.append("residual_table_layout")
    if ARTICLE_RE.search(text) and not LAW_ID_RE.search(text):
        flags.append("residual_article_without_law_identity")
    return {
        "chunk_id": row.get("chunk_id"),
        "source_path": row.get("source_path"),
        "chunk_index": row.get("chunk_index"),
        "diacritic_ratio": round(dr, 5),
        "ascii_legal_hits": ascii_hits,
        "flags": flags,
        "preview": re.sub(r"\s+", " ", text[:500]).strip(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/interim/gate_v2/ready_textbook_contexts.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("research/results/audit/ready_contexts_residual_audit.json"))
    ap.add_argument("--sample", type=int, default=120)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    audited = [audit_row(r) for r in rows]
    flagged = [a for a in audited if a["flags"]]
    flag_counts = Counter(f for a in flagged for f in a["flags"])
    ratios = [a["diacritic_ratio"] for a in audited]
    random.seed(42)
    sample = random.sample(audited, min(args.sample, len(audited)))
    sample_flagged = [a for a in sample if a["flags"]]

    report = {
        "input": str(args.input),
        "total_ready_contexts": len(rows),
        "residual_flagged_count": len(flagged),
        "residual_flagged_ratio": round(len(flagged) / max(len(rows), 1), 6),
        "flag_counts": dict(flag_counts),
        "diacritic_ratio": {
            "min": min(ratios) if ratios else None,
            "median": statistics.median(ratios) if ratios else None,
            "mean": statistics.mean(ratios) if ratios else None,
        },
        "sample_size": len(sample),
        "sample_flagged_count": len(sample_flagged),
        "sample_flagged_ratio": round(len(sample_flagged) / max(len(sample), 1), 6),
        "flagged_examples": flagged[:100],
        "sample_flagged_examples": sample_flagged[:30],
        "recommendation": "ready_textbook_contexts is acceptable for a small QAG pilot after excluding residual_flagged rows; it is not yet a fully clean final corpus.",
    }
    dump_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
