#!/usr/bin/env python3
"""Build strict-clean BloomDepth contexts for QAG pilot/paper-grade reporting.

Input:  data/interim/gate_v2/ready_textbook_contexts.jsonl
Output: data/interim/gate_v2/ready_textbook_contexts_strict.jsonl
        data/interim/gate_v2/ready_textbook_contexts_strict_excluded.jsonl
        research/results/audit/strict_clean_contexts_report.json

This script removes residual leakage from the gate_v2 ready layer:
- non-diacritic/degraded OCR text still present in ready contexts;
- image/cover/encoding artifacts;
- review-question/exercise-like chunks;
- front matter/bibliographic/editorial metadata;
- residual table/layout or article-without-law-identity risks.

It does not rewrite, restore, or invent legal text.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

VI_DIACRITICS = set(
    "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ĂÂĐÊÔƠƯÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)
ASCII_LEGAL_RE = re.compile(r"\b(luat|phap luat|dieu|khoan|quyen|nghia vu|nha nuoc|hinh su|dan su|hanh chinh)\b", re.I)
NONCONTENT_RE = re.compile(
    r"(mục\s*lục|muc\s*luc|tài\s*liệu\s*tham\s*khảo|tai\s*lieu\s*tham\s*khao|"
    r"nhà\s*xuất\s*bản|nha\s*xuat\s*ban|lời\s*nói\s*đầu|loi\s*noi\s*dau|"
    r"lời\s*giới\s*thiệu|loi\s*gioi\s*thieu|chủ\s*biên|chu\s*bien|biên\s*soạn|bien\s*soan|"
    r"tổng\s*biên\s*tập|tong\s*bien\s*tap|chịu\s*trách\s*nhiệm|chiu\s*trach\s*nhiem)",
    re.I,
)
REVIEW_RE = re.compile(r"(câu\s*hỏi|cau\s*hoi|bài\s*tập|bai\s*tap|thảo\s*luận|thao\s*luan|ôn\s*tập|on\s*tap)", re.I)
TABLE_RE = re.compile(r"\|[-: ]{3,}\||(?:\|[^\n]*){4,}")
ARTICLE_RE = re.compile(r"\bĐiều\s+\d+", re.I)
LAW_ID_RE = re.compile(r"\b(Luật|Bộ\s+luật|Hiến\s+pháp|Nghị\s+định|Thông\s+tư|Nghị\s+quyết|Pháp\s+lệnh)\b", re.I)
ENCODING_RE = re.compile(r"(Toµ|thÈm|d©n|quyÒn|chÊp|hîp|®|¬|È|µ|Ð|ð)")
IMAGE_RE = re.compile(r"<!--\s*image\s*-->", re.I)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def diacritic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in VI_DIACRITICS for c in letters) / len(letters)


def residual_flags(row: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    text = str(row.get("text") or "")
    dr = diacritic_ratio(text)
    ascii_hits = len(ASCII_LEGAL_RE.findall(text))
    flags: List[str] = []
    first_900 = text[:900]
    if dr < 0.08 and ascii_hits >= 1:
        flags.append("residual_low_diacritics")
    if dr < 0.12 and IMAGE_RE.search(text):
        flags.append("image_or_cover_low_diacritics")
    if ENCODING_RE.search(text):
        flags.append("mojibake_or_encoding_artifact")
    if NONCONTENT_RE.search(first_900):
        flags.append("residual_front_matter")
    if REVIEW_RE.search(first_900):
        flags.append("residual_review_or_exercise")
    if TABLE_RE.search(text):
        flags.append("residual_table_layout")
    if ARTICLE_RE.search(text) and not LAW_ID_RE.search(text):
        flags.append("residual_article_without_law_identity")
    return flags, {
        "diacritic_ratio": round(dr, 5),
        "ascii_legal_hits": ascii_hits,
        "char_count": len(text),
        "word_count": len(re.findall(r"[\wÀ-ỹ]+", text, re.UNICODE)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/interim/gate_v2/ready_textbook_contexts.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/interim/gate_v2/ready_textbook_contexts_strict.jsonl"))
    ap.add_argument("--excluded", type=Path, default=Path("data/interim/gate_v2/ready_textbook_contexts_strict_excluded.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("research/results/audit/strict_clean_contexts_report.json"))
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    clean: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    source_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    ratios_clean: List[float] = []

    for row in rows:
        flags, metrics = residual_flags(row)
        flag_counts.update(flags)
        source = str(row.get("source_path") or row.get("source_doc") or "unknown")
        if flags:
            out = dict(row)
            out["strict_clean"] = {
                "accepted": False,
                "residual_flags": flags,
                **metrics,
            }
            excluded.append(out)
            source_counts[source]["excluded"] += 1
        else:
            out = dict(row)
            out["strict_clean"] = {
                "accepted": True,
                "residual_flags": [],
                **metrics,
            }
            clean.append(out)
            source_counts[source]["accepted"] += 1
            ratios_clean.append(metrics["diacritic_ratio"])

    write_jsonl(args.output, clean)
    write_jsonl(args.excluded, excluded)
    total = len(rows)
    report = {
        "input": str(args.input),
        "outputs": {
            "strict_clean": str(args.output),
            "strict_excluded": str(args.excluded),
        },
        "counts": {
            "input_ready": total,
            "strict_clean": len(clean),
            "strict_excluded": len(excluded),
        },
        "ratios": {
            "strict_clean": round(len(clean) / max(total, 1), 6),
            "strict_excluded": round(len(excluded) / max(total, 1), 6),
        },
        "residual_flag_counts": dict(flag_counts),
        "strict_clean_diacritic_ratio": {
            "min": min(ratios_clean) if ratios_clean else None,
            "median": statistics.median(ratios_clean) if ratios_clean else None,
            "mean": statistics.mean(ratios_clean) if ratios_clean else None,
        },
        "by_source_top": {
            src: dict(c) for src, c in sorted(source_counts.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:30]
        },
        "excluded_examples": [
            {
                "chunk_id": r.get("chunk_id"),
                "source_path": r.get("source_path"),
                "residual_flags": r.get("strict_clean", {}).get("residual_flags", []),
                "diacritic_ratio": r.get("strict_clean", {}).get("diacritic_ratio"),
                "preview": re.sub(r"\s+", " ", str(r.get("text") or "")[:400]).strip(),
            }
            for r in excluded[:100]
        ],
        "paper_note": "Strict clean contexts are a conservative QAG pilot subset, not the final full legal QA corpus. Non-clean contexts are retained for OCR repair and official-source validation, not discarded from the project.",
    }
    dump_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
