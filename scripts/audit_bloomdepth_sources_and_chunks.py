#!/usr/bin/env python3
"""Audit BloomDepth raw PDFs and extracted chunks.

This is a lightweight, non-mutating audit. It does not run OCR, VLM captioning,
or legal-source restoration. It checks:
- raw PDF inventory and size/page counts when PyMuPDF is available;
- whether existing chunks map back to raw files;
- chunk length/domain/source-category distributions;
- legal anchors and common chunk-quality risks;
- multimodal readiness: whether PDF images exist and whether enriched outputs exist.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None

VI_DIACRITICS = set(
    "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
    "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ĂÂĐÊÔƠƯÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)

LEGAL_ANCHOR_RE = re.compile(
    r"\b(Điều|Khoản|Điểm|Chương|Mục|Luật|Bộ\s+luật|Hiến\s+pháp|Nghị\s+định|Thông\s+tư|"
    r"Nghị\s+quyết|Pháp\s+lệnh|Quyết\s+định|Quốc\s+hội|Chính\s+phủ|Tòa\s+án|Toà\s+án|"
    r"Viện\s+kiểm\s+sát|quyền|nghĩa\s+vụ|trách\s+nhiệm|xử\s+phạt|hợp\s+đồng)\b",
    re.IGNORECASE,
)
ARTICLE_RE = re.compile(r"\bĐiều\s+(\d+[a-zA-Z]?)\b", re.IGNORECASE)
DOC_NO_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ-]+(?:-[A-ZĐ]+)*\b")
NON_CONTENT_RE = re.compile(
    r"\b(mục\s*lục|danh\s*mục\s*tài\s*liệu|tài\s*liệu\s*tham\s*khảo|"
    r"tập\s*thể\s*tác\s*giả|lời\s*nói\s*đầu|lời\s*giới\s*thiệu|nhà\s*xuất\s*bản|"
    r"lưu\s*hành\s*nội\s*bộ)\b",
    re.IGNORECASE,
)
REVIEW_Q_RE = re.compile(r"\b(câu\s*hỏi\s*(hướng\s*dẫn|ôn\s*tập|thảo\s*luận)|bài\s*tập)\b", re.IGNORECASE)
TABLE_RE = re.compile(r"\|[-: ]{3,}\||(?:\|[^\n]*){4,}")
OCR_GARBAGE_RE = re.compile(r"(�|□|_{3,}|\.{6,}|-{12,}|[ЛΩ])")
ASCII_LEGAL_RE = re.compile(r"\b(luat|bo luat|hien phap|nghi dinh|thong tu|dieu|khoan|diem)\b", re.I)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def word_count(text: str) -> int:
    return len(re.findall(r"[\wÀ-ỹ]+", text, re.UNICODE))


def diacritic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in VI_DIACRITICS for c in letters) / len(letters)


def quantiles(values: List[int], qs=(0.25, 0.5, 0.75)) -> Dict[str, float]:
    if not values:
        return {}
    vals = sorted(values)
    out = {"min": vals[0], "max": vals[-1]}
    for q in qs:
        idx = int(round((len(vals) - 1) * q))
        out[f"p{int(q*100)}"] = vals[idx]
    return out


def pdf_info(path: Path, root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "mb": round(path.stat().st_size / 1024 / 1024, 3),
        "pages": None,
        "image_count_sample": None,
        "text_chars_sample": None,
        "is_probably_scan_heavy_sample": None,
        "error": None,
    }
    if fitz is None:
        info["error"] = "PyMuPDF not available"
        return info
    try:
        with fitz.open(path) as doc:
            info["pages"] = doc.page_count
            sample_pages = [0]
            if doc.page_count > 2:
                sample_pages.append(doc.page_count // 2)
            if doc.page_count > 1:
                sample_pages.append(doc.page_count - 1)
            text_chars = 0
            image_count = 0
            for i in sorted(set(sample_pages)):
                page = doc.load_page(i)
                text_chars += len(page.get_text("text") or "")
                image_count += len(page.get_images(full=True))
            info["image_count_sample"] = image_count
            info["text_chars_sample"] = text_chars
            info["is_probably_scan_heavy_sample"] = bool(text_chars < 300 and image_count > 0)
    except Exception as e:
        info["error"] = str(e)[:300]
    return info


def chunk_flags(row: Dict[str, Any]) -> List[str]:
    text = str(row.get("text") or "")
    flags: List[str] = []
    wc = word_count(text)
    dr = diacritic_ratio(text)
    if wc < 40:
        flags.append("too_short")
    if NON_CONTENT_RE.search(text[:600]):
        flags.append("front_matter_or_bibliographic")
    if REVIEW_Q_RE.search(text[:800]):
        flags.append("review_questions")
    if TABLE_RE.search(text):
        flags.append("table_or_layout_heavy")
    if OCR_GARBAGE_RE.search(text):
        flags.append("ocr_garbage_tokens")
    if dr < 0.035 and len(ASCII_LEGAL_RE.findall(text)) >= 2:
        flags.append("possible_missing_diacritics")
    if not LEGAL_ANCHOR_RE.search(text):
        flags.append("no_legal_anchor")
    if ARTICLE_RE.search(text) and not re.search(r"\b(Luật|Bộ\s+luật|Hiến\s+pháp|Nghị\s+định|Thông\s+tư|Nghị\s+quyết|Pháp\s+lệnh)\b", text, re.I):
        flags.append("article_without_law_identity")
    if DOC_NO_RE.search(text) or ARTICLE_RE.search(text) or re.search(r"\b(Luật|Bộ\s+luật|Hiến\s+pháp|Nghị\s+định|Thông\s+tư|Nghị\s+quyết|Pháp\s+lệnh)\b", text, re.I):
        flags.append("needs_state_source_check")
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--sample-pdfs", type=int, default=0, help="0 = inspect all PDFs for metadata; set smaller for speed")
    args = ap.parse_args()
    root = args.root.resolve()
    raw = root / "data" / "raw"
    chunks_path = root / "data" / "interim" / "extracted_chunks.jsonl"
    out_dir = root / "research" / "results" / "audit"

    pdfs = sorted(raw.rglob("*.pdf"))
    inspect_pdfs = pdfs if args.sample_pdfs <= 0 else pdfs[: args.sample_pdfs]
    pdf_infos = [pdf_info(p, root) for p in inspect_pdfs]

    rows = read_jsonl(chunks_path) if chunks_path.exists() else []
    by_source = Counter(str(r.get("source_path") or r.get("source_doc") or "unknown") for r in rows)
    by_category = Counter(str(r.get("source_category") or "unknown") for r in rows)
    by_domain = Counter(str(r.get("legal_domain") or "unknown") for r in rows)
    words = [word_count(str(r.get("text") or "")) for r in rows]
    chars = [len(str(r.get("text") or "")) for r in rows]

    flags_counter: Counter[str] = Counter()
    detail_rows: List[Dict[str, Any]] = []
    for r in rows:
        flags = chunk_flags(r)
        flags_counter.update(flags)
        if flags:
            detail_rows.append({
                "chunk_id": r.get("chunk_id"),
                "source_doc": r.get("source_doc"),
                "source_path": r.get("source_path"),
                "chunk_index": r.get("chunk_index"),
                "words": word_count(str(r.get("text") or "")),
                "chars": len(str(r.get("text") or "")),
                "diacritic_ratio": round(diacritic_ratio(str(r.get("text") or "")), 5),
                "flags": flags,
                "preview": re.sub(r"\s+", " ", str(r.get("text") or "")[:500]).strip(),
            })

    raw_total_bytes = sum(p.stat().st_size for p in pdfs)
    report = {
        "root": str(root),
        "raw_pdf_count": len(pdfs),
        "raw_total_mb": round(raw_total_bytes / 1024 / 1024, 3),
        "raw_by_top_level": dict(Counter(str(p.relative_to(raw).parts[0]) if p.relative_to(raw).parts else "root" for p in pdfs)),
        "pdfs_inspected": len(pdf_infos),
        "pdf_page_total_inspected": sum(x.get("pages") or 0 for x in pdf_infos),
        "scan_heavy_sample_count": sum(1 for x in pdf_infos if x.get("is_probably_scan_heavy_sample")),
        "largest_pdfs": sorted(pdf_infos, key=lambda x: x["bytes"], reverse=True)[:15],
        "existing_multimodal_outputs": {
            "images_dir_exists": (root / "data" / "interim" / "images").exists(),
            "images_count": len([p for p in (root / "data" / "interim" / "images").rglob("*") if p.is_file()]) if (root / "data" / "interim" / "images").exists() else 0,
            "enriched_chunks_exists": (root / "data" / "interim" / "extracted_chunks_enriched.jsonl").exists(),
        },
        "chunks": {
            "path": str(chunks_path.relative_to(root)) if chunks_path.exists() else None,
            "count": len(rows),
            "file_mb": round(chunks_path.stat().st_size / 1024 / 1024, 3) if chunks_path.exists() else 0,
            "word_stats": quantiles(words),
            "char_stats": quantiles(chars),
            "source_category_counts": dict(by_category),
            "legal_domain_counts_top": dict(by_domain.most_common(30)),
            "source_count": len(by_source),
            "top_sources_by_chunks": dict(by_source.most_common(20)),
            "flag_counts": dict(flags_counter),
            "flagged_chunks": len(detail_rows),
        },
        "recommendation": {
            "multimodal": "Run a pilot first on 3-5 PDFs or selected pages. The raw corpus is large; full VLM enrichment may be slow even if feasible.",
            "chunks": "Refine chunk gates before QAG: exclude front matter/review-only chunks; route legal anchors to state-source verification; sample audit flagged chunks.",
        },
    }

    dump_json(out_dir / "bloomdepth_source_chunk_audit_report.json", report)
    dump_json(out_dir / "bloomdepth_pdf_inventory.json", pdf_infos)
    dump_json(out_dir / "bloomdepth_flagged_chunk_samples.json", detail_rows[:5000])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
