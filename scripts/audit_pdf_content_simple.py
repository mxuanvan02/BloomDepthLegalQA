#!/usr/bin/env python3
"""
Simplified PDF content audit without pymupdf dependency.
Reads raw PDF bytes and searches for Vietnamese text patterns to classify.

Usage:
    python scripts/audit_pdf_content_simple.py [--target gap_2026]
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)


def extract_text_from_raw_bytes(pdf_path: Path, max_bytes: int = 150000) -> str:
    """
    Extract searchable text from raw PDF bytes.
    PDFs store text as Latin-1 or UTF-8 in stream objects.
    This is crude but works for Vietnamese diacritic patterns.
    """
    try:
        with open(pdf_path, 'rb') as f:
            raw = f.read(max_bytes)
        
        # Try decode as latin-1 (common in PDF streams)
        try:
            text = raw.decode('latin-1', errors='ignore')
        except:
            text = raw.decode('utf-8', errors='ignore')
        
        # PDF has a lot of binary noise — keep only printable Vietnamese-compatible chars
        # Keep: letters, numbers, spaces, Vietnamese diacritics (combining chars)
        text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t')
        
        return text
    except Exception as e:
        logger.debug(f"Failed reading {pdf_path.name}: {e}")
        return ""


def classify_content(text: str, filename: str) -> Dict:
    """
    Classify based on text patterns found in raw PDF bytes.
    """
    if not text or len(text) < 500:
        return {"type": "unknown", "confidence": 0, "signals": []}

    text_lower = text.lower()
    fn_lower = filename.lower()
    
    signals = []

    # === Textbook signals ===
    textbook_signals = []
    if 'giao trinh' in text_lower or 'giáo trình' in fn_lower:
        textbook_signals.append('giáo_trình_keyword')
    if 'muc luc' in text_lower or 'mục lục' in text_lower:
        textbook_signals.append('mục_lục')
    if re.search(r'chuong\s*\d+', text_lower) or re.search(r'chương\s*\d+', text_lower):
        textbook_signals.append('chương_structure')
    if 'cau hoi' in text_lower or 'bai tap' in text_lower or 'câu hỏi' in text_lower or 'bài tập' in text_lower:
        textbook_signals.append('exercises')
    if 'nha xuat ban' in text_lower or 'nhà xuất bản' in text_lower:
        textbook_signals.append('publisher')
    if 'chu bien' in text_lower or 'chủ biên' in text_lower or 'bien soan' in text_lower:
        textbook_signals.append('editor')
    if 'tai lieu tham khao' in text_lower or 'tài liệu tham khảo' in text_lower:
        textbook_signals.append('references')

    # === Lecture notes ===
    lecture_signals = []
    if 'tap bai giang' in text_lower or 'tập bài giảng' in fn_lower or 'tbg' in fn_lower:
        lecture_signals.append('lecture_collection')
    if 'bai giang' in text_lower and 'giao trinh' not in text_lower[:3000]:
        lecture_signals.append('lecture_keyword')

    # === Legal document signals ===
    legal_signals = []
    # Law number pattern: "Luật số 12/2023/QH15"
    if re.search(r'luat\s*so\s*\d+', text_lower) or re.search(r'luật\s*số\s*\d+', text_lower):
        legal_signals.append('law_number')
    # Dense article numbering
    article_matches = re.findall(r'dieu\s*\d+', text_lower) + re.findall(r'điều\s*\d+', text_lower)
    article_count = len(article_matches)
    if article_count >= 8:
        legal_signals.append(f'article_dense_{article_count}')
    # Khoản pattern
    khoan_count = len(re.findall(r'khoan\s*\d+', text_lower)) + len(re.findall(r'khoản\s*\d+', text_lower))
    if khoan_count >= 5:
        legal_signals.append(f'khoan_dense_{khoan_count}')
    # Issuer
    if 'quoc hoi' in text_lower or 'quốc hội' in text_lower:
        legal_signals.append('quốc_hội')
    if 'nghi dinh' in text_lower or 'nghị định' in text_lower:
        legal_signals.append('decree')
    if 'thong tu' in text_lower or 'thông tư' in text_lower:
        legal_signals.append('circular')

    # === Reference / monograph ===
    ref_signals = []
    if 'chuyen khao' in text_lower or 'chuyên khảo' in fn_lower:
        ref_signals.append('monograph')
    if 'sach tham khao' in text_lower or 'sách tham khảo' in fn_lower:
        ref_signals.append('reference_book')

    # === Scoring ===
    textbook_score = len(textbook_signals)
    lecture_score = len(lecture_signals)
    legal_score = len(legal_signals)
    ref_score = len(ref_signals)

    # Hybrid: strong textbook + heavy legal citations
    is_hybrid = textbook_score >= 2 and legal_score >= 2 and article_count >= 15

    if is_hybrid:
        doc_type = "textbook_with_legal_appendix"
        confidence = min(85, (textbook_score + legal_score) * 8)
    elif textbook_score >= 2:
        doc_type = "textbook"
        confidence = min(90, textbook_score * 15)
    elif lecture_score >= 1 and lecture_score >= textbook_score:
        doc_type = "lecture_notes"
        confidence = min(80, lecture_score * 25)
    elif legal_score >= 3:
        doc_type = "legal_document"
        confidence = min(95, legal_score * 12)
    elif ref_score >= 1:
        doc_type = "reference"
        confidence = min(75, ref_score * 30)
    elif textbook_score >= 1:
        doc_type = "textbook_weak"
        confidence = textbook_score * 20
    elif legal_score >= 1:
        doc_type = "legal_document_weak"
        confidence = legal_score * 15
    else:
        doc_type = "unknown"
        confidence = 0

    return {
        "type": doc_type,
        "confidence": confidence,
        "textbook_signals": textbook_signals,
        "lecture_signals": lecture_signals,
        "legal_signals": legal_signals,
        "reference_signals": ref_signals,
        "article_count": article_count,
        "text_sample_length": len(text),
    }


def assess_quality(classification: Dict, filename: str) -> Dict:
    """Assess quality and give recommendation."""
    doc_type = classification["type"]
    conf = classification["confidence"]
    
    if doc_type == "textbook" and conf >= 70:
        return {"score": 90, "recommendation": "primary_use"}
    elif doc_type == "textbook_with_legal_appendix" and conf >= 60:
        return {"score": 80, "recommendation": "primary_use"}
    elif doc_type in ("textbook", "textbook_weak") and conf >= 40:
        return {"score": 65, "recommendation": "secondary_use"}
    elif doc_type == "lecture_notes":
        return {"score": 60, "recommendation": "secondary_use"}
    elif doc_type == "reference":
        return {"score": 55, "recommendation": "supplementary"}
    elif doc_type in ("legal_document", "legal_document_weak"):
        return {"score": 30, "recommendation": "raw_law_text_caution"}
    else:
        return {"score": 0, "recommendation": "manual_review"}


def audit_pdf(pdf_path: Path, base_dir: Path) -> Dict:
    """Audit one PDF."""
    rel_path = pdf_path.relative_to(base_dir)
    size_mb = round(pdf_path.stat().st_size / 1024 / 1024, 2)

    text = extract_text_from_raw_bytes(pdf_path)
    
    classification = classify_content(text, pdf_path.name)
    quality = assess_quality(classification, pdf_path.name)

    return {
        "filename": pdf_path.name,
        "relative_path": str(rel_path),
        "size_mb": size_mb,
        "content_classification": classification,
        "quality_score": quality["score"],
        "recommendation": quality["recommendation"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("research/results/pdf_content_audit.json"))
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Find PDFs
    if args.target:
        pdfs = []
        for path in args.data_dir.rglob("*"):
            if path.is_dir() and args.target in path.name:
                pdfs.extend(sorted(path.glob("*.pdf")))
        if not pdfs:
            pdfs = list(args.data_dir.rglob(f"*{args.target}*/*.pdf"))
    else:
        pdfs = sorted(args.data_dir.rglob("*.pdf"))

    if args.limit:
        pdfs = pdfs[:args.limit]

    logger.info(f"Auditing {len(pdfs)} PDFs")

    records = []
    for i, pdf in enumerate(pdfs, 1):
        logger.info(f"[{i}/{len(pdfs)}] {pdf.name[:55]}")
        record = audit_pdf(pdf, args.data_dir)
        records.append(record)
        cls = record["content_classification"]
        logger.info(f"   → {cls['type']} (conf={cls['confidence']}, qual={record['quality_score']}) [{record['recommendation']}]")

    # Summary
    by_type = {}
    by_rec = {}
    for r in records:
        t = r["content_classification"]["type"]
        by_type[t] = by_type.get(t, 0) + 1
        by_rec[r["recommendation"]] = by_rec.get(r["recommendation"], 0) + 1

    summary = {
        "total_pdfs": len(records),
        "by_content_type": by_type,
        "by_recommendation": by_rec,
        "records": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\nReport: {args.output}")

    print(f"\n{'='*70}")
    print("CONTENT-BASED AUDIT SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {len(records)}")
    print(f"\nBy content type:")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {c:3} | {t}")
    print(f"\nBy recommendation:")
    for r, c in sorted(by_rec.items(), key=lambda x: -x[1]):
        print(f"  {c:3} | {r}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    sys.exit(main())
