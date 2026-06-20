#!/usr/bin/env python3
"""
Content-based PDF audit: distinguish textbooks (giáo trình) from legal documents (văn bản luật).

Workflow:
1. Extract first 5 pages of each PDF using pdf_reader
2. Apply content-based classifier (textbook vs legal_doc vs hybrid)
3. Score textbook quality (structure: chapters, exercises, examples)
4. Output detailed audit JSON + summary report

Usage:
    python scripts/audit_pdf_content.py [--limit N] [--target gap_2026]
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)

PDF_READER = Path("/home/node/.openclaw/workspace/tools/pdf_reader/parse_pdf.py")


def extract_text_first_pages(pdf_path: Path, num_pages: int = 5) -> Optional[str]:
    """Extract first N pages of PDF as plain text via pdf_reader."""
    try:
        result = subprocess.run(
            [
                "python3", str(PDF_READER),
                str(pdf_path),
                "--backend", "pymupdf",
                "--format", "text",
                "--pages", f"1-{num_pages}",
                "--no-cache",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.debug(f"pdf_reader failed for {pdf_path.name}: {result.stderr[:200]}")
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout extracting {pdf_path.name}")
        return None
    except Exception as e:
        logger.warning(f"Error extracting {pdf_path.name}: {e}")
        return None


def classify_content(text: str) -> Dict:
    """
    Classify based on content signals:

    - textbook: giáo trình, chương N, mục lục, câu hỏi ôn tập, bài tập, ví dụ
    - lecture_notes: bài giảng, slide, tập bài giảng
    - legal_document: luật số .../QH, điều N, khoản N, nghị định, thông tư
    - reference: sách chuyên khảo, monograph
    - hybrid: contains both textbook and legal-doc signals
    """
    if not text:
        return {"type": "unknown", "confidence": 0, "signals": []}

    text_lower = text.lower()
    sample = text_lower[:8000]  # Front-matter is most informative

    signals = []

    # === Textbook signals ===
    textbook_signals = []
    if re.search(r'\bgiáo\s+trình\b', sample) or 'giao trinh' in sample:
        textbook_signals.append('giáo_trình_keyword')
    if re.search(r'\bmục\s+lục\b', sample) or 'muc luc' in sample:
        textbook_signals.append('mục_lục')
    if re.search(r'chương\s+\d+', sample, re.IGNORECASE) or re.search(r'chuong\s+\d+', sample):
        textbook_signals.append('chương_cấu_trúc')
    if re.search(r'(câu\s+hỏi\s+ôn\s+tập|bài\s+tập\s+chương|cau hoi on tap)', sample):
        textbook_signals.append('bài_tập_câu_hỏi')
    if re.search(r'(tài\s+liệu\s+tham\s+khảo|tai lieu tham khao)', sample):
        textbook_signals.append('tài_liệu_tham_khảo')
    if re.search(r'(ví\s+dụ\s+\d+|vi du \d+)', sample):
        textbook_signals.append('ví_dụ_đánh_số')
    if re.search(r'(nhà\s+xuất\s+bản|nha xuat ban)', sample):
        textbook_signals.append('nxb_publisher')
    if re.search(r'(chủ\s+biên|chu bien|biên\s+soạn)', sample):
        textbook_signals.append('chủ_biên')

    # === Lecture notes signals ===
    lecture_signals = []
    if re.search(r'\b(tập\s+bài\s+giảng|tap bai giang|tbg)\b', sample):
        lecture_signals.append('tập_bài_giảng')
    if re.search(r'\b(bài\s+giảng|bai giang)\b', sample) and 'giáo trình' not in sample[:2000]:
        lecture_signals.append('bài_giảng_keyword')
    if re.search(r'(slide|powerpoint)', sample):
        lecture_signals.append('slide_format')

    # === Legal document signals ===
    legal_signals = []
    # Vietnamese law citation: "Luật số 12/2023/QH15"
    if re.search(r'luật\s+số\s+\d+\s*[/-]\s*\d{4}', sample) or re.search(r'luat so \d+', sample):
        legal_signals.append('luật_số_citation')
    # Article numbering pattern dense (Điều 1, Điều 2...)
    article_count = len(re.findall(r'điều\s+\d+\b', sample, re.IGNORECASE))
    if article_count >= 5:
        legal_signals.append(f'điều_dense_{article_count}')
    # Khoản (paragraph) pattern
    if len(re.findall(r'khoản\s+\d+', sample, re.IGNORECASE)) >= 3:
        legal_signals.append('khoản_dense')
    # Chính phủ / Quốc hội issuer
    if re.search(r'(quốc\s+hội|quoc hoi)\b', sample) and 'điều' in sample:
        legal_signals.append('quốc_hội_issuer')
    if re.search(r'(nghị\s+định|nghi dinh)', sample):
        legal_signals.append('nghị_định')
    if re.search(r'(thông\s+tư|thong tu)\s+\d+', sample):
        legal_signals.append('thông_tư_numbered')

    # === Reference book signals ===
    reference_signals = []
    if re.search(r'(sách\s+chuyên\s+khảo|sach chuyen khao|monograph)', sample):
        reference_signals.append('sách_chuyên_khảo')
    if re.search(r'(sách\s+tham\s+khảo|sach tham khao)', sample):
        reference_signals.append('sách_tham_khảo')

    # Decision logic
    textbook_score = len(textbook_signals)
    lecture_score = len(lecture_signals)
    legal_score = len(legal_signals)
    reference_score = len(reference_signals)

    # Hybrid case: both substantial textbook + heavy legal citations
    is_hybrid = textbook_score >= 2 and legal_score >= 2 and article_count >= 10

    if is_hybrid:
        doc_type = "textbook_with_legal_text"
        confidence = min(90, (textbook_score + legal_score) * 10)
    elif textbook_score >= 2 and textbook_score > legal_score:
        doc_type = "textbook"
        confidence = min(95, textbook_score * 15)
    elif lecture_score >= 1 and lecture_score >= textbook_score:
        doc_type = "lecture_notes"
        confidence = min(85, lecture_score * 25)
    elif legal_score >= 2 and legal_score > textbook_score:
        doc_type = "legal_document"
        confidence = min(95, legal_score * 15)
    elif reference_score >= 1:
        doc_type = "reference"
        confidence = min(80, reference_score * 30)
    elif textbook_score >= 1:
        doc_type = "textbook_weak"
        confidence = textbook_score * 20
    elif legal_score >= 1:
        doc_type = "legal_document_weak"
        confidence = legal_score * 20
    else:
        doc_type = "unknown"
        confidence = 0

    return {
        "type": doc_type,
        "confidence": confidence,
        "textbook_signals": textbook_signals,
        "lecture_signals": lecture_signals,
        "legal_signals": legal_signals,
        "reference_signals": reference_signals,
        "article_count_first_pages": article_count,
        "text_length": len(text),
    }


def assess_textbook_quality(text: str) -> int:
    """Score textbook structural quality (0-100)."""
    if not text:
        return 0

    text_lower = text.lower()
    score = 0

    # Has table of contents
    if re.search(r'mục\s+lục', text_lower):
        score += 20

    # Multiple chapters
    chapter_count = len(set(re.findall(r'chương\s+(\d+)', text_lower)))
    score += min(30, chapter_count * 5)

    # Has authors/editors
    if re.search(r'(chủ biên|chu bien|biên soạn|tác giả)', text_lower):
        score += 10

    # Has publisher
    if re.search(r'(nhà xuất bản|nxb)', text_lower):
        score += 10

    # Has references section
    if re.search(r'(tài liệu tham khảo|references)', text_lower):
        score += 10

    # Has exercises/questions
    if re.search(r'(câu hỏi ôn tập|bài tập|exercise)', text_lower):
        score += 15

    # Has examples
    if re.search(r'(ví dụ \d+|example \d+)', text_lower):
        score += 5

    return min(100, score)


def audit_pdf(pdf_path: Path, base_dir: Path) -> Dict:
    """Run full content audit on one PDF."""
    rel_path = pdf_path.relative_to(base_dir)
    size_mb = pdf_path.stat().st_size / 1024 / 1024

    record = {
        "filename": pdf_path.name,
        "relative_path": str(rel_path),
        "size_mb": round(size_mb, 2),
        "extracted": False,
        "content_classification": None,
        "quality_score": 0,
        "recommendation": "skip_extraction_failed",
    }

    text = extract_text_first_pages(pdf_path)
    if not text:
        return record

    record["extracted"] = True
    record["text_sample_length"] = len(text)

    classification = classify_content(text)
    record["content_classification"] = classification

    quality = assess_textbook_quality(text)
    record["quality_score"] = quality

    # Recommendation
    doc_type = classification["type"]
    if doc_type in ("textbook", "textbook_with_legal_text") and quality >= 50:
        record["recommendation"] = "primary_use"
    elif doc_type == "lecture_notes" and quality >= 30:
        record["recommendation"] = "secondary_use"
    elif doc_type in ("textbook", "textbook_weak"):
        record["recommendation"] = "secondary_use"
    elif doc_type == "reference":
        record["recommendation"] = "supplementary"
    elif doc_type in ("legal_document", "legal_document_weak"):
        record["recommendation"] = "raw_law_text_caution"
    else:
        record["recommendation"] = "manual_review"

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("research/results/pdf_content_audit.json"))
    parser.add_argument("--target", type=str, default=None,
                        help="Subdirectory to limit audit to (e.g. 'gap_2026')")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Find PDFs
    if args.target:
        pdfs = list(args.data_dir.rglob(f"*{args.target}*/**/*.pdf"))
        if not pdfs:
            # Try direct subfolder match
            for path in args.data_dir.rglob("*"):
                if path.is_dir() and args.target in path.name:
                    pdfs.extend(path.glob("*.pdf"))
    else:
        pdfs = sorted(args.data_dir.rglob("*.pdf"))

    if args.limit:
        pdfs = pdfs[:args.limit]

    logger.info(f"Auditing {len(pdfs)} PDFs from {args.data_dir}")

    records = []
    for i, pdf in enumerate(pdfs, 1):
        logger.info(f"[{i}/{len(pdfs)}] {pdf.name[:60]}")
        record = audit_pdf(pdf, args.data_dir)
        records.append(record)

        if record["extracted"]:
            cls = record["content_classification"]
            logger.info(f"   → {cls['type']} (conf={cls['confidence']}, qual={record['quality_score']}) [{record['recommendation']}]")

    # Build summary
    by_type = {}
    by_recommendation = {}
    for r in records:
        cls = r.get("content_classification") or {}
        t = cls.get("type", "extraction_failed")
        by_type[t] = by_type.get(t, 0) + 1
        by_recommendation[r["recommendation"]] = by_recommendation.get(r["recommendation"], 0) + 1

    summary = {
        "total_pdfs": len(records),
        "extracted_successfully": sum(1 for r in records if r["extracted"]),
        "by_content_type": by_type,
        "by_recommendation": by_recommendation,
        "records": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\nReport written: {args.output}")

    print(f"\n{'='*70}")
    print(f"CONTENT-BASED AUDIT SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {summary['total_pdfs']}, extracted: {summary['extracted_successfully']}")
    print(f"\nBy content type:")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {count:4} | {t}")
    print(f"\nBy recommendation:")
    for r, count in sorted(by_recommendation.items(), key=lambda x: -x[1]):
        print(f"  {count:4} | {r}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
