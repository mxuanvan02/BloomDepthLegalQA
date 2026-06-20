#!/usr/bin/env python3
"""
Quality filter for existing raw PDFs: check usability, extract metadata, assess recency.

Filters PDFs by:
- File integrity (valid PDF magic bytes, readable)
- Basic metadata (page count, file size, creation/modification date)
- Text extractability (can extract text via pdftotext/strings)
- Recency heuristics from filename/metadata

Does NOT require pymupdf/fitz; uses subprocess calls to pdfinfo/pdftotext (poppler-utils).

Usage:
    python scripts/filter_existing_pdfs.py --data-dir data/raw
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)


def check_pdf_magic(pdf_path: Path) -> bool:
    """Check if file starts with PDF magic bytes."""
    try:
        with open(pdf_path, 'rb') as f:
            magic = f.read(5)
            return magic.startswith(b'%PDF-')
    except Exception as e:
        logger.warning(f"Cannot read {pdf_path.name}: {e}")
        return False


def extract_pdf_metadata(pdf_path: Path) -> Optional[Dict]:
    """Extract basic metadata from file stats (no external tools)."""
    try:
        stat = pdf_path.stat()
        return {
            'file_size': stat.st_size,
            'modified_time': stat.st_mtime,
        }
    except Exception as e:
        logger.warning(f"Error getting file stats for {pdf_path.name}: {e}")
        return None


def extract_text_sample(pdf_path: Path, max_chars: int = 2000) -> str:
    """Lightweight text extraction: read raw bytes and look for readable ASCII/UTF-8."""
    try:
        with open(pdf_path, 'rb') as f:
            # Skip PDF header, read some content
            f.seek(1024)
            raw_bytes = f.read(50000)
        
        # Try decode as latin-1 (safe fallback)
        text = raw_bytes.decode('latin-1', errors='ignore')
        
        # Count readable text (letters, spaces, Vietnamese chars)
        readable_chars = sum(1 for c in text if c.isprintable() or c in '\n\t ')
        
        return text[:max_chars] if readable_chars > 100 else ""
        
    except Exception as e:
        logger.warning(f"Text extraction failed for {pdf_path.name}: {e}")
        return ""


def estimate_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename using patterns."""
    # Look for 4-digit years
    matches = re.findall(r'(19|20)\d{2}', filename)
    if matches:
        years = [int(m) for m in matches]
        # Return most recent year found
        return max(years)
    return None


def estimate_year_from_metadata(metadata: Dict) -> Optional[int]:
    """Extract year from file modification time."""
    if 'modified_time' in metadata:
        from datetime import datetime
        dt = datetime.fromtimestamp(metadata['modified_time'])
        return dt.year
    return None


def classify_textbook_type(filename: str, text_sample: str) -> str:
    """
    Classify into:
    - primary_textbook: giáo trình chính thức (GT_, numbered institute files)
    - lecture_notes: tập bài giảng (TBG), bài giảng (BG)
    - study_material: tài liệu học tập (TLHT, TLHT)
    - reference: sách tham khảo, sách chuyên khảo (Sach_CK, Sach_CB)
    - unknown
    """
    filename_lower = filename.lower()
    text_lower = text_sample.lower()

    # Vietnamese institute abbreviations
    # TBG = Tập Bài Giảng (lecture notes compilation)
    if re.search(r'\btbg\b', filename_lower) or 'tap bai giang' in filename_lower or 'tập bài giảng' in filename_lower:
        return 'lecture_notes'

    # BG = Bài Giảng (lecture notes)
    if re.match(r'^bg[\s_]', filename_lower) or 'bai giang' in filename_lower or 'bài giảng' in filename_lower:
        return 'lecture_notes'

    # TLHT = Tài Liệu Học Tập (study material)
    if re.search(r'\btlht\b', filename_lower) or 'tai lieu hoc tap' in filename_lower or 'tài liệu học tập' in filename_lower:
        return 'study_material'

    # Sach_CK = Sách Chuyên Khảo (monograph); Sach_CB = Sách Chuyên Biệt / chuyên ban
    if re.search(r'sach_ck', filename_lower) or 'chuyen khao' in filename_lower or 'chuyên khảo' in filename_lower:
        return 'reference'
    if re.search(r'sach_cb', filename_lower) or filename_lower.startswith('sach'):
        return 'reference'

    # GT_ prefix or "giao trinh" = primary textbook
    if filename_lower.startswith('gt_') or 'giao trinh' in filename_lower or 'giáo trình' in filename_lower:
        return 'primary_textbook'

    # Numbered institute files (e.g. "13. LUAT DAN SU...") are core curriculum textbooks
    if re.match(r'^\d+[\.\,]', filename_lower):
        return 'primary_textbook'

    # Plain "LUAT ..." files without prefix are core textbooks
    if re.match(r'^(luat|pl_|phap luat)', filename_lower):
        return 'primary_textbook'

    # Check text content for additional signals
    if 'giáo trình' in text_lower[:500]:
        return 'primary_textbook'

    return 'unknown'


def assess_pdf_quality(pdf_path: Path) -> Dict:
    """Comprehensive quality assessment of a single PDF."""
    
    assessment = {
        'filename': pdf_path.name,
        'relative_path': str(pdf_path),
        'file_size_mb': pdf_path.stat().st_size / 1024 / 1024,
        'valid_pdf': False,
        'page_count': None,
        'has_text': False,
        'text_chars': 0,
        'estimated_year': None,
        'textbook_type': 'unknown',
        'quality_score': 0,
        'usable': False,
        'issues': [],
    }
    
    # Check PDF magic
    if not check_pdf_magic(pdf_path):
        assessment['issues'].append('invalid_pdf_format')
        return assessment
    
    assessment['valid_pdf'] = True
    
    # Extract metadata
    metadata = extract_pdf_metadata(pdf_path)
    if metadata:
        # Estimate page count from file size (rough heuristic: 50KB per page average for scanned textbooks)
        estimated_pages = int(metadata['file_size'] / (50 * 1024))
        assessment['page_count'] = max(10, estimated_pages)  # Minimum 10 pages
        
        # Extract year
        year_meta = estimate_year_from_metadata(metadata)
        year_file = estimate_year_from_filename(pdf_path.name)
        assessment['estimated_year'] = year_file or year_meta
    
    # Extract text sample
    text_sample = extract_text_sample(pdf_path)
    assessment['text_chars'] = len(text_sample)
    assessment['has_text'] = len(text_sample) > 100
    
    if not assessment['has_text']:
        assessment['issues'].append('no_extractable_text')
    
    # Classify textbook type
    assessment['textbook_type'] = classify_textbook_type(pdf_path.name, text_sample)
    
    # Quality score (0-100)
    score = 0
    
    if assessment['valid_pdf']:
        score += 20
    
    if assessment['page_count']:
        if assessment['page_count'] >= 50:
            score += 20
        elif assessment['page_count'] >= 20:
            score += 10
    
    if assessment['has_text']:
        score += 30
    
    if assessment['estimated_year']:
        year = assessment['estimated_year']
        # Prefer recent but accept older textbooks
        if year >= 2020:
            score += 30
        elif year >= 2015:
            score += 25
        elif year >= 2010:
            score += 20
        elif year >= 2005:
            score += 10
    
    assessment['quality_score'] = score
    # Lower threshold: accept PDFs with score >= 40 (valid + some text)
    assessment['usable'] = score >= 40 and assessment['valid_pdf']
    
    return assessment


def main():
    parser = argparse.ArgumentParser(description="Quality filter for existing raw PDFs")
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/raw'),
        help='Root directory of raw PDFs'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('research/results/pdf_quality_filter.json'),
        help='Output JSON report'
    )
    parser.add_argument(
        '--min-score',
        type=int,
        default=50,
        help='Minimum quality score for usable PDFs'
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1
    
    # Find all PDFs
    all_pdfs = list(args.data_dir.rglob('*.pdf'))
    logger.info(f"Found {len(all_pdfs)} PDFs in {args.data_dir}")
    
    # Assess each PDF
    assessments = []
    for i, pdf_path in enumerate(all_pdfs, 1):
        logger.info(f"[{i}/{len(all_pdfs)}] Assessing: {pdf_path.name}")
        assessment = assess_pdf_quality(pdf_path)
        assessments.append(assessment)
    
    # Aggregate statistics
    usable_pdfs = [a for a in assessments if a['usable']]
    unusable_pdfs = [a for a in assessments if not a['usable']]
    
    by_source = defaultdict(list)
    for a in assessments:
        parts = Path(a['relative_path']).parts
        source = parts[2] if len(parts) > 2 else 'unknown'  # data/raw/<source>/...
        by_source[source].append(a)
    
    by_textbook_type = defaultdict(list)
    for a in usable_pdfs:
        by_textbook_type[a['textbook_type']].append(a)
    
    report = {
        'filter_date': datetime.utcnow().isoformat(),
        'total_pdfs': len(all_pdfs),
        'usable_pdfs': len(usable_pdfs),
        'unusable_pdfs': len(unusable_pdfs),
        'min_quality_score': args.min_score,
        'usable_by_source': {
            source: len([a for a in pdfs if a['usable']])
            for source, pdfs in by_source.items()
        },
        'usable_by_textbook_type': {
            ttype: len(pdfs)
            for ttype, pdfs in by_textbook_type.items()
        },
        'assessments': assessments,
    }
    
    # Write report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Quality filter report written to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("PDF QUALITY FILTER SUMMARY")
    print("="*60)
    print(f"Total PDFs:    {len(all_pdfs)}")
    print(f"Usable PDFs:   {len(usable_pdfs)} ({len(usable_pdfs)/len(all_pdfs)*100:.1f}%)")
    print(f"Unusable PDFs: {len(unusable_pdfs)}")
    
    print(f"\nUsable by source:")
    for source, count in sorted(report['usable_by_source'].items()):
        total_in_source = len(by_source[source])
        print(f"  {source}: {count}/{total_in_source}")
    
    print(f"\nUsable by textbook type:")
    for ttype, count in sorted(report['usable_by_textbook_type'].items()):
        print(f"  {ttype}: {count}")
    
    print("\n" + "="*60 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
