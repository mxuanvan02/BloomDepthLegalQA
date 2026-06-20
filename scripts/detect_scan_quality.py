#!/usr/bin/env python3
"""
Detect scan-quality / OCR-need PDFs by structural analysis of raw bytes.

No pymupdf/poppler needed. Heuristics:
- Count /Image XObjects vs /Font objects
- Detect text operators (Tj, TJ, BT/ET) presence
- Detect /Type/Font and font encoding
- Estimate text-layer presence
- Compute size-per-page-estimate

Classification:
- digital_text: has fonts + text operators (good for pymupdf extraction)
- scanned_no_ocr: many images, no/few fonts, no text operators (NEEDS OCR)
- scanned_with_ocr: images + text layer present (OCR already applied)
- hybrid: mix
- encrypted: encrypted streams

Usage:
    python scripts/detect_scan_quality.py [--data-dir data/raw]
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)


def analyze_pdf_structure(pdf_path: Path, scan_bytes: int = 3_000_000) -> Dict:
    """
    Analyze PDF structure from raw bytes.
    Reads up to scan_bytes (default 3MB) — enough to sample structure
    of even large PDFs (image streams are huge but markers appear throughout).
    """
    try:
        size = pdf_path.stat().st_size
        with open(pdf_path, 'rb') as f:
            # Read beginning chunk
            head = f.read(min(scan_bytes, size))
            # Also read tail for trailer/xref (where object types are summarized)
            tail = b""
            if size > scan_bytes:
                f.seek(max(0, size - 500_000))
                tail = f.read(500_000)
        
        raw = head + tail
        text = raw.decode('latin-1', errors='ignore')

        # === Count structural markers ===
        # Image XObjects
        image_count = len(re.findall(r'/Subtype\s*/Image', text))
        # Font objects
        font_count = len(re.findall(r'/Type\s*/Font', text))
        font_subtype = len(re.findall(r'/Subtype\s*/(TrueType|Type0|Type1|CIDFontType)', text))
        # Text show operators (Tj, TJ) — indicate text layer
        tj_count = len(re.findall(r'\bTj\b', text)) + len(re.findall(r'\bTJ\b', text))
        # BT/ET text blocks
        bt_count = len(re.findall(r'\bBT\b', text))
        # ToUnicode CMap (indicates extractable text)
        tounicode = len(re.findall(r'/ToUnicode', text))
        # Encryption
        encrypted = '/Encrypt' in text
        # Page count estimate
        page_markers = len(re.findall(r'/Type\s*/Page\b', text))

        # Estimate total pages (from /Count in pages tree, or page markers)
        count_match = re.search(r'/Count\s+(\d+)', text)
        estimated_pages = int(count_match.group(1)) if count_match else max(page_markers, 1)

        size_mb = size / 1024 / 1024
        size_per_page = size_mb / max(estimated_pages, 1)

        # === Classification logic ===
        has_text_layer = (tj_count > 5 or bt_count > 5) and (font_count > 0 or font_subtype > 0)
        has_tounicode = tounicode > 0
        image_heavy = image_count >= max(2, estimated_pages * 0.3)

        if encrypted:
            doc_class = "encrypted"
            ocr_need = "unknown_encrypted"
        elif has_text_layer and has_tounicode:
            if image_heavy:
                doc_class = "scanned_with_ocr"
                ocr_need = "ocr_present_good"
            else:
                doc_class = "digital_text"
                ocr_need = "no_ocr_needed"
        elif has_text_layer and not has_tounicode:
            # Has fonts/text ops but no ToUnicode — text may extract garbled (CID without mapping)
            doc_class = "digital_text_risky"
            ocr_need = "extraction_may_garble"
        elif image_heavy and not has_text_layer:
            doc_class = "scanned_no_ocr"
            ocr_need = "ocr_required"
        elif font_count == 0 and font_subtype == 0 and tj_count == 0:
            doc_class = "scanned_no_ocr"
            ocr_need = "ocr_required"
        else:
            doc_class = "uncertain"
            ocr_need = "manual_check"

        return {
            "size_mb": round(size_mb, 2),
            "estimated_pages": estimated_pages,
            "size_per_page_mb": round(size_per_page, 3),
            "image_xobjects": image_count,
            "font_objects": font_count,
            "font_subtypes": font_subtype,
            "text_show_ops": tj_count,
            "bt_blocks": bt_count,
            "tounicode_maps": tounicode,
            "encrypted": encrypted,
            "has_text_layer": has_text_layer,
            "has_tounicode": has_tounicode,
            "image_heavy": image_heavy,
            "doc_class": doc_class,
            "ocr_need": ocr_need,
        }
    except Exception as e:
        logger.warning(f"Error analyzing {pdf_path.name}: {e}")
        return {"doc_class": "error", "ocr_need": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("research/results/scan_quality_audit.json"))
    parser.add_argument("--exclude-dir", type=str, default="_excluded")
    args = parser.parse_args()

    pdfs = sorted(args.data_dir.rglob("*.pdf"))
    # Exclude _excluded folder
    pdfs = [p for p in pdfs if args.exclude_dir not in p.parts]

    logger.info(f"Analyzing {len(pdfs)} PDFs for scan quality")

    records = []
    for i, pdf in enumerate(pdfs, 1):
        rel = pdf.relative_to(args.data_dir)
        source = rel.parts[0] if len(rel.parts) > 1 else "root"
        result = analyze_pdf_structure(pdf)
        result["filename"] = pdf.name
        result["relative_path"] = str(rel)
        result["source"] = source
        records.append(result)
        if i % 20 == 0:
            logger.info(f"  ...{i}/{len(pdfs)}")

    # Summary
    by_class = {}
    by_ocr = {}
    for r in records:
        by_class[r["doc_class"]] = by_class.get(r["doc_class"], 0) + 1
        by_ocr[r["ocr_need"]] = by_ocr.get(r["ocr_need"], 0) + 1

    # Flag problematic
    needs_ocr = [r for r in records if r["ocr_need"] == "ocr_required"]
    risky = [r for r in records if r["ocr_need"] in ("extraction_may_garble", "manual_check")]
    encrypted = [r for r in records if r["doc_class"] == "encrypted"]

    summary = {
        "total": len(records),
        "by_doc_class": by_class,
        "by_ocr_need": by_ocr,
        "needs_ocr_count": len(needs_ocr),
        "risky_count": len(risky),
        "encrypted_count": len(encrypted),
        "records": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("SCAN QUALITY / OCR-NEED AUDIT")
    print(f"{'='*70}")
    print(f"Total: {len(records)} PDFs\n")
    print("By document class:")
    for c, n in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {n:4} | {c}")
    print("\nBy OCR need:")
    for c, n in sorted(by_ocr.items(), key=lambda x: -x[1]):
        print(f"  {n:4} | {c}")

    if needs_ocr:
        print(f"\n⚠️  SCAN PDFs NEEDING OCR ({len(needs_ocr)}):")
        for r in sorted(needs_ocr, key=lambda x: -x.get("size_mb", 0)):
            print(f"  [{r['source'][:10]:10}] {r['size_mb']:5.1f}MB ~{r['estimated_pages']}p | {r['filename'][:50]}")

    if risky:
        print(f"\n⚠️  RISKY EXTRACTION ({len(risky)}):")
        for r in risky:
            print(f"  [{r['source'][:10]:10}] {r['doc_class']:20} | {r['filename'][:50]}")

    if encrypted:
        print(f"\n🔒 ENCRYPTED ({len(encrypted)}):")
        for r in encrypted:
            print(f"  {r['filename'][:60]}")

    print(f"\n{'='*70}")
    print(f"Report: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    sys.exit(main())
