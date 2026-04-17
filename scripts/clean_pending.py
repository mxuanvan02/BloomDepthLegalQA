"""
Post-processing: clean metadata/header entries from pending.json or final QA dataset.

Usage (on Colab after Phase A completes):
    python3 scripts/clean_pending.py --input path/to/pending.json --output path/to/cleaned.json
    python3 scripts/clean_pending.py --input path/to/final_dataset.jsonl --output path/to/final_dataset_clean.jsonl

Applies the same _is_metadata_chunk heuristics as prepare_data.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("bloom_depth.clean_pending")

METADATA_KEYWORDS = [
    "NHÀ XUẤT BẢN", "Chủ biên", "GIÁO TRÌNH", "HỌC VIỆN",
    "TRƯỜNG ĐẠI HỌC", "TÁC GIẢ", "BAN BIÊN SOẠN",
    "MỤC LỤC", "TABLE OF CONTENTS", "ISBN", "Lưu hành nội bộ",
]


def _is_metadata_chunk(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    # Heuristic 1: header-dominated (>60% lines start with #)
    header_lines = sum(1 for l in lines if l.startswith("#"))
    if header_lines / len(lines) > 0.6:
        return True
    # Heuristic 2: ≥2 metadata keywords
    text_upper = text.upper()
    hits = sum(1 for kw in METADATA_KEYWORDS if kw.upper() in text_upper)
    if hits >= 2:
        return True
    # Heuristic 3: <80 chars of non-header content
    non_header = " ".join(l for l in lines if not l.startswith("#"))
    if len(non_header) < 80:
        return True
    return False


def _is_bad_ocr_chunk(text: str) -> bool:
    tokens = text.split()
    if not tokens: return True
        
    bad_tokens = 0
    for t in tokens:
        t = t.strip('.,;:"()[]{}<>!?\'-')
        if not t: continue
        if '/' in t or '-' in t or '_' in t or t.isupper(): continue
            
        if len(t) >= 3 and any(c.islower() for c in t) and any(c.isdigit() for c in t):
            bad_tokens += 1
        elif any(c in t for c in ';:!<>*&^%$#@~=+|\\') and any(c.isalpha() for c in t):
            bad_tokens += 1

    if (bad_tokens / len(tokens)) > 0.05:
        return True
    return False


def load_records(path: Path) -> list[dict]:
    """Load both .json (list) and .jsonl (one record per line) formats."""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def save_records(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    else:  # .jsonl
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), path)


def main():
    parser = argparse.ArgumentParser(description="Clean metadata and OCR chunks from QA dataset")
    parser.add_argument("--input",  type=Path, required=True,  help="Input file (.json or .jsonl)")
    parser.add_argument("--output", type=Path, required=True,  help="Output file (.json or .jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="Report stats only, don't write")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    records = load_records(args.input)
    logger.info("Loaded %d records from %s", len(records), args.input)

    sample_keys = set(records[0].keys()) if records else set()
    ctx_field = "context_text" if "context_text" in sample_keys else "text"
    logger.info("Using context field: '%s'", ctx_field)

    kept, dropped_meta, dropped_ocr, dropped_no_qa = [], [], [], []
    for r in records:
        ctx = r.get(ctx_field, "")
        if _is_metadata_chunk(ctx):
            dropped_meta.append(r)
        elif _is_bad_ocr_chunk(ctx):
            dropped_ocr.append(r)
        elif r.get("qa") is None:
            dropped_no_qa.append(r)
        else:
            kept.append(r)

    logger.info(
        "Results: %d kept | %d dropped (metadata) | %d dropped (ocr) | %d dropped (no-qa)",
        len(kept), len(dropped_meta), len(dropped_ocr), len(dropped_no_qa),
    )

    if dropped_ocr:
        logger.info("Sample dropped (OCR) chunks:")
        for r in dropped_ocr[:3]:
            preview = r.get(ctx_field, "")[:120].replace("\n", " ")
            logger.info("  [OCR] %s...", preview)

    if not args.dry_run:
        save_records(kept, args.output)
        logger.info("Done. Reduction: %.1f%%", (1 - len(kept) / len(records)) * 100)
    else:
        logger.info("Dry-run mode — nothing written.")


if __name__ == "__main__":
    main()
