"""
Corpus extraction script — runs Docling pipeline on all raw PDFs.

Usage:
    python scripts/extract_corpus.py
    python scripts/extract_corpus.py --dry-run
    python scripts/extract_corpus.py --output data/interim/test_chunks.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CFG
from src.document_extractor import DocumentExtractionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("bloom_depth.extract_corpus")


def main():
    parser = argparse.ArgumentParser(description="Extract text corpus from legal PDF textbooks")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--dry-run", action="store_true", help="Only list PDFs, don't extract")
    args = parser.parse_args()

    pipeline = DocumentExtractionPipeline(config=CFG)

    if args.dry_run:
        pdfs = pipeline.discover_pdfs()
        domains = Counter()
        sources = Counter()

        for pdf in pdfs:
            domain = pipeline._classify_domain(pdf.name)
            source = pipeline._classify_source(pdf)
            domains[domain] += 1
            sources[source] += 1
            logger.info("  [%-8s] [%-20s] %s", domain, source, pdf.name)

        logger.info("Total: %d PDFs", len(pdfs))
        for d, c in domains.most_common():
            logger.info("  %-15s: %d", d, c)
        return

    output_path = args.output or CFG.paths.extracted_chunks
    chunks = pipeline.run(output_path=output_path)

    if not chunks:
        logger.error("No chunks extracted!")
        sys.exit(1)

    domains = Counter(c["legal_domain"] for c in chunks)
    logger.info("Total: %d chunks, avg %.0f chars",
                len(chunks), sum(len(c["text"]) for c in chunks) / len(chunks))
    for d, c in domains.most_common():
        logger.info("  %-15s: %d chunks", d, c)


if __name__ == "__main__":
    main()
