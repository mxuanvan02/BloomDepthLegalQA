"""
Stage 1 — CLI Runner
=====================
Command-line entrypoint for the extraction pipeline.

Usage:
    # Check environment first
    python -m src.stage1.check_env

    # Dry-run: classify all PDFs without extracting
    python -m src.stage1.run --input data/raw --classify-only

    # Full extraction
    python -m src.stage1.run --input data/raw --output data/interim/extracted --workers 2

    # Resume from checkpoint
    python -m src.stage1.run --input data/raw --output data/interim/extracted --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Allow running as script or module
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.stage1.classifier import PDFClassifier
    from src.stage1.pipeline import PipelineConfig, Stage1Pipeline
else:
    from .classifier import PDFClassifier
    from .pipeline import PipelineConfig, Stage1Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage1")


def classify_only(input_dir: Path, output: Path | None, exclude: tuple[str, ...]):
    """Classify all PDFs without extraction (dry-run)."""
    classifier = PDFClassifier()
    pdfs = [p for p in sorted(input_dir.rglob("*.pdf"))
            if not any(e in p.parts for e in exclude)]

    logger.info(f"Classifying {len(pdfs)} PDFs...")
    results = []
    class_counter = Counter()
    ocr_counter = Counter()

    for i, pdf in enumerate(pdfs, 1):
        r = classifier.classify(pdf)
        results.append(r.to_dict())
        class_counter[r.doc_class] += 1
        ocr_counter[r.ocr_need] += 1
        logger.info(f"[{i}/{len(pdfs)}] {r.doc_class:20} | {pdf.name[:50]}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print("By document class:")
    for c, n in class_counter.most_common():
        print(f"  {n:4} | {c}")
    print("\nBy OCR need:")
    for c, n in ocr_counter.most_common():
        print(f"  {n:4} | {c}")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "total": len(results),
            "by_class": dict(class_counter),
            "by_ocr_need": dict(ocr_counter),
            "records": results,
        }, ensure_ascii=False, indent=2))
        logger.info(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="BloomDepth Stage 1 extraction")
    parser.add_argument("--input", type=Path, default=Path("data/raw"),
                        help="Input directory of raw PDFs")
    parser.add_argument("--output", type=Path, default=Path("data/interim/extracted"),
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers")
    parser.add_argument("--classify-only", action="store_true",
                        help="Only classify, don't extract")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip processed)")
    parser.add_argument("--exclude", nargs="*", default=["_excluded"],
                        help="Subdirectories to exclude")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)

    exclude = tuple(args.exclude)

    if args.classify_only:
        out = args.output / "classification.json" if args.output else None
        classify_only(args.input, out, exclude)
        return

    config = PipelineConfig(
        input_dir=args.input,
        output_dir=args.output,
        n_workers=args.workers,
        skip_existing=args.resume,
        exclude_dirs=exclude,
    )
    pipeline = Stage1Pipeline(config)
    results = pipeline.run()

    # Final summary
    passed = sum(1 for r in results if r.quality_passed)
    failed = len(results) - passed
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total processed: {len(results)}")
    print(f"Quality passed:  {passed}")
    print(f"Quality failed:  {failed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
