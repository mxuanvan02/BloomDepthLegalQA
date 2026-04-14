"""
Data preparation — validates extracted chunks and creates domain splits.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --include-v1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CFG
from src.drive_sync import get_drive_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("bloom_depth.prepare_data")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path, drive_sync: Any = None, drive_subpath: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), path)
    if drive_sync and drive_subpath:
        drive_sync.sync_file(path, drive_subpath)


def validate_chunk(record: dict) -> bool:
    """Required: chunk_id, text (≥100 chars), source_doc."""
    required = ["chunk_id", "text", "source_doc"]
    return all(record.get(f) for f in required) and len(record.get("text", "")) >= 100


def main():
    parser = argparse.ArgumentParser(description="Prepare BloomDepth data for experiments")
    parser.add_argument("--source", type=Path, default=None, help="Path to extracted_chunks.jsonl")
    parser.add_argument("--include-v1", action="store_true", help="Merge VDTM-LegalQA baseline")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or CFG.paths.root / "data"
    source_path = args.source or CFG.paths.extracted_chunks
    
    # Initialize Drive Sync
    try:
        drive_sync = get_drive_sync()
    except Exception as e:
        logger.warning("Drive sync unavailable: %s (continuing without)", e)
        drive_sync = None

    if not source_path.exists():
        logger.error("Chunks not found: %s (run extract_corpus.py first)", source_path)
        sys.exit(1)

    # Load & validate
    chunks = load_jsonl(source_path)
    valid = [r for r in chunks if validate_chunk(r)]
    logger.info("Loaded %d chunks, %d valid (%d dropped)", len(chunks), len(valid), len(chunks) - len(valid))

    # Distribution
    domain_dist = Counter(r.get("legal_domain", "unknown") for r in valid)
    source_dist = Counter(r.get("source_category", "unknown") for r in valid)
    doc_count = len(set(r.get("source_doc", "") for r in valid))
    lengths = [len(r["text"]) for r in valid]

    logger.info("Domains: %s", dict(domain_dist.most_common()))
    logger.info("Sources: %s", dict(source_dist.most_common()))
    logger.info("Unique docs: %d, chunk length: min=%d avg=%.0f max=%d",
                doc_count, min(lengths), sum(lengths) / len(lengths), max(lengths))

    # Optionally merge v1
    if args.include_v1:
        v1_path = CFG.paths.vdtm_dataset
        if v1_path.exists():
            v1_records = load_jsonl(v1_path)
            target_v1 = output_dir / "processed" / "v1_baseline.jsonl"
            save_jsonl(v1_records, target_v1, drive_sync, "data/processed/v1_baseline.jsonl")
            logger.info("Merged %d v1 QA records", len(v1_records))
        else:
            logger.warning("v1 dataset not found: %s", v1_path)

    # Save validated corpus
    corpus_target = output_dir / "processed" / "corpus_validated.jsonl"
    save_jsonl(valid, corpus_target, drive_sync, "data/processed/corpus_validated.jsonl")

    # Domain splits
    splits_dir = output_dir / "interim" / "domain_splits"
    domain_groups: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        domain_groups[r.get("legal_domain", "general")].append(r)
    for domain, records in domain_groups.items():
        domain_file = splits_dir / f"{domain}.jsonl"
        save_jsonl(records, domain_file, drive_sync, f"data/interim/domain_splits/{domain}.jsonl")

    logger.info("Done: %d chunks → %d domains", len(valid), len(domain_groups))


if __name__ == "__main__":
    main()
