#!/usr/bin/env python3
"""T1 — pdf_extraction → [B] extracted_chunks.jsonl  (CONTRACT.md §4).

This is the REAL T1 transform. It produces ONE merged chunk file for all 118
source documents using a SINGLE chunking strategy (BloomDepth's own
``chunk_legal_text``), so there is no intra-dataset chunking drift.

Two ingestion paths, merged into one schema:

1. TQA-reuse adapter (48 institute docs)
   The 48 'institute' giáo trình were already digitized by the sibling
   TQA_Pipeline (marker-pdf). The .md intermediates were gitignored/deleted,
   but the clean chunk text survives in
   ``TQA_Pipeline/data/output/processed/dataset_eval_ready.clean.jsonl``
   (field ``context_text``, keyed by ``doc_id``). We REUSE that text rather
   than re-running marker, then RE-CHUNK it with BloomDepth's chunker so it
   shares one format with the universities docs. This path uses ONLY stdlib —
   no docling / no GPU.

2. Docling extractor (70 universities docs)
   The 70 new GT_/Sach_ PDFs under ``data/raw/universities/`` are extracted via
   the existing docling-based ``DocumentExtractionPipeline``
   (``src/document_extractor.py``). Requires docling installed.

Every emitted chunk carries a REAL ``source_doc``, ``chunk_id`` and
``content_hash`` (this is the bug the broken Phase-A run had: empty source_doc).

Usage
-----
    # TQA-reuse only (no heavy deps) — dry run on a few docs:
    python scripts/build_extracted_chunks.py --tqa-only --limit-docs 3 --dry-run

    # TQA-reuse only, write merged file:
    python scripts/build_extracted_chunks.py --tqa-only

    # Full T1 (TQA reuse + docling on 70 universities PDFs):
    python scripts/build_extracted_chunks.py

Output
------
    data/interim/extracted_chunks.jsonl                       [B]
    research/results/audit/extracted_chunks_t1_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# chunk_legal_text is pure-stdlib (regex only) — safe to import without docling.
from src.document_extractor import chunk_legal_text  # noqa: E402

logger = logging.getLogger("bloom_depth.t1_extract")

DEFAULT_TQA_CLEAN = (
    PROJECT_ROOT.parent
    / "TQA_Pipeline"
    / "data"
    / "output"
    / "processed"
    / "dataset_eval_ready.clean.jsonl"
)

# Chunking parameters mirror configs.config.ExtractionConfig so the TQA-reuse
# path and the docling path share IDENTICAL chunk geometry.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 200
MAX_CHUNK_LENGTH = 5000


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _classify_domain(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["hinh_su", "hinh su", "hình sự", "toi pham", "tội phạm", "hinh su"]):
        return "hinh_su"
    if any(k in n for k in ["hanh_chinh", "hanh chinh", "hành chính", "to tung hanh"]):
        return "hanh_chinh"
    if any(k in n for k in ["dan_su", "dan su", "dân sự", "hon nhan", "hôn nhân",
                            "thua ke", "thừa kế", "hop dong", "hợp đồng", "thuong mai", "thương mại"]):
        return "dan_su"
    return "general"


def _slug(name: str) -> str:
    base = name.replace(".pdf", "").strip()
    return re.sub(r"\s+", "_", base)


# ─────────────────────────────────────────────────────────────────────────────
# Path 1 — TQA-reuse adapter (48 institute docs, stdlib only)
# ─────────────────────────────────────────────────────────────────────────────
def build_institute_pdf_index(institute_dir: Path) -> Dict[str, str]:
    """Map a normalized doc_id → real institute PDF filename (for source_doc).

    TQA doc_id looks like '10. XAY DUNG VAN BAN PHAPLUAT'; the institute PDF is
    '10. XAY DUNG VAN BAN PHAPLUAT.pdf'. Match on case/space-insensitive stem.
    """
    index: Dict[str, str] = {}
    if not institute_dir.exists():
        return index
    for pdf in institute_dir.glob("*.pdf"):
        key = re.sub(r"\s+", " ", pdf.stem).strip().lower()
        index[key] = pdf.name
    return index


def resolve_source_doc(doc_id: str, pdf_index: Dict[str, str]) -> str:
    key = re.sub(r"\s+", " ", str(doc_id)).strip().lower()
    if key in pdf_index:
        return pdf_index[key]
    # Some doc_ids merge numbers ('11.12.LY LUAN...'). Try a looser prefix match.
    for k, v in pdf_index.items():
        if k.startswith(key) or key.startswith(k):
            return v
    # Fall back to a deterministic .pdf name derived from the doc_id.
    return f"{str(doc_id).strip()}.pdf"


def adapt_tqa_clean(
    tqa_path: Path,
    institute_dir: Path,
    limit_docs: int | None = None,
) -> List[Dict[str, Any]]:
    """Read TQA clean rows → reconstruct per-doc text → BloomDepth re-chunk.

    For each TQA ``doc_id`` we collect its DISTINCT ``context_text`` values in
    first-seen order, join them into one document stream, then re-chunk with
    ``chunk_legal_text``. This guarantees the institute docs use the exact same
    chunk geometry as the docling-extracted universities docs.
    """
    pdf_index = build_institute_pdf_index(institute_dir)

    # doc_id -> ordered unique context_text  (+ multimodal flag carried per doc)
    per_doc: "OrderedDict[str, OrderedDict[str, bool]]" = OrderedDict()
    for row in read_jsonl(tqa_path):
        doc_id = row.get("doc_id")
        ctx = (row.get("context_text") or "").strip()
        if not doc_id or not ctx:
            continue
        bucket = per_doc.setdefault(doc_id, OrderedDict())
        # OrderedDict over text dedups within-doc while preserving order.
        if ctx not in bucket:
            bucket[ctx] = bool(row.get("is_multimodal"))

    chunks: List[Dict[str, Any]] = []
    doc_ids = list(per_doc.keys())
    if limit_docs:
        doc_ids = doc_ids[:limit_docs]

    for doc_id in doc_ids:
        ctx_map = per_doc[doc_id]
        source_doc = resolve_source_doc(doc_id, pdf_index)
        doc_stream = "\n\n".join(ctx_map.keys())
        raw_chunks = chunk_legal_text(
            doc_stream,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            min_chunk_length=MIN_CHUNK_LENGTH,
            max_chunk_length=MAX_CHUNK_LENGTH,
        )
        domain = _classify_domain(source_doc)
        slug = _slug(source_doc)
        had_multimodal = any(ctx_map.values())
        for i, ch in enumerate(raw_chunks):
            text = ch["text"]
            chunks.append({
                "chunk_id": f"{slug}_chunk_{i:04d}",
                "text": text,
                "source_doc": source_doc,
                "source_path": f"institute/{source_doc}",
                "source_category": "institute_textbook",
                "legal_domain": domain,
                "chunk_index": i,
                "content_hash": _content_hash(text),
                "extraction_origin": "tqa_reuse",
                "tqa_doc_id": doc_id,
                "metadata": {
                    "start_char": ch.get("start_char"),
                    "end_char": ch.get("end_char"),
                    "doc_had_multimodal_context": had_multimodal,
                },
            })
        logger.info("  [tqa] %-55s → %4d chunks (source_doc=%s)",
                    str(doc_id)[:55], len(raw_chunks), source_doc)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Path 2 — Docling extractor (70 universities docs)
# ─────────────────────────────────────────────────────────────────────────────
def extract_universities_docling(universities_dir: Path) -> List[Dict[str, Any]]:
    """Extract the 70 universities PDFs via the docling-based pipeline.

    Imported lazily so the TQA-reuse path works without docling installed.
    Reuses ``DocumentExtractionPipeline.process_single_pdf`` which already emits
    the canonical chunk schema (chunk_id/source_doc/content_hash/...).
    """
    from configs.config import CFG
    from src.document_extractor import DocumentExtractionPipeline

    pipeline = DocumentExtractionPipeline(config=CFG)
    pdfs = sorted(universities_dir.rglob("*.pdf"))
    logger.info("Docling: %d universities PDFs discovered under %s", len(pdfs), universities_dir)

    chunks: List[Dict[str, Any]] = []
    for j, pdf in enumerate(pdfs, 1):
        try:
            recs = pipeline.process_single_pdf(pdf)
        except Exception as exc:  # noqa: BLE001
            logger.error("  [docling] FAILED %s: %s", pdf.name, exc)
            continue
        for r in recs:
            r.setdefault("extraction_origin", "docling_universities")
            # Normalize source_path to be relative to data/raw for consistency.
            try:
                r["source_path"] = str(pdf.relative_to(universities_dir.parent))
            except ValueError:
                r["source_path"] = str(pdf)
        chunks.extend(recs)
        logger.info("  [docling] [%d/%d] %-50s → %4d chunks", j, len(pdfs), pdf.name[:50], len(recs))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Merge + report
# ─────────────────────────────────────────────────────────────────────────────
def build_report(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_origin = Counter(c.get("extraction_origin") for c in chunks)
    by_category = Counter(c.get("source_category") for c in chunks)
    by_domain = Counter(c.get("legal_domain") for c in chunks)
    docs = {c.get("source_doc") for c in chunks}
    empty_source = sum(1 for c in chunks if not str(c.get("source_doc") or "").strip())
    empty_text = sum(1 for c in chunks if not str(c.get("text") or "").strip())
    missing_hash = sum(1 for c in chunks if not c.get("content_hash"))
    dup_ids = sum(cnt - 1 for cnt in Counter(c.get("chunk_id") for c in chunks).values() if cnt > 1)
    return {
        "artifact": "extracted_chunks.jsonl [B] (CONTRACT.md §4 T1)",
        "total_chunks": len(chunks),
        "distinct_source_docs": len(docs),
        "by_extraction_origin": dict(by_origin),
        "by_source_category": dict(by_category),
        "by_legal_domain": dict(by_domain),
        "integrity": {
            "empty_source_doc": empty_source,   # MUST be 0 (was 10000 in broken run)
            "empty_text": empty_text,
            "missing_content_hash": missing_hash,
            "duplicate_chunk_ids": dup_ids,
        },
        "exit_gate_pass": len(chunks) > 0 and empty_source == 0,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tqa-clean", type=Path, default=DEFAULT_TQA_CLEAN,
                    help="TQA dataset_eval_ready.clean.jsonl path")
    ap.add_argument("--institute-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "institute")
    ap.add_argument("--universities-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "universities")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "interim" / "extracted_chunks.jsonl")
    ap.add_argument("--report", type=Path,
                    default=PROJECT_ROOT / "research" / "results" / "audit" / "extracted_chunks_t1_report.json")
    ap.add_argument("--tqa-only", action="store_true",
                    help="Skip docling; only build the 48 institute docs from TQA reuse (no heavy deps).")
    ap.add_argument("--docling-only", action="store_true",
                    help="Skip TQA reuse; only run docling on the 70 universities PDFs.")
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="Limit number of TQA docs (debug/dry-run).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build chunks in memory and print the report, but do NOT write the JSONL.")
    args = ap.parse_args()

    if not args.tqa_clean.exists() and not args.docling_only:
        raise FileNotFoundError(
            f"TQA clean file not found: {args.tqa_clean}\n"
            "  → This is the reused institute text. Check the TQA_Pipeline sibling repo path."
        )

    all_chunks: List[Dict[str, Any]] = []

    if not args.docling_only:
        logger.info("T1 path 1/2 — TQA-reuse adapter (institute docs)")
        all_chunks.extend(adapt_tqa_clean(args.tqa_clean, args.institute_dir, args.limit_docs))

    if not args.tqa_only:
        logger.info("T1 path 2/2 — docling extraction (universities docs)")
        all_chunks.extend(extract_universities_docling(args.universities_dir))

    report = build_report(all_chunks)

    if args.dry_run:
        logger.info("[DRY RUN] %d chunks built in memory (not written).", len(all_chunks))
        if all_chunks:
            logger.info("[DRY RUN] sample chunk:\n%s",
                        json.dumps({k: (v[:160] if isinstance(v, str) else v)
                                    for k, v in all_chunks[0].items()},
                                   ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if not report["exit_gate_pass"]:
        dump_json(args.report, report)
        raise RuntimeError(
            f"T1 EXIT GATE FAILED: chunks={report['total_chunks']} "
            f"empty_source_doc={report['integrity']['empty_source_doc']} (must be >0 chunks, 0 empty source)."
        )

    write_jsonl(args.output, all_chunks)
    dump_json(args.report, report)
    logger.info("T1 done → %s (%d chunks, %d docs)",
                args.output, report["total_chunks"], report["distinct_source_docs"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
