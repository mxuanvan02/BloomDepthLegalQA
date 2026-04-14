"""
BloomDepth — Document Extraction Module
=========================================
Extracts text from legal PDF textbooks using Docling (layout-aware)
and filters non-Vietnamese noise via FastText LID.

Input:  PDF files from data/raw/
Output: data/interim/extracted_chunks.jsonl
        data/interim/extracted_markdown/*.md
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("bloom_depth.document_extractor")


# ─────────────────────────────────────────────
# FastText Language Identification Filter
# ─────────────────────────────────────────────
class LanguageFilter:
    """FastText-based language identification for filtering OCR noise."""

    def __init__(self, model_path: str = "lid.176.bin", min_confidence: float = 0.5,
                 fallback_keep_all: bool = True) -> None:
        self.min_confidence = min_confidence
        self.fallback_keep_all = fallback_keep_all
        self._model = None

        try:
            import fasttext
            fasttext.FastText.eprint = lambda x: None

            resolved = Path(model_path)
            if not resolved.exists():
                for candidate in [
                    Path.home() / ".fasttext" / model_path,
                    Path("/tmp") / model_path,
                ]:
                    if candidate.exists():
                        resolved = candidate
                        break

            if resolved.exists():
                self._model = fasttext.load_model(str(resolved))
                logger.info("FastText LID loaded from: %s", resolved)
            else:
                logger.warning("FastText model not found at %s", model_path)
        except ImportError:
            logger.warning("FastText not installed. Install: pip install fasttext-wheel")

    def is_vietnamese(self, text: str) -> tuple[bool, float]:
        """Check if text is Vietnamese. Returns (is_vi, confidence)."""
        if self._model is None:
            return (self.fallback_keep_all, 1.0 if self.fallback_keep_all else 0.0)

        clean = text.replace("\n", " ").strip()
        if len(clean) < 20:
            return (True, 1.0)

        labels, scores = self._model.predict(clean, k=3)
        for label, score in zip(labels, scores):
            if label == "__label__vi":
                return (score >= self.min_confidence, score)
        return (False, 0.0)


# ─────────────────────────────────────────────
# Legal-Hierarchy-Aware Chunker
# ─────────────────────────────────────────────
LEGAL_SECTION_PATTERNS = [
    r"^#{1,3}\s+",                              # Markdown headings
    r"^(?:PHẦN|Phần)\s+(?:THỨ\s+)?\w+",        # Phần thứ nhất/hai...
    r"^(?:CHƯƠNG|Chương)\s+[IVXLCDM\d]+",      # Chương I, II, III...
    r"^(?:MỤC|Mục)\s+\d+",                     # Mục 1, 2, 3...
    r"^(?:TIỂU MỤC|Tiểu mục)\s+\d+",          # Tiểu mục
    r"^(?:Điều|ĐIỀU)\s+\d+[\.\:]",             # Điều 1. / Điều 2:
]

COMPILED_LEGAL_PATTERNS = [re.compile(p, re.MULTILINE) for p in LEGAL_SECTION_PATTERNS]


def _is_section_boundary(line: str) -> bool:
    """Check if a line is a legal document section boundary."""
    stripped = line.strip()
    return any(p.match(stripped) for p in COMPILED_LEGAL_PATTERNS)


def chunk_legal_text(
    text: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    min_chunk_length: int = 200,
    max_chunk_length: int = 5000,
) -> list[dict[str, Any]]:
    """Split legal text into chunks respecting hierarchy boundaries.

    Priority: legal boundaries > paragraph breaks > chunk_size limit.
    """
    if not text or len(text.strip()) < min_chunk_length:
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[dict[str, Any]] = []
    current_chunk: list[str] = []
    current_length = 0
    char_offset = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if _is_section_boundary(para) and current_length > min_chunk_length:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text.strip()) >= min_chunk_length:
                chunks.append({
                    "text": chunk_text.strip(),
                    "start_char": char_offset - current_length,
                    "end_char": char_offset,
                })
            current_chunk = []
            current_length = 0

        if current_length + len(para) > max_chunk_length and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text.strip()) >= min_chunk_length:
                chunks.append({
                    "text": chunk_text.strip(),
                    "start_char": char_offset - current_length,
                    "end_char": char_offset,
                })
            overlap_text = chunk_text[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = [overlap_text] if overlap_text else []
            current_length = len(overlap_text)

        current_chunk.append(para)
        current_length += len(para)
        char_offset += len(para) + 2

        if current_length >= chunk_size and not _is_section_boundary(para):
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text.strip()) >= min_chunk_length:
                chunks.append({
                    "text": chunk_text.strip(),
                    "start_char": char_offset - current_length,
                    "end_char": char_offset,
                })
            overlap_text = chunk_text[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = [overlap_text] if overlap_text else []
            current_length = len(overlap_text)

    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        if len(chunk_text.strip()) >= min_chunk_length:
            chunks.append({
                "text": chunk_text.strip(),
                "start_char": char_offset - current_length,
                "end_char": char_offset,
            })

    return chunks


# ─────────────────────────────────────────────
# Docling PDF Extraction
# ─────────────────────────────────────────────
def extract_single_pdf(pdf_path: Path, ocr_enabled: bool = True,
                       table_structure: bool = True) -> str | None:
    """Extract text from a single PDF using Docling's DocumentConverter."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()

        if not markdown or len(markdown.strip()) < 100:
            logger.warning("Near-empty output for %s", pdf_path.name)
            return None
        return markdown

    except ImportError:
        logger.error("Docling not installed. Install: pip install docling")
        return None
    except Exception as e:
        logger.error("Extraction failed for %s: %s", pdf_path.name, e)
        return None


# ─────────────────────────────────────────────
# Full Extraction Pipeline
# ─────────────────────────────────────────────
class DocumentExtractionPipeline:
    """PDF → Chunks pipeline: Docling extraction → legal chunking → FastText filter → JSONL."""

    def __init__(self, config: Any = None) -> None:
        if config is None:
            from configs.config import CFG
            config = CFG

        self.cfg = config
        self.paths = config.paths
        self.ext_cfg = config.extraction
        self.lang_filter = LanguageFilter(
            model_path=self.ext_cfg.fasttext_model,
            min_confidence=self.ext_cfg.min_vietnamese_confidence,
            fallback_keep_all=self.ext_cfg.fallback_on_no_fasttext,
        )

    def discover_pdfs(self) -> list[Path]:
        """Discover all PDF files across configured source directories."""
        pdfs: list[Path] = []
        for source_dir in self.ext_cfg.source_dirs:
            full_path = self.paths.raw / source_dir
            if not full_path.exists():
                logger.warning("Source directory not found: %s", full_path)
                continue
            found = sorted(full_path.glob("*.pdf"))
            logger.info("Found %d PDFs in %s", len(found), source_dir)
            pdfs.extend(found)
        return pdfs

    def _generate_chunk_id(self, pdf_name: str, chunk_index: int) -> str:
        base = pdf_name.replace(".pdf", "").replace(" ", "_")
        return f"{base}_chunk_{chunk_index:04d}"

    def _classify_domain(self, pdf_name: str) -> str:
        """Heuristic domain classification from filename."""
        name_lower = pdf_name.lower()
        if any(kw in name_lower for kw in ["hinh_su", "hinh su", "hình sự", "toi pham", "tội phạm"]):
            return "hinh_su"
        if any(kw in name_lower for kw in ["hanh_chinh", "hanh chinh", "hành chính", "to tung hanh"]):
            return "hanh_chinh"
        if any(kw in name_lower for kw in ["dan_su", "dan su", "dân sự", "hon nhan", "hôn nhân",
                                            "thua ke", "thừa kế", "hop dong", "hợp đồng"]):
            return "dan_su"
        return "general"

    def _classify_source(self, pdf_path: Path) -> str:
        path_str = str(pdf_path)
        if "institute" in path_str:
            return "institute_textbook"
        if "fdvn" in path_str:
            return "university_textbook"
        if "archive_org" in path_str:
            return "reference_book"
        return "unknown"

    def process_single_pdf(self, pdf_path: Path) -> list[dict[str, Any]]:
        """Process a single PDF: extract → chunk → filter → records."""
        start = time.monotonic()
        pdf_name = pdf_path.name
        logger.info("Processing: %s", pdf_name)

        markdown = extract_single_pdf(
            pdf_path,
            ocr_enabled=self.ext_cfg.ocr_enabled,
            table_structure=self.ext_cfg.table_structure,
        )
        if markdown is None:
            return []

        if self.ext_cfg.save_markdown:
            md_path = self.paths.extracted_markdown / f"{pdf_path.stem}.md"
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(markdown, encoding="utf-8")

        raw_chunks = chunk_legal_text(
            markdown,
            chunk_size=self.ext_cfg.chunk_size,
            chunk_overlap=self.ext_cfg.chunk_overlap,
            min_chunk_length=self.ext_cfg.min_chunk_length,
            max_chunk_length=self.ext_cfg.max_chunk_length,
        )

        domain = self._classify_domain(pdf_name)
        source = self._classify_source(pdf_path)
        valid_chunks: list[dict[str, Any]] = []

        for i, chunk in enumerate(raw_chunks):
            is_vi, confidence = self.lang_filter.is_vietnamese(chunk["text"])
            if not is_vi:
                continue

            valid_chunks.append({
                "chunk_id": self._generate_chunk_id(pdf_name, i),
                "text": chunk["text"],
                "source_doc": pdf_name,
                "source_path": str(pdf_path.relative_to(self.paths.raw)),
                "source_category": source,
                "legal_domain": domain,
                "chunk_index": i,
                "char_start": chunk.get("start_char", 0),
                "char_end": chunk.get("end_char", 0),
                "lang_confidence": round(confidence, 4),
                "content_hash": hashlib.md5(chunk["text"].encode()).hexdigest()[:12],
            })

        elapsed = time.monotonic() - start
        logger.info(
            "  %s: %d chunks (%d filtered) in %.1fs",
            pdf_name, len(valid_chunks), len(raw_chunks) - len(valid_chunks), elapsed,
        )
        return valid_chunks

    def run(self, output_path: Path | None = None) -> list[dict[str, Any]]:
        """Run extraction on all discovered PDFs. Returns all chunk records."""
        self.paths.ensure_dirs()
        output_path = output_path or self.paths.extracted_chunks

        pdfs = self.discover_pdfs()
        if not pdfs:
            logger.error("No PDFs found in configured source directories!")
            return []

        logger.info("=" * 60)
        logger.info("BloomDepth Document Extraction")
        logger.info("  %d PDFs across %d directories", len(pdfs), len(self.ext_cfg.source_dirs))
        logger.info("=" * 60)

        all_chunks: list[dict[str, Any]] = []
        stats = {"processed": 0, "failed": 0, "total_chunks": 0}

        for pdf in pdfs:
            try:
                chunks = self.process_single_pdf(pdf)
                all_chunks.extend(chunks)
                stats["processed"] += 1
                stats["total_chunks"] += len(chunks)
            except Exception as e:
                logger.error("Fatal error processing %s: %s", pdf.name, e)
                stats["failed"] += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        logger.info("Extraction complete: %d/%d PDFs, %d chunks → %s",
                     stats["processed"], len(pdfs), stats["total_chunks"], output_path)
        return all_chunks
