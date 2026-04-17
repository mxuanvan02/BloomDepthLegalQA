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

import gc
import hashlib
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger("bloom_depth.document_extractor")

_WORKER_PIPELINE: Any | None = None


def _init_pdf_worker(project_root: str, drive_base: str, docling_device: str, docling_num_threads: int) -> None:
    """Initialize one long-lived Docling pipeline per worker process."""
    global _WORKER_PIPELINE

    os.environ["BLOOMDEPTH_ROOT"] = project_root
    if drive_base:
        os.environ["BLOOMDEPTH_DRIVE"] = drive_base
    os.environ["DOCLING_DEVICE"] = docling_device
    os.environ["DOCLING_NUM_THREADS"] = str(docling_num_threads)
    os.environ["OMP_NUM_THREADS"] = str(docling_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(docling_num_threads)

    from configs.config import CFG

    _WORKER_PIPELINE = DocumentExtractionPipeline(config=CFG)


def _process_pdf_in_worker(pdf_path: str) -> tuple[str, list[dict[str, Any]]]:
    """Process one PDF in a worker process and return records to the parent."""
    if _WORKER_PIPELINE is None:
        raise RuntimeError("PDF worker was not initialized")
    
    pdf = Path(pdf_path)
    try:
        results = _WORKER_PIPELINE.process_single_pdf(pdf)
        return pdf.name, results
    finally:
        # Aggressive memory cleanup inside worker to prevent OOM across multiple PDFs
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


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
        self._warned_runtime_failure = False

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

        try:
            labels, scores = self._model.predict(clean, k=3)
            for label, score in zip(labels, scores):
                if label == "__label__vi":
                    return (score >= self.min_confidence, score)
            return (False, 0.0)
        except Exception as exc:
            if not self._warned_runtime_failure:
                logger.warning(
                    "FastText LID runtime error (likely NumPy 2.0 incompatibility). "
                    "Falling back to keeping all chunks. Error: %s",
                    exc,
                )
                self._warned_runtime_failure = True
            return (self.fallback_keep_all, 1.0 if self.fallback_keep_all else 0.0)


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
            # Overlap using the last paragraph to ensure clean sentence boundaries
            current_chunk = [current_chunk[-1]] if (chunk_overlap > 0 and current_chunk) else []
            current_length = len(current_chunk[0]) if current_chunk else 0

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
            # Overlap using the last paragraph added (which is 'para')
            current_chunk = [para] if chunk_overlap > 0 else []
            current_length = len(para) if chunk_overlap > 0 else 0

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
def _accelerator_device(device_name: str, accelerator_device: Any) -> Any:
    """Resolve a stable Docling accelerator device across Docling versions."""
    normalized = device_name.strip().lower()
    if normalized == "cuda":
        return getattr(accelerator_device, "CUDA", "cuda")
    if normalized == "cpu":
        return getattr(accelerator_device, "CPU", "cpu")
    if normalized == "mps":
        return getattr(accelerator_device, "MPS", "mps")
    if normalized == "auto":
        return getattr(accelerator_device, "AUTO", "auto")
    return device_name


def create_docling_converter(
    ocr_enabled: bool = True,
    table_structure: bool = True,
    device: str = "auto",
    num_threads: int = 12,
) -> Any:
    """Create a Docling converter with explicit accelerator settings.

    Falls back to Docling defaults if the installed Docling version exposes a
    different accelerator API.
    """
    from docling.document_converter import DocumentConverter

    try:
        from docling.datamodel.base_models import InputFormat
        try:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        except ImportError:
            from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.settings import settings
        from docling.document_converter import PdfFormatOption

        settings.debug.profile_pipeline_timings = True
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr_enabled
        pipeline_options.do_table_structure = table_structure
        if hasattr(pipeline_options, "table_structure_options"):
            table_opts = pipeline_options.table_structure_options
            if hasattr(table_opts, "do_cell_matching"):
                table_opts.do_cell_matching = table_structure
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=_accelerator_device(device, AcceleratorDevice),
        )
        logger.info(
            "Docling accelerator: device=%s, num_threads=%d, ocr=%s, table_structure=%s",
            device,
            num_threads,
            ocr_enabled,
            table_structure,
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
    except Exception as exc:
        logger.warning("Docling accelerator config unavailable, using defaults: %s", exc)
        return DocumentConverter()


def extract_single_pdf(pdf_path: Path, ocr_enabled: bool = True,
                       table_structure: bool = True,
                       converter: Any | None = None) -> str | None:
    """Extract text from a single PDF using Docling's DocumentConverter."""
    try:
        if converter is None:
            converter = create_docling_converter(
                ocr_enabled=ocr_enabled,
                table_structure=table_structure,
                device=os.environ.get("DOCLING_DEVICE", "auto"),
                num_threads=int(os.environ.get("DOCLING_NUM_THREADS", "12")),
            )
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()
        timings = getattr(result, "timings", None)
        if timings:
            compact_timings = {}
            for name, timing in timings.items():
                times = getattr(timing, "times", None)
                if times:
                    compact_timings[name] = round(sum(times), 3)
            if compact_timings:
                logger.info("  Docling timings for %s: %s", pdf_path.name, compact_timings)

        if not markdown or len(markdown.strip()) < 100:
            logger.warning("Near-empty output for %s", pdf_path.name)
            return None
        return markdown

    except ImportError:
        logger.error("Docling not installed. Install: pip install docling")
        return None
    except Exception as e:
        logger.error("Extraction failed for %s: %s", pdf_path.name, e, exc_info=True)
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
        self.docling_converter: Any | None = None

    def _get_docling_converter(self) -> Any:
        if self.docling_converter is None:
            self.docling_converter = create_docling_converter(
                ocr_enabled=self.ext_cfg.ocr_enabled,
                table_structure=self.ext_cfg.table_structure,
                device=self.ext_cfg.docling_device,
                num_threads=self.ext_cfg.docling_num_threads,
            )
        return self.docling_converter

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

    def _drive_checkpoint_path(self, local_path: Path) -> Path | None:
        from src.drive_sync import get_drive_sync
        ds = get_drive_sync()
        if not ds or not ds.enabled or not ds.drive_base:
            return None
        try:
            rel_path = local_path.resolve().relative_to(self.paths.root.resolve())
        except ValueError:
            rel_path = Path(local_path.name)
        return ds.drive_base / rel_path

    def _restore_checkpoint(self, local_path: Path) -> None:
        drive_path = self._drive_checkpoint_path(local_path)
        if local_path.exists() or drive_path is None or not drive_path.exists():
            return
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_path, local_path)
        logger.info("Restored extraction checkpoint from Drive: %s", drive_path)

    def _sync_checkpoint(self, local_path: Path) -> None:
        drive_path = self._drive_checkpoint_path(local_path)
        if drive_path is None or not local_path.exists():
            return
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, drive_path)
        logger.info("Checkpoint synced to Drive: %s", drive_path)

    def _manifest_path(self, output_path: Path) -> Path:
        return output_path.with_suffix(output_path.suffix + ".manifest.json")

    def _save_manifest(self, manifest_path: Path, completed_docs: set[str]) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps({"completed_docs": sorted(completed_docs)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(manifest_path)
        self._sync_checkpoint(manifest_path)

    def _load_existing_chunks(self, output_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
        if not output_path.exists():
            return [], set()

        chunks: list[dict[str, Any]] = []
        completed_docs: set[str] = set()
        with open(output_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupt checkpoint line %d in %s", line_num, output_path)
                    continue
                chunks.append(record)
                source_doc = record.get("source_doc")
                if source_doc:
                    completed_docs.add(source_doc)

        manifest_path = self._manifest_path(output_path)
        self._restore_checkpoint(manifest_path)
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                completed_docs.update(manifest.get("completed_docs", []))
            except json.JSONDecodeError:
                logger.warning("Ignoring corrupt extraction manifest: %s", manifest_path)
        return chunks, completed_docs

    def _persist_pdf_chunks(
        self,
        output_file: Any,
        output_path: Path,
        manifest_path: Path,
        completed_docs: set[str],
        pdf_name: str,
        chunks: list[dict[str, Any]],
        force_sync: bool = False,
    ) -> None:
        """Write chunks to local file and sync progress to Drive.
        
        Manifest is synced every time (small). Heavy .jsonl is synced every 5 PDFs.
        """
        for chunk in chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())
        
        completed_docs.add(pdf_name)
        self._save_manifest(manifest_path, completed_docs)
        
        # Optimization: Don't sync the massive .jsonl file to Drive every single PDF
        # unless forced or at intervals to avoid I/O blocking.
        should_sync = force_sync or (len(completed_docs) % 5 == 0)
        if should_sync:
            self._sync_checkpoint(output_path)

    def process_single_pdf(self, pdf_path: Path) -> list[dict[str, Any]]:
        """Process a single PDF: extract → chunk → filter → records."""
        start = time.monotonic()
        pdf_name = pdf_path.name
        logger.info("Processing: %s", pdf_name)

        md_path = self.paths.extracted_markdown / f"{pdf_path.stem}.md"
        if self.ext_cfg.save_markdown and md_path.exists():
            markdown = md_path.read_text(encoding="utf-8")
            logger.info("  Reusing cached markdown: %s", md_path)
        else:
            markdown = extract_single_pdf(
                pdf_path,
                ocr_enabled=self.ext_cfg.ocr_enabled,
                table_structure=self.ext_cfg.table_structure,
                converter=self._get_docling_converter(),
            )
            if markdown is None:
                return []

            if self.ext_cfg.save_markdown:
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(markdown, encoding="utf-8")

        if self.ext_cfg.save_markdown:
            self._sync_checkpoint(md_path)

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
        manifest_path = self._manifest_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._restore_checkpoint(output_path)
        self._restore_checkpoint(manifest_path)

        pdfs = self.discover_pdfs()
        if not pdfs:
            logger.error("No PDFs found in configured source directories!")
            return []

        logger.info("=" * 60)
        logger.info("BloomDepth Document Extraction")
        logger.info("  %d PDFs across %d directories", len(pdfs), len(self.ext_cfg.source_dirs))
        logger.info("=" * 60)

        all_chunks, completed_docs = self._load_existing_chunks(output_path)
        stats = {
            "processed": len(completed_docs),
            "skipped": 0,
            "failed": 0,
            "total_chunks": len(all_chunks),
        }
        if completed_docs:
            logger.info(
                "Resuming extraction checkpoint: %d PDFs already complete, %d chunks loaded from %s",
                len(completed_docs),
                len(all_chunks),
                output_path,
            )

        pending_pdfs: list[Path] = []
        for pdf in pdfs:
            if pdf.name in completed_docs:
                stats["skipped"] += 1
                logger.info("Skipping already extracted PDF: %s", pdf.name)
                continue
            pending_pdfs.append(pdf)

        n_workers = max(1, int(self.ext_cfg.n_workers))
        if pending_pdfs:
            logger.info("Extraction workers: %d (%d pending PDFs)", n_workers, len(pending_pdfs))

        with open(output_path, "a", encoding="utf-8") as f:
            if n_workers == 1 or len(pending_pdfs) <= 1:
                for pdf in pending_pdfs:
                    try:
                        chunks = self.process_single_pdf(pdf)
                        all_chunks.extend(chunks)
                        self._persist_pdf_chunks(f, output_path, manifest_path, completed_docs, pdf.name, chunks)
                        stats["processed"] += 1
                        stats["total_chunks"] += len(chunks)
                        logger.info(
                            "Checkpoint: %d/%d PDFs complete, %d chunks persisted",
                            len(completed_docs),
                            len(pdfs),
                            stats["total_chunks"],
                        )
                    except Exception as e:
                        logger.error("Fatal error processing %s: %s", pdf.name, e, exc_info=True)
                        stats["failed"] += 1
                
                # Final sync for single-threaded path
                self._sync_checkpoint(output_path)
            else:
                worker_threads = max(1, int(self.ext_cfg.docling_num_threads) // n_workers)
                logger.info(
                    "Parallel extraction enabled: %d workers × %d Docling threads/worker",
                    n_workers,
                    worker_threads,
                )
                import multiprocessing as mp
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=mp.get_context("spawn"),
                    initializer=_init_pdf_worker,
                    initargs=(
                        str(self.paths.root),
                        os.environ.get("BLOOMDEPTH_DRIVE", ""),
                        self.ext_cfg.docling_device,
                        worker_threads,
                    ),
                ) as executor:
                    future_to_pdf = {
                        executor.submit(_process_pdf_in_worker, str(pdf)): pdf
                        for pdf in pending_pdfs
                    }
                    for future in as_completed(future_to_pdf):
                        pdf = future_to_pdf[future]
                        try:
                            pdf_name, chunks = future.result()
                            all_chunks.extend(chunks)
                            self._persist_pdf_chunks(f, output_path, manifest_path, completed_docs, pdf_name, chunks)
                            stats["processed"] += 1
                            stats["total_chunks"] += len(chunks)
                            logger.info(
                                "Checkpoint: %d/%d PDFs complete, %d chunks persisted",
                                len(completed_docs),
                                len(pdfs),
                                stats["total_chunks"],
                            )
                        except Exception as e:
                            logger.error("Fatal worker error for %s: %s", pdf.name, e, exc_info=True)
                            stats["failed"] += 1

                # Final sync at the end of parallel block
                self._sync_checkpoint(output_path)

        logger.info("Extraction complete: %d/%d PDFs, %d skipped, %d failed, %d chunks → %s",
                    len(completed_docs), len(pdfs), stats["skipped"],
                    stats["failed"], stats["total_chunks"], output_path)
        return all_chunks
