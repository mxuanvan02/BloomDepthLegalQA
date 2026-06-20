"""
Stage 1 — Pipeline Orchestrator
================================
End-to-end PDF→Markdown extraction with:
- Parallel processing
- Real-time progress logging
- Checkpoint/resume support
- Error handling and retry
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .router import ExtractionResult, ExtractionRouter

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    input_dir: Path
    output_dir: Path
    n_workers: int = 2
    checkpoint_path: Path | None = None
    skip_existing: bool = True
    exclude_dirs: tuple[str, ...] = ("_excluded",)


class Stage1Pipeline:
    """Orchestrates parallel PDF extraction with checkpointing."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.router = ExtractionRouter()
        self.checkpoint_path = config.checkpoint_path or config.output_dir / "checkpoint.json"
        self.metadata_path = config.output_dir / "metadata.jsonl"
        self.markdown_dir = config.output_dir / "markdown"
        self._processed: set[str] = set()

    def run(self) -> list[ExtractionResult]:
        """Run full pipeline with progress logging."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint
        self._load_checkpoint()

        # Discover PDFs
        pdfs = list(self._discover_pdfs())
        total = len(pdfs)
        logger.info(f"Found {total} PDFs to process")

        if self.config.skip_existing:
            pdfs = [p for p in pdfs if p.name not in self._processed]
            logger.info(f"Skipping {total - len(pdfs)} already processed, {len(pdfs)} remaining")

        if not pdfs:
            logger.info("Nothing to process")
            return []

        # Process in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {executor.submit(self._process_one, pdf): pdf for pdf in pdfs}
            
            for i, future in enumerate(as_completed(futures), 1):
                pdf = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Save markdown
                    if result.markdown_text and not result.error:
                        md_name = f"{result.sha256[:8]}_{result.filename}.md"
                        md_path = self.markdown_dir / md_name
                        md_path.write_text(result.markdown_text, encoding="utf-8")
                    
                    # Append metadata
                    with open(self.metadata_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                    
                    # Update checkpoint
                    self._processed.add(result.filename)
                    self._save_checkpoint()
                    
                    # Log progress
                    status = "✓" if result.quality_passed else "⚠"
                    logger.info(
                        f"[{i}/{len(pdfs)}] {status} {result.filename[:50]} | "
                        f"{result.extractor_used} | {len(result.markdown_text):,} chars | "
                        f"{result.processing_time_sec:.1f}s"
                    )
                    
                except Exception as e:
                    logger.error(f"[{i}/{len(pdfs)}] ✗ {pdf.name}: {e}")
        
        logger.info(f"Pipeline complete: {len(results)} PDFs processed")
        return results

    def _discover_pdfs(self) -> Iterator[Path]:
        """Discover all PDFs in input directory."""
        for pdf in self.config.input_dir.rglob("*.pdf"):
            if any(excl in pdf.parts for excl in self.config.exclude_dirs):
                continue
            yield pdf

    def _process_one(self, pdf_path: Path) -> ExtractionResult:
        """Process one PDF (called in worker process)."""
        router = ExtractionRouter()
        return router.extract(pdf_path)

    def _load_checkpoint(self):
        """Load checkpoint of processed files."""
        if self.checkpoint_path.exists():
            data = json.loads(self.checkpoint_path.read_text())
            self._processed = set(data.get("processed", []))
            logger.info(f"Loaded checkpoint: {len(self._processed)} files already processed")

    def _save_checkpoint(self):
        """Save checkpoint."""
        self.checkpoint_path.write_text(
            json.dumps({"processed": sorted(self._processed)}, ensure_ascii=False, indent=2)
        )
