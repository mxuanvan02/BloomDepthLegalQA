"""
Stage 1 — Extraction Router
============================
Routes PDF to appropriate extractor based on classification.
Handles fallback chain and quality validation.

Extractor chain:
  digital_text      → pymupdf
  digital_text_risky → pymupdf → gate → marker fallback
  scanned_no_ocr    → marker (force OCR) → ocrmypdf fallback
  scanned_with_ocr  → pymupdf → gate → marker fallback
  encrypted         → decrypt → reclassify → route
  uncertain         → marker (safe default)
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .classifier import ClassificationResult, PDFClassifier
from .quality_gates import GateResult, QualityGates

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of PDF extraction."""
    filename: str
    sha256: str
    doc_class: str
    extractor_used: str
    fallback_used: bool
    quality_passed: bool
    quality: GateResult | None
    markdown_text: str
    processing_time_sec: float
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "doc_class": self.doc_class,
            "extractor_used": self.extractor_used,
            "fallback_used": self.fallback_used,
            "quality_passed": self.quality_passed,
            "quality": self.quality.to_dict() if self.quality else None,
            "text_length": len(self.markdown_text),
            "processing_time_sec": round(self.processing_time_sec, 2),
            "error": self.error,
        }


class ExtractionRouter:
    """Routes PDFs to extractors and handles fallback."""

    def __init__(
        self,
        marker_cmd: str | None = None,
        ocrmypdf_cmd: str | None = None,
        qpdf_cmd: str | None = None,
    ):
        """
        Args:
            marker_cmd: Path to marker_single CLI (default: auto-detect).
            ocrmypdf_cmd: Path to ocrmypdf CLI (default: auto-detect).
            qpdf_cmd: Path to qpdf CLI (default: auto-detect).
        """
        self.classifier = PDFClassifier()
        self.quality_gates = QualityGates()
        self.marker_cmd = marker_cmd or self._find_cmd("marker_single")
        self.ocrmypdf_cmd = ocrmypdf_cmd or self._find_cmd("ocrmypdf")
        self.qpdf_cmd = qpdf_cmd or self._find_cmd("qpdf")

    def _find_cmd(self, name: str) -> str | None:
        """Find command in PATH."""
        result = subprocess.run(["which", name], capture_output=True, text=True)
        return result.stdout.strip() or None

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Extract text from PDF with automatic routing and fallback."""
        pdf_path = Path(pdf_path)
        start_time = time.time()

        # Compute SHA256
        with open(pdf_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        # Classify
        classification = self.classifier.classify(pdf_path)
        if classification.error:
            return ExtractionResult(
                filename=pdf_path.name, sha256=sha256,
                doc_class="error", extractor_used="none", fallback_used=False,
                quality_passed=False, quality=None, markdown_text="",
                processing_time_sec=time.time() - start_time,
                error=classification.error,
            )

        # Decrypt if needed
        if classification.encrypted:
            decrypted = self._decrypt(pdf_path)
            if not decrypted:
                return ExtractionResult(
                    filename=pdf_path.name, sha256=sha256,
                    doc_class="encrypted", extractor_used="none", fallback_used=False,
                    quality_passed=False, quality=None, markdown_text="",
                    processing_time_sec=time.time() - start_time,
                    error="decryption_failed",
                )
            pdf_path = decrypted
            classification = self.classifier.classify(pdf_path)

        # Route to primary extractor
        doc_class = classification.doc_class
        if doc_class in ("digital_text", "scanned_with_ocr", "digital_text_risky"):
            text, extractor = self._extract_pymupdf(pdf_path)
        elif doc_class == "scanned_no_ocr":
            text, extractor = self._extract_marker(pdf_path)
        else:  # uncertain
            text, extractor = self._extract_marker(pdf_path)

        if not text:
            return ExtractionResult(
                filename=pdf_path.name, sha256=sha256, doc_class=doc_class,
                extractor_used=extractor, fallback_used=False,
                quality_passed=False, quality=None, markdown_text="",
                processing_time_sec=time.time() - start_time,
                error=f"{extractor}_failed",
            )

        # Quality gate
        quality = self.quality_gates.evaluate(text)
        fallback_used = False

        # Fallback if quality fails
        if not quality.passed and doc_class in ("digital_text_risky", "scanned_with_ocr"):
            logger.info(f"Quality failed for {pdf_path.name}, trying marker fallback")
            fallback_text, fallback_extractor = self._extract_marker(pdf_path)
            if fallback_text:
                fallback_quality = self.quality_gates.evaluate(fallback_text)
                if fallback_quality.passed or fallback_quality.char_count > quality.char_count:
                    text = fallback_text
                    extractor = fallback_extractor
                    quality = fallback_quality
                    fallback_used = True

        return ExtractionResult(
            filename=pdf_path.name, sha256=sha256, doc_class=doc_class,
            extractor_used=extractor, fallback_used=fallback_used,
            quality_passed=quality.passed, quality=quality,
            markdown_text=text,
            processing_time_sec=time.time() - start_time,
        )

    def _decrypt(self, pdf_path: Path) -> Path | None:
        """Decrypt PDF with qpdf (assumes no password)."""
        if not self.qpdf_cmd:
            logger.warning("qpdf not found, cannot decrypt")
            return None
        try:
            temp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp.close()
            result = subprocess.run(
                [self.qpdf_cmd, "--decrypt", str(pdf_path), temp.name],
                capture_output=True, timeout=60,
            )
            if result.returncode == 0:
                return Path(temp.name)
            return None
        except Exception as e:
            logger.warning(f"Decrypt failed: {e}")
            return None

    def _extract_pymupdf(self, pdf_path: Path) -> tuple[str, str]:
        """Extract with pymupdf (fitz)."""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            return text, "pymupdf"
        except Exception as e:
            logger.warning(f"pymupdf failed for {pdf_path.name}: {e}")
            return "", "pymupdf"

    def _extract_marker(self, pdf_path: Path) -> tuple[str, str]:
        """Extract with marker_single CLI."""
        if not self.marker_cmd:
            logger.warning("marker_single not found")
            return "", "marker"
        try:
            result = subprocess.run(
                [self.marker_cmd, str(pdf_path), "--output_format", "markdown"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                # marker outputs to same dir with .md extension
                md_path = pdf_path.with_suffix(".md")
                if md_path.exists():
                    text = md_path.read_text(encoding="utf-8")
                    md_path.unlink()  # cleanup
                    return text, "marker"
            return "", "marker"
        except Exception as e:
            logger.warning(f"marker failed for {pdf_path.name}: {e}")
            return "", "marker"
