"""
Stage 1 — PDF Structural Classifier
=====================================
Classify PDFs by extraction strategy using raw-byte structural analysis.
No pymupdf/poppler dependency — works anywhere with stdlib only.

Classes:
    - digital_text:       fonts + text ops + ToUnicode → pymupdf direct
    - digital_text_risky: fonts + text ops, no ToUnicode → may garble, gate+fallback
    - scanned_no_ocr:     images, no text layer → OCR required
    - scanned_with_ocr:   images + text layer → OCR present, verify quality
    - encrypted:          /Encrypt present → decrypt first
    - uncertain:          ambiguous → default to marker
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClassificationResult:
    """Result of structural PDF classification."""
    filename: str
    doc_class: str
    ocr_need: str
    size_mb: float
    estimated_pages: int
    image_xobjects: int
    font_objects: int
    text_show_ops: int
    tounicode_maps: int
    encrypted: bool
    has_text_layer: bool
    has_tounicode: bool
    image_heavy: bool
    error: str | None = None
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "doc_class": self.doc_class,
            "ocr_need": self.ocr_need,
            "size_mb": self.size_mb,
            "estimated_pages": self.estimated_pages,
            "image_xobjects": self.image_xobjects,
            "font_objects": self.font_objects,
            "text_show_ops": self.text_show_ops,
            "tounicode_maps": self.tounicode_maps,
            "encrypted": self.encrypted,
            "has_text_layer": self.has_text_layer,
            "has_tounicode": self.has_tounicode,
            "image_heavy": self.image_heavy,
            "error": self.error,
        }


class PDFClassifier:
    """Structural PDF classifier using raw-byte heuristics."""

    def __init__(self, scan_bytes: int = 3_000_000, tail_bytes: int = 500_000):
        """
        Args:
            scan_bytes: How many bytes to read from PDF head (default 3MB).
            tail_bytes: How many bytes to read from PDF tail (xref/trailer).
        """
        self.scan_bytes = scan_bytes
        self.tail_bytes = tail_bytes

    def classify(self, pdf_path: Path) -> ClassificationResult:
        """Classify a single PDF by structure."""
        pdf_path = Path(pdf_path)
        try:
            size = pdf_path.stat().st_size
            with open(pdf_path, "rb") as f:
                head = f.read(min(self.scan_bytes, size))
                tail = b""
                if size > self.scan_bytes:
                    f.seek(max(0, size - self.tail_bytes))
                    tail = f.read(self.tail_bytes)
            raw = head + tail
            text = raw.decode("latin-1", errors="ignore")
            return self._analyze(pdf_path, size, text)
        except Exception as e:
            return ClassificationResult(
                filename=pdf_path.name, doc_class="error", ocr_need="error",
                size_mb=0.0, estimated_pages=0, image_xobjects=0, font_objects=0,
                text_show_ops=0, tounicode_maps=0, encrypted=False,
                has_text_layer=False, has_tounicode=False, image_heavy=False,
                error=str(e),
            )

    def _analyze(self, pdf_path: Path, size: int, text: str) -> ClassificationResult:
        """Apply structural heuristics to classify."""
        image_count = len(re.findall(r"/Subtype\s*/Image", text))
        font_count = len(re.findall(r"/Type\s*/Font", text))
        font_subtype = len(re.findall(r"/Subtype\s*/(TrueType|Type0|Type1|CIDFontType)", text))
        tj_count = len(re.findall(r"\bTj\b", text)) + len(re.findall(r"\bTJ\b", text))
        bt_count = len(re.findall(r"\bBT\b", text))
        tounicode = len(re.findall(r"/ToUnicode", text))
        encrypted = "/Encrypt" in text
        page_markers = len(re.findall(r"/Type\s*/Page\b", text))

        count_match = re.search(r"/Count\s+(\d+)", text)
        estimated_pages = int(count_match.group(1)) if count_match else max(page_markers, 1)

        size_mb = round(size / 1024 / 1024, 2)

        has_text_layer = (tj_count > 5 or bt_count > 5) and (font_count > 0 or font_subtype > 0)
        has_tounicode = tounicode > 0
        image_heavy = image_count >= max(2, estimated_pages * 0.3)

        # Decision logic
        if encrypted:
            doc_class, ocr_need = "encrypted", "decrypt_first"
        elif has_text_layer and has_tounicode:
            if image_heavy:
                doc_class, ocr_need = "scanned_with_ocr", "ocr_present_good"
            else:
                doc_class, ocr_need = "digital_text", "no_ocr_needed"
        elif has_text_layer and not has_tounicode:
            doc_class, ocr_need = "digital_text_risky", "extraction_may_garble"
        elif image_heavy and not has_text_layer:
            doc_class, ocr_need = "scanned_no_ocr", "ocr_required"
        elif font_count == 0 and font_subtype == 0 and tj_count == 0:
            doc_class, ocr_need = "scanned_no_ocr", "ocr_required"
        else:
            doc_class, ocr_need = "uncertain", "manual_check"

        return ClassificationResult(
            filename=pdf_path.name, doc_class=doc_class, ocr_need=ocr_need,
            size_mb=size_mb, estimated_pages=estimated_pages,
            image_xobjects=image_count, font_objects=font_count + font_subtype,
            text_show_ops=tj_count, tounicode_maps=tounicode, encrypted=encrypted,
            has_text_layer=has_text_layer, has_tounicode=has_tounicode,
            image_heavy=image_heavy,
        )
