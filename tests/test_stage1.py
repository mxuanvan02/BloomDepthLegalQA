"""
Tests for Stage 1 Extraction Pipeline
======================================
Unit tests for classifier, quality_gates, router.
Integration test with synthetic PDFs.
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import pytest

# Ensure project root is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage1.classifier import PDFClassifier, ClassificationResult
from src.stage1.quality_gates import QualityGates, GateResult


# ─────────────────────────────────────────────
# Fixtures: synthetic PDF files
# ─────────────────────────────────────────────

def _make_minimal_pdf(content: bytes) -> bytes:
    """Create minimal valid PDF with embedded content for testing."""
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
    )
    pdf += content
    pdf += (
        b">>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000108 00000 n \n"
        b"trailer<</Root 1 0 R/Size 4>>\n"
        b"startxref\n0\n%%EOF"
    )
    return pdf


@pytest.fixture
def digital_pdf(tmp_path) -> Path:
    """PDF with fonts + text ops + ToUnicode (digital text)."""
    content = (
        b"/Contents 4 0 R"
        b">>endobj\n"
        b"4 0 obj<</Length 50>>stream\n"
        b"BT /F1 12 Tf (Hello World) Tj ET\n"
        b"BT /F1 12 Tf (Test text) Tj ET\n"
        b"BT /F1 12 Tf (More text) Tj ET\n"
        b"BT /F1 12 Tf (Content) Tj ET\n"
        b"BT /F1 12 Tf (Another) Tj ET\n"
        b"BT /F1 12 Tf (Final line) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj<</Type/Font/Subtype/TrueType/ToUnicode 6 0 R>>endobj\n"
        b"6 0 obj<</Length 5>>stream\ntest\nendstream endobj\n"
    )
    # Build full PDF
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/" + content +
        b"xref\n0 7\n"
        b"trailer<</Root 1 0 R/Size 7>>\nstartxref\n0\n%%EOF"
    )
    path = tmp_path / "digital.pdf"
    path.write_bytes(pdf)
    return path


@pytest.fixture
def scanned_pdf(tmp_path) -> Path:
    """PDF with images, no fonts/text (scan without OCR)."""
    content = (
        b"/Resources<</XObject<</Im0 4 0 R/Im1 5 0 R/Im2 6 0 R>>>>"
        b">>endobj\n"
        b"4 0 obj<</Type/XObject/Subtype/Image/Width 100/Height 100>>endobj\n"
        b"5 0 obj<</Type/XObject/Subtype/Image/Width 100/Height 100>>endobj\n"
        b"6 0 obj<</Type/XObject/Subtype/Image/Width 100/Height 100>>endobj\n"
    )
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/" + content +
        b"xref\n0 7\n"
        b"trailer<</Root 1 0 R/Size 7>>\nstartxref\n0\n%%EOF"
    )
    path = tmp_path / "scanned.pdf"
    path.write_bytes(pdf)
    return path


@pytest.fixture
def encrypted_pdf(tmp_path) -> Path:
    """PDF with /Encrypt marker."""
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R>>endobj\n"
        b"4 0 obj<</Type/Encrypt/Filter/Standard>>endobj\n"
        b"xref\n0 5\n"
        b"trailer<</Root 1 0 R/Size 5/Encrypt 4 0 R>>\nstartxref\n0\n%%EOF"
    )
    path = tmp_path / "encrypted.pdf"
    path.write_bytes(pdf)
    return path


# ─────────────────────────────────────────────
# Tests: PDFClassifier
# ─────────────────────────────────────────────

class TestPDFClassifier:
    """Tests for PDFClassifier."""

    def test_classify_digital_text(self, digital_pdf):
        c = PDFClassifier()
        result = c.classify(digital_pdf)
        assert result.doc_class == "digital_text"
        assert result.ocr_need == "no_ocr_needed"
        assert result.has_text_layer is True
        assert result.has_tounicode is True
        assert result.encrypted is False

    def test_classify_scanned_no_ocr(self, scanned_pdf):
        c = PDFClassifier()
        result = c.classify(scanned_pdf)
        assert result.doc_class == "scanned_no_ocr"
        assert result.ocr_need == "ocr_required"
        assert result.has_text_layer is False
        assert result.image_heavy is True

    def test_classify_encrypted(self, encrypted_pdf):
        c = PDFClassifier()
        result = c.classify(encrypted_pdf)
        assert result.doc_class == "encrypted"
        assert result.encrypted is True

    def test_classify_nonexistent(self, tmp_path):
        c = PDFClassifier()
        result = c.classify(tmp_path / "nonexistent.pdf")
        assert result.doc_class == "error"
        assert result.error is not None

    def test_to_dict(self, digital_pdf):
        c = PDFClassifier()
        result = c.classify(digital_pdf)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "doc_class" in d
        assert "filename" in d


# ─────────────────────────────────────────────
# Tests: QualityGates
# ─────────────────────────────────────────────

class TestQualityGates:
    """Tests for QualityGates."""

    def test_good_vietnamese_text(self):
        # Vietnamese legal text with diacritics and anchors
        text = """
        Chương 1: Những quy định chung về pháp luật dân sự Việt Nam
        Điều 1. Phạm vi điều chỉnh
        Bộ luật này quy định địa vị pháp lý, chuẩn mực pháp lý về cách ứng xử
        của cá nhân, pháp nhân; quyền, nghĩa vụ về nhân thân và tài sản.
        Điều 2. Công nhận, tôn trọng, bảo vệ và bảo đảm quyền dân sự
        Khoản 1. Ở nước Cộng hòa xã hội chủ nghĩa Việt Nam, các quyền dân sự
        được công nhận, tôn trọng, bảo vệ và bảo đảm theo Hiến pháp và pháp luật.
        Điều 3. Các nguyên tắc cơ bản
        Khoản 2. Cá nhân, pháp nhân xác lập, thực hiện, chấm dứt quyền.
        Điều 4. Áp dụng Bộ luật dân sự
        Khoản 3. Trường hợp luật khác có liên quan không quy định.
        """ * 5  # Repeat for length

        gates = QualityGates()
        result = gates.evaluate(text)
        assert result.passed is True
        assert result.diacritic_ratio > 0.18
        assert result.legal_anchor_density > 0.002
        assert result.recommendation == "accept"

    def test_empty_text(self):
        gates = QualityGates()
        result = gates.evaluate("")
        assert result.passed is False
        assert result.recommendation == "no_content"

    def test_garbled_no_diacritics(self):
        # ASCII-only text (OCR failed on diacritics)
        text = "Dieu 1. Pham vi dieu chinh. " * 100
        gates = QualityGates()
        result = gates.evaluate(text)
        assert result.passed is False
        assert result.diacritic_ratio < 0.05
        assert result.recommendation == "ocr_fallback_needed"

    def test_non_legal_text(self):
        # Vietnamese but no legal anchors
        text = "Hôm nay trời đẹp quá, chúng tôi đi dạo trong công viên. " * 50
        gates = QualityGates(min_legal_anchor_density=0.002)
        result = gates.evaluate(text)
        assert result.passed is False
        assert "legal_anchor_density" in result.details.get("failed_gates", [])

    def test_page_empty_rate(self):
        text = "Điều 1. Nội dung luật pháp luật dân sự Việt Nam. " * 100
        pages = ["Nội dung trang " * 50] * 8 + ["", ""]  # 20% empty
        gates = QualityGates(max_empty_page_rate=0.05)
        result = gates.evaluate(text, page_texts=pages)
        assert result.passed is False
        assert "empty_page_rate" in result.details.get("failed_gates", [])

    def test_to_dict(self):
        gates = QualityGates()
        result = gates.evaluate("Điều 1. Pháp luật Việt Nam. " * 100)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "passed" in d
        assert "diacritic_ratio" in d


# ─────────────────────────────────────────────
# Tests: ExtractionRouter (unit, no real extraction)
# ─────────────────────────────────────────────

class TestExtractionRouter:
    """Test router logic (mocked extractors)."""

    def test_router_classifies_and_routes(self, digital_pdf):
        """Test that router correctly classifies PDF."""
        from src.stage1.router import ExtractionRouter
        router = ExtractionRouter()
        # The result may fail (no pymupdf in test env) but should classify
        result = router.extract(digital_pdf)
        assert result.filename == "digital.pdf"
        assert result.sha256  # Should compute hash
        assert result.doc_class == "digital_text"

    def test_router_encrypted(self, encrypted_pdf):
        from src.stage1.router import ExtractionRouter
        router = ExtractionRouter()
        result = router.extract(encrypted_pdf)
        # Should detect encryption, fail gracefully (no qpdf)
        assert result.doc_class == "encrypted"
        assert result.error is not None or result.markdown_text == ""


# ─────────────────────────────────────────────
# Integration test: Pipeline dry-run
# ─────────────────────────────────────────────

class TestPipelineIntegration:
    """Integration test with synthetic PDFs."""

    def test_pipeline_processes_directory(self, tmp_path, digital_pdf, scanned_pdf):
        """Pipeline should process PDFs and produce output."""
        from src.stage1.pipeline import PipelineConfig, Stage1Pipeline

        # Copy test PDFs to input dir
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "digital.pdf").write_bytes(digital_pdf.read_bytes())
        (input_dir / "scanned.pdf").write_bytes(scanned_pdf.read_bytes())

        output_dir = tmp_path / "output"

        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            n_workers=1,
            skip_existing=False,
        )
        pipeline = Stage1Pipeline(config)
        results = pipeline.run()

        # Should process both
        assert len(results) == 2
        # Checkpoint should exist
        assert (output_dir / "checkpoint.json").exists()
        # Metadata should exist
        assert (output_dir / "metadata.jsonl").exists()
