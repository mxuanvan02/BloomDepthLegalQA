# Stage 1 Extraction Pipeline — Architecture Design

## Overview
Quality-routed PDF→Markdown extraction for 118 Vietnamese legal textbooks.
Handles digital text, scanned PDFs (OCR-needed), encrypted files, and quality validation.

## Decision Tree

```
PDF → [Decrypt if encrypted]
    → [Classify: digital_text | scanned_no_ocr | scanned_with_ocr | risky | uncertain]
    → Route to extractor:
        digital_text      → pymupdf (fast, native)
        scanned_no_ocr    → TQA/marker (force OCR) → OCRmyPDF fallback
        scanned_with_ocr  → pymupdf → quality gate → marker fallback if bad
        risky/uncertain   → pymupdf → quality gate → marker fallback
    → Quality gates (diacritic ratio, legal anchors, empty pages)
    → Output: Markdown + metadata
```

## Modules

1. **classifier.py** — PDFClassifier
   - Input: PDF path
   - Output: {doc_class, ocr_need, confidence, signals}
   - Method: structural analysis (fonts, images, text ops, ToUnicode)

2. **quality_gates.py** — QualityGates
   - Input: extracted Markdown text
   - Output: {passed, scores, recommendation}
   - Gates: diacritic_ratio ≥ 0.18, legal_anchor_density ≥ 0.002, empty_page_rate ≤ 0.05

3. **router.py** — ExtractionRouter
   - Input: PDF path + classification
   - Output: Markdown text + metadata
   - Logic: route to extractor, run fallback if quality fails

4. **pipeline.py** — Stage1Pipeline
   - Input: data/raw/ directory
   - Output: data/interim/extracted/ (markdown + metadata JSONL)
   - Features: parallel workers, progress logging, resume from checkpoint

## Extractors (external dependencies)

- **pymupdf**: native text extraction (fitz module)
- **TQA/marker**: OCR-based extraction (assumes marker CLI available)
- **OCRmyPDF + Tesseract**: transparent QC fallback (vie+eng)
- **qpdf**: decrypt encrypted PDFs (assumes no password)

## Quality Gates

From configs/config.py ExtractionConfig:
- min_vn_diacritic_ratio: 0.18
- min_legal_anchor_density: 0.002  
- max_empty_page_rate: 0.05

## Outputs

```
data/interim/extracted/
  markdown/
    {sha256[:8]}_{filename}.md
  metadata.jsonl   # one record per PDF
  errors.log
  checkpoint.json  # resume state
```

Metadata schema:
```json
{
  "filename": "GT_Luat_Dan_Su.pdf",
  "sha256": "abc123...",
  "doc_class": "digital_text",
  "extractor_used": "pymupdf",
  "quality_passed": true,
  "quality_scores": {...},
  "text_length": 125000,
  "processing_time_sec": 12.5,
  "error": null
}
```
