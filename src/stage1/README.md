# BloomDepth Stage 1: PDF → Markdown Extraction Pipeline

Quality-routed extraction cho 118 giáo trình pháp luật Việt Nam.

## Quick Start

```bash
# 1. Kiểm tra môi trường
python -m src.stage1.check_env

# 2. Cài đặt dependencies
pip install -r src/stage1/requirements.txt

# 3. Chạy classification (dry-run)
python -m src.stage1.run --classify-only --input data/raw --output data/interim/extracted

# 4. Chạy full extraction
python -m src.stage1.run --input data/raw --output data/interim/extracted --workers 2
```

## Hoặc dùng Notebook

```bash
jupyter notebook notebooks/Stage1_Extraction_Pipeline.ipynb
```

## Test

```bash
# Với pytest
pytest tests/test_stage1.py -v

# Hoặc không cần pytest
python tests/run_stdlib_tests.py
```

## Documentation

- **Architecture:** [src/stage1/ARCHITECTURE.md](src/stage1/ARCHITECTURE.md)
- **Acceptance Report:** [src/stage1/ACCEPTANCE_REPORT.md](src/stage1/ACCEPTANCE_REPORT.md)
- **Notebook:** [notebooks/Stage1_Extraction_Pipeline.ipynb](notebooks/Stage1_Extraction_Pipeline.ipynb)

## Features

✓ Structural PDF classifier (stdlib only, no external deps)  
✓ Quality gates: Vietnamese diacritics + legal anchors  
✓ Auto-routing: digital text / scanned / encrypted  
✓ Fallback chain: pymupdf → marker → ocrmypdf  
✓ Parallel workers + checkpoint/resume  
✓ Real-time progress logs  

## Results (118 PDFs)

| Document Class | Count | Strategy |
|----------------|-------|----------|
| scanned_no_ocr | 43 | marker OCR |
| digital_text | 35 | pymupdf direct |
| digital_text_risky | 21 | pymupdf + fallback |
| uncertain | 9 | marker default |
| scanned_with_ocr | 7 | pymupdf + verify |
| encrypted | 3 | qpdf decrypt |

Tests: **16/16 PASS** ✓
