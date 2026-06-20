"""Standalone test runner — no pytest/pymupdf needed. Validates pure-stdlib logic."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage1.classifier import PDFClassifier
from src.stage1.quality_gates import QualityGates

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS | {name}")
    else:
        FAIL += 1
        print(f"  FAIL | {name}")

import tempfile
tmp = Path(tempfile.mkdtemp())

# --- Classifier: digital text PDF ---
digital = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
           b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Count 1>>endobj\n"
           b"4 0 obj<</Length 50>>stream\n"
           b"BT /F1 12 Tf (Hello) Tj ET\nBT (Test) Tj ET\nBT (More) Tj ET\n"
           b"BT (Content) Tj ET\nBT (Line) Tj ET\nBT (End) Tj ET\nendstream endobj\n"
           b"5 0 obj<</Type/Font/Subtype/TrueType/ToUnicode 6 0 R>>endobj\n"
           b"trailer<</Root 1 0 R>>\n%%EOF")
p = tmp / "digital.pdf"; p.write_bytes(digital)
c = PDFClassifier()
r = c.classify(p)
check("digital_text classification", r.doc_class == "digital_text")
check("digital has_text_layer", r.has_text_layer is True)
check("digital has_tounicode", r.has_tounicode is True)

# --- Classifier: scanned PDF (images, no fonts) ---
scanned = (b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n"
           b"3 0 obj<</Type/Page/Count 1>>endobj\n"
           b"4 0 obj<</Subtype/Image/Width 100>>endobj\n"
           b"5 0 obj<</Subtype/Image/Width 100>>endobj\n"
           b"6 0 obj<</Subtype/Image/Width 100>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF")
p = tmp / "scan.pdf"; p.write_bytes(scanned)
r = c.classify(p)
check("scanned_no_ocr classification", r.doc_class == "scanned_no_ocr")
check("scanned ocr_required", r.ocr_need == "ocr_required")

# --- Classifier: encrypted ---
enc = (b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n"
       b"4 0 obj<</Type/Encrypt/Filter/Standard>>endobj\n"
       b"trailer<</Root 1 0 R/Encrypt 4 0 R>>\n%%EOF")
p = tmp / "enc.pdf"; p.write_bytes(enc)
r = c.classify(p)
check("encrypted detection", r.encrypted is True and r.doc_class == "encrypted")

# --- Classifier: error on missing file ---
r = c.classify(tmp / "nope.pdf")
check("error on missing file", r.doc_class == "error" and r.error is not None)

# --- QualityGates: good Vietnamese legal text ---
g = QualityGates()
good = ("Chương 1: Những quy định chung về pháp luật dân sự Việt Nam. "
        "Điều 1. Phạm vi điều chỉnh. Bộ luật này quy định địa vị pháp lý. "
        "Điều 2. Quyền dân sự được công nhận theo Hiến pháp và pháp luật. "
        "Khoản 1. Cá nhân, pháp nhân xác lập quyền và nghĩa vụ về tài sản. "
        "Điều 3. Nguyên tắc cơ bản của hợp đồng dân sự. ") * 10
res = g.evaluate(good)
check("good VN text passes", res.passed is True)
check("good VN diacritic ratio > 0.18", res.diacritic_ratio > 0.18)
check("good VN legal density > 0.002", res.legal_anchor_density > 0.002)
check("good VN recommendation=accept", res.recommendation == "accept")

# --- QualityGates: empty ---
res = g.evaluate("")
check("empty text fails", res.passed is False and res.recommendation == "no_content")

# --- QualityGates: garbled (no diacritics) ---
res = g.evaluate("Dieu 1. Pham vi dieu chinh phap luat dan su. " * 100)
check("garbled fails", res.passed is False)
check("garbled low diacritic", res.diacritic_ratio < 0.05)
check("garbled recommends ocr", res.recommendation == "ocr_fallback_needed")

# --- QualityGates: empty page rate ---
text = "Điều 1. Nội dung pháp luật dân sự Việt Nam quy định. " * 100
pages = ["Nội dung trang pháp luật " * 30] * 8 + ["", ""]
res = g.evaluate(text, page_texts=pages)
check("empty page rate detected", "empty_page_rate" in res.details.get("failed_gates", []))

print(f"\n{'='*50}\nRESULTS: {PASS} passed, {FAIL} failed\n{'='*50}")
sys.exit(1 if FAIL else 0)
