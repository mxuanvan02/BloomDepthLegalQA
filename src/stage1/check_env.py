"""
Stage 1 — Environment Checker
===============================
Verifies all dependencies are available before running the pipeline.
Run this first on any new machine.

Usage:
    python -m src.stage1.check_env
    python src/stage1/check_env.py
"""

from __future__ import annotations

import shutil
import sys


def check_python_version() -> tuple[bool, str]:
    v = sys.version_info
    ok = v >= (3, 10)
    return ok, f"Python {v.major}.{v.minor}.{v.micro} {'✓' if ok else '✗ (need >=3.10)'}"


def check_pymupdf() -> tuple[bool, str]:
    try:
        import fitz
        return True, f"PyMuPDF {fitz.__doc__.split()[1] if fitz.__doc__ else '?'} ✓"
    except ImportError:
        return False, "PyMuPDF ✗ — pip install PyMuPDF"


def check_marker() -> tuple[bool, str]:
    cmd = shutil.which("marker_single") or shutil.which("marker")
    if cmd:
        return True, f"marker ✓ ({cmd})"
    return False, "marker ✗ (optional) — pip install marker-pdf"


def check_ocrmypdf() -> tuple[bool, str]:
    cmd = shutil.which("ocrmypdf")
    if cmd:
        return True, f"ocrmypdf ✓ ({cmd})"
    return False, "ocrmypdf ✗ (optional) — pip install ocrmypdf"


def check_tesseract() -> tuple[bool, str]:
    cmd = shutil.which("tesseract")
    if cmd:
        return True, f"tesseract ✓ ({cmd})"
    return False, "tesseract ✗ (optional) — apt install tesseract-ocr tesseract-ocr-vie"


def check_qpdf() -> tuple[bool, str]:
    cmd = shutil.which("qpdf")
    if cmd:
        return True, f"qpdf ✓ ({cmd})"
    return False, "qpdf ✗ (optional, for encrypted PDFs) — apt install qpdf"


def check_pytest() -> tuple[bool, str]:
    try:
        import pytest
        return True, f"pytest {pytest.__version__} ✓"
    except ImportError:
        return False, "pytest ✗ — pip install pytest"


def main() -> int:
    print("=" * 60)
    print("BloomDepth Stage 1 — Environment Check")
    print("=" * 60)

    checks = [
        ("Python", check_python_version),
        ("PyMuPDF", check_pymupdf),
        ("marker", check_marker),
        ("ocrmypdf", check_ocrmypdf),
        ("tesseract", check_tesseract),
        ("qpdf", check_qpdf),
        ("pytest", check_pytest),
    ]

    required = {"Python", "PyMuPDF"}
    all_ok = True
    missing_required = []

    for name, fn in checks:
        ok, msg = fn()
        tag = "REQUIRED" if name in required else "optional"
        print(f"  [{tag:8}] {msg}")
        if not ok and name in required:
            missing_required.append(name)
            all_ok = False

    print()
    if missing_required:
        print(f"✗ BLOCKED: Missing required: {', '.join(missing_required)}")
        print("  Install with: pip install -r src/stage1/requirements.txt")
        return 1
    else:
        print("✓ Ready to run Stage 1 pipeline")
        if not check_marker()[0]:
            print("  NOTE: marker not found — scanned PDFs will fail extraction.")
            print("        Install for full coverage: pip install marker-pdf")
        return 0


if __name__ == "__main__":
    sys.exit(main())
