"""
Stage 1 — Quality Gates for Extracted Markdown
===============================================
Validates Vietnamese legal textbook extraction quality using:
- Vietnamese diacritic ratio (tonal marks present)
- Legal anchor density (Điều, Khoản, Luật keywords)
- Empty page detection
- Minimum content length

Thresholds from configs/config.py ExtractionConfig.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GateResult:
    """Quality gate evaluation result."""
    passed: bool
    diacritic_ratio: float
    legal_anchor_density: float
    empty_page_rate: float
    char_count: int
    word_count: int
    recommendation: str
    details: dict

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "diacritic_ratio": round(self.diacritic_ratio, 4),
            "legal_anchor_density": round(self.legal_anchor_density, 5),
            "empty_page_rate": round(self.empty_page_rate, 4),
            "char_count": self.char_count,
            "word_count": self.word_count,
            "recommendation": self.recommendation,
        }


class QualityGates:
    """Vietnamese legal text quality validation."""

    # Vietnamese diacritics (tonal marks and special chars)
    VN_DIACRITICS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
                       "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ")

    # Legal anchors (Vietnamese legal document keywords)
    LEGAL_ANCHORS = [
        r"\bđiều\s+\d+",      # Điều 1, Điều 2...
        r"\bkhoản\s+\d+",     # Khoản 1, Khoản 2...
        r"\bđiểm\s+[a-z]\b",  # Điểm a, Điểm b...
        r"\bluật\s+\w+",      # Luật dân sự, Luật hình sự...
        r"\bnghị\s+định",     # Nghị định
        r"\bthông\s+tư",      # Thông tư
        r"\bquyết\s+định",    # Quyết định
        r"\bbộ\s+luật",       # Bộ luật
        r"\bhiến\s+pháp",     # Hiến pháp
        r"\bpháp\s+luật",     # Pháp luật
        r"\bquyền\b",         # Quyền
        r"\bnghĩa\s+vụ",      # Nghĩa vụ
        r"\bhợp\s+đồng",      # Hợp đồng
        r"\btài\s+sản",       # Tài sản
        r"\bchương\s+\d+",    # Chương 1, Chương 2...
        r"\bmục\s+\d+",       # Mục 1, Mục 2...
    ]

    def __init__(
        self,
        min_diacritic_ratio: float = 0.18,
        min_legal_anchor_density: float = 0.002,
        max_empty_page_rate: float = 0.05,
        min_chars: int = 500,
    ):
        """
        Args:
            min_diacritic_ratio: Min ratio of Vietnamese diacritics to alpha chars.
            min_legal_anchor_density: Min legal anchors per 1000 chars.
            max_empty_page_rate: Max fraction of near-empty pages (< 100 chars).
            min_chars: Minimum total character count.
        """
        self.min_diacritic_ratio = min_diacritic_ratio
        self.min_legal_anchor_density = min_legal_anchor_density
        self.max_empty_page_rate = max_empty_page_rate
        self.min_chars = min_chars
        self._legal_pattern = re.compile("|".join(self.LEGAL_ANCHORS), re.IGNORECASE)

    def evaluate(self, text: str, page_texts: list[str] | None = None) -> GateResult:
        """
        Evaluate quality gates on extracted text.

        Args:
            text: Full extracted text.
            page_texts: Optional list of per-page texts for empty page detection.

        Returns:
            GateResult with pass/fail and detailed scores.
        """
        if not text:
            return GateResult(
                passed=False, diacritic_ratio=0.0, legal_anchor_density=0.0,
                empty_page_rate=1.0, char_count=0, word_count=0,
                recommendation="no_content", details={"error": "empty text"},
            )

        char_count = len(text)
        word_count = len(text.split())
        alpha_chars = sum(1 for c in text if c.isalpha())

        # Diacritic ratio
        diacritic_count = sum(1 for c in text if c in self.VN_DIACRITICS)
        diacritic_ratio = diacritic_count / max(alpha_chars, 1)

        # Legal anchor density (per 1000 chars)
        legal_matches = len(self._legal_pattern.findall(text.lower()))
        legal_anchor_density = (legal_matches / max(char_count, 1)) * 1000

        # Empty page rate
        empty_page_rate = 0.0
        if page_texts:
            empty_pages = sum(1 for p in page_texts if len(p.strip()) < 100)
            empty_page_rate = empty_pages / max(len(page_texts), 1)

        # Gate checks
        details = {}
        failed_gates = []

        if char_count < self.min_chars:
            failed_gates.append("min_chars")
            details["min_chars_expected"] = self.min_chars

        if diacritic_ratio < self.min_diacritic_ratio:
            failed_gates.append("diacritic_ratio")
            details["diacritic_threshold"] = self.min_diacritic_ratio

        if legal_anchor_density < self.min_legal_anchor_density:
            failed_gates.append("legal_anchor_density")
            details["legal_anchor_threshold"] = self.min_legal_anchor_density

        if empty_page_rate > self.max_empty_page_rate:
            failed_gates.append("empty_page_rate")
            details["empty_page_threshold"] = self.max_empty_page_rate

        passed = len(failed_gates) == 0
        details["failed_gates"] = failed_gates

        # Recommendation
        if passed:
            recommendation = "accept"
        elif "diacritic_ratio" in failed_gates and diacritic_ratio < 0.05:
            recommendation = "ocr_fallback_needed"
        elif "min_chars" in failed_gates:
            recommendation = "extraction_failed"
        else:
            recommendation = "quality_marginal"

        return GateResult(
            passed=passed,
            diacritic_ratio=diacritic_ratio,
            legal_anchor_density=legal_anchor_density,
            empty_page_rate=empty_page_rate,
            char_count=char_count,
            word_count=word_count,
            recommendation=recommendation,
            details=details,
        )
