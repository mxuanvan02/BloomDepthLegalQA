"""
6-Level Bloom Taxonomy Classifier
===================================
Who:    LLM-as-Judge (Gemma-2-2b-it) or rule-based heuristics
Where:  BloomDepth/src/bloom_classifier.py
How:    Classifies a given question into one of 6 Bloom levels.
        Extended from Paper 1's 3-level classifier to full taxonomy.

Bloom Levels (ordered by cognitive complexity):
    1. Remember    — Factual recall
    2. Understand  — Explain, summarize
    3. Apply       — Use in new situation
    4. Analyze     — Break down, compare
    5. Evaluate    — Judge, justify
    6. Create      — Design, propose
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("bloom_depth.bloom_classifier")


# ─────────────────────────────────────────────
# Classification Prompt Template
# ─────────────────────────────────────────────
BLOOM_CLASSIFY_6_TEMPLATE = """\
Câu hỏi: {question}

NHIỆM VỤ: Phân loại câu hỏi này vào MỘT TRONG 6 cấp độ Bloom:

1. 'Remember': Hỏi trực tiếp về định nghĩa, số liệu, quy định cụ thể trong luật.
   (Từ khóa: "Theo Điều...", "Ai có thẩm quyền...", "Khi nào...")
2. 'Understand': Yêu cầu giải thích ý nghĩa, tóm tắt, hoặc diễn giải nội dung luật.
   (Từ khóa: "Giải thích...", "Nêu ý nghĩa...", "Tóm tắt...")
3. 'Apply': Đưa ra tình huống thực tế mới và hỏi cách áp dụng luật.
   (Từ khóa: "Trong tình huống...", "Áp dụng Điều X...", "A muốn...")
4. 'Analyze': Yêu cầu so sánh, phân biệt các khái niệm, tìm mối quan hệ.
   (Từ khóa: "So sánh...", "Phân biệt...", "Mối quan hệ giữa...")
5. 'Evaluate': Yêu cầu nhận xét, đánh giá tính hợp lý, phản biện.
   (Từ khóa: "Đánh giá...", "Nhận xét...", "Có hợp lý không...")
6. 'Create': Yêu cầu đề xuất, soạn thảo, thiết kế giải pháp pháp lý.
   (Từ khóa: "Hãy soạn thảo...", "Đề xuất...", "Thiết kế quy trình...")

CHỈ TRẢ VỀ MỘT TỪ DUY NHẤT: [Remember|Understand|Apply|Analyze|Evaluate|Create]
"""

VALID_BLOOM_LEVELS = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}


# ─────────────────────────────────────────────
# Rule-Based Heuristic Classifier
# ─────────────────────────────────────────────
# Vietnamese keyword patterns for each Bloom level (ordered by priority: highest first)
BLOOM_PATTERNS: list[tuple[str, list[str]]] = [
    ("Create", [
        r"soạn\s+thảo", r"đề\s+xuất", r"thiết\s+kế", r"xây\s+dựng\s+(?:phương|quy)",
        r"hãy\s+viết", r"hãy\s+lập", r"tạo\s+ra", r"đưa\s+ra\s+giải\s+pháp",
    ]),
    ("Evaluate", [
        r"đánh\s+giá", r"nhận\s+xét", r"có\s+hợp\s+lý", r"phản\s+biện",
        r"có\s+đúng\s+không", r"bạn\s+(?:đồng\s+ý|nghĩ\s+sao)", r"chứng\s+minh",
        r"lập\s+luận", r"biện\s+minh",
    ]),
    ("Analyze", [
        r"so\s+sánh", r"phân\s+biệt", r"phân\s+tích", r"mối\s+quan\s+hệ",
        r"điểm\s+(?:giống|khác)", r"nguyên\s+nhân", r"hệ\s+quả",
        r"tại\s+sao\s+(?:lại|có)", r"(?:giống|khác)\s+nhau",
    ]),
    ("Apply", [
        r"trong\s+tình\s+huống", r"(?:anh|chị|ông|bà|A|B)\s+(?:muốn|đã|cần)",
        r"áp\s+dụng\s+(?:điều|quy)", r"giải\s+quyết\s+(?:vụ|tình)",
        r"(?:nếu|khi)\s+.*?thì\s+(?:phải|cần|được)", r"xử\s+lý\s+(?:như|thế)",
    ]),
    ("Understand", [
        r"giải\s+thích", r"nêu\s+ý\s+nghĩa", r"tóm\s+tắt", r"diễn\s+giải",
        r"hiểu\s+(?:như\s+thế\s+nào|thế\s+nào)", r"(?:có\s+nghĩa|nghĩa)\s+là",
        r"trình\s+bày", r"mô\s+tả",
    ]),
    ("Remember", [
        r"theo\s+(?:điều|khoản|luật)", r"(?:ai|cơ\s+quan\s+nào)\s+có\s+thẩm\s+quyền",
        r"khi\s+nào", r"(?:bao\s+gồm|gồm\s+(?:có|những))",
        r"(?:quy\s+định|điều\s+kiện)\s+(?:nào|gì)",
        r"(?:liệt\s+kê|nêu\s+(?:tên|các))", r"là\s+gì",
    ]),
]

# Compile patterns
_COMPILED_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    (level, [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns])
    for level, patterns in BLOOM_PATTERNS
]


def classify_bloom_heuristic(question: str) -> str:
    """Classify a question into Bloom level using rule-based heuristics.

    Patterns are checked highest-first (Create → Remember).
    First match wins.

    Args:
        question: Vietnamese question text.

    Returns:
        Bloom level string. Defaults to "Remember" if no match.
    """
    for level, patterns in _COMPILED_PATTERNS:
        for pattern in patterns:
            if pattern.search(question):
                return level

    return "Remember"  # Default fallback


def classify_bloom_llm(
    question: str,
    llm_engine: Any = None,
) -> str:
    """Classify a question using LLM-as-classifier.

    Args:
        question: Vietnamese question text.
        llm_engine: Callable that takes prompt → returns text.

    Returns:
        Bloom level string.
    """
    if llm_engine is None:
        logger.warning("No LLM engine provided, falling back to heuristic.")
        return classify_bloom_heuristic(question)

    prompt = BLOOM_CLASSIFY_6_TEMPLATE.format(question=question)
    response = llm_engine(prompt).strip()

    # Extract the classification from response
    for level in VALID_BLOOM_LEVELS:
        if level.lower() in response.lower():
            return level

    # Fallback to heuristic
    logger.warning("LLM classification unrecognized ('%s'), using heuristic.", response)
    return classify_bloom_heuristic(question)
