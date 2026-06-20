import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_context_readiness as acr


class ContextReadinessTests(unittest.TestCase):
    def test_missing_diacritic_legal_text_is_flagged(self):
        text = "phap luat nha nuoc quy dinh quyen va nghia vu cua cong dan trong quan he xa hoi. " * 20
        scores, risks, _ = acr.score_context(text, {"chunk_id": "c1", "source_doc": "s.pdf", "content_hash": "h"}, 1)
        self.assertIn("possible_residual_missing_diacritics", risks)
        self.assertLess(scores["vietnamese_text_health"], 0.8)

    def test_clean_text_gets_eligible_lower_bloom_levels(self):
        text = (
            "Pháp luật là hệ thống quy tắc xử sự chung do Nhà nước ban hành hoặc thừa nhận. "
            "Nội dung này giải thích vai trò của pháp luật trong việc điều chỉnh quan hệ xã hội. "
            "Khi một chủ thể thực hiện quyền và nghĩa vụ, các điều kiện áp dụng cần được xem xét. "
        ) * 8
        scores, risks, _ = acr.score_context(text, {"chunk_id": "c2", "source_doc": "s.pdf", "content_hash": "h2"}, 1)
        quality = sum(scores[k] * acr.WEIGHTS[k] for k in acr.WEIGHTS)
        eligible, _, _ = acr.route_bloom(text, quality, risks)
        self.assertIn("Remember", eligible)
        self.assertIn("Apply", eligible)
        self.assertGreater(quality, 0.65)

    def test_front_matter_is_rejected(self):
        text = "Nhà xuất bản, chủ biên, tái bản, lời nói đầu, danh mục tài liệu tham khảo. " * 30
        scores, risks, review = acr.score_context(text, {"chunk_id": "c3", "source_doc": "s.pdf", "content_hash": "h3"}, 1)
        quality = sum(scores[k] * acr.WEIGHTS[k] for k in acr.WEIGHTS)
        eligible, _, _ = acr.route_bloom(text, quality, risks)
        self.assertIn("front_matter_or_bibliographic", risks)
        self.assertEqual(acr.tier_of(quality, risks, review, eligible), "reject")

    def test_spaced_ocr_fragmentation_is_rejected(self):
        text = "Tr ườ ng h ợ p ng ườ i ph ạ m t ộ i đ ã b ị k ế t án v ề t ộ i. " * 35
        scores, risks, review = acr.score_context(text, {"chunk_id": "c4", "source_doc": "s.pdf", "content_hash": "h4"}, 1)
        quality = sum(scores[k] * acr.WEIGHTS[k] for k in acr.WEIGHTS)
        eligible, _, _ = acr.route_bloom(text, quality, risks)
        self.assertIn("spaced_ocr_word_fragmentation", risks)
        self.assertEqual(acr.tier_of(quality, risks, review, eligible), "reject")


if __name__ == "__main__":
    unittest.main()
