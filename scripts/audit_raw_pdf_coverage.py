#!/usr/bin/env python3
"""
Audit raw PDF coverage: classify into law taxonomy and report gaps.

Usage:
    python scripts/audit_raw_pdf_coverage.py --output research/results/corpus_coverage_audit.json
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)


# Law domain taxonomy with Vietnamese/English keywords
LAW_TAXONOMY = {
    "foundation_theory": {
        "name": "Lý luận nền tảng / Foundation & Theory",
        "keywords": [
            "lý luận", "ly luan", "lịch sử", "lich su", "luật học so sánh", "luat hoc so sanh",
            "comparative", "la mã", "la ma", "roman", "xây dựng văn bản", "xay dung van ban",
            "quyền con người", "quyen con nguoi", "human rights", "thực hành nghề", "thuc hanh nghe",
            "legal profession", "đạo đức", "dao duc", "ethics"
        ]
    },
    "constitutional": {
        "name": "Luật Hiến pháp / Constitutional Law",
        "keywords": [
            "hiến pháp", "hien phap", "constitutional", "hiến chính", "hien chinh"
        ]
    },
    "administrative": {
        "name": "Luật Hành chính / Administrative Law",
        "keywords": [
            "hành chính", "hanh chinh", "administrative", "tố tụng hành chính", "to tung hanh chinh",
            "cán bộ công chức", "can bo cong chuc", "civil servant", "khiếu nại", "khieu nai",
            "complaint", "tố cáo", "to cao", "thanh tra", "inspection", "kiểm toán", "kiem toan", "audit"
        ]
    },
    "criminal": {
        "name": "Luật Hình sự / Criminal Law",
        "keywords": [
            "hình sự", "hinh su", "criminal", "tội phạm", "toi pham", "crime",
            "điều tra", "dieu tra", "investigation", "khoa học điều tra", "forensic",
            "tâm lý học tư pháp", "tam ly hoc tu phap", "legal psychology"
        ]
    },
    "civil": {
        "name": "Luật Dân sự / Civil Law",
        "keywords": [
            "dân sự", "dan su", "civil law", "tố tụng dân sự", "to tung dan su",
            "hôn nhân", "hon nhan", "marriage", "gia đình", "gia dinh", "family",
            "thừa kế", "thua ke", "inheritance", "hợp đồng dân sự", "hop dong dan su",
            "bồi thường", "boi thuong", "compensation", "tài sản", "tai san", "property",
            "công chứng", "cong chung", "notary", "chứng thực", "chung thuc", "authentication",
            "hộ tịch", "ho tich", "civil status", "thi hành án dân sự", "thi hanh an dan su"
        ]
    },
    "commercial": {
        "name": "Luật Thương mại / Commercial Law",
        "keywords": [
            "thương mại", "thuong mai", "commercial", "trade", "doanh nghiệp", "doanh nghiep",
            "enterprise", "đầu tư", "dau tu", "investment", "cạnh tranh", "canh tranh",
            "competition", "chứng khoán", "chung khoan", "securities", "phá sản", "pha san",
            "bankruptcy", "bảo hiểm", "bao hiem", "insurance", "trọng tài", "trong tai",
            "arbitration", "hòa giải", "hoa giai", "mediation", "wto"
        ]
    },
    "finance_tax": {
        "name": "Tài chính, Thuế, Ngân hàng / Finance, Tax, Banking",
        "keywords": [
            "thuế", "thue", "tax", "ngân sách", "ngan sach", "budget", "ngân hàng", "ngan hang",
            "banking", "tài chính", "tai chinh", "finance", "kế toán", "ke toan", "accounting"
        ]
    },
    "land_environment": {
        "name": "Đất đai, Môi trường / Land & Environment",
        "keywords": [
            "đất đai", "dat dai", "land", "bất động sản", "bat dong san", "real estate",
            "môi trường", "moi truong", "environment", "tài nguyên", "tai nguyen", "resource",
            "khoáng sản", "khoang san", "mineral", "lâm nghiệp", "lam nghiep", "forestry",
            "biển", "bien", "marine", "hải đảo", "hai dao", "năng lượng", "nang luong", "energy"
        ]
    },
    "labor_social": {
        "name": "Lao động, Xã hội / Labor & Social Welfare",
        "keywords": [
            "lao động", "lao dong", "labor", "labour", "công đoàn", "cong doan", "trade union",
            "việc làm", "viec lam", "employment", "bảo hiểm xã hội", "bao hiem xa hoi",
            "social insurance", "an sinh", "social welfare", "bình đẳng giới", "binh dang gioi",
            "gender equality", "trẻ em", "tre em", "children", "người cao tuổi", "nguoi cao tuoi",
            "elderly", "người khuyết tật", "nguoi khuyet tat", "disability"
        ]
    },
    "international": {
        "name": "Luật Quốc tế / International Law",
        "keywords": [
            "quốc tế", "quoc te", "international", "công pháp quốc tế", "cong phap quoc te",
            "public international", "tư pháp quốc tế", "tu phap quoc te", "private international",
            "kinh tế quốc tế", "kinh te quoc te", "international economic", "liên hợp quốc",
            "lien hop quoc", "united nations", "điều ước", "dieu uoc", "treaty",
            "asean", "conflict of laws"
        ]
    },
    "ip": {
        "name": "Sở hữu trí tuệ / Intellectual Property",
        "keywords": [
            "sở hữu trí tuệ", "so huu tri tue", "intellectual property", "ip",
            "thương mại hóa", "thuong mai hoa", "commercialization"
        ]
    },
    "technology": {
        "name": "Công nghệ, Mạng, Dữ liệu / Technology, Cyber, Data",
        "keywords": [
            "công nghệ thông tin", "cong nghe thong tin", "information technology",
            "an ninh mạng", "an ninh mang", "cybersecurity", "dữ liệu cá nhân", "du lieu ca nhan",
            "personal data", "thương mại điện tử", "thuong mai dien tu", "e-commerce",
            "fintech", "tài sản số", "tai san so", "digital asset", "blockchain"
        ]
    },
    "health_education": {
        "name": "Y tế, Giáo dục / Health & Education",
        "keywords": [
            "y tế", "y te", "health", "dược", "duoc", "pharmaceutical", "pháp y", "phap y",
            "forensic medicine", "giáo dục", "giao duc", "education", "khoa học công nghệ",
            "khoa hoc cong nghe", "science and technology"
        ]
    },
    "media_culture": {
        "name": "Truyền thông, Văn hóa / Media & Culture",
        "keywords": [
            "báo chí", "bao chi", "press", "xuất bản", "xuat ban", "publishing",
            "truyền thông", "truyen thong", "media", "quảng cáo", "quang cao", "advertising",
            "văn hóa", "van hoa", "culture", "di sản", "di san", "heritage",
            "thể thao", "the thao", "sports", "du lịch", "du lich", "tourism"
        ]
    },
    "transport_construction": {
        "name": "Giao thông, Xây dựng / Transport & Construction",
        "keywords": [
            "giao thông", "giao thong", "transport", "vận tải", "van tai", "hàng không",
            "hang khong", "aviation", "hàng hải", "hang hai", "maritime", "shipping",
            "xây dựng", "xay dung", "construction", "quy hoạch", "quy hoach", "planning",
            "nhà ở", "nha o", "housing"
        ]
    },
    "agriculture": {
        "name": "Nông nghiệp, Thực phẩm / Agriculture & Food",
        "keywords": [
            "nông nghiệp", "nong nghiep", "agriculture", "thực phẩm", "thuc pham", "food",
            "an toàn thực phẩm", "an toan thuc pham", "food safety", "thú y", "thu y", "veterinary"
        ]
    },
    "legal_english": {
        "name": "Tiếng Anh pháp lý / Legal English",
        "keywords": [
            "legal english", "tiếng anh pháp lý", "tieng anh phap ly"
        ]
    },
    "other_specialized": {
        "name": "Chuyên ngành khác / Other Specialized",
        "keywords": []  # Catch-all for unclassified
    }
}


# Prefix mapping for gap_2026 downloads (filename starts with law_domain key from discovery)
GAP_PREFIX_MAP = {
    "technology_cyber": "technology",
    "health_education": "health_education",
    "media_culture": "media_culture",
    "agriculture_food": "agriculture",
    "legal_english": "legal_english",
    "transport_constructi": "transport_construction",
    "transport_construction": "transport_construction",
}


def classify_pdf_by_filename(filename: str) -> List[str]:
    """Classify PDF into one or more taxonomy categories by filename."""
    filename_lower = filename.lower()

    # First: check gap_2026 download prefix (authoritative for curated downloads)
    for prefix, category_id in GAP_PREFIX_MAP.items():
        if filename_lower.startswith(prefix.lower()):
            return [category_id]

    matched_categories = []

    for category_id, category_info in LAW_TAXONOMY.items():
        keywords = category_info["keywords"]
        if any(kw in filename_lower for kw in keywords):
            matched_categories.append(category_id)

    # Default to other_specialized if no match
    if not matched_categories:
        matched_categories.append("other_specialized")

    return matched_categories


def audit_corpus(data_raw_dir: Path) -> Dict:
    """Audit all PDFs in data/raw and classify by taxonomy."""
    
    # Find all PDFs
    all_pdfs = list(data_raw_dir.rglob('*.pdf'))
    
    logger.info(f"Found {len(all_pdfs)} PDFs in {data_raw_dir}")
    
    # Classify each PDF
    corpus_by_category = defaultdict(list)
    pdf_records = []
    
    for pdf_path in all_pdfs:
        relative_path = pdf_path.relative_to(data_raw_dir)
        source_dir = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        
        categories = classify_pdf_by_filename(pdf_path.name)
        
        record = {
            "filename": pdf_path.name,
            "relative_path": str(relative_path),
            "source": source_dir,
            "categories": categories,
            "size_mb": pdf_path.stat().st_size / 1024 / 1024,
        }
        
        pdf_records.append(record)
        
        for cat in categories:
            corpus_by_category[cat].append(pdf_path.name)
    
    # Build coverage summary
    category_counts = {cat: len(pdfs) for cat, pdfs in corpus_by_category.items()}
    
    # Identify gaps (categories with 0 or very few PDFs)
    gap_threshold = 2
    gaps = []
    well_covered = []
    moderate_covered = []
    
    for category_id, category_info in LAW_TAXONOMY.items():
        count = category_counts.get(category_id, 0)
        
        if count == 0:
            gaps.append({
                "category_id": category_id,
                "name": category_info["name"],
                "count": 0,
                "severity": "critical"
            })
        elif count < gap_threshold:
            gaps.append({
                "category_id": category_id,
                "name": category_info["name"],
                "count": count,
                "severity": "moderate"
            })
        elif count < 5:
            moderate_covered.append({
                "category_id": category_id,
                "name": category_info["name"],
                "count": count
            })
        else:
            well_covered.append({
                "category_id": category_id,
                "name": category_info["name"],
                "count": count
            })
    
    # Sort by count
    gaps.sort(key=lambda x: x['count'])
    well_covered.sort(key=lambda x: x['count'], reverse=True)
    moderate_covered.sort(key=lambda x: x['count'])
    
    # Build audit report
    audit_report = {
        "audit_date": "2026-06-08",
        "total_pdfs": len(all_pdfs),
        "source_distribution": dict(Counter(r['source'] for r in pdf_records)),
        "category_distribution": category_counts,
        "well_covered": well_covered,
        "moderate_covered": moderate_covered,
        "gaps": gaps,
        "pdf_records": pdf_records,
    }
    
    return audit_report


def main():
    parser = argparse.ArgumentParser(description="Audit raw PDF corpus coverage")
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/raw'),
        help='Root directory of raw PDF corpus'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('research/results/corpus_coverage_audit.json'),
        help='Output JSON audit report'
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1
    
    # Run audit
    audit_report = audit_corpus(args.data_dir)
    
    # Write JSON report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Audit report written to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("CORPUS COVERAGE AUDIT SUMMARY")
    print("="*60)
    print(f"Total PDFs: {audit_report['total_pdfs']}")
    print(f"\nSource distribution:")
    for source, count in sorted(audit_report['source_distribution'].items()):
        print(f"  {source}: {count}")
    
    print(f"\nWell-covered categories ({len(audit_report['well_covered'])} categories with 5+ PDFs):")
    for item in audit_report['well_covered'][:10]:
        print(f"  [{item['count']:2d}] {item['name']}")
    
    print(f"\nModerate coverage ({len(audit_report['moderate_covered'])} categories with 2-4 PDFs):")
    for item in audit_report['moderate_covered']:
        print(f"  [{item['count']:2d}] {item['name']}")
    
    print(f"\nGaps ({len(audit_report['gaps'])} categories with <2 PDFs):")
    for item in audit_report['gaps']:
        severity_icon = "⚠️" if item['severity'] == "moderate" else "❌"
        print(f"  {severity_icon} [{item['count']:2d}] {item['name']}")
    
    print("="*60 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
