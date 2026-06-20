#!/usr/bin/env python3
"""
Discover Vietnamese law textbook PDF sources using 9Router (Tavily/Exa) search API.

Searches for downloadable PDF textbooks across priority law domains,
ranks by recency + authority, and outputs a candidate CSV for download.

Usage:
    python scripts/discover_law_pdf_sources.py --output data/interim/pdf_discovery_candidates.csv
    python scripts/discover_law_pdf_sources.py --gaps-only   # Only search gap domains
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from ninerouter_search import NineRouterSearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)


# Authority ranking: preferred source domains scored 1-5
AUTHORITY_SCORES = {
    "fdvn.vn": 5,
    "hocluat.vn": 4,
    "luatvietnam.vn": 4,
    "thuvienphapluat.vn": 3,
    "repository.vnu.edu.vn": 5,
    "hlu.edu.vn": 5,
    "hcmulaw.edu.vn": 5,
    "ou.edu.vn": 4,
    "ctu.edu.vn": 4,
    "uel.edu.vn": 4,
    "neu.edu.vn": 4,
    "hvtc.edu.vn": 4,
    "ftu.edu.vn": 4,
    "archive.org": 3,
    "scribd.com": 2,
    "123doc.net": 2,
    "tailieu.vn": 2,
    "doc.edu.vn": 3,
    "nxbgd.vn": 5,  # NXB Giáo dục
    "nxbctqg.org.vn": 5,  # NXB Chính trị Quốc gia
}

# Domains to exclude (low quality, pirate, irrelevant)
EXCLUDED_DOMAINS = {
    "facebook.com", "tiktok.com", "youtube.com", "zalo.me",
    "shopee.vn", "lazada.vn", "tiki.vn",
}

# Search queries organized by law domain
# Focus on gap domains identified in audit + expansion of moderate coverage
SEARCH_QUERIES = {
    # Critical gaps (0 PDFs currently)
    "technology_cyber": [
        'giáo trình luật công nghệ thông tin pdf',
        'giáo trình an ninh mạng pháp luật pdf',
        'giáo trình luật thương mại điện tử pdf',
        'giáo trình pháp luật bảo vệ dữ liệu cá nhân pdf',
    ],
    "health_education": [
        'giáo trình pháp luật y tế pdf',
        'giáo trình luật giáo dục Việt Nam pdf',
        'giáo trình pháp luật dược pdf',
    ],
    "media_culture": [
        'giáo trình luật báo chí xuất bản pdf',
        'giáo trình pháp luật di sản văn hóa pdf',
        'giáo trình luật du lịch Việt Nam pdf',
    ],
    "agriculture_food": [
        'giáo trình luật nông nghiệp pdf',
        'giáo trình pháp luật an toàn thực phẩm pdf',
    ],
    "legal_english": [
        'giáo trình tiếng Anh pháp lý pdf',
        'legal English textbook Vietnamese pdf',
    ],
    "transport_construction": [
        'giáo trình luật xây dựng pdf',
        'giáo trình luật giao thông vận tải pdf',
        'giáo trình luật hàng hải Việt Nam pdf',
        'giáo trình luật nhà ở pdf',
    ],
    # Moderate coverage — expand
    "enterprise_investment": [
        'giáo trình luật doanh nghiệp 2024 pdf',
        'giáo trình luật đầu tư 2023 pdf',
        'giáo trình luật phá sản Việt Nam pdf',
    ],
    "consumer_protection": [
        'giáo trình luật bảo vệ người tiêu dùng pdf',
        'giáo trình trọng tài thương mại pdf',
    ],
    "anticorruption_inspection": [
        'giáo trình phòng chống tham nhũng pdf',
        'giáo trình luật thanh tra Việt Nam pdf',
    ],
    # Refresh existing — prefer newer editions
    "civil_refresh": [
        'giáo trình luật dân sự Việt Nam 2023 2024 pdf',
        'giáo trình luật tố tụng dân sự mới nhất pdf',
    ],
    "criminal_refresh": [
        'giáo trình luật hình sự Việt Nam 2023 2024 pdf',
        'giáo trình luật tố tụng hình sự mới nhất pdf',
    ],
    "administrative_refresh": [
        'giáo trình luật hành chính 2023 2024 pdf',
    ],
    "international_refresh": [
        'giáo trình công pháp quốc tế 2023 pdf',
        'giáo trình tư pháp quốc tế mới nhất pdf',
    ],
}

# Only gap domains
GAP_DOMAINS = [
    "technology_cyber", "health_education", "media_culture",
    "agriculture_food", "legal_english", "transport_construction",
]


def extract_domain(url: str) -> str:
    """Extract root domain from URL."""
    parsed = urlparse(url)
    parts = parsed.netloc.lower().split('.')
    # Keep last 2-3 parts (e.g., hlu.edu.vn, fdvn.vn)
    if len(parts) >= 3 and parts[-2] in ('edu', 'com', 'org', 'gov', 'ac'):
        return '.'.join(parts[-3:])
    return '.'.join(parts[-2:]) if len(parts) >= 2 else parsed.netloc


def is_excluded(url: str) -> bool:
    """Check if URL is from excluded domain."""
    domain = extract_domain(url)
    return any(excl in domain for excl in EXCLUDED_DOMAINS)


def is_likely_pdf_link(url: str, title: str, snippet: str) -> bool:
    """Heuristic: is this result likely to lead to a PDF textbook?"""
    url_lower = url.lower()
    title_lower = title.lower()
    snippet_lower = snippet.lower()
    combined = f"{url_lower} {title_lower} {snippet_lower}"

    # Positive signals: direct PDF URL or page hosting PDF download
    has_pdf = (
        url_lower.endswith('.pdf')
        or '/pdf' in url_lower
        or 'download' in url_lower
        or 'ebook' in url_lower
        or 'wp-content/uploads' in url_lower
    )

    has_textbook = any(sig in combined for sig in [
        'giáo trình', 'giao trinh', 'bài giảng', 'bai giang',
        'textbook', 'ebook', 'pdf', 'download', 'tải'
    ])

    has_law = any(sig in combined for sig in [
        'luật', 'luat', 'pháp luật', 'phap luat', 'law', 'legal',
    ])

    return (has_pdf or has_textbook) and has_law


def compute_authority_score(url: str) -> int:
    """Score URL by domain authority (1-5)."""
    domain = extract_domain(url)
    for auth_domain, score in AUTHORITY_SCORES.items():
        if auth_domain in domain:
            return score
    # Default: unknown domain
    return 1


def compute_recency_score(published_at: Optional[str], title: str, url: str) -> int:
    """Score recency from published date or year mentions in title/URL (0-30)."""
    year = None

    # Try published_at field
    if published_at:
        match = re.search(r'(20\d{2})', published_at)
        if match:
            year = int(match.group(1))

    # Try title/URL year mentions
    if not year:
        combined = f"{title} {url}"
        matches = re.findall(r'(202[0-6]|201[5-9])', combined)
        if matches:
            year = max(int(y) for y in matches)

    if not year:
        return 5  # Unknown year, neutral score

    current_year = 2026
    age = current_year - year
    if age <= 1:
        return 30
    elif age <= 3:
        return 25
    elif age <= 5:
        return 20
    elif age <= 8:
        return 10
    else:
        return 5


def rank_candidate(result: Dict) -> float:
    """Composite ranking score: authority * 10 + recency + search_score * 20."""
    authority = compute_authority_score(result['url'])
    recency = compute_recency_score(
        result.get('published_at'),
        result.get('title', ''),
        result.get('url', '')
    )
    search_score = result.get('score', 0.5)

    return authority * 10 + recency + search_score * 20


def run_discovery(
    client: NineRouterSearch,
    domains: Optional[List[str]] = None,
    provider: str = "tavily",
    limit_per_query: int = 10,
    delay: float = 1.5,
) -> List[Dict]:
    """Run search discovery across specified law domains."""

    queries_to_run = {}
    if domains:
        for d in domains:
            if d in SEARCH_QUERIES:
                queries_to_run[d] = SEARCH_QUERIES[d]
    else:
        queries_to_run = SEARCH_QUERIES

    all_candidates = []
    seen_urls: Set[str] = set()

    total_queries = sum(len(qs) for qs in queries_to_run.values())
    query_num = 0

    for domain_name, queries in queries_to_run.items():
        logger.info(f"Searching domain: {domain_name} ({len(queries)} queries)")

        for query in queries:
            query_num += 1
            logger.info(f"  [{query_num}/{total_queries}] {query}")

            results = client.search(query, provider=provider, limit=limit_per_query)

            for r in results:
                url = r.get('url', '')
                title = r.get('title', '')
                snippet = r.get('snippet', '')

                if not url or url in seen_urls:
                    continue
                if is_excluded(url):
                    continue
                if not is_likely_pdf_link(url, title, snippet):
                    continue

                seen_urls.add(url)
                candidate = {
                    'url': url,
                    'title': title,
                    'snippet': snippet[:200],
                    'domain': extract_domain(url),
                    'law_domain': domain_name,
                    'authority_score': compute_authority_score(url),
                    'recency_score': compute_recency_score(
                        r.get('published_at'), title, url
                    ),
                    'search_score': r.get('score', 0),
                    'composite_score': rank_candidate(r),
                    'is_direct_pdf': url.lower().endswith('.pdf'),
                    'provider': provider,
                    'access_date': datetime.now(timezone.utc).isoformat(),
                    'status': 'discovered',
                }
                all_candidates.append(candidate)

            # Polite delay
            time.sleep(delay)

    # Sort by composite score descending
    all_candidates.sort(key=lambda x: x['composite_score'], reverse=True)

    return all_candidates


def main():
    parser = argparse.ArgumentParser(description="Discover Vietnamese law textbook PDFs via 9Router")
    parser.add_argument(
        '--output', type=Path,
        default=Path('data/interim/pdf_discovery_candidates.csv'),
    )
    parser.add_argument(
        '--provider', type=str, default='tavily',
        choices=['tavily', 'exa'],
    )
    parser.add_argument(
        '--limit-per-query', type=int, default=10,
    )
    parser.add_argument(
        '--gaps-only', action='store_true',
        help='Only search critical gap domains',
    )
    parser.add_argument(
        '--domains', nargs='+', default=None,
        help='Specific domain names to search (keys from SEARCH_QUERIES)',
    )
    parser.add_argument(
        '--delay', type=float, default=1.5,
        help='Delay between search API calls (seconds)',
    )

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Init client
    try:
        client = NineRouterSearch()
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    # Determine which domains to search
    domains = args.domains
    if args.gaps_only:
        domains = GAP_DOMAINS

    # Run discovery
    candidates = run_discovery(
        client,
        domains=domains,
        provider=args.provider,
        limit_per_query=args.limit_per_query,
        delay=args.delay,
    )

    logger.info(f"Total candidates discovered: {len(candidates)}")

    if not candidates:
        logger.warning("No candidates found. Check API connectivity and queries.")
        return 1

    # Write CSV
    fieldnames = [
        'url', 'title', 'domain', 'law_domain', 'authority_score',
        'recency_score', 'search_score', 'composite_score',
        'is_direct_pdf', 'snippet', 'provider', 'access_date', 'status',
    ]
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(candidates)

    logger.info(f"Wrote {len(candidates)} candidates to {args.output}")

    # Print top candidates summary
    print(f"\n{'='*70}")
    print(f"DISCOVERY RESULTS: {len(candidates)} candidates")
    print(f"{'='*70}")

    by_domain = {}
    for c in candidates:
        by_domain.setdefault(c['law_domain'], []).append(c)

    for domain_name, items in sorted(by_domain.items()):
        direct_pdf = sum(1 for i in items if i['is_direct_pdf'])
        print(f"\n{domain_name}: {len(items)} candidates ({direct_pdf} direct PDFs)")
        for item in items[:3]:
            auth = "★" * item['authority_score']
            print(f"  [{item['composite_score']:.0f}] {auth} {item['title'][:60]}")
            print(f"       {item['url'][:80]}")

    print(f"\n{'='*70}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
