#!/usr/bin/env python3
"""
Download Vietnamese law textbook PDFs from discovery manifest with SHA256 deduplication.

Usage:
    python scripts/download_raw_pdfs.py --input data/interim/pdf_discovery_candidates.csv
"""

import argparse
import csv
import hashlib
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, Set, Optional
from urllib.parse import urlparse, unquote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)


def create_session() -> requests.Session:
    """Create requests session with retry logic and realistic headers."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/pdf,application/octet-stream,*/*',
        'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    })
    
    return session


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def normalize_filename(title: str, source: str, field: str) -> str:
    """
    Normalize filename to: GT_<field>_<short_title>_<source>.pdf
    
    Examples:
        "Giáo trình Luật Dân sự" -> GT_civil_Luat_Dan_Su_fdvn.pdf
        "Luật Hành chính Việt Nam" -> GT_administrative_Luat_Hanh_Chinh_VN_vnu.pdf
    """
    # Remove common prefixes
    title_clean = title
    for prefix in ['Giáo trình', 'Giao trinh', 'Bài giảng', 'Bai giang', 'Sách', 'Sach']:
        title_clean = re.sub(f'^{prefix}\\s*', '', title_clean, flags=re.IGNORECASE)
    
    # Remove special chars, keep Vietnamese alphanumeric + space
    title_clean = re.sub(r'[^\w\s]', '', title_clean)
    
    # Condense spaces
    title_clean = re.sub(r'\s+', '_', title_clean.strip())
    
    # Limit length
    if len(title_clean) > 50:
        title_clean = title_clean[:50]
    
    # Build filename
    filename = f"GT_{field}_{title_clean}_{source}.pdf"
    
    return filename


def download_pdf(
    url: str,
    output_path: Path,
    session: requests.Session,
    timeout: int = 60
) -> bool:
    """Download PDF from URL to output_path. Return success bool."""
    try:
        logger.info(f"Downloading: {url}")
        response = session.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'pdf' not in content_type and 'octet-stream' not in content_type:
            logger.warning(f"Unexpected content type: {content_type}")
        
        # Write to temp, then move to final
        temp_path = output_path.with_suffix('.pdf.tmp')
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify it's actually a PDF
        with open(temp_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'%PDF':
                logger.error(f"Downloaded file is not a PDF: {url}")
                temp_path.unlink()
                return False
        
        temp_path.rename(output_path)
        logger.info(f"Saved: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Download failed for {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {e}")
        return False


def load_existing_hashes(data_raw_dir: Path) -> Set[str]:
    """Load SHA256 hashes of existing PDFs to avoid re-downloading."""
    existing_hashes = set()
    
    for pdf_path in data_raw_dir.rglob('*.pdf'):
        try:
            file_hash = compute_sha256(pdf_path)
            existing_hashes.add(file_hash)
            logger.debug(f"Existing: {pdf_path.name} -> {file_hash[:8]}")
        except Exception as e:
            logger.warning(f"Could not hash {pdf_path}: {e}")
    
    logger.info(f"Loaded {len(existing_hashes)} existing PDF hashes")
    return existing_hashes


def main():
    parser = argparse.ArgumentParser(description="Download Vietnamese law textbook PDFs")
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/interim/pdf_discovery_candidates.csv'),
        help='Input CSV from discovery'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/raw'),
        help='Root directory for downloaded PDFs'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Polite delay between downloads (seconds)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Max PDFs to download (for testing)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download even if SHA256 exists'
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        logger.error(f"Input CSV not found: {args.input}")
        return 1
    
    # Load existing hashes for deduplication
    existing_hashes = set() if args.force else load_existing_hashes(args.output_dir)
    
    # Create session
    session = create_session()
    
    # Read candidates
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        candidates = list(reader)
    
    logger.info(f"Loaded {len(candidates)} candidates from {args.input}")
    
    if args.limit:
        candidates = candidates[:args.limit]
        logger.info(f"Limited to {args.limit} candidates")
    
    # Download manifest
    manifest_rows = []
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    for i, candidate in enumerate(candidates, 1):
        url = candidate['url']
        source = candidate['source']
        title = candidate.get('title', 'untitled')
        field = candidate.get('field', 'general')
        
        # Determine output subdir
        source_dir = args.output_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        
        # Normalize filename
        normalized_name = normalize_filename(title, source, field)
        output_path = source_dir / normalized_name
        
        # Skip if file already exists by name
        if output_path.exists():
            logger.info(f"[{i}/{len(candidates)}] Skip (exists by name): {output_path.name}")
            skipped_count += 1
            manifest_rows.append({
                **candidate,
                'local_path': str(output_path.relative_to(args.output_dir.parent)),
                'sha256': compute_sha256(output_path),
                'download_status': 'skipped_exists',
            })
            continue
        
        # Download
        success = download_pdf(url, output_path, session)
        
        if success:
            file_hash = compute_sha256(output_path)
            
            # Check if duplicate by hash
            if file_hash in existing_hashes and not args.force:
                logger.warning(f"Duplicate by SHA256: {output_path.name} -> {file_hash[:8]}")
                output_path.unlink()
                skipped_count += 1
                manifest_rows.append({
                    **candidate,
                    'local_path': '',
                    'sha256': file_hash,
                    'download_status': 'skipped_duplicate_hash',
                })
            else:
                existing_hashes.add(file_hash)
                downloaded_count += 1
                manifest_rows.append({
                    **candidate,
                    'local_path': str(output_path.relative_to(args.output_dir.parent)),
                    'sha256': file_hash,
                    'download_status': 'success',
                })
                logger.info(f"[{i}/{len(candidates)}] Downloaded: {output_path.name}")
        else:
            failed_count += 1
            manifest_rows.append({
                **candidate,
                'local_path': '',
                'sha256': '',
                'download_status': 'failed',
            })
        
        # Polite delay
        if i < len(candidates):
            time.sleep(args.delay)
    
    # Write manifest
    manifest_path = args.output_dir.parent / 'data' / 'interim' / 'download_manifest.csv'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = list(candidates[0].keys()) + ['local_path', 'sha256', 'download_status']
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    
    logger.info(f"Manifest written to {manifest_path}")
    logger.info(f"Summary: {downloaded_count} downloaded, {skipped_count} skipped, {failed_count} failed")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
