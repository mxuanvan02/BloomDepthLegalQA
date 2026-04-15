#!/usr/bin/env python3
"""
BloomDepth — Restore Workspace Script
=======================================
Tự động đồng bộ ngược các dữ liệu đã xử lý từ Google Drive về máy ảo Colab 
(Local Path) nếu Runtime bị reset. Đảo bảo tính liền mạch giữa các Stage.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CFG
from src.drive_sync import get_drive_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("bloom_depth.restore")

def copy_if_exists(src: Path, dest: Path) -> None:
    if src.exists():
        if src.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            # Dùng dirs_exist_ok để gộp đè
            shutil.copytree(src, dest, dirs_exist_ok=True)
            logger.info("✅ Restored Directory: %s ➔ %s", src.name, dest.relative_to(PROJECT_ROOT))
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            logger.info("✅ Restored File: %s ➔ %s", src.name, dest.relative_to(PROJECT_ROOT))
    else:
        logger.warning("⚠️ Bỏ qua (Chưa có trên Drive): %s", src)

def main():
    ds = get_drive_sync()
    if not ds or not ds.enabled or not ds.drive_base:
        logger.error("Không tìm thấy Google Drive được kết nối hoặc Drive Sync bị vô hiệu hóa.")
        return

    drive_base = ds.drive_base
    logger.info("Bắt đầu quy trình Khôi phục Môi trường từ: %s", drive_base)

    # Các thư mục cốt lõi cần restore để chạy Stage 2, 3, 4
    sync_targets = [
        ("data/interim", CFG.paths.interim),
        ("data/processed", CFG.paths.processed),
        ("research/results", CFG.paths.results),
    ]

    for drive_subpath, local_dest in sync_targets:
        drive_src = drive_base / drive_subpath
        copy_if_exists(drive_src, local_dest)
        
    logger.info("Hoàn tất quy trình đồng bộ Workspace! Có thể chạy trực tiếp các Stage tiếp theo.")

if __name__ == "__main__":
    main()
