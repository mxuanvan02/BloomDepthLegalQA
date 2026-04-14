"""
Drive Sync Utility — Crash-Safe Checkpointing
================================================
Who:    All phases (Refinement, Benchmark, Analysis)
Where:  BloomDepth/src/drive_sync.py
How:    Maintains progress.json to track checkpoints.
        Native execution on Google Drive means file copying is redundant.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("bloom_depth.drive_sync")


class DriveSync:
    """Manages execution state for crash-safe execution.
    
    File copying methods (sync_dir/sync_file) are no-ops because 
    the Colab environment executes directly on the mounted Google Drive, 
    making native File I/O automatically synced.

    Usage:
        sync = DriveSync(drive_base="research/results/checkpoints")

        # Check if step already completed
        if sync.is_completed("benchmark", "qwen3-8b_remember"):
            print("Skipping — already done")

        # After completing a step
        sync.mark_completed("benchmark", "qwen3-8b_remember")
    """

    def __init__(
        self,
        drive_base: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        if drive_base is None:
            # Checkpoints are saved natively to the workspace directory
            from configs.config import CFG
            self.drive_base = CFG.paths.results.parent / "checkpoints"
        else:
            self.drive_base = Path(drive_base)
            
        self.enabled = enabled
        self._progress_file = self.drive_base / "progress.json"
        self._progress: dict[str, dict[str, Any]] = {}

        if self.enabled:
            self.drive_base.mkdir(parents=True, exist_ok=True)
            self._load_progress()

    def _load_progress(self) -> None:
        """Load progress tracker from Drive."""
        if self._progress_file.exists():
            try:
                with open(self._progress_file, encoding="utf-8") as f:
                    self._progress = json.load(f)
                logger.info("Loaded progress: %d completed steps", self._count_completed())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load progress: %s — starting fresh", e)
                self._progress = {}
        else:
            self._progress = {}

    def _save_progress(self) -> None:
        """Persist progress tracker to Drive."""
        if not self.enabled:
            return
        self._progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._progress_file, "w", encoding="utf-8") as f:
            json.dump(self._progress, f, ensure_ascii=False, indent=2, default=str)

    def _count_completed(self) -> int:
        """Count total completed steps across all phases."""
        return sum(
            1 for phase in self._progress.values()
            if isinstance(phase, dict)
            for step_data in phase.values()
            if isinstance(step_data, dict) and step_data.get("completed")
        )

    def is_completed(self, phase: str, step: str) -> bool:
        """Check if a specific step has already been completed."""
        return bool(
            self._progress.get(phase, {}).get(step, {}).get("completed", False)
        )

    def mark_completed(self, phase: str, step: str, metadata: dict | None = None) -> None:
        """Mark a step as completed with optional metadata."""
        if phase not in self._progress:
            self._progress[phase] = {}

        self._progress[phase][step] = {
            "completed": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(metadata or {}),
        }
        self._save_progress()
        logger.info("✅ Checkpoint: %s/%s saved", phase, step)

    def sync_dir(self, local_dir: Path, drive_subpath: str) -> None:
        """Sync a local directory to Drive."""
        if not self.enabled:
            return

        drive_target = self.drive_base / drive_subpath
        drive_target.mkdir(parents=True, exist_ok=True)

        try:
            import shutil
            shutil.copytree(local_dir, drive_target, dirs_exist_ok=True)
            logger.info("📁 Synced %s → %s", local_dir, drive_target)
        except OSError as e:
            logger.error("Sync failed: %s", e)

    def sync_file(self, local_file: Path, drive_subpath: str) -> None:
        """Sync a single file to Drive."""
        if not self.enabled:
            return

        drive_target = self.drive_base / drive_subpath
        drive_target.parent.mkdir(parents=True, exist_ok=True)

        try:
            import shutil
            shutil.copy2(local_file, drive_target)
            logger.info("📄 Synced %s → %s", local_file.name, drive_target)
        except OSError as e:
            logger.error("Sync failed: %s", e)

    def restore_from_drive(self, drive_subpath: str, local_dir: Path) -> bool:
        """Restore local data from Drive backup."""
        drive_source = self.drive_base / drive_subpath
        if drive_source.exists() and not local_dir.exists():
            import shutil
            shutil.copytree(drive_source, local_dir, dirs_exist_ok=True)
            logger.info("♻️  Restored %s from Drive", drive_subpath)
            return True
        return False

    def get_resume_point(self, phase: str) -> list[str]:
        """Get list of completed steps for a phase."""
        phase_data = self._progress.get(phase, {})
        return [
            step for step, data in phase_data.items()
            if isinstance(data, dict) and data.get("completed")
        ]

    def print_status(self) -> None:
        """Print current progress summary."""
        logger.info("=" * 50)
        logger.info("  Checkpoints Status")
        logger.info("  Base: %s", self.drive_base)
        logger.info("=" * 50)

        for phase, steps in self._progress.items():
            if not isinstance(steps, dict):
                continue
            completed = sum(1 for s in steps.values() if isinstance(s, dict) and s.get("completed"))
            total = len(steps)
            logger.info("  %s: %d/%d steps completed", phase, completed, total)
            for step, data in steps.items():
                if isinstance(data, dict):
                    status = "✅" if data.get("completed") else "⏳"
                    logger.info("    %s %s (%s)", status, step, data.get("timestamp", ""))

        logger.info("=" * 50)


def get_drive_sync() -> DriveSync | None:
    """Initialize DriveSync with auto-resolved path for Mac or Colab."""
    import os
    from configs.config import CFG

    # 1. Check User-Provided Env Var
    env_path = os.environ.get("BLOOMDEPTH_DRIVE")
    if env_path:
        return DriveSync(drive_base=env_path)

    # 2. Check Local Mac Drive Setup (Academic/Research structure)
    mac_drive = Path("/Users/van/Library/CloudStorage/GoogleDrive-mxuanvan159@gmail.com/My Drive/02_Academic_Research/DHH_Projects/DHH2026/BloomDepth_Backup")
    if mac_drive.parent.exists():
        return DriveSync(drive_base=mac_drive)

    # 3. Check Colab Drive Setup
    colab_drive = Path("/content/drive/MyDrive/02_Academic_Research/DHH_Projects/DHH2026/BloomDepth_Backup")
    if colab_drive.parent.exists():
        return DriveSync(drive_base=colab_drive)

    # 4. Fallback natively inline
    return DriveSync(drive_base=CFG.paths.results.parent / "checkpoints")
