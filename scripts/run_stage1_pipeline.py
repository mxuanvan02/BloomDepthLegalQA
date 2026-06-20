#!/usr/bin/env python3
"""BloomDepth Stage-1 data pipeline runner — T1 → T5 (CONTRACT.md §4).

One ordered, fail-loud runner that turns raw sources into the Phase-A gate input
``context_bloom_suitability.jsonl`` ([F]). All five transforms are CPU-only.

    T1 build_extracted_chunks.py     →  [B] data/interim/extracted_chunks.jsonl
    T2 build_chunk_gate_v2.py        →  [C0..C3] data/interim/gate_v2/*.jsonl
    T3 build_strict_clean_contexts   →  [D] ready_textbook_contexts_strict.jsonl
    T4 build_paper_contexts.py       →  [E] paper_qag_contexts.jsonl
    T5 score_context_bloom_suitabil. →  [F] context_bloom_suitability.jsonl

Counts are printed at EVERY stage (CONTRACT.md §5: losses must be VISIBLE, not
silent). The run ends by asserting [F] exists and that EVERY row carries
``eligible_bloom_levels`` — the exact precondition of the Phase-A gate in
run_experiments.py.

Backups: before overwriting an existing artifact, the original is copied to
``_backups/<stamp>/<relative_path>`` first (no deletes).

Usage
-----
    # Full pipeline (T1 runs docling on the 70 universities PDFs — needs docling):
    python scripts/run_stage1_pipeline.py

    # Reuse-only T1 (48 institute docs from TQA; no heavy deps), then T2-T5:
    python scripts/run_stage1_pipeline.py --tqa-only

    # Skip T1 (extracted_chunks.jsonl already exists), run T2-T5 only:
    python scripts/run_stage1_pipeline.py --skip-t1
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"

INTERIM = PROJECT_ROOT / "data" / "interim"
GATE = INTERIM / "gate_v2"

EXTRACTED = INTERIM / "extracted_chunks.jsonl"                     # [B]
C0_READY = GATE / "ready_textbook_contexts.jsonl"                 # [C0]
D_STRICT = GATE / "ready_textbook_contexts_strict.jsonl"         # [D]
E_PAPER = GATE / "paper_qag_contexts.jsonl"                       # [E]
F_SUITABILITY = GATE / "context_bloom_suitability.jsonl"          # [F]

STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BACKUP_ROOT = PROJECT_ROOT / "_backups" / "20260608_realrun"


def banner(msg: str) -> None:
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72, flush=True)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def backup_if_exists(path: Path) -> None:
    """Copy an existing artifact into _backups before it gets overwritten."""
    if not path.exists():
        return
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        rel = Path(path.name)
    dest = BACKUP_ROOT / f"{STAMP}" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    print(f"  [backup] {rel}  →  {dest.relative_to(PROJECT_ROOT)}")


def run_step(label: str, argv: list[str]) -> None:
    banner(label)
    print("$ " + " ".join(argv), flush=True)
    result = subprocess.run(argv, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"STEP FAILED ({result.returncode}): {label}")


def assert_file_grew(path: Path, label: str) -> int:
    n = count_lines(path)
    if n == 0:
        raise SystemExit(f"EXIT GATE FAILED: {label} produced 0 rows at {path}")
    print(f"  → {label}: {n:,} rows  ({path.relative_to(PROJECT_ROOT)})")
    return n


def verify_phase_a_gate(path: Path) -> None:
    """Final assertion: [F] exists and EVERY row has eligible_bloom_levels."""
    banner("VERIFY — Phase-A gate precondition on [F] context_bloom_suitability.jsonl")
    if not path.exists():
        raise SystemExit(f"DELIVERABLE MISSING: {path}")
    total = 0
    missing = 0
    bad_len = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            elig = row.get("eligible_bloom_levels")
            if not isinstance(elig, list) or len(elig) == 0:
                missing += 1
            elif not (1 <= len(elig) <= 6):
                bad_len += 1
    print(f"  rows                         : {total:,}")
    print(f"  rows WITHOUT eligible_bloom  : {missing}")
    print(f"  rows with bad list length    : {bad_len}")
    if total == 0:
        raise SystemExit("DELIVERABLE EMPTY: context_bloom_suitability.jsonl has 0 rows")
    if missing > 0 or bad_len > 0:
        raise SystemExit(
            "PHASE-A GATE WOULD FAIL: some rows lack a valid eligible_bloom_levels list. "
            "T5 did not route every context."
        )
    print("  ✓ Every row carries a valid eligible_bloom_levels list (1-6 items).")
    print("  ✓ Deliverable is ready as the ONLY legal input to T6 (run_experiments.py).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tqa-only", action="store_true",
                    help="T1 reuses only the 48 institute docs from TQA (no docling).")
    ap.add_argument("--docling-only", action="store_true",
                    help="T1 runs docling only on the 70 universities PDFs.")
    ap.add_argument("--skip-t1", action="store_true",
                    help="Skip T1; assume data/interim/extracted_chunks.jsonl already exists.")
    args = ap.parse_args()

    py = sys.executable

    # ── T1 ────────────────────────────────────────────────────────────────
    if not args.skip_t1:
        backup_if_exists(EXTRACTED)
        t1 = [py, str(SCRIPTS / "build_extracted_chunks.py"),
              "--output", str(EXTRACTED)]
        if args.tqa_only:
            t1.append("--tqa-only")
        if args.docling_only:
            t1.append("--docling-only")
        run_step("T1 — pdf_extraction → [B] extracted_chunks.jsonl", t1)
    else:
        banner("T1 — SKIPPED (using existing extracted_chunks.jsonl)")
    n_b = assert_file_grew(EXTRACTED, "[B] extracted_chunks")

    # ── T2 ────────────────────────────────────────────────────────────────
    for p in (C0_READY, GATE / "needs_diacritic_ocr_repair.jsonl",
              GATE / "needs_state_source_check.jsonl", GATE / "excluded_from_qag.jsonl"):
        backup_if_exists(p)
    run_step("T2 — gate_v2 partition → [C0..C3]",
             [py, str(SCRIPTS / "build_chunk_gate_v2.py"),
              "--chunks", str(EXTRACTED), "--out-dir", str(GATE)])
    n_c0 = assert_file_grew(C0_READY, "[C0] ready_textbook_contexts")
    for name in ("needs_diacritic_ocr_repair", "needs_state_source_check", "excluded_from_qag"):
        print(f"  → [held-out] {name}: {count_lines(GATE / (name + '.jsonl')):,} rows")

    # ── T3 ────────────────────────────────────────────────────────────────
    backup_if_exists(D_STRICT)
    run_step("T3 — strict_clean on [C0] → [D]",
             [py, str(SCRIPTS / "build_strict_clean_contexts.py"),
              "--input", str(C0_READY), "--output", str(D_STRICT)])
    n_d = assert_file_grew(D_STRICT, "[D] strict_clean_contexts")

    # ── T4 ────────────────────────────────────────────────────────────────
    backup_if_exists(E_PAPER)
    run_step("T4 — dedup_and_paper_subset → [E]",
             [py, str(SCRIPTS / "build_paper_contexts.py"),
              "--input", str(D_STRICT), "--output", str(E_PAPER)])
    n_e = assert_file_grew(E_PAPER, "[E] paper_qag_contexts")

    # ── T5 ────────────────────────────────────────────────────────────────
    backup_if_exists(F_SUITABILITY)
    run_step("T5 — bloom_routing → [F] context_bloom_suitability.jsonl  (DELIVERABLE)",
             [py, str(SCRIPTS / "score_context_bloom_suitability.py"),
              "--input", str(E_PAPER), "--output", str(F_SUITABILITY)])
    n_f = assert_file_grew(F_SUITABILITY, "[F] context_bloom_suitability")

    # ── Final verification ────────────────────────────────────────────────
    verify_phase_a_gate(F_SUITABILITY)

    banner("STAGE-1 PIPELINE SUMMARY  (CONTRACT.md §4 T1→T5)")
    print(f"  [B] extracted_chunks            : {n_b:,}")
    print(f"  [C0] ready_textbook_contexts    : {n_c0:,}")
    print(f"  [D] strict_clean_contexts       : {n_d:,}")
    print(f"  [E] paper_qag_contexts          : {n_e:,}")
    print(f"  [F] context_bloom_suitability   : {n_f:,}   ← Phase-A gate input")
    print("\n  ✓ Pipeline complete. Run Phase A with:")
    print(f"      python scripts/run_experiments.py --phase a --contexts {F_SUITABILITY.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
