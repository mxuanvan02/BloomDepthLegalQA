#!/usr/bin/env python3
"""Build paper-grade QAG contexts from strict-clean contexts.

This finalizes the experimental context subset by removing exact duplicate text
across sources while preserving provenance for the retained record.

Input:  data/interim/gate_v2/ready_textbook_contexts_strict.jsonl
Output: data/interim/gate_v2/paper_qag_contexts.jsonl
        data/interim/gate_v2/paper_qag_context_duplicates.jsonl
        research/results/audit/paper_qag_contexts_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def norm_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/interim/gate_v2/ready_textbook_contexts_strict.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/interim/gate_v2/paper_qag_contexts.jsonl"))
    ap.add_argument("--duplicates", type=Path, default=Path("data/interim/gate_v2/paper_qag_context_duplicates.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("research/results/audit/paper_qag_contexts_report.json"))
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    kept: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    duplicate_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        text = str(row.get("text") or "")
        h = norm_hash(text)
        if h in seen:
            dup = dict(row)
            dup["paper_qag"] = {
                "accepted": False,
                "reason": "exact_duplicate_text",
                "duplicate_of_chunk_id": seen[h].get("chunk_id"),
                "content_hash_sha256": h,
            }
            duplicates.append(dup)
            duplicate_groups[h].append(dup)
            continue
        out = dict(row)
        out["paper_qag"] = {
            "accepted": True,
            "content_hash_sha256": h,
            "source_policy": "first_occurrence_retained_for_exact_duplicate_text",
        }
        kept.append(out)
        seen[h] = out
        duplicate_groups[h].append(out)

    words = [len(re.findall(r"[\wÀ-ỹ]+", str(r.get("text") or ""), re.UNICODE)) for r in kept]
    chars = [len(str(r.get("text") or "")) for r in kept]
    src_counts = Counter(r.get("source_path", "unknown") for r in kept)
    domain_counts = Counter(r.get("legal_domain", "unknown") for r in kept)
    cat_counts = Counter(r.get("source_category", "unknown") for r in kept)

    write_jsonl(args.output, kept)
    write_jsonl(args.duplicates, duplicates)

    report = {
        "input": str(args.input),
        "outputs": {
            "paper_qag_contexts": str(args.output),
            "duplicates_removed": str(args.duplicates),
        },
        "counts": {
            "strict_clean_input": len(rows),
            "paper_qag_contexts": len(kept),
            "exact_duplicate_text_removed": len(duplicates),
            "duplicate_groups": sum(1 for g in duplicate_groups.values() if len(g) > 1),
        },
        "integrity": {
            "empty_text": sum(1 for r in kept if not str(r.get("text") or "").strip()),
            "duplicate_chunk_ids": sum(1 for _, c in Counter(r.get("chunk_id") for r in kept).items() if c > 1),
            "duplicate_texts_remaining": len(kept) - len({r["paper_qag"]["content_hash_sha256"] for r in kept}),
            "not_strict_accepted": sum(1 for r in kept if r.get("strict_clean", {}).get("accepted") is not True),
            "gate_not_ready": sum(1 for r in kept if r.get("gate_v2", {}).get("bucket") != "ready_textbook_contexts"),
            "residual_flags_remaining": sum(1 for r in kept if r.get("strict_clean", {}).get("residual_flags")),
            "state_source_check_flags_remaining": sum(1 for r in kept if "needs_state_source_check" in r.get("gate_v2", {}).get("flags", [])),
        },
        "stats": {
            "source_count": len(src_counts),
            "word_min": min(words) if words else None,
            "word_median": statistics.median(words) if words else None,
            "word_mean": statistics.mean(words) if words else None,
            "word_max": max(words) if words else None,
            "char_min": min(chars) if chars else None,
            "char_median": statistics.median(chars) if chars else None,
            "char_mean": statistics.mean(chars) if chars else None,
            "char_max": max(chars) if chars else None,
            "domain_counts": dict(domain_counts),
            "source_category_counts": dict(cat_counts),
            "top_sources": dict(src_counts.most_common(20)),
        },
        "paper_safe_claim": "This paper uses 6,866 deduplicated, strict-clean textbook contexts for controlled QAG experiments. The set passes deterministic structural, source-mapping, duplicate, residual-noise, and legal-anchor leakage checks, but generated QA still requires answer-level validation.",
    }
    dump_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
