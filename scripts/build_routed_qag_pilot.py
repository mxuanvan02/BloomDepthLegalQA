#!/usr/bin/env python3
"""Build a cost-controlled QAG pilot from routed gold contexts."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ORDER = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/interim/gate_v2/readiness/gold_contexts.jsonl")
    ap.add_argument("--out-contexts", default="data/interim/gate_v2/readiness/qag_pilot_gold_contexts.jsonl")
    ap.add_argument("--out-jobs", default="data/interim/gate_v2/readiness/qag_pilot_jobs.jsonl")
    ap.add_argument("--report", default="research/results/audit/qag_pilot_plan.json")
    ap.add_argument("--max-contexts", type=int, default=200)
    ap.add_argument("--per-source-cap", type=int, default=20)
    ap.add_argument("--min-quality", type=float, default=0.86)
    args = ap.parse_args()

    candidates = []
    for row in iter_jsonl(Path(args.gold)):
        audit = row["readiness_audit"]
        if audit["readiness_quality_score"] < args.min_quality:
            continue
        if not audit["eligible_bloom_levels"]:
            continue
        candidates.append(row)

    # Prioritize quality while preventing a single source document from dominating.
    candidates.sort(key=lambda r: (-r["readiness_audit"]["readiness_quality_score"], r.get("source_doc", ""), r.get("chunk_id", "")))
    source_counts: Counter[str] = Counter()
    selected = []
    bloom_counts: Counter[str] = Counter()
    for row in candidates:
        source = row.get("source_doc", "unknown")
        if source_counts[source] >= args.per_source_cap:
            continue
        selected.append(row)
        source_counts[source] += 1
        bloom_counts.update(row["readiness_audit"]["eligible_bloom_levels"])
        if len(selected) >= args.max_contexts:
            break

    jobs = []
    for row in selected:
        levels = [lvl for lvl in ORDER if lvl in row["readiness_audit"]["eligible_bloom_levels"]]
        for lvl in levels:
            jobs.append({
                "job_id": f"{row['chunk_id']}::{lvl}",
                "chunk_id": row["chunk_id"],
                "bloom_level": lvl,
                "source_doc": row.get("source_doc"),
                "source_path": row.get("source_path"),
                "content_hash": row.get("content_hash"),
                "readiness_quality_score": row["readiness_audit"]["readiness_quality_score"],
                "text": row.get("text", ""),
            })

    report = {
        "gold_input": args.gold,
        "candidate_contexts_after_quality_filter": len(candidates),
        "selected_contexts": len(selected),
        "qag_jobs": len(jobs),
        "max_contexts": args.max_contexts,
        "per_source_cap": args.per_source_cap,
        "min_quality": args.min_quality,
        "bloom_job_counts": dict((lvl, sum(1 for j in jobs if j["bloom_level"] == lvl)) for lvl in ORDER),
        "source_counts_top20": dict(source_counts.most_common(20)),
    }
    write_jsonl(Path(args.out_contexts), selected)
    write_jsonl(Path(args.out_jobs), jobs)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
