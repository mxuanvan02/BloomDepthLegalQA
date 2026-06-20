#!/usr/bin/env python3
"""Score context-to-Bloom suitability for Vietnamese legal textbook QA.

Purpose
-------
Avoid forcing every context to generate all six Bloom levels. For university-use
Vietnamese legal textbook QA, a context should only be used for Bloom levels it
can support with sufficient textual evidence.

Input
-----
data/interim/gate_v2/paper_qag_contexts.jsonl

Output
------
data/interim/gate_v2/context_bloom_suitability.jsonl
research/results/audit/context_bloom_suitability_report.json

This is a deterministic heuristic router. It is intentionally conservative for
Evaluate/Create: those levels require stronger textual signals.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

BLOOM_LEVELS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")

PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "definition": [
        re.compile(p, re.I | re.U) for p in [
            r"\blà\b", r"khái\s+niệm", r"được\s+hiểu", r"bao\s+gồm", r"gồm\s+có", r"đặc\s+điểm",
            r"chức\s+năng", r"vai\s+trò", r"nguyên\s+tắc", r"điều\s+kiện",
        ]
    ],
    "explanation": [
        re.compile(p, re.I | re.U) for p in [
            r"có\s+nghĩa\s+là", r"điều\s+này", r"do\s+đó", r"vì\s+vậy", r"nhằm", r"mục\s+đích",
            r"ý\s+nghĩa", r"thể\s+hiện", r"làm\s+rõ", r"được\s+giải\s+thích",
        ]
    ],
    "scenario": [
        re.compile(p, re.I | re.U) for p in [
            r"ví\s+dụ", r"trường\s+hợp", r"tình\s+huống", r"khi\s+.*?thì", r"nếu\s+.*?thì",
            r"áp\s+dụng", r"thực\s+tiễn", r"xử\s+lý", r"giải\s+quyết",
        ]
    ],
    "comparison": [
        re.compile(p, re.I | re.U) for p in [
            r"so\s+sánh", r"phân\s+biệt", r"khác\s+với", r"giống", r"khác\s+nhau", r"mối\s+quan\s+hệ",
            r"thứ\s+nhất", r"thứ\s+hai", r"một\s+mặt", r"mặt\s+khác", r"tuy\s+nhiên", r"ngược\s+lại",
        ]
    ],
    "evaluation": [
        re.compile(p, re.I | re.U) for p in [
            r"hợp\s+lý", r"bất\s+cập", r"hạn\s+chế", r"ưu\s+điểm", r"nhược\s+điểm", r"đánh\s+giá",
            r"cần\s+thiết", r"phù\s+hợp", r"không\s+phù\s+hợp", r"tranh\s+luận", r"quan\s+điểm",
            r"lập\s+luận", r"cơ\s+sở", r"chứng\s+minh", r"phản\s+biện",
        ]
    ],
    "creation": [
        re.compile(p, re.I | re.U) for p in [
            r"đề\s+xuất", r"giải\s+pháp", r"kiến\s+nghị", r"xây\s+dựng", r"thiết\s+kế", r"quy\s+trình",
            r"biện\s+pháp", r"phương\s+án", r"soạn\s+thảo", r"hoàn\s+thiện", r"sửa\s+đổi", r"bổ\s+sung",
        ]
    ],
}

BLOCKERS: dict[str, list[re.Pattern[str]]] = {
    "metadata": [re.compile(p, re.I | re.U) for p in [r"mục\s+lục", r"nhà\s+xuất\s+bản", r"chủ\s+biên", r"lời\s+nói\s+đầu"]],
    "low_content": [re.compile(p, re.I | re.U) for p in [r"<!--\s*image\s*-->"]],
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def count_hits(text: str, key: str) -> int:
    return sum(len(p.findall(text)) for p in PATTERNS[key])


def blocker_hits(text: str) -> List[str]:
    hits = []
    for key, pats in BLOCKERS.items():
        if any(p.search(text[:1000]) for p in pats):
            hits.append(key)
    return hits


def score_context(row: Dict[str, Any]) -> Dict[str, Any]:
    text = str(row.get("text") or "")
    words = len(re.findall(r"[\wÀ-ỹ]+", text, re.U))
    hits = {k: count_hits(text, k) for k in PATTERNS}
    blockers = blocker_hits(text)

    # Normalized heuristic scores. Remember/Understand are broadly supported by
    # clean textbook prose; higher levels require stronger evidence markers.
    scores = {
        "Remember": min(1.0, 0.35 + 0.12 * hits["definition"] + 0.04 * (words >= 250)),
        "Understand": min(1.0, 0.30 + 0.10 * hits["definition"] + 0.12 * hits["explanation"] + 0.04 * (words >= 300)),
        "Apply": min(1.0, 0.10 + 0.18 * hits["scenario"] + 0.06 * hits["definition"] + 0.04 * (words >= 400)),
        "Analyze": min(1.0, 0.08 + 0.17 * hits["comparison"] + 0.05 * hits["explanation"] + 0.04 * (words >= 450)),
        "Evaluate": min(1.0, 0.04 + 0.18 * hits["evaluation"] + 0.05 * hits["comparison"] + 0.04 * (words >= 500)),
        "Create": min(1.0, 0.02 + 0.20 * hits["creation"] + 0.05 * hits["scenario"] + 0.04 * (words >= 500)),
    }

    thresholds = {
        "Remember": 0.35,
        "Understand": 0.42,
        "Apply": 0.34,
        "Analyze": 0.34,
        "Evaluate": 0.38,
        "Create": 0.38,
    }
    eligible = [lvl for lvl in BLOOM_LEVELS if scores[lvl] >= thresholds[lvl]]

    # Educational QA floor: if a clean textbook context has no strong markers,
    # allow Remember/Understand rather than dropping it entirely.
    if not eligible and not blockers:
        eligible = ["Remember", "Understand"]
        scores["Remember"] = max(scores["Remember"], thresholds["Remember"])
        scores["Understand"] = max(scores["Understand"], thresholds["Understand"])

    blocked = {
        lvl: reason
        for lvl, reason in {
            "Apply": "insufficient scenario/application markers",
            "Analyze": "insufficient comparison/relationship markers",
            "Evaluate": "insufficient evaluative/argumentative markers",
            "Create": "insufficient design/proposal/procedural markers",
        }.items()
        if lvl not in eligible
    }

    if blockers:
        # In principle paper_qag_contexts should not trigger blockers. If it does,
        # keep only low-level comprehension and surface the issue.
        eligible = [lvl for lvl in eligible if lvl in {"Remember", "Understand"}]
        for lvl in set(BLOOM_LEVELS) - set(eligible):
            blocked[lvl] = "context blocker: " + ",".join(blockers)

    return {
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "eligible_bloom_levels": eligible,
        "blocked_bloom_levels": blocked,
        "signals": hits,
        "blockers": blockers,
        "word_count": words,
        "routing_policy": "Generate QA only for eligible_bloom_levels unless --force-all-blooms is used.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/interim/gate_v2/paper_qag_contexts.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/interim/gate_v2/context_bloom_suitability.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("research/results/audit/context_bloom_suitability_report.json"))
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    out_rows: List[Dict[str, Any]] = []
    eligible_counts: Counter[str] = Counter()
    eligible_set_counts: Counter[str] = Counter()
    per_context_counts: List[int] = []
    blocked_counts: Counter[str] = Counter()
    source_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        routing = score_context(row)
        out = dict(row)
        out["bloom_suitability"] = routing
        out["eligible_bloom_levels"] = routing["eligible_bloom_levels"]
        out_rows.append(out)
        eligible_counts.update(routing["eligible_bloom_levels"])
        eligible_set_counts.update(["+".join(routing["eligible_bloom_levels"])])
        per_context_counts.append(len(routing["eligible_bloom_levels"]))
        blocked_counts.update(routing["blocked_bloom_levels"].keys())
        src = str(row.get("source_path") or row.get("source_doc") or "unknown")
        source_counts[src].update(routing["eligible_bloom_levels"])

    write_jsonl(args.output, out_rows)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "contexts": len(out_rows),
        "total_qag_jobs_routed": sum(per_context_counts),
        "total_qag_jobs_force_all": len(out_rows) * len(BLOOM_LEVELS),
        "job_reduction_ratio": round(1 - (sum(per_context_counts) / max(len(out_rows) * len(BLOOM_LEVELS), 1)), 6),
        "eligible_counts": dict(eligible_counts),
        "blocked_counts": dict(blocked_counts),
        "eligible_set_counts_top": dict(eligible_set_counts.most_common(30)),
        "eligible_per_context": {
            "min": min(per_context_counts) if per_context_counts else None,
            "median": statistics.median(per_context_counts) if per_context_counts else None,
            "mean": statistics.mean(per_context_counts) if per_context_counts else None,
            "max": max(per_context_counts) if per_context_counts else None,
        },
        "by_source_top": {src: dict(c) for src, c in sorted(source_counts.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:20]},
        "paper_note": "Bloom routing avoids forcing high-level Bloom QA on contexts that lack supporting evidence. This improves educational QA quality but changes the design from contexts×6 to context-qualified Bloom generation.",
    }
    dump_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
