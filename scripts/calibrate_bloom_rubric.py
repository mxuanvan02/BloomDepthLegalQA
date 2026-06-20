#!/usr/bin/env python3
"""
Glass-Box Calibration for the Quantitative Bloom Rubric
=======================================================
Who:    Methodology step that turns bloom_rubric.py from "mechanism only" into
        "data-grounded". Run BEFORE trusting any rubric Bloom label.
Where:  BloomDepth/scripts/calibrate_bloom_rubric.py
How:    Given a LABELLED set of MCQ items (each with a human/expert/tagger
        'bloom_level'), it:
          1. Computes the continuous demand_score for every item (no thresholds).
          2. Grid-searches the 5 monotone cut points that maximise a transparent
             objective (default: quadratic-weighted Cohen's kappa vs labels).
          3. Reports accuracy, within-±1-level agreement, weighted kappa, and a
             full 6x6 confusion matrix.
          4. Writes the chosen thresholds + the entire report to JSON so the
             calibration is reproducible and auditable (glass-box).

Why glass-box (CONTRACT.md + user requirement):
    No hidden fitting. The objective, the search grid, the metric formulas, and
    the resulting thresholds are all written to disk. Anyone can re-run and get
    the identical cut points. No model, no RNG.

DATA HONESTY (CONTRACT.md §0, claim-inflation guard):
    Phase A QA generation has NOT produced labelled qa_pairs yet. Until it does,
    this script runs on a SEED set whose labels follow the textbook Bloom
    definitions. The seed is explicitly marked and MUST be replaced by
    human-/tagger-labelled items before any calibration number enters the paper.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

# Make src/ importable when run from repo root.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from bloom_rubric import (  # noqa: E402
    BLOOM_ORDER,
    BLOOM_TO_INT,
    INT_TO_BLOOM,
    score_item,
)


# ─────────────────────────────────────────────
# Transparent metric formulas (no library black box)
# ─────────────────────────────────────────────
def confusion_matrix(true_ints: list[int], pred_ints: list[int]) -> list[list[int]]:
    """6x6 matrix M[t-1][p-1] = count of (true=t, pred=p)."""
    m = [[0] * 6 for _ in range(6)]
    for t, p in zip(true_ints, pred_ints):
        m[t - 1][p - 1] += 1
    return m


def accuracy(true_ints: list[int], pred_ints: list[int]) -> float:
    if not true_ints:
        return 0.0
    hits = sum(1 for t, p in zip(true_ints, pred_ints) if t == p)
    return hits / len(true_ints)


def within_one(true_ints: list[int], pred_ints: list[int]) -> float:
    """Fraction predicted within ±1 Bloom level (ordinal tolerance)."""
    if not true_ints:
        return 0.0
    ok = sum(1 for t, p in zip(true_ints, pred_ints) if abs(t - p) <= 1)
    return ok / len(true_ints)


def quadratic_weighted_kappa(true_ints: list[int], pred_ints: list[int]) -> float:
    """Cohen's kappa with quadratic weights, computed from first principles.

    QWK rewards predictions that are CLOSE on the ordinal scale, not just exact.
    Range: 1.0 perfect, 0.0 chance-level, <0 worse than chance.
    """
    n = len(true_ints)
    if n == 0:
        return 0.0
    k = 6
    obs = [[0.0] * k for _ in range(k)]
    for t, p in zip(true_ints, pred_ints):
        obs[t - 1][p - 1] += 1.0

    row = [sum(obs[i]) for i in range(k)]            # true marginals
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]  # pred marginals

    w = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]

    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * (row[i] * col[j] / n) for i in range(k) for j in range(k))
    if den == 0:
        return 1.0
    return 1.0 - num / den


# ─────────────────────────────────────────────
# Threshold mapping + monotone grid search
# ─────────────────────────────────────────────
def map_with_cuts(score: float, cuts: tuple[float, ...]) -> int:
    """cuts = 5 ascending boundaries between the 6 levels.

    Returns Bloom int 1..6. level = 1 + (#cuts the score has crossed).
    """
    lvl = 1
    for c in cuts:
        if score >= c:
            lvl += 1
        else:
            break
    return lvl


def _candidate_cut_values(scores: list[float], n_steps: int) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [lo]
    step = (hi - lo) / n_steps
    return [round(lo + step * i, 4) for i in range(n_steps + 1)]


def grid_search_cuts(
    scores: list[float],
    true_ints: list[int],
    n_steps: int = 12,
    objective: str = "qwk",
) -> tuple[tuple[float, ...], float]:
    """Search 5 ASCENDING cut points maximising the chosen objective.

    Objective options (all transparent):
        'qwk'      — quadratic weighted kappa (default, ordinal-aware)
        'acc'      — exact accuracy
        'within1'  — within-±1 agreement
    Monotonicity (c1<=c2<=...<=c5) is enforced so the mapping stays ordinal.
    """
    grid = _candidate_cut_values(scores, n_steps)
    obj_fn = {
        "qwk": quadratic_weighted_kappa,
        "acc": accuracy,
        "within1": within_one,
    }[objective]

    best_cuts: tuple[float, ...] = tuple(sorted(grid)[1:6]) if len(grid) >= 6 else (0, 1, 2, 3, 4)
    best_score = -1e9
    for combo in itertools.combinations_with_replacement(grid, 5):
        preds = [map_with_cuts(s, combo) for s in scores]
        val = obj_fn(true_ints, preds)
        if val > best_score:
            best_score, best_cuts = val, combo
    return best_cuts, best_score


# ─────────────────────────────────────────────
# SEED labelled set — placeholder, MUST be replaced
# ─────────────────────────────────────────────
# Each item's "bloom_level" follows the textbook Bloom definition for the
# *question stem*. This exists ONLY to prove the calibrator runs end-to-end.
# Replace with human-/tagger-labelled qa_pairs once Phase A produces them.
SEED_LABELLED: list[dict[str, Any]] = [
    {"bloom_level": "Remember",
     "question": "Theo Điều 429 Bộ luật Dân sự, thời hiệu khởi kiện hợp đồng là bao nhiêu năm?",
     "candidate_answers": ["A. 2 năm", "B. 3 năm", "C. 5 năm", "D. 10 năm"],
     "ground_truth": "B", "context_text": "thời hiệu khởi kiện hợp đồng là 3 năm"},
    {"bloom_level": "Remember",
     "question": "Cơ quan nào có thẩm quyền cấp giấy chứng nhận quyền sử dụng đất?",
     "candidate_answers": ["A. Ủy ban nhân dân cấp tỉnh", "B. Tòa án nhân dân",
                            "C. Viện kiểm sát", "D. Quốc hội"],
     "ground_truth": "A", "context_text": ""},
    {"bloom_level": "Understand",
     "question": "Hãy giải thích ý nghĩa của nguyên tắc thiện chí trong giao kết hợp đồng.",
     "candidate_answers": ["A. Các bên phải trung thực, hợp tác vì lợi ích chung",
                            "B. Một bên được quyền ép buộc bên kia",
                            "C. Hợp đồng luôn vô hiệu",
                            "D. Chỉ bên mua có nghĩa vụ"],
     "ground_truth": "A", "context_text": ""},
    {"bloom_level": "Understand",
     "question": "Nêu ý nghĩa của việc đăng ký biện pháp bảo đảm.",
     "candidate_answers": ["A. Để xác lập hiệu lực đối kháng với người thứ ba",
                            "B. Để hủy bỏ hợp đồng", "C. Để trốn thuế",
                            "D. Không có ý nghĩa pháp lý"],
     "ground_truth": "A", "context_text": ""},
    {"bloom_level": "Apply",
     "question": "Trong tình huống A ký hợp đồng khi chưa đủ 18 tuổi, hợp đồng được xử lý thế nào?",
     "candidate_answers": ["A. Có hiệu lực hoàn toàn",
                            "B. Vô hiệu hoặc cần người đại diện theo quy định",
                            "C. A bị phạt tù", "D. Hợp đồng tự động gia hạn"],
     "ground_truth": "B", "context_text": ""},
    {"bloom_level": "Apply",
     "question": "Nếu bên thuê chậm trả tiền 3 tháng thì bên cho thuê có quyền làm gì theo hợp đồng?",
     "candidate_answers": ["A. Đơn phương chấm dứt hợp đồng theo thỏa thuận và luật",
                            "B. Bắt giữ bên thuê", "C. Tịch thu toàn bộ tài sản",
                            "D. Không được làm gì"],
     "ground_truth": "A", "context_text": ""},
    {"bloom_level": "Analyze",
     "question": "So sánh điểm khác nhau giữa hợp đồng vô hiệu tuyệt đối và vô hiệu tương đối?",
     "candidate_answers": ["A. Vô hiệu tuyệt đối có thể khắc phục, tương đối thì không",
                            "B. Vô hiệu tuyệt đối vi phạm điều cấm, tương đối do ý chí chủ thể",
                            "C. Cả hai đều do tòa án tuyên và không thể khắc phục",
                            "D. Vô hiệu tương đối vi phạm điều cấm của luật"],
     "ground_truth": "B", "context_text": ""},
    {"bloom_level": "Analyze",
     "question": "Phân tích mối quan hệ giữa nghĩa vụ và biện pháp bảo đảm thực hiện nghĩa vụ.",
     "candidate_answers": ["A. Biện pháp bảo đảm là nghĩa vụ phụ gắn với nghĩa vụ chính",
                            "B. Hai cái hoàn toàn độc lập",
                            "C. Biện pháp bảo đảm thay thế nghĩa vụ chính",
                            "D. Nghĩa vụ chính phụ thuộc biện pháp bảo đảm"],
     "ground_truth": "A", "context_text": ""},
    {"bloom_level": "Evaluate",
     "question": "Lập luận nào sau đây hợp lý nhất để bảo vệ quyền lợi bên mua ngay tình?",
     "candidate_answers": ["A. Bên mua luôn được bảo vệ vô điều kiện",
                            "B. Bên mua ngay tình được bảo vệ nếu đã đăng ký theo quy định",
                            "C. Bên mua không bao giờ được bảo vệ",
                            "D. Bên mua chỉ được bảo vệ khi bên bán đồng ý"],
     "ground_truth": "B", "context_text": ""},
    {"bloom_level": "Evaluate",
     "question": "Nhận xét nào sau đây đánh giá đúng nhất tính hợp lý của quy định phạt vi phạm?",
     "candidate_answers": ["A. Phạt vi phạm là vô lý và nên bãi bỏ",
                            "B. Mức phạt do các bên thỏa thuận trong giới hạn luật là hợp lý",
                            "C. Phạt vi phạm luôn cao hơn bồi thường",
                            "D. Không bên nào được thỏa thuận phạt"],
     "ground_truth": "B", "context_text": ""},
    {"bloom_level": "Create",
     "question": "Hãy đề xuất phương án nào sau đây phù hợp nhất để thiết kế điều khoản giải quyết tranh chấp?",
     "candidate_answers": ["A. Bỏ trống, không quy định gì",
                            "B. Kết hợp thương lượng, hòa giải rồi trọng tài/tòa án theo thứ tự",
                            "C. Chỉ dùng vũ lực", "D. Giao toàn quyền cho một bên"],
     "ground_truth": "B", "context_text": ""},
    {"bloom_level": "Create",
     "question": "Đề xuất quy trình soạn thảo hợp đồng nào sau đây bảo đảm chặt chẽ nhất về pháp lý?",
     "candidate_answers": ["A. Soạn nhanh, không rà soát",
                            "B. Xác định chủ thể, đối tượng, rà soát điều cấm, kiểm tra hiệu lực rồi ký",
                            "C. Sao chép mẫu bất kỳ trên mạng",
                            "D. Chỉ thỏa thuận miệng"],
     "ground_truth": "B", "context_text": ""},
]


def load_labelled(path: Path | None) -> tuple[list[dict[str, Any]], bool]:
    """Return (items, is_seed). If path is None or missing -> seed set."""
    if path is None or not path.exists():
        return SEED_LABELLED, True
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    else:
        data = json.load(path.open(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("qa_pairs", [])
    rows = [r for r in rows if r.get("bloom_level") in BLOOM_TO_INT]
    return rows, False


def _fmt_confusion(m: list[list[int]]) -> str:
    head = "true\\pred " + " ".join(f"{INT_TO_BLOOM[j+1][:3]:>4}" for j in range(6))
    lines = [head]
    for i in range(6):
        lines.append(f"{INT_TO_BLOOM[i+1][:8]:>9} " + " ".join(f"{m[i][j]:>4}" for j in range(6)))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Glass-box calibration for the Bloom rubric.")
    ap.add_argument("--input", type=Path, default=None,
                    help="Labelled QA file (.jsonl/.json with bloom_level). Omit to use SEED set.")
    ap.add_argument("--objective", choices=["qwk", "acc", "within1"], default="qwk")
    ap.add_argument("--n-steps", type=int, default=12, help="Grid resolution per cut.")
    ap.add_argument("--report", type=Path,
                    default=Path("research/results/calibration/bloom_rubric_calibration.json"))
    args = ap.parse_args()

    items, is_seed = load_labelled(args.input)
    if not items:
        print("No labelled items found. Provide --input with bloom_level labels.")
        sys.exit(1)

    true_ints = [BLOOM_TO_INT[r["bloom_level"]] for r in items]
    scored = [score_item(r) for r in items]
    scores = [s.demand_score for s in scored]

    # Baseline: rubric's own DEFAULT_THRESHOLDS (pre-calibration).
    base_preds = [s.bloom_int for s in scored]
    base = {
        "accuracy": round(accuracy(true_ints, base_preds), 4),
        "within_1": round(within_one(true_ints, base_preds), 4),
        "qwk": round(quadratic_weighted_kappa(true_ints, base_preds), 4),
    }

    # Calibrated: search cuts on the demand scores.
    cuts, best_obj = grid_search_cuts(scores, true_ints, args.n_steps, args.objective)
    cal_preds = [map_with_cuts(s, cuts) for s in scores]
    cal = {
        "accuracy": round(accuracy(true_ints, cal_preds), 4),
        "within_1": round(within_one(true_ints, cal_preds), 4),
        "qwk": round(quadratic_weighted_kappa(true_ints, cal_preds), 4),
    }

    cuts_named = [
        {"boundary": INT_TO_BLOOM[i + 1] + "/" + INT_TO_BLOOM[i + 2], "min_score": round(c, 4)}
        for i, c in enumerate(cuts)
    ]
    report = {
        "data_source": "SEED_PLACEHOLDER (NOT publishable)" if is_seed else str(args.input),
        "is_seed_data": is_seed,
        "n_items": len(items),
        "objective": args.objective,
        "grid_steps_per_cut": args.n_steps,
        "score_range": {"min": round(min(scores), 4), "max": round(max(scores), 4)},
        "calibrated_cuts_ascending": [round(c, 4) for c in cuts],
        "calibrated_cuts_named": cuts_named,
        "metrics_baseline_default_thresholds": base,
        "metrics_calibrated": cal,
        "best_objective_value": round(best_obj, 4),
        "confusion_matrix_calibrated": confusion_matrix(true_ints, cal_preds),
        "per_item": [
            {"question": r["question"][:70], "true": r["bloom_level"],
             "demand_score": round(s.demand_score, 4),
             "pred_calibrated": INT_TO_BLOOM[p]}
            for r, s, p in zip(items, scored, cal_preds)
        ],
        "interpretation_gates": {
            "qwk>=0.80": "strong agreement (target for publishable tagger)",
            "qwk 0.60-0.79": "moderate; usable with caveats",
            "qwk<0.60": "weak; do not use as ground-truth proxy",
        },
        "honesty_note": (
            "Cut points are fit to maximise the objective on THIS labelled set. "
            "On seed data the numbers only prove the mechanism runs; they are not "
            "evidence about real corpus performance. Replace with human-/tagger-"
            "labelled qa_pairs and re-run before citing any number."
        ),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== Bloom Rubric Calibration ({'SEED' if is_seed else 'REAL'} data, n={len(items)}) ===")
    print(f"objective={args.objective}  best={best_obj:.4f}")
    print(f"baseline   : {base}")
    print(f"calibrated : {cal}")
    print(f"cuts (asc) : {[round(c,4) for c in cuts]}")
    print("\nConfusion matrix (calibrated):")
    print(_fmt_confusion(confusion_matrix(true_ints, cal_preds)))
    print(f"\nReport written -> {args.report}")
    if is_seed:
        print("\n[WARNING] SEED placeholder data — numbers are mechanism-proof only, NOT publishable.")


if __name__ == "__main__":
    main()
