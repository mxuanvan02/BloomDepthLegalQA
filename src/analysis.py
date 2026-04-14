"""
Statistical Analysis: Bloom × Inference Depth Interaction
===========================================================
Who:    scipy, statsmodels, matplotlib
Where:  BloomDepth/src/analysis.py
How:    Performs two-way ANOVA, interaction plots, and depth heatmaps
        to test H₁: Bloom level × Strategy interaction is significant.

Statistical Tests:
    1. Two-way ANOVA: Bloom_level × Strategy → Accuracy
    2. Post-hoc Tukey HSD: Which strategy pairs differ per Bloom level
    3. Effect size: η² (eta-squared) for practical significance

Visualizations:
    1. Interaction plot: Bloom level (x) × Strategy (lines) → Accuracy (y)
    2. Heatmap: Bloom × Strategy → Accuracy (color intensity)
    3. Depth-gap bar chart: (CoT - Standard) per Bloom level
    4. Refinement convergence curves per Bloom level
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("bloom_depth.analysis")

# Ordered Bloom levels for consistent plotting
BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
STRATEGY_ORDER = ["standard", "few_shot", "cot", "self_consistency"]

STRATEGY_LABELS = {
    "standard": "Standard (0-shot)",
    "few_shot": "Few-shot (3-shot)",
    "cot": "Chain-of-Thought",
    "self_consistency": "Self-Consistency",
}


# ─────────────────────────────────────────────
# 1 · Data Preparation
# ─────────────────────────────────────────────
def load_results_matrix(
    results: list[dict[str, Any]],
    condition: str = "with_context",
) -> dict[str, dict[str, float]]:
    """Reshape flat results list into Bloom × Strategy accuracy matrix.

    Args:
        results: List of BenchmarkResult dicts (from depth_benchmark.py).
        condition: Filter to this condition only.

    Returns:
        Nested dict: {bloom_level: {strategy: accuracy}}
    """
    matrix: dict[str, dict[str, float]] = defaultdict(dict)

    for r in results:
        if r.get("condition", "") != condition:
            continue
        bloom = r.get("bloom_level", "Unknown")
        strategy = r.get("strategy", "Unknown")
        accuracy = r.get("accuracy", 0.0)
        matrix[bloom][strategy] = accuracy

    return dict(matrix)


def compute_depth_gap(matrix: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute the reasoning depth gap: CoT - Standard per Bloom level.

    This is the key metric for H₁: if gap increases with Bloom level,
    it supports the hypothesis that higher cognitive tasks benefit more
    from deeper reasoning.

    Returns:
        Dict mapping bloom_level → (cot_accuracy - standard_accuracy)
    """
    gaps = {}
    for bloom in BLOOM_ORDER:
        if bloom in matrix:
            std = matrix[bloom].get("standard", 0.0)
            cot = matrix[bloom].get("cot", 0.0)
            gaps[bloom] = cot - std

    return gaps


# ─────────────────────────────────────────────
# 2 · Statistical Tests
# ─────────────────────────────────────────────
def run_two_way_anova(
    results: list[dict[str, Any]],
    condition: str = "with_context",
) -> dict[str, Any]:
    """Run two-way ANOVA: Bloom_level × Strategy → Accuracy.

    Tests three hypotheses:
        H₀₁: No main effect of Bloom level on accuracy
        H₀₂: No main effect of Strategy on accuracy
        H₀₃: No interaction between Bloom level and Strategy

    H₁ (our hypothesis): H₀₃ is rejected → significant interaction.

    Returns:
        Dict with F-statistics, p-values, and η² for each effect.
    """
    try:
        import pandas as pd
        from scipy import stats
    except ImportError:
        logger.error("pandas and scipy are required for ANOVA. Install them first.")
        return {"error": "Missing dependencies"}

    # Build long-format dataframe
    rows = []
    for r in results:
        if r.get("condition", "") != condition:
            continue
        # Create one row per question prediction
        for pred in r.get("predictions", []):
            rows.append({
                "bloom_level": r.get("bloom_level", "Unknown"),
                "strategy": r.get("strategy", "Unknown"),
                "correct": int(pred.get("correct", False)),
            })

    if not rows:
        logger.warning("No prediction data found for ANOVA. Using aggregate results.")
        for r in results:
            if r.get("condition", "") != condition:
                continue
            rows.append({
                "bloom_level": r.get("bloom_level", "Unknown"),
                "strategy": r.get("strategy", "Unknown"),
                "correct": r.get("accuracy", 0.0),
            })

    df = pd.DataFrame(rows)

    if len(df) < 10:
        logger.warning("Too few data points (%d) for reliable ANOVA", len(df))
        return {"error": f"Too few data points: {len(df)}"}

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        # Fit OLS model with interaction term
        model = ols("correct ~ C(bloom_level) * C(strategy)", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Compute η² (eta-squared) effect sizes
        ss_total = anova_table["sum_sq"].sum()
        anova_table["eta_sq"] = anova_table["sum_sq"] / ss_total

        result = {
            "anova_table": anova_table.to_dict(),
            "bloom_effect": {
                "F": float(anova_table.loc["C(bloom_level)", "F"]),
                "p_value": float(anova_table.loc["C(bloom_level)", "PR(>F)"]),
                "eta_sq": float(anova_table.loc["C(bloom_level)", "eta_sq"]),
            },
            "strategy_effect": {
                "F": float(anova_table.loc["C(strategy)", "F"]),
                "p_value": float(anova_table.loc["C(strategy)", "PR(>F)"]),
                "eta_sq": float(anova_table.loc["C(strategy)", "eta_sq"]),
            },
            "interaction_effect": {
                "F": float(anova_table.loc["C(bloom_level):C(strategy)", "F"]),
                "p_value": float(anova_table.loc["C(bloom_level):C(strategy)", "PR(>F)"]),
                "eta_sq": float(anova_table.loc["C(bloom_level):C(strategy)", "eta_sq"]),
            },
            "n_observations": len(df),
            "r_squared": float(model.rsquared),
        }

        # Interpret interaction
        p_interaction = result["interaction_effect"]["p_value"]
        if p_interaction < 0.001:
            result["interpretation"] = "STRONG support for H₁: Highly significant Bloom×Strategy interaction (p < .001)"
        elif p_interaction < 0.05:
            result["interpretation"] = "MODERATE support for H₁: Significant Bloom×Strategy interaction (p < .05)"
        else:
            result["interpretation"] = "WEAK/NO support for H₁: Bloom×Strategy interaction not significant (p ≥ .05)"

        logger.info("ANOVA Interaction: F=%.3f, p=%.6f, η²=%.4f → %s",
                     result["interaction_effect"]["F"],
                     p_interaction,
                     result["interaction_effect"]["eta_sq"],
                     result["interpretation"])

        return result

    except Exception as e:
        logger.error("ANOVA computation failed: %s", e)
        return {"error": str(e)}


# ─────────────────────────────────────────────
# 3 · Refinement Analysis
# ─────────────────────────────────────────────
def analyze_refinement_traces(
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze refinement traces to understand convergence patterns.

    Computes per-Bloom-level:
        - Average loops to convergence
        - Convergence rate (% that converged before max_loops)
        - Quality improvement per loop

    Returns:
        Dict with summary statistics per Bloom level.
    """
    by_bloom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        bloom = trace.get("bloom_level", "Unknown")
        by_bloom[bloom].append(trace)

    analysis = {}
    for bloom in BLOOM_ORDER:
        bloom_traces = by_bloom.get(bloom, [])
        if not bloom_traces:
            continue

        loops = [t.get("total_loops", 0) for t in bloom_traces]
        converged = [t.get("converged", False) for t in bloom_traces]

        analysis[bloom] = {
            "count": len(bloom_traces),
            "avg_loops": float(np.mean(loops)) if loops else 0.0,
            "median_loops": float(np.median(loops)) if loops else 0.0,
            "max_loops": max(loops) if loops else 0,
            "convergence_rate": sum(converged) / max(len(converged), 1),
        }

    # Compute correlation: Bloom position × avg loops
    bloom_positions = []
    avg_loops_list = []
    for i, bloom in enumerate(BLOOM_ORDER):
        if bloom in analysis:
            bloom_positions.append(i)
            avg_loops_list.append(analysis[bloom]["avg_loops"])

    if len(bloom_positions) >= 3:
        from scipy import stats
        corr, p_val = stats.spearmanr(bloom_positions, avg_loops_list)
        analysis["_correlation"] = {
            "spearman_rho": float(corr),
            "p_value": float(p_val),
            "interpretation": (
                f"{'Positive' if corr > 0 else 'Negative'} correlation (ρ={corr:.3f}, p={p_val:.4f}): "
                f"Higher Bloom levels {'require' if corr > 0 else 'do not require'} more refinement loops."
            ),
        }

    return analysis


# ─────────────────────────────────────────────
# 4 · Visualization
# ─────────────────────────────────────────────
def plot_interaction(
    matrix: dict[str, dict[str, float]],
    output_path: Path,
    title: str = "Bloom Level × Inference Strategy Interaction",
) -> None:
    """Generate interaction plot: Bloom (x) × Strategy (lines) → Accuracy (y).

    This is the KEY visualization for H₁.
    If lines diverge at higher Bloom levels → supports the hypothesis.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "standard": "#94a3b8",
        "few_shot": "#60a5fa",
        "cot": "#34d399",
        "self_consistency": "#f472b6",
    }

    x_labels = [b for b in BLOOM_ORDER if b in matrix]
    x_pos = range(len(x_labels))

    for strategy in STRATEGY_ORDER:
        y_vals = [matrix.get(b, {}).get(strategy, 0.0) for b in x_labels]
        if any(v > 0 for v in y_vals):
            ax.plot(
                x_pos, y_vals,
                marker="o", linewidth=2, markersize=8,
                color=colors.get(strategy, "#666"),
                label=STRATEGY_LABELS.get(strategy, strategy),
            )

    ax.set_xlabel("Bloom Cognitive Level →", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved interaction plot to %s", output_path)


def plot_depth_gap(
    gaps: dict[str, float],
    output_path: Path,
    title: str = "Reasoning Depth Gap (CoT − Standard) per Bloom Level",
) -> None:
    """Bar chart showing CoT improvement over Standard per Bloom level.

    Hypothesis: bars should get taller from left (Remember) to right (Create).
    """
    import matplotlib.pyplot as plt

    x_labels = [b for b in BLOOM_ORDER if b in gaps]
    y_vals = [gaps[b] for b in x_labels]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#6ee7b7" if v >= 0 else "#fca5a5" for v in y_vals]
    bars = ax.bar(range(len(x_labels)), y_vals, color=colors, edgecolor="#333", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, y_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:+.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Bloom Cognitive Level →", fontsize=12)
    ax.set_ylabel("Δ Accuracy (CoT − Standard)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.axhline(y=0, color="#333", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved depth gap plot to %s", output_path)


def plot_heatmap(
    matrix: dict[str, dict[str, float]],
    output_path: Path,
    title: str = "Accuracy Heatmap: Bloom Level × Strategy",
) -> None:
    """Heatmap visualization of Bloom × Strategy accuracy matrix."""
    import matplotlib.pyplot as plt

    x_labels = [b for b in BLOOM_ORDER if b in matrix]
    y_labels = [STRATEGY_LABELS.get(s, s) for s in STRATEGY_ORDER]

    data = np.zeros((len(STRATEGY_ORDER), len(x_labels)))
    for j, bloom in enumerate(x_labels):
        for i, strategy in enumerate(STRATEGY_ORDER):
            data[i, j] = matrix.get(bloom, {}).get(strategy, 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    # Add text annotations
    for i in range(len(STRATEGY_ORDER)):
        for j in range(len(x_labels)):
            val = data[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Bloom Cognitive Level →", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap to %s", output_path)


def plot_refinement_convergence(
    analysis: dict[str, Any],
    output_path: Path,
    title: str = "Refinement Convergence: Avg Loops per Bloom Level",
) -> None:
    """Bar chart of average refinement loops per Bloom level.

    Hypothesis: Higher Bloom levels need more loops (positive correlation).
    """
    import matplotlib.pyplot as plt

    x_labels = [b for b in BLOOM_ORDER if b in analysis]
    avg_loops = [analysis[b]["avg_loops"] for b in x_labels]
    conv_rates = [analysis[b]["convergence_rate"] for b in x_labels]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bars: average loops
    bars = ax1.bar(range(len(x_labels)), avg_loops, color="#818cf8", edgecolor="#333", linewidth=0.5, alpha=0.8)
    ax1.set_xlabel("Bloom Cognitive Level →", fontsize=12)
    ax1.set_ylabel("Avg. Refinement Loops", fontsize=12, color="#818cf8")
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, fontsize=10)

    # Line: convergence rate (secondary axis)
    ax2 = ax1.twinx()
    ax2.plot(range(len(x_labels)), conv_rates, "o-", color="#f472b6", linewidth=2, markersize=8)
    ax2.set_ylabel("Convergence Rate", fontsize=12, color="#f472b6")
    ax2.set_ylim(0, 1.1)

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved refinement convergence plot to %s", output_path)


# ─────────────────────────────────────────────
# 5 · Report Generation
# ─────────────────────────────────────────────
def generate_analysis_report(
    benchmark_results: list[dict[str, Any]],
    refinement_traces: list[dict[str, Any]] | None = None,
    output_dir: Path = Path("research/artifacts"),
    condition: str = "with_context",
) -> dict[str, Any]:
    """Run full analysis pipeline and generate all outputs.

    Returns:
        Dict with ANOVA results, depth gaps, and file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build matrix
    matrix = load_results_matrix(benchmark_results, condition)
    logger.info("Results matrix: %d Bloom levels × %d strategies",
                len(matrix), max(len(v) for v in matrix.values()) if matrix else 0)

    # 2. Compute depth gap
    gaps = compute_depth_gap(matrix)

    # 3. Run ANOVA
    anova_results = run_two_way_anova(benchmark_results, condition)

    # 4. Generate visualizations
    plot_interaction(matrix, output_dir / "interaction_plot.png")
    plot_depth_gap(gaps, output_dir / "depth_gap.png")
    plot_heatmap(matrix, output_dir / "accuracy_heatmap.png")

    # 5. Refinement analysis (if traces provided)
    ref_analysis = None
    if refinement_traces:
        ref_analysis = analyze_refinement_traces(refinement_traces)
        plot_refinement_convergence(ref_analysis, output_dir / "refinement_convergence.png")

    # 6. Save summary JSON
    summary = {
        "condition": condition,
        "accuracy_matrix": matrix,
        "depth_gaps": gaps,
        "anova": anova_results,
        "refinement_analysis": ref_analysis,
    }

    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Full analysis report generated in %s", output_dir)
    return summary
