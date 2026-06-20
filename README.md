# 🔬 BloomDepth: Bloom-Stratified Reasoning Depth Analysis for Vietnamese Legal QA

> **Full Title:** *Bloom-Stratified Reasoning Depth: How Iterative Refinement and Inference-Time Compute Scale Across Cognitive Levels in Vietnamese Legal QA*

## Overview

This project investigates the relationship between **Bloom's Taxonomy cognitive levels** and **inference-time compute depth** for automated Question-Answering in Vietnamese legal domain.

**Core Hypothesis (H₁):** Higher Bloom cognitive levels (Analyze → Evaluate → Create) require proportionally more reasoning depth (more CoT steps, more refinement loops) from LLMs to achieve equivalent accuracy to lower levels (Remember → Understand → Apply).

## Relationship to VDTM-LegalQA (Paper 1)

This is a **follow-up study** that builds upon the VDTM-LegalQA dataset and benchmark (DHH2026). It extends the work by:

1. Expanding Bloom taxonomy from 3 levels → 6 levels
2. Introducing iterative refinement loops for QA generation
3. Benchmarking inference-time compute strategies per Bloom level
4. Expanding the dataset with additional sources

## Project Structure

```
BloomDepth/
├── README.md                 # This file
├── configs/
│   └── config.py             # Central configuration (extends TQA_Pipeline/config.py)
├── src/
│   ├── iterative_qag.py      # Iterative refinement QA generation (Loop pipeline)
│   ├── bloom_classifier.py   # 6-level Bloom taxonomy classifier
│   ├── depth_benchmark.py    # Inference-time compute depth benchmark
│   └── analysis.py           # Statistical analysis (Bloom × Depth interaction)
├── scripts/
│   ├── run_refinement_ablation.sh   # Ablation: 1-pass vs. 2-loop vs. 3-loop
│   └── run_depth_benchmark.sh       # Benchmark: Standard/Few-shot/CoT/Self-Consistency
├── data/
│   ├── raw/                  # Source PDFs/documents (extended dataset)
│   ├── interim/              # Intermediate QA pairs, refinement logs
│   └── processed/            # Final 6-level Bloom dataset
├── research/
│   ├── paper/                # LaTeX manuscript
│   ├── artifacts/            # Figures, tables, analysis outputs
│   └── results/              # Benchmark JSON results
└── requirements.txt
```

## Research Questions

| RQ | Question |
|---|---|
| **RQ1** | Does iterative refinement improve auto-generated QA quality? At which Bloom level is it most effective? |
| **RQ2** | How much reasoning depth does a model need per Bloom level? |
| **RQ3** | Is the Standard→CoT performance gap proportional to Bloom level? |

## Models (Canonical — see CONTRACT.md §3)

| Role | Model | Notes |
|---|---|---|
| QA Generator | `Qwen/Qwen3-8B-AWQ` | Construction only |
| QA Judge | `google/gemma-3-4b-it` | Construction only, cross-family |
| Benchmark (independent) | Llama-3-8B-Instruct, Mistral-7B-Instruct, Phi-4 | Primary RQ3 stats |
| Benchmark (diagnostic) | Qwen3-8B, gemma-3-4b-it | Construction-family, flagged |
| VLM | — | Deferred to future multimodal work |

## Key References

1. Madaan et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. *NeurIPS 2023*.
2. Snell et al. (2024). Scaling LLM Test-Time Compute Optimally. *arXiv:2408.03314*.
3. Anderson & Krathwohl (2001). *A Taxonomy for Learning, Teaching, and Assessing*.
4. VDTM-LegalQA v1 (2026). DHH2026 Workshop Paper.
