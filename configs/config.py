"""
BloomDepth — Central Configuration Module
==========================================
Who:    All BloomDepth pipeline stages (iterative QAG, benchmark, analysis).
Where:  BloomDepth/configs/config.py
How:    Extends VDTM-LegalQA pipeline config with 6-level Bloom taxonomy,
        iterative refinement parameters, and depth benchmark strategies.
Input:  None (pure configuration).
Output: Importable constants & dataclass singletons.

Relationship: This config INHERITS model choices from TQA_Pipeline/src/config.py
              but adds refinement loop and depth benchmark parameters.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("bloom_depth")


# ─────────────────────────────────────────────
# 1 · Path Configuration
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class PathConfig:
    """Directory layout for BloomDepth project."""

    root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("BLOOMDEPTH_ROOT", str(Path(__file__).resolve().parent.parent))
        )
    )

    # Data directories
    raw: Path = field(default=Path("data/raw"))
    interim: Path = field(default=Path("data/interim"))
    processed: Path = field(default=Path("data/processed"))

    # Interim sub-paths
    extracted_markdown: Path = field(default=Path("data/interim/extracted_markdown"))
    extracted_chunks: Path = field(default=Path("data/interim/extracted_chunks.jsonl"))
    refinement_logs: Path = field(default=Path("data/interim/refinement_logs"))
    bloom6_qa_pairs: Path = field(default=Path("data/interim/bloom6_qa_pairs.json"))
    dataset_jsonl: Path = field(default=Path("data/processed/bloom6_dataset.jsonl"))

    # Research outputs
    results: Path = field(default=Path("research/results"))
    artifacts: Path = field(default=Path("research/artifacts"))

    # Link to parent VDTM-LegalQA dataset
    vdtm_dataset: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "VDTM_DATASET_PATH",
                str(Path(__file__).resolve().parent.parent.parent / "TQA_Pipeline" / "data" / "processed" / "dataset.jsonl"),
            )
        )
    )

    def __post_init__(self) -> None:
        root = Path(os.environ.get("BLOOMDEPTH_ROOT", str(self.root))).expanduser()
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "raw", root / "data" / "raw")
        object.__setattr__(self, "interim", root / "data" / "interim")
        object.__setattr__(self, "processed", root / "data" / "processed")
        object.__setattr__(self, "extracted_markdown", root / "data" / "interim" / "extracted_markdown")
        object.__setattr__(self, "extracted_chunks", root / "data" / "interim" / "extracted_chunks.jsonl")
        object.__setattr__(self, "refinement_logs", root / "data" / "interim" / "refinement_logs")
        object.__setattr__(self, "bloom6_qa_pairs", root / "data" / "interim" / "bloom6_qa_pairs.json")
        object.__setattr__(self, "dataset_jsonl", root / "data" / "processed" / "bloom6_dataset.jsonl")
        object.__setattr__(self, "results", root / "research" / "results")
        object.__setattr__(self, "artifacts", root / "research" / "artifacts")

    def ensure_dirs(self) -> None:
        """Create every directory if it does not already exist."""
        for d in (self.raw, self.interim, self.processed, self.extracted_markdown,
                  self.refinement_logs, self.results, self.artifacts):
            d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 2 · Document Extraction Configuration
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class ExtractionConfig:
    """PDF extraction pipeline: Docling + FastText LID.

    Docling handles layout-aware parsing (DocLayNet + TableFormer).
    FastText LID filters non-Vietnamese noise from OCR artifacts.
    """

    # Chunking parameters (tuned for Vietnamese legal textbooks)
    chunk_size: int = 1500            # Characters per chunk (legal articles are verbose)
    chunk_overlap: int = 200          # Overlap to avoid cutting mid-sentence
    min_chunk_length: int = 200       # Skip chunks shorter than this (headers, footers)
    max_chunk_length: int = 5000      # Safety cap for abnormally long paragraphs

    # Docling settings
    ocr_enabled: bool = True          # Enable OCR fallback for scanned pages
    export_format: str = "markdown"   # "markdown" or "json"
    table_structure: bool = True      # Use TableFormer for table extraction
    docling_device: str = field(default_factory=lambda: os.environ.get("DOCLING_DEVICE", "auto"))
    docling_num_threads: int = field(default_factory=lambda: int(os.environ.get("DOCLING_NUM_THREADS", "12")))

    # FastText Language Identification
    fasttext_model: str = "lid.176.bin"  # Pre-trained LID model
    min_vietnamese_confidence: float = 0.5  # Minimum P(vi) to keep a chunk
    fallback_on_no_fasttext: bool = True    # If FastText unavailable, keep all chunks

    # Processing
    n_workers: int = field(default_factory=lambda: int(os.environ.get("DOCLING_EXTRACT_WORKERS", "8")))
    save_markdown: bool = True        # Also save per-document .md files

    # Source directories to scan (relative to data/raw/)
    source_dirs: tuple[str, ...] = (
        "institute",
        "universities/fdvn",
        "universities/archive_org",
    )


# ─────────────────────────────────────────────
# 3 · Model Configuration (Upgraded for Paper 2)
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class GeneratorConfig:
    """LLM for QA generation AND refinement (Self-Refine loop).

    Same model used for both GENERATE and REFINE steps.
    Paper 2 upgrade: Qwen3-14B (vs Paper 1's Qwen2.5-7B)
    → ~10GB VRAM (AWQ), fits L4 alone. Sequential load with Critic.
    """

    model_name: str = "Qwen/Qwen3-14B-Instruct-AWQ"
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    use_vllm: bool = True
    max_new_tokens: int = 1024  # Higher for Bloom 5-6 (longer reasoning chains)
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.15

    # Paper 1 baseline (for cross-generation comparison)
    paper1_model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"


@dataclass(frozen=True)
class CriticConfig:
    """LLM-as-Judge for CRITIQUE step in refinement loop.

    Cross-family design: Generator = Qwen, Critic = Gemma (anti-bias).
    Paper 2 upgrade: Gemma3-12B (vs Paper 1's Gemma-2-2b)
    → 6× larger judge for more rigorous filtering.
    → ~9GB VRAM (AWQ). Sequential load with Generator.
    """

    model_name: str = "google/gemma-3-12b-it"
    fallback_model_name: str = "google/gemma-3-4b-it"
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    max_new_tokens: int = 768  # More room for detailed critique at higher Bloom
    temperature: float = 0.1  # Low temperature for consistent judgment

    # Paper 1 baseline
    paper1_model_name: str = "google/gemma-2-2b-it"


@dataclass(frozen=True)
class VLMConfig:
    """Vision-Language Model (inherited from TQA_Pipeline)."""

    model_name: str = "5CD-AI/Vintern-1B-v3_5"
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    batch_size: int = 16


# ─────────────────────────────────────────────
# 3 · Extended Bloom Taxonomy (6 Levels)
# ─────────────────────────────────────────────
# fmt: off
BLOOM_LEVELS_3 = ("Remember", "Understand", "Apply")   # Paper 1 (baseline)
BLOOM_LEVELS_6 = (
    "Remember",      # Level 1 — Factual recall
    "Understand",    # Level 2 — Explain, summarize, interpret
    "Apply",         # Level 3 — Use knowledge in new situations
    "Analyze",       # Level 4 — Break down, compare, differentiate
    "Evaluate",      # Level 5 — Judge, justify, critique
    "Create",        # Level 6 — Design, propose, construct
)
# fmt: on

BLOOM_DESCRIPTIONS: dict[str, str] = {
    "Remember": "Nhớ lại — Hỏi trực tiếp về định nghĩa, số liệu, quy định cụ thể trong luật.",
    "Understand": "Hiểu — Giải thích ý nghĩa, tóm tắt nội dung, diễn giải điều luật bằng lời khác.",
    "Apply": "Áp dụng — Đưa ra tình huống thực tế mới và yêu cầu áp dụng quy định pháp luật.",
    "Analyze": "Phân tích — So sánh các điều khoản, phân biệt các khái niệm pháp lý, xác định mối quan hệ.",
    "Evaluate": "Đánh giá — Nhận xét tính hợp lý của một quyết định pháp lý, luận chứng ủng hộ/phản đối.",
    "Create": "Sáng tạo — Soạn thảo văn bản pháp lý, đề xuất giải pháp, thiết kế quy trình tuân thủ.",
}


# ─────────────────────────────────────────────
# 4 · Iterative Refinement Configuration
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class RefinementConfig:
    """Self-Refine loop parameters.

    Pipeline: GENERATE → CRITIQUE → REFINE → CRITIQUE → ... → ACCEPT/MAX_LOOPS
    Reference: Madaan et al. (2023). Self-Refine. NeurIPS 2023.
    """

    max_loops: int = 3                    # Maximum refinement iterations
    min_loops: int = 1                    # Minimum (always at least 1 critique)
    adaptive: bool = True                 # If True, stop early when critic approves
    batch_size: int = 32                  # QA pairs per batch

    # Critique dimensions (binary pass/fail per dimension)
    critique_dimensions: tuple[str, ...] = (
        "bloom_alignment",                # Does the question match the target Bloom level?
        "factual_grounding",              # Is the answer grounded in the context?
        "distractor_quality",             # Are distractors plausible but wrong?
        "question_clarity",               # Is the question unambiguous?
        "legal_accuracy",                 # Is the legal reasoning correct?
    )

    # Convergence threshold: all dimensions must pass
    convergence_threshold: float = 1.0    # 100% of dimensions must pass to accept

    # Ablation modes for paper experiments
    ablation_modes: tuple[str, ...] = (
        "single_pass",                    # Baseline: no refinement (1 pass only)
        "fixed_2",                        # Fixed 2 loops
        "fixed_3",                        # Fixed 3 loops
        "adaptive",                       # Adaptive: stop when converged
    )


# ─────────────────────────────────────────────
# 5 · Depth Benchmark Configuration
# ─────────────────────────────────────────────
InferenceStrategy = Literal["standard", "few_shot", "cot", "self_consistency"]

@dataclass(frozen=True)
class DepthBenchmarkConfig:
    """Benchmark configuration for inference-time compute depth analysis.

    Tests each model × strategy × Bloom level combination.
    Reference: Snell et al. (2024). Scaling LLM Test-Time Compute. arXiv:2408.03314.
    """

    # Benchmark models — fit L4 24GB with AWQ quantization
    # NOTE: Always verify model exists on HuggingFace before running!
    # MoE: VRAM ≈ total_params × 0.6GB (4-bit). Check TOTAL not active params.
    benchmark_models: tuple[str, ...] = (
        "Qwen/Qwen3-8B-Instruct-AWQ",                       # ~5GB  — Qwen gen3 small
        "Qwen/Qwen3-14B-Instruct-AWQ",                      # ~10GB — Qwen gen3 large
        "google/gemma-3-12b-it",                             # ~9GB  — Google gen3 dense (bf16, fits L4 tight)
        "microsoft/Phi-4-reasoning-plus",                    # ~10GB — Microsoft reasoning (bf16)
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506-AWQ", # ~14GB — Mistral dense
    )

    # Paper 1 baseline models (for cross-generation comparison in RQ3)
    paper1_benchmark_models: tuple[str, ...] = (
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-7B-Instruct",
        "google/gemma-2-9b-it",
    )

    # Inference strategies (ordered by compute depth)
    strategies: tuple[InferenceStrategy, ...] = (
        "standard",           # 0-shot, direct answer (1× compute)
        "few_shot",           # 3-5 exemplars, direct answer (~1.2× compute)
        "cot",                # Chain-of-Thought prompting (~3-5× compute)
        "self_consistency",   # 10 CoT paths → majority vote (~10-20× compute)
    )

    # Self-consistency parameters
    sc_num_paths: int = 10            # Number of independent CoT paths
    sc_temperature: float = 0.7       # Sampling temperature for diversity

    # Few-shot parameters
    few_shot_k: int = 3               # Number of exemplars per Bloom level

    # Conditions (inherited from Paper 1)
    conditions: tuple[str, ...] = (
        "none_context",       # No context provided
        "with_context",       # Full context provided
    )


# ─────────────────────────────────────────────
# 6 · QA Generation Configuration
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class QAGConfig:
    """Extended QAG config for 6-level Bloom taxonomy."""

    bloom_levels: tuple[str, ...] = BLOOM_LEVELS_6
    questions_per_level: int = 1      # Per chunk, per Bloom level
    batch_size: int = 64
    merge_bloom_levels: bool = True   # Merge all levels into one GPU call


# ─────────────────────────────────────────────
# 7 · Dataset Expansion Configuration
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetExpansionConfig:
    """Configuration for expanding dataset beyond original Viện sources.

    Paper 2 expands to multi-source, multi-domain:
    - Multiple universities across Vietnam
    - Three legal domains: Civil, Criminal, Administrative
    """

    # Legal domains (expanded from Paper 1's Civil-only)
    legal_domains: tuple[str, ...] = (
        "dan_su",              # Dân sự (Civil) — Paper 1 baseline + expansion
        "hinh_su",             # Hình sự (Criminal) — NEW
        "hanh_chinh",          # Hành chính (Administrative) — NEW
    )

    legal_domain_labels: dict[str, str] = field(default_factory=lambda: {
        "dan_su": "Luật Dân sự",
        "hinh_su": "Luật Hình sự",
        "hanh_chinh": "Luật Hành chính",
    })

    # Source categories
    source_categories: tuple[str, ...] = (
        "institute_textbook",      # Original Viện giáo trình (baseline)
        "university_textbook",     # Giáo trình ĐH khác
        "legal_document",          # Văn bản pháp luật chính thức
        "exam_paper",              # Đề thi pháp luật
        "reference_book",          # Sách tham khảo
    )

    # Target source universities
    target_universities: tuple[str, ...] = (
        "Viện (baseline)",
        "ĐH Luật Hà Nội",
        "ĐH Luật TP.HCM",
        "Học viện Tư pháp",
        "ĐH Quốc gia Hà Nội",
        "ĐH Quốc gia TP.HCM",
        "ĐH Cần Thơ",
        "ĐH Huế",
        "ĐH Đà Nẵng",
    )

    # Quality control thresholds
    min_ocr_confidence: float = 0.8
    max_domain_shift_score: float = 0.3   # Cosine distance from baseline vocabulary
    require_vietnamese: bool = True


# ─────────────────────────────────────────────
# 8 · Drive Sync Configuration (Anti-data-loss)
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class DriveSyncConfig:
    """Google Drive sync for crash-safe Colab execution.

    Strategy: Sync to Drive after each atomic step.
    If Colab dies, lose at most 1 model × 1 bloom level.
    """

    enabled: bool = True
    drive_base: Path = field(
        default_factory=lambda: Path(
            os.environ.get("BLOOMDEPTH_DRIVE", "/content/drive/MyDrive/BloomDepth_Backup")
        )
    )
    sync_interval: str = "per_model_per_bloom"  # Sync granularity
    progress_file: str = "progress.json"        # Resume tracker


# ─────────────────────────────────────────────
# 9 · Colab Runtime Configuration
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class ColabConfig:
    """GPU-specific settings for Colab L4 (24GB VRAM)."""

    gpu_memory_gb: int = 24
    max_model_vram_gb: float = 22.0     # Leave 2GB headroom for CUDA context
    sequential_loading: bool = True     # Cannot fit gen + critic simultaneously
    enable_torch_compile: bool = False  # L4 Ada Lovelace has limited compile gains
    vllm_gpu_memory_utilization: float = 0.85


# ─────────────────────────────────────────────
# 10 · Aggregate Config Singleton
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class BloomDepthConfig:
    """Top-level configuration for entire BloomDepth project."""

    paths: PathConfig = field(default_factory=PathConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    qag: QAGConfig = field(default_factory=QAGConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    depth_benchmark: DepthBenchmarkConfig = field(default_factory=DepthBenchmarkConfig)
    dataset_expansion: DatasetExpansionConfig = field(default_factory=DatasetExpansionConfig)
    drive_sync: DriveSyncConfig = field(default_factory=DriveSyncConfig)
    colab: ColabConfig = field(default_factory=ColabConfig)


# Instantiate the global config
CFG = BloomDepthConfig()
