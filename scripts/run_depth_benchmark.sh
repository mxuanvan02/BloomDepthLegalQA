#!/usr/bin/env bash
# ============================================================
# BloomDepth — Inference Depth Benchmark (RQ2 & RQ3)
# ============================================================
# Benchmarks 4 models × 4 strategies × 6 Bloom levels × 2 conditions
#
# Usage:
#   bash scripts/run_depth_benchmark.sh [--limit N] [--model MODEL]
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export BLOOMDEPTH_ROOT="$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

EXTRA_FLAGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            EXTRA_FLAGS="$EXTRA_FLAGS --limit $2"
            shift 2
            ;;
        --model)
            EXTRA_FLAGS="$EXTRA_FLAGS --model $2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 2
            ;;
    esac
done

echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  BloomDepth — Depth Benchmark (RQ2 & RQ3)       ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""

# ── Step 1: Prepare data ──
if [ ! -f "data/processed/v1_baseline.jsonl" ]; then
    echo -e "${YELLOW}▶ Preparing baseline data...${NC}"
    python scripts/prepare_data.py
    echo ""
fi

# ── Step 2: Run benchmark ──
echo -e "${YELLOW}▶ Running depth benchmark (Phase B)...${NC}"
START=$(date +%s)
python scripts/run_experiments.py --phase b $EXTRA_FLAGS
END=$(date +%s)
echo ""

# ── Step 3: Run analysis ──
echo -e "${YELLOW}▶ Running statistical analysis (Phase C)...${NC}"
python scripts/run_experiments.py --phase c
echo ""

echo -e "${GREEN}✅ Benchmark + Analysis complete in $(( END - START ))s${NC}"
echo -e "${GREEN}   Results:   research/results/benchmark/${NC}"
echo -e "${GREEN}   Artifacts: research/artifacts/${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
