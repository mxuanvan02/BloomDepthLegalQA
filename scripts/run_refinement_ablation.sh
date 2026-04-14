#!/usr/bin/env bash
# ============================================================
# BloomDepth — Iterative Refinement Ablation Study (RQ1)
# ============================================================
# Runs 4 ablation modes to measure self-refinement effectiveness:
#   single_pass → fixed_2 → fixed_3 → adaptive
#
# Usage:
#   bash scripts/run_refinement_ablation.sh [--limit N]
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

LIMIT_FLAG=""
if [[ "${1:-}" == "--limit" ]] && [[ -n "${2:-}" ]]; then
    LIMIT_FLAG="--limit $2"
fi

echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  BloomDepth — Refinement Ablation Study (RQ1)    ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""

# ── Step 1: Prepare data ──
if [ ! -f "data/processed/v1_baseline.jsonl" ]; then
    echo -e "${YELLOW}▶ Preparing baseline data from VDTM-LegalQA v1...${NC}"
    python scripts/prepare_data.py
    echo ""
fi

# ── Step 2: Run ablation ──
echo -e "${YELLOW}▶ Running refinement ablation (Phase A)...${NC}"
START=$(date +%s)
python scripts/run_experiments.py --phase a $LIMIT_FLAG
END=$(date +%s)
echo ""
echo -e "${GREEN}✅ Ablation complete in $(( END - START ))s${NC}"
echo -e "${GREEN}   Results: research/results/refinement/${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
