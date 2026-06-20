#!/usr/bin/env bash
# ==============================================================================
# BloomDepth Sync Script: Local Workspace -> Google Drive
# ==============================================================================
# Focus: Ensure Google Drive has the latest code and key inputs (like notebooks
# and context_bloom_suitability.jsonl) while excluding huge output results,
# logs, and preventing accidental deletion of Colab run results on Drive.
# ==============================================================================

LOCAL_ROOT="/Users/van/Workspace/Research/07_Projects/DHH2026/BloomDepth"
DRIVE_ROOT="/Users/van/Library/CloudStorage/GoogleDrive-mxuanvan159@gmail.com/My Drive/02_Academic_Research/DHH_Projects/DHH2026/BloomDepth"

if [ ! -d "$LOCAL_ROOT" ]; then
    echo "❌ Local root directory does not exist: $LOCAL_ROOT"
    exit 1
fi

if [ ! -d "$DRIVE_ROOT" ]; then
    echo "❌ Google Drive root directory does not exist or is not mounted:"
    echo "   $DRIVE_ROOT"
    echo "   Please make sure Google Drive Desktop is running and mounted."
    exit 1
fi

echo "🔄 Synchronizing BloomDepth Local -> Google Drive..."
echo "Local source:  $LOCAL_ROOT"
echo "Drive target:  $DRIVE_ROOT"
echo ""

# Run rsync command
rsync -rv --delete --size-only \
  --exclude=".git" \
  --exclude=".DS_Store" \
  --exclude="*__pycache__*" \
  --exclude=".ipynb_checkpoints" \
  --exclude=".pytest_cache" \
  --exclude="_backups" \
  --exclude="experiments" \
  --exclude="results" \
  --exclude="data/processed" \
  --exclude="data/interim/domain_splits" \
  --exclude="refinement/adaptive/pending.json" \
  --exclude="runtime_monitor" \
  "$LOCAL_ROOT/" \
  "$DRIVE_ROOT/"

echo ""
echo "✅ BloomDepth Sync complete!"
