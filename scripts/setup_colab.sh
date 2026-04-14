#!/bin/bash
# BloomDepth — Colab setup script
# Run this in the first cell of your Colab notebook:
#   !bash scripts/setup_colab.sh

set -e

echo "═══════════════════════════════════════════"
echo "  BloomDepth — Colab Environment Setup"
echo "═══════════════════════════════════════════"

# 1. Mount Google Drive
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "⚠ Mount Google Drive first: from google.colab import drive; drive.mount('/content/drive')"
    exit 1
fi
echo "✓ Google Drive mounted"

# 2. Clone or restore repo
REPO_DIR="/content/BloomDepth"
DRIVE_BACKUP="/content/drive/MyDrive/BloomDepth_Backup"

if [ -d "$DRIVE_BACKUP/repo" ]; then
    echo "Restoring from Drive backup..."
    cp -r "$DRIVE_BACKUP/repo" "$REPO_DIR"
elif [ ! -d "$REPO_DIR" ]; then
    echo "Clone repo manually or upload to $REPO_DIR"
    exit 1
fi

cd "$REPO_DIR"

# 3. Install dependencies
pip install -q -U pip
pip install -q vllm>=0.4.0 docling>=2.0 fasttext-wheel>=0.9.2
pip install -q torch transformers pydantic openai
pip install -q scipy statsmodels scikit-learn matplotlib seaborn tqdm python-dotenv pandas numpy

# 4. Download FastText LID model
FASTTEXT_DIR="$HOME/.fasttext"
mkdir -p "$FASTTEXT_DIR"
if [ ! -f "$FASTTEXT_DIR/lid.176.bin" ]; then
    echo "Downloading FastText LID model..."
    wget -q -O "$FASTTEXT_DIR/lid.176.bin" \
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
fi
echo "✓ FastText LID ready"

# 5. Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.0f}GB)')"

# 6. Create data directories
python -c "
import sys; sys.path.insert(0, '.')
from configs.config import CFG
CFG.paths.ensure_dirs()
print('✓ Data directories created')
"

# 7. Restore progress from Drive (if any)
if [ -f "$DRIVE_BACKUP/progress.json" ]; then
    echo "✓ Found existing progress — will resume from last checkpoint"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  Setup complete! Run pipeline:"
echo "  1. !python scripts/extract_corpus.py"
echo "  2. !python scripts/prepare_data.py"
echo "  3. !python scripts/run_experiments.py --phase a"
echo "  4. !python scripts/run_experiments.py --phase b"
echo "  5. !python scripts/run_experiments.py --phase c"
echo "═══════════════════════════════════════════"
