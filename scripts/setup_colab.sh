#!/bin/bash
# BloomDepth — Colab setup script.
# Code runs from /content/BloomDepth; Google Drive is used only for checkpoint/result sync.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export BLOOMDEPTH_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export BLOOMDEPTH_DRIVE="${BLOOMDEPTH_DRIVE:-/content/drive/MyDrive/02_Academic_Research/DHH_Projects/DHH2026/BloomDepth}"
export DOCLING_DEVICE="${DOCLING_DEVICE:-cuda}"
export DOCLING_NUM_THREADS="${DOCLING_NUM_THREADS:-12}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"

echo "═══════════════════════════════════════════"
echo "  BloomDepth — Môi trường Colab"
echo "═══════════════════════════════════════════"
echo "Project root: $BLOOMDEPTH_ROOT"
echo "Drive sync:   $BLOOMDEPTH_DRIVE"
echo "Docling:      device=$DOCLING_DEVICE threads=$DOCLING_NUM_THREADS"

echo "1. Đang cài đặt thư viện..."
python -m pip install -q -U pip
python -m pip install -q -r requirements.txt

echo "2. Tải FastText LID model..."
FASTTEXT_DIR="$HOME/.fasttext"
mkdir -p "$FASTTEXT_DIR"
if [ ! -f "$FASTTEXT_DIR/lid.176.bin" ]; then
    wget -q -O "$FASTTEXT_DIR/lid.176.bin" \
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
fi
echo "✓ FastText chuẩn bị xong."

echo "3. Kiểm tra Card Đồ họa:"
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Không thấy CUDA GPU. Hãy đổi Runtime type sang GPU/L4 trong Colab.")

props = torch.cuda.get_device_properties(0)
print(f"GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.0f}GB)")
PY

echo "4. Chuẩn bị thư mục dữ liệu/kết quả..."
python - <<'PY'
from configs.config import CFG

CFG.paths.ensure_dirs()
print("✓ Data directories created")
PY

echo "═══════════════════════════════════════════"
echo "  Setup thành công! Sẵn sàng chạy pipeline."
echo "═══════════════════════════════════════════"
