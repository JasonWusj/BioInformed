#!/bin/bash
# run_experiments.sh
# 在RTX 3080服务baseline (Dice only) 和 default (Dice + PDE + BC)
# 用法: bash run_experiments.sh

set -e

echo "============================================================"
echo "  Biophysics-Informed Segmentation - RTX 3080 Experiments"
echo "  $(date)"
echo "============================================================"

# 检查GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"

# 检查预处理数据
if [ ! -d "./data/preprocessed" ]; then
    echo "[INFO] Preprocessed data not found, running preprocessing..."
    python src/preprocess.py --data_dir ./data/BraTS2023 --output_dir ./data/preprocessed
fi

echo ""
echo "============================================================"
echo "  Experiment 1/2: Baseline (Dice loss only)"
echo "============================================================"
echo ""

python src/train.py --config configs/baseline.yaml

echo ""
echo "============================================================"
echo "  Experiment 2/2: Biophysics-Informed (Dice + PDE + BC)"
echo "============================================================"
echo ""

python src/train.py --config configs/default.yaml

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Results:"
echo "    Baseline:    ./outputs_baseline/"
echo "    Biophysics:  ./outputs/"
echo "  $(date)"
echo "============================================================"
