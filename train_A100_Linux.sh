#!/bin/bash
# COAT + FLUX LoRA 训练启动脚本 - A100 40GB Linux版

echo "============================================"
echo "   COAT + FLUX LoRA 训练启动器"
echo "   A100 40GB 优化版 - Linux系统"
echo "============================================"
echo ""

# 设置HuggingFace镜像（如果在中国）
# export HF_ENDPOINT=https://hf-mirror.com
# echo "✅ 已设置 HuggingFace 镜像"

# 设置CUDA内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "✅ 已启用 CUDA 内存优化"

# 设置多线程
export OMP_NUM_THREADS=8
echo "✅ 已设置 OMP 线程数: 8"

# 检测 CUDA
echo ""
echo "🔍 检测 CUDA 环境..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "🚀 开始训练..."
echo ""

# 启动训练
python3 train_flux_lora_with_coat.py ai_toolkit_integration/coat_config_a100.yaml

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 训练完成！"
    echo "📁 输出目录: output/flux_lora_clothing_coat_a100/"
else
    echo ""
    echo "❌ 训练出错！"
    exit 1
fi

