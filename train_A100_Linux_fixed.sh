#!/bin/bash
# COAT + FLUX LoRA 训练启动脚本 - A100 40GB Linux版
# 包含 libGL 错误修复

echo "============================================"
echo "   COAT + FLUX LoRA 训练启动器"
echo "   A100 40GB 优化版 - Linux系统"
echo "============================================"
echo ""

# 修复 Python 路径（确保能找到 cv2 和 COAT）
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
echo "✅ 已设置 Python 模块路径"
echo "   项目路径: $PROJECT_ROOT"

# 修复 libGL.so.1 错误 (无头服务器)
export QT_QPA_PLATFORM=offscreen
export OPENCV_IO_ENABLE_OPENEXR=0
export OPENCV_VIDEOIO_PRIORITY_MSMF=0
echo "✅ 已设置 OpenCV 无头模式"

# 设置HuggingFace镜像（如果在中国）
# export HF_ENDPOINT=https://hf-mirror.com
# echo "✅ 已设置 HuggingFace 镜像"

# 设置CUDA内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "✅ 已启用 CUDA 内存优化"

# 设置多线程
export OMP_NUM_THREADS=8
echo "✅ 已设置 OMP 线程数: 8"

# 禁用不必要的 GUI 后端
export MPLBACKEND=Agg
echo "✅ 已设置 Matplotlib 后端为 Agg"

# 检查 HuggingFace Token
echo ""
echo "🔍 检查 HuggingFace 认证..."
if [ -z "$HF_TOKEN" ]; then
    if [ -f ~/.huggingface/token ]; then
        export HF_TOKEN=$(cat ~/.huggingface/token)
        echo "✅ 从 ~/.huggingface/token 加载 Token"
    else
        echo "⚠️  未设置 HF_TOKEN"
        echo "💡 FLUX.1-dev 需要 HuggingFace Token"
        echo ""
        echo "快速设置:"
        echo "  bash setup_huggingface_token.sh"
        echo ""
        read -p "是否现在设置 Token? (y/n): " setup_token
        if [[ "$setup_token" == "y" || "$setup_token" == "Y" ]]; then
            bash setup_huggingface_token.sh
            if [ $? -ne 0 ]; then
                exit 1
            fi
        fi
    fi
else
    echo "✅ HF_TOKEN 已设置"
fi

# 检测 CUDA
echo ""
echo "🔍 检测 CUDA 环境..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  PyTorch 检测失败，继续尝试训练..."
fi

# 检查 OpenCV
echo ""
echo "🔍 检测 OpenCV..."
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  OpenCV 导入失败"
    echo "💡 建议运行: pip3 install opencv-python-headless"
    echo ""
    read -p "是否现在安装 opencv-python-headless? (y/n): " install_cv2
    if [[ "$install_cv2" == "y" || "$install_cv2" == "Y" ]]; then
        echo "📦 安装 opencv-python-headless..."
        pip3 uninstall -y opencv-python opencv-contrib-python 2>/dev/null
        pip3 install opencv-python-headless
    fi
fi

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
    echo ""
    echo "💡 常见问题排查:"
    echo "   1. libGL 错误 → 运行: bash fix_libgl_error.sh"
    echo "   2. CUDA 不可用 → 检查: nvidia-smi"
    echo "   3. 显存不足 → 减少 batch_size"
    echo ""
    exit 1
fi

