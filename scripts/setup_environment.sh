#!/bin/bash
# COAT + FLUX LoRA 环境设置脚本 (Linux/Mac)

set -e

echo "=================================="
echo "COAT + FLUX LoRA 环境设置"
echo "=================================="
echo ""

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 创建虚拟环境
echo ""
echo "创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 虚拟环境已创建"
else
    echo "⚠️  虚拟环境已存在"
fi

# 激活虚拟环境
echo ""
echo "激活虚拟环境..."
source venv/bin/activate

# 检查CUDA
echo ""
echo "检查CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "CUDA版本: $cuda_version"
else
    echo "⚠️  未检测到NVIDIA GPU"
fi

# 安装PyTorch
echo ""
echo "安装PyTorch..."
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
echo ""
echo "安装其他依赖..."
pip install -r requirements.txt

# 验证安装
echo ""
echo "验证安装..."
python3 -c "
import torch
print(f'✅ PyTorch版本: {torch.__version__}')
print(f'✅ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print(f'✅ FP8支持: {hasattr(torch, \"float8_e4m3fn\")}')
"

# 克隆ai-toolkit (可选)
echo ""
read -p "是否克隆ai-toolkit? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "ai-toolkit" ]; then
        echo "克隆ai-toolkit..."
        git clone https://github.com/ostris/ai-toolkit.git
        echo "✅ ai-toolkit已克隆"
    else
        echo "⚠️  ai-toolkit已存在"
    fi
fi

# 创建数据集目录
echo ""
echo "创建目录结构..."
mkdir -p datasets/clothing
mkdir -p output
mkdir -p logs

echo ""
echo "=================================="
echo "✅ 环境设置完成!"
echo "=================================="
echo ""
echo "下一步:"
echo "1. 将服装图片和标注放入 datasets/clothing/"
echo "2. 阅读快速开始: cat QUICKSTART_CN.md"
echo "3. 运行训练或基准测试"
echo ""
echo "激活虚拟环境: source venv/bin/activate"







