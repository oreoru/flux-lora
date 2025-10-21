#!/bin/bash
# 修复 COAT 模块导入问题

echo "============================================"
echo "   修复 COAT 模块导入问题"
echo "============================================"
echo ""

# 获取项目根目录
PROJECT_ROOT=$(pwd)
echo "项目目录: $PROJECT_ROOT"

# 1. 检查 coat_implementation 目录
echo ""
echo "🔍 步骤 1: 检查 coat_implementation 目录"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "coat_implementation" ]; then
    echo "✅ coat_implementation 目录存在"
    echo ""
    echo "目录内容:"
    ls -lh coat_implementation/
else
    echo "❌ coat_implementation 目录不存在！"
    exit 1
fi

# 2. 检查必要文件
echo ""
echo "🔍 步骤 2: 检查必要文件"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

files=(
    "coat_implementation/__init__.py"
    "coat_implementation/fp8_optimizer.py"
    "coat_implementation/fp8_activation.py"
    "coat_implementation/coat_trainer.py"
)

all_files_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file 不存在"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo ""
    echo "❌ 缺少必要文件！"
    exit 1
fi

# 3. 运行诊断脚本
echo ""
echo "🔍 步骤 3: 运行导入诊断"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 check_coat_import.py

# 4. 修复方案
echo ""
echo "🔧 步骤 4: 应用修复"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 方案 1: 设置 PYTHONPATH
echo ""
echo "方案 1: 设置 PYTHONPATH"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\""
echo "✅ PYTHONPATH 已设置"

# 方案 2: 添加到 bashrc (永久)
echo ""
echo "方案 2: 添加到 ~/.bashrc (永久生效)"
BASHRC_LINE="export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\""

if grep -q "PYTHONPATH.*$PROJECT_ROOT" ~/.bashrc 2>/dev/null; then
    echo "⚠️  ~/.bashrc 中已存在项目路径"
else
    echo "" >> ~/.bashrc
    echo "# COAT FLUX LoRA 项目路径" >> ~/.bashrc
    echo "$BASHRC_LINE" >> ~/.bashrc
    echo "✅ 已添加到 ~/.bashrc"
    echo "   运行 'source ~/.bashrc' 使其生效"
fi

# 5. 验证修复
echo ""
echo "🧪 步骤 5: 验证修复"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 << 'EOF'
import sys
import os

# 添加项目路径
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from coat_implementation import FP8AdamW, FP8QuantizationConfig
    print("✅ COAT 模块导入成功！")
    print(f"   FP8AdamW: {FP8AdamW}")
    print(f"   FP8QuantizationConfig: {FP8QuantizationConfig}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "   ✅ 修复成功！"
    echo "============================================"
    echo ""
    echo "💡 下一步:"
    echo "   1. 运行: source ~/.bashrc"
    echo "   2. 启动训练: ./train_A100_Linux_fixed.sh"
else
    echo ""
    echo "============================================"
    echo "   ❌ 修复失败"
    echo "============================================"
    echo ""
    echo "💡 手动解决方案:"
    echo "   export PYTHONPATH=$(pwd):\$PYTHONPATH"
    echo "   python3 train_flux_lora_with_coat.py ai_toolkit_integration/coat_config_a100.yaml"
fi


