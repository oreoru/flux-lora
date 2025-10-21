#!/bin/bash

echo "════════════════════════════════════════════════════════════"
echo "  🔧 COAT 导入问题修复工具"
echo "════════════════════════════════════════════════════════════"
echo ""

# 获取项目根目录的绝对路径
PROJECT_ROOT=$(pwd)
echo "📁 项目根目录: $PROJECT_ROOT"
echo ""

# 检查 coat_implementation 是否存在
if [ ! -d "coat_implementation" ]; then
    echo "❌ coat_implementation 目录不存在！"
    echo "   请先上传 coat_implementation 目录到当前路径"
    exit 1
fi

echo "✅ coat_implementation 目录已找到"
echo ""

# 检查必需文件
echo "🔍 检查必需文件..."
required_files=(
    "coat_implementation/__init__.py"
    "coat_implementation/fp8_optimizer.py"
    "coat_implementation/fp8_activation.py"
    "coat_implementation/coat_trainer.py"
)

all_files_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (缺失)"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo ""
    echo "❌ 缺少必需文件，请重新上传完整的 coat_implementation 目录"
    exit 1
fi

echo ""
echo "✅ 所有必需文件都存在"
echo ""

# 修复文件权限
echo "🔧 修复文件权限..."
chmod -R 755 coat_implementation/
echo "  ✅ 权限已更新为 755"
echo ""

# 检查 __init__.py 文件大小
init_size=$(wc -c < coat_implementation/__init__.py)
if [ "$init_size" -lt 100 ]; then
    echo "⚠️  警告: __init__.py 文件太小 ($init_size 字节)"
    echo "   文件可能损坏，建议重新上传"
fi

# 设置 PYTHONPATH
echo "🔧 配置 PYTHONPATH..."
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "  export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\""
echo ""

# 测试导入
echo "🧪 测试 Python 导入..."
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from coat_implementation import FP8AdamW, FP8QuantizationConfig
    print('✅ COAT 模块导入成功!')
    print('   可用类: FP8AdamW, FP8QuantizationConfig')
except ImportError as e:
    print(f'❌ 导入失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ✅ COAT 修复成功！"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "📝 在运行训练脚本前，请先设置环境变量："
    echo ""
    echo "    export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\""
    echo ""
    echo "或者在训练脚本中已经包含此设置（推荐）"
    echo ""
    echo "🚀 现在可以启动训练："
    echo "    bash train_A100_Linux_fixed.sh"
    echo ""
else
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ❌ 修复失败"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "可能的原因："
    echo "1. Python 依赖包缺失（torch, transformers 等）"
    echo "2. coat_implementation 文件内容损坏"
    echo "3. Python 版本不兼容"
    echo ""
    echo "建议操作："
    echo "1. 运行诊断脚本查看详细错误："
    echo "   bash 诊断COAT导入问题_Linux.sh"
    echo ""
    echo "2. 检查 Python 包："
    echo "   pip list | grep -E 'torch|transformers'"
    echo ""
    echo "3. 重新上传 coat_implementation 目录"
    echo ""
fi

echo "════════════════════════════════════════════════════════════"

