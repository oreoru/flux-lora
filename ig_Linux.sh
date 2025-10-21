#!/bin/bash

echo "════════════════════════════════════════════════════════════"
echo "  🔍 COAT 导入问题诊断工具"
echo "════════════════════════════════════════════════════════════"
echo ""

# 检查当前目录
echo "📁 当前目录:"
pwd
echo ""

# 检查 coat_implementation 目录
echo "📂 coat_implementation 目录内容:"
ls -lah coat_implementation/
echo ""

# 检查 __init__.py 文件内容（前20行）
echo "📄 __init__.py 文件内容:"
head -20 coat_implementation/__init__.py
echo ""

# 检查文件权限
echo "🔐 文件权限:"
ls -l coat_implementation/*.py
echo ""

# 检查 Python 能否找到模块
echo "🐍 Python 导入测试:"
python3 << 'PYEOF'
import sys
from pathlib import Path

# 添加当前目录到 Python 路径
current_dir = Path.cwd()
print(f"当前工作目录: {current_dir}")
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.path[:3]}...")
print("")

# 添加 coat_implementation 的父目录到路径
sys.path.insert(0, str(current_dir))
print(f"✅ 已添加到 sys.path: {current_dir}")
print("")

# 尝试导入
print("📦 尝试导入 coat_implementation...")
try:
    import coat_implementation
    print("✅ 导入成功!")
    print(f"   模块位置: {coat_implementation.__file__}")
    print(f"   版本: {coat_implementation.__version__}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("")
    print("详细错误追踪:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("")

# 尝试导入具体类
print("🎯 尝试导入 FP8AdamW...")
try:
    from coat_implementation import FP8AdamW, FP8QuantizationConfig
    print("✅ FP8AdamW 和 FP8QuantizationConfig 导入成功!")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("")
print("🎉 所有导入测试通过!")
PYEOF

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  💡 诊断建议"
echo "════════════════════════════════════════════════════════════"

if [ $? -eq 0 ]; then
    echo "✅ COAT 模块可以正常导入！"
    echo ""
    echo "如果训练脚本仍然报错，可能的原因："
    echo "1. 训练脚本的工作目录不正确"
    echo "2. PYTHONPATH 环境变量未设置"
    echo "3. 使用了不同的 Python 环境（虚拟环境）"
    echo ""
    echo "解决方案："
    echo "  export PYTHONPATH=\"\$(pwd):\$PYTHONPATH\""
    echo "  然后重新运行训练脚本"
else
    echo "❌ COAT 模块导入失败！"
    echo ""
    echo "可能的原因："
    echo "1. __init__.py 文件损坏或不完整"
    echo "2. fp8_optimizer.py、fp8_activation.py 或 coat_trainer.py 有语法错误"
    echo "3. 缺少依赖的 Python 包"
    echo ""
    echo "解决方案："
    echo "1. 重新上传 coat_implementation 目录"
    echo "2. 检查 Python 包依赖: pip list | grep -E 'torch|transformers'"
    echo "3. 查看上方的详细错误信息"
fi

echo ""
echo "════════════════════════════════════════════════════════════"

