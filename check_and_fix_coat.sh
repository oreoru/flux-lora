#!/bin/bash
# COAT 导入问题检查和修复脚本
# 在 Linux 服务器上运行

echo "════════════════════════════════════════════════════════════"
echo "  🔧 COAT 导入问题 - 自动检查和修复"
echo "════════════════════════════════════════════════════════════"
echo ""

# 获取项目根目录
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
COAT_DIR="$PROJECT_ROOT/coat_implementation"

echo "📂 项目路径: $PROJECT_ROOT"
echo "📂 COAT路径: $COAT_DIR"
echo ""

# 步骤 1: 检查目录和文件
echo "════════════════════════════════════════════════════════════"
echo "  步骤 1: 检查文件完整性"
echo "════════════════════════════════════════════════════════════"

if [ ! -d "$COAT_DIR" ]; then
    echo "❌ coat_implementation 目录不存在!"
    exit 1
fi

echo "✅ coat_implementation 目录存在"
echo ""
echo "目录内容:"
ls -lh "$COAT_DIR"
echo ""

# 检查必需文件
FILES=("__init__.py" "fp8_optimizer.py" "fp8_activation.py" "coat_trainer.py")
ALL_FILES_EXIST=true

for file in "${FILES[@]}"; do
    if [ -f "$COAT_DIR/$file" ]; then
        size=$(stat -f%z "$COAT_DIR/$file" 2>/dev/null || stat -c%s "$COAT_DIR/$file" 2>/dev/null)
        echo "✅ $file ($size bytes)"
    else
        echo "❌ $file 不存在"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    echo ""
    echo "❌ 缺少必需文件，请确保所有文件都已上传到服务器"
    exit 1
fi

# 步骤 2: 测试 Python 导入
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  步骤 2: 测试 Python 导入"
echo "════════════════════════════════════════════════════════════"

python3 << EOF
import sys
from pathlib import Path

# 设置路径
project_root = Path("$PROJECT_ROOT")
sys.path.insert(0, str(project_root))

print(f"Python 版本: {sys.version}")
print(f"项目路径: {project_root}")
print("")

# 测试导入
try:
    from coat_implementation import FP8AdamW, FP8QuantizationConfig
    print("✅ COAT 导入成功!")
    print(f"   FP8AdamW: {FP8AdamW}")
    print(f"   FP8QuantizationConfig: {FP8QuantizationConfig}")
    exit(0)
except ImportError as e:
    print(f"❌ COAT 导入失败: {e}")
    print("")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

IMPORT_STATUS=$?

if [ $IMPORT_STATUS -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ✅ COAT 导入正常，可以开始训练"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "启动训练:"
    echo "  bash train_A100_Linux_fixed.sh"
    exit 0
fi

# 步骤 3: 修复 __init__.py
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  步骤 3: 修复 __init__.py"
echo "════════════════════════════════════════════════════════════"

echo "正在备份原文件..."
cp "$COAT_DIR/__init__.py" "$COAT_DIR/__init__.py.backup"
echo "✅ 已备份到: __init__.py.backup"

echo ""
echo "正在创建新的 __init__.py..."

cat > "$COAT_DIR/__init__.py" << 'INITEOF'
"""
COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training
"""

# 使用绝对导入，避免相对导入问题
import sys
from pathlib import Path

# 确保当前目录在 sys.path 中
coat_dir = Path(__file__).parent
if str(coat_dir) not in sys.path:
    sys.path.insert(0, str(coat_dir))

# 导入核心模块
try:
    from fp8_optimizer import (
        FP8AdamW,
        FP8QuantizationConfig,
        FP8Quantizer,
        DynamicRangeExpansion
    )
    _optimizer_available = True
except ImportError as e:
    print(f"警告: 无法导入 fp8_optimizer: {e}")
    FP8AdamW = None
    FP8QuantizationConfig = None
    FP8Quantizer = None
    DynamicRangeExpansion = None
    _optimizer_available = False

try:
    from fp8_activation import (
        FP8ActivationQuantizer,
        FP8PrecisionFlow,
        FP8LinearWrapper,
        replace_linear_with_fp8,
        MemoryEfficientCheckpoint
    )
    _activation_available = True
except ImportError as e:
    print(f"警告: 无法导入 fp8_activation: {e}")
    FP8ActivationQuantizer = None
    FP8PrecisionFlow = None
    FP8LinearWrapper = None
    replace_linear_with_fp8 = None
    MemoryEfficientCheckpoint = None
    _activation_available = False

try:
    from coat_trainer import (
        COATConfig,
        COATTrainer,
        create_coat_trainer_for_flux_lora
    )
    _trainer_available = True
except ImportError as e:
    print(f"警告: 无法导入 coat_trainer: {e}")
    COATConfig = None
    COATTrainer = None
    create_coat_trainer_for_flux_lora = None
    _trainer_available = False

__version__ = "0.1.0"

__all__ = [
    'FP8AdamW',
    'FP8QuantizationConfig',
    'FP8Quantizer',
    'DynamicRangeExpansion',
    'FP8ActivationQuantizer',
    'FP8PrecisionFlow',
    'FP8LinearWrapper',
    'replace_linear_with_fp8',
    'MemoryEfficientCheckpoint',
    'COATConfig',
    'COATTrainer',
    'create_coat_trainer_for_flux_lora',
]

# 检查核心功能是否可用
if _optimizer_available and _activation_available:
    print("✅ COAT 模块加载成功")
else:
    print("⚠️  COAT 模块部分加载失败")
INITEOF

echo "✅ 已创建新的 __init__.py"

# 步骤 4: 再次测试
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  步骤 4: 再次测试导入"
echo "════════════════════════════════════════════════════════════"

python3 << EOF
import sys
from pathlib import Path

project_root = Path("$PROJECT_ROOT")
sys.path.insert(0, str(project_root))

try:
    from coat_implementation import FP8AdamW, FP8QuantizationConfig
    print("✅ COAT 导入成功!")
    print(f"   FP8AdamW: {FP8AdamW}")
    print(f"   FP8QuantizationConfig: {FP8QuantizationConfig}")
    exit(0)
except ImportError as e:
    print(f"❌ COAT 导入仍然失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

FINAL_STATUS=$?

echo ""
if [ $FINAL_STATUS -eq 0 ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "  ✅ 修复成功！现在可以开始训练"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "启动训练:"
    echo "  bash train_A100_Linux_fixed.sh"
    exit 0
else
    echo "════════════════════════════════════════════════════════════"
    echo "  ❌ 修复失败，需要进一步诊断"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "请运行详细诊断:"
    echo "  python3 -c 'import sys; print(sys.path)'"
    echo "  python3 -c 'import coat_implementation'"
    exit 1
fi
