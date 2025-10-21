#!/usr/bin/env python3
"""
检查 COAT 模块导入问题
"""

import sys
import os
from pathlib import Path

print("="*60)
print("  COAT 模块导入诊断")
print("="*60)

# 1. 检查当前目录
print("\n1. 当前工作目录:")
print(f"   {os.getcwd()}")

# 2. 检查 coat_implementation 目录
coat_path = Path("coat_implementation")
print(f"\n2. coat_implementation 目录:")
print(f"   存在: {coat_path.exists()}")
if coat_path.exists():
    print(f"   绝对路径: {coat_path.absolute()}")
    print(f"   是目录: {coat_path.is_dir()}")
    
    # 列出目录内容
    print(f"\n   目录内容:")
    for item in sorted(coat_path.iterdir()):
        print(f"     - {item.name}")

# 3. 检查 __init__.py
init_file = coat_path / "__init__.py"
print(f"\n3. __init__.py:")
print(f"   存在: {init_file.exists()}")
if init_file.exists():
    print(f"   大小: {init_file.stat().st_size} bytes")

# 4. 检查 Python 路径
print(f"\n4. Python 模块搜索路径:")
for i, p in enumerate(sys.path[:5]):
    print(f"   [{i}] {p}")

# 5. 尝试导入
print(f"\n5. 尝试导入 COAT 模块:")

# 添加当前目录到路径
current_dir = str(Path.cwd())
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"   ✅ 已添加当前目录到 sys.path")

try:
    from coat_implementation import FP8AdamW, FP8QuantizationConfig
    print(f"   ✅ 成功导入 FP8AdamW")
    print(f"   ✅ 成功导入 FP8QuantizationConfig")
    print(f"\n   FP8AdamW 位置: {FP8AdamW.__module__}")
    
except ImportError as e:
    print(f"   ❌ 导入失败: {e}")
    print(f"\n   详细错误:")
    import traceback
    traceback.print_exc()
    
    # 尝试手动导入
    print(f"\n6. 尝试手动导入子模块:")
    try:
        import coat_implementation
        print(f"   ✅ coat_implementation 模块导入成功")
        print(f"   模块路径: {coat_implementation.__file__}")
        print(f"   可用属性: {dir(coat_implementation)}")
    except Exception as e2:
        print(f"   ❌ coat_implementation 导入失败: {e2}")
        import traceback
        traceback.print_exc()

# 6. 检查相对路径导入
print(f"\n7. 检查其他必要模块:")
try:
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
except:
    print(f"   ❌ PyTorch 未安装")

try:
    import numpy
    print(f"   ✅ NumPy: {numpy.__version__}")
except:
    print(f"   ❌ NumPy 未安装")

print("\n" + "="*60)
print("  诊断完成")
print("="*60)


