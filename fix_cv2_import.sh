#!/bin/bash
# 修复 cv2 导入问题

echo "============================================"
echo "   修复 cv2 导入问题"
echo "============================================"
echo ""

echo "🔍 步骤 1: 检查当前 Python 和 OpenCV 状态"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查 Python 版本
echo "Python 版本:"
python3 --version

# 检查 pip 安装位置
echo ""
echo "pip3 安装的 opencv-python-headless:"
pip3 list | grep opencv

# 检查 Python 路径
echo ""
echo "Python 模块搜索路径:"
python3 -c "import sys; print('\n'.join(sys.path))"

echo ""
echo "🔧 步骤 2: 强制重新安装 opencv-python-headless"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 完全卸载所有 opencv 相关包
echo "卸载所有 OpenCV 包..."
pip3 uninstall -y opencv-python opencv-contrib-python opencv-python-headless 2>/dev/null

# 清理缓存
echo "清理 pip 缓存..."
pip3 cache purge 2>/dev/null || true

# 重新安装
echo "重新安装 opencv-python-headless..."
pip3 install --no-cache-dir opencv-python-headless

echo ""
echo "🧪 步骤 3: 测试导入"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 测试导入
python3 << 'EOF'
import sys
print("Python 路径:")
for p in sys.path:
    print(f"  {p}")

print("\n尝试导入 cv2...")
try:
    import cv2
    print(f"✅ 成功! OpenCV 版本: {cv2.__version__}")
except ImportError as e:
    print(f"❌ 失败: {e}")
    print("\n检查已安装的包:")
    import subprocess
    result = subprocess.run(['pip3', 'list'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'opencv' in line.lower():
            print(f"  {line}")
EOF

echo ""
echo "🔍 步骤 4: 检查安装位置"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 查找 cv2.so 文件
echo "查找 cv2 模块文件:"
find ~/.local/lib/python3.*/site-packages -name "cv2*.so" 2>/dev/null | head -5

echo ""
echo "============================================"
echo "   诊断完成"
echo "============================================"
echo ""

# 提供建议
python3 -c "import cv2" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ cv2 可以正常导入！"
else
    echo "❌ cv2 仍然无法导入"
    echo ""
    echo "💡 尝试以下解决方案:"
    echo ""
    echo "方案 1: 使用 --user 参数重新安装"
    echo "  pip3 install --user --force-reinstall opencv-python-headless"
    echo ""
    echo "方案 2: 使用系统 pip (如果有 sudo)"
    echo "  sudo pip3 install opencv-python-headless"
    echo ""
    echo "方案 3: 添加 PYTHONPATH"
    echo "  export PYTHONPATH=\$HOME/.local/lib/python3.10/site-packages:\$PYTHONPATH"
    echo ""
    echo "方案 4: 使用 conda (如果有)"
    echo "  conda install opencv"
    echo ""
fi

