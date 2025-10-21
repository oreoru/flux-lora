#!/bin/bash
# 修复 libGL.so.1 错误 - Linux 无头服务器解决方案

echo "============================================"
echo "   修复 libGL.so.1 错误"
echo "============================================"
echo ""

echo "🔍 检测系统..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    echo "系统: $OS"
else
    echo "无法检测系统类型"
    OS="unknown"
fi

echo ""
echo "📦 方案 1: 安装 OpenGL 库（推荐）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    echo "检测到 Ubuntu/Debian 系统"
    echo ""
    echo "请运行以下命令（需要 sudo 权限）:"
    echo ""
    echo "sudo apt-get update"
    echo "sudo apt-get install -y libgl1-mesa-glx libglib2.0-0"
    echo ""
    
    read -p "是否现在安装? (y/n): " choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        sudo apt-get update
        sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
        if [ $? -eq 0 ]; then
            echo "✅ 安装成功！"
        else
            echo "❌ 安装失败，可能需要管理员权限"
        fi
    fi

elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
    echo "检测到 CentOS/RHEL 系统"
    echo ""
    echo "请运行以下命令（需要 sudo 权限）:"
    echo ""
    echo "sudo yum install -y mesa-libGL"
    echo ""
    
    read -p "是否现在安装? (y/n): " choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        sudo yum install -y mesa-libGL
        if [ $? -eq 0 ]; then
            echo "✅ 安装成功！"
        else
            echo "❌ 安装失败，可能需要管理员权限"
        fi
    fi

else
    echo "未知系统类型，请手动安装 OpenGL 库"
fi

echo ""
echo "📦 方案 2: 使用 opencv-python-headless（无需 sudo）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "如果没有 sudo 权限，可以使用无头版本的 OpenCV:"
echo ""
echo "pip3 uninstall -y opencv-python opencv-contrib-python"
echo "pip3 install opencv-python-headless"
echo ""

read -p "是否现在安装 opencv-python-headless? (y/n): " choice2
if [[ "$choice2" == "y" || "$choice2" == "Y" ]]; then
    pip3 uninstall -y opencv-python opencv-contrib-python 2>/dev/null
    pip3 install opencv-python-headless
    if [ $? -eq 0 ]; then
        echo "✅ opencv-python-headless 安装成功！"
    else
        echo "❌ 安装失败"
    fi
fi

echo ""
echo "📦 方案 3: 设置环境变量（临时方案）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "如果只是临时使用，可以设置环境变量禁用 GUI:"
echo ""
echo "export QT_QPA_PLATFORM=offscreen"
echo "export OPENCV_IO_ENABLE_OPENEXR=0"
echo ""

echo ""
echo "============================================"
echo "   修复完成"
echo "============================================"
echo ""
echo "💡 推荐方案:"
echo "   1. 有 sudo 权限 → 方案 1 (安装系统库)"
echo "   2. 无 sudo 权限 → 方案 2 (opencv-python-headless)"
echo "   3. 临时测试   → 方案 3 (环境变量)"
echo ""

