#!/bin/bash
# 设置 HuggingFace Token - Linux 版本

echo "============================================"
echo "   HuggingFace Token 设置"
echo "============================================"
echo ""

echo "📝 步骤 1: 获取 HuggingFace Token"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. 访问: https://huggingface.co/settings/tokens"
echo "2. 登录您的账号"
echo "3. 点击 'New token'"
echo "4. 选择 'Read' 权限"
echo "5. 复制生成的 token (格式: hf_xxxxxxxxxxxxx)"
echo ""

echo "📝 步骤 2: 接受 FLUX.1-dev 使用协议"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "访问: https://huggingface.co/black-forest-labs/FLUX.1-dev"
echo "点击 'Agree and access repository' 接受协议"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "请输入您的 HuggingFace Token: " HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo "❌ Token 不能为空！"
    exit 1
fi

echo ""
echo "🔧 设置 Token..."
echo ""

# 方法 1: 写入配置文件
mkdir -p ~/.huggingface
echo "$HF_TOKEN" > ~/.huggingface/token
chmod 600 ~/.huggingface/token
echo "✅ Token 已保存到 ~/.huggingface/token"

# 方法 2: 添加到 bashrc
if ! grep -q "HF_TOKEN" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# HuggingFace Token" >> ~/.bashrc
    echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc
    echo "✅ Token 已添加到 ~/.bashrc"
else
    echo "⚠️  ~/.bashrc 中已存在 HF_TOKEN"
fi

# 方法 3: 当前会话
export HF_TOKEN="$HF_TOKEN"
echo "✅ 当前会话 Token 已设置"

echo ""
echo "🧪 测试连接..."
python3 << EOF
import os
from huggingface_hub import HfApi

token = os.getenv('HF_TOKEN') or '$HF_TOKEN'
api = HfApi()

try:
    user_info = api.whoami(token=token)
    print(f"✅ 认证成功!")
    print(f"   用户: {user_info.get('name', 'N/A')}")
    print(f"   Token 有效")
except Exception as e:
    print(f"❌ 认证失败: {e}")
    print("   请检查 Token 是否正确")
EOF

echo ""
echo "============================================"
echo "   设置完成"
echo "============================================"
echo ""
echo "💡 下一步:"
echo "   1. 确保已接受 FLUX.1-dev 使用协议"
echo "   2. 运行: source ~/.bashrc"
echo "   3. 启动训练: ./train_A100_Linux_fixed.sh"
echo ""

