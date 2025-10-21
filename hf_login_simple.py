"""
简单的HuggingFace Token设置脚本
无需git命令
"""

import os
from pathlib import Path

def setup_huggingface_token():
    """直接设置HuggingFace Token到配置文件"""
    
    print("=" * 60)
    print("🤗 HuggingFace Token 设置")
    print("=" * 60)
    print()
    
    print("📋 步骤说明：")
    print("1. 访问: https://huggingface.co/black-forest-labs/FLUX.1-dev")
    print("   点击 'Agree and access repository'")
    print()
    print("2. 访问: https://huggingface.co/settings/tokens")
    print("   创建新Token（Read权限即可）")
    print()
    print("3. 复制Token并粘贴到下面")
    print()
    print("-" * 60)
    
    # 获取Token
    token = input("请粘贴你的HuggingFace Token (hf_xxx...): ").strip()
    
    if not token:
        print("❌ Token为空，已取消")
        return False
    
    if not token.startswith("hf_"):
        print("⚠️  警告: Token通常以 'hf_' 开头")
        confirm = input("是否继续? (y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return False
    
    # 保存到HuggingFace配置目录
    hf_home = Path.home() / ".cache" / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    
    token_file = hf_home / "token"
    
    try:
        # 写入Token
        token_file.write_text(token, encoding='utf-8')
        print()
        print("✅ Token已保存到:", token_file)
        
        # 验证Token
        print()
        print("🔍 验证Token...")
        from huggingface_hub import HfApi
        
        api = HfApi(token=token)
        user_info = api.whoami()
        
        print("✅ 登录成功!")
        print(f"   用户名: {user_info.get('name', 'N/A')}")
        print(f"   类型: {user_info.get('type', 'N/A')}")
        print()
        
        # 设置环境变量（当前会话）
        os.environ['HF_TOKEN'] = token
        print("✅ 已设置环境变量 HF_TOKEN")
        print()
        
        print("=" * 60)
        print("🎉 配置完成！现在可以开始训练了")
        print("=" * 60)
        print()
        print("运行以下命令开始训练：")
        print("  python train_flux_lora_with_coat.py")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print()
        print("请检查：")
        print("1. Token是否正确")
        print("2. 是否已申请FLUX.1-dev访问权限")
        print("3. 网络连接是否正常")
        return False

if __name__ == "__main__":
    try:
        setup_huggingface_token()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()






