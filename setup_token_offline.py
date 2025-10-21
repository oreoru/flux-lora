"""
离线设置HuggingFace Token（不验证网络连接）
"""

import os
from pathlib import Path

def setup_token_offline():
    """直接保存Token到本地，跳过在线验证"""
    
    print("=" * 60)
    print("🤗 HuggingFace Token 离线设置")
    print("=" * 60)
    print()
    
    print("📋 首先获取Token（浏览器操作）：")
    print("1. 访问: https://huggingface.co/settings/tokens")
    print("2. 创建新Token（Read权限）")
    print("3. 复制Token")
    print()
    print("⚠️  注意：需要先在浏览器中申请FLUX.1-dev访问权限")
    print("   https://huggingface.co/black-forest-labs/FLUX.1-dev")
    print()
    print("-" * 60)
    
    # 获取Token
    token = input("请粘贴你的HuggingFace Token: ").strip()
    
    if not token:
        print("❌ Token为空，已取消")
        return False
    
    if not token.startswith("hf_"):
        print("⚠️  警告: Token通常以 'hf_' 开头")
        confirm = input("是否继续? (y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return False
    
    # 保存到3个位置确保生效
    saved_locations = []
    
    # 位置1: HuggingFace缓存目录
    try:
        hf_home = Path.home() / ".cache" / "huggingface"
        hf_home.mkdir(parents=True, exist_ok=True)
        token_file = hf_home / "token"
        token_file.write_text(token, encoding='utf-8')
        saved_locations.append(str(token_file))
        print(f"✅ 已保存到: {token_file}")
    except Exception as e:
        print(f"⚠️  位置1保存失败: {e}")
    
    # 位置2: HuggingFace Hub配置目录
    try:
        hf_hub = Path.home() / ".huggingface"
        hf_hub.mkdir(parents=True, exist_ok=True)
        token_file2 = hf_hub / "token"
        token_file2.write_text(token, encoding='utf-8')
        saved_locations.append(str(token_file2))
        print(f"✅ 已保存到: {token_file2}")
    except Exception as e:
        print(f"⚠️  位置2保存失败: {e}")
    
    # 位置3: 当前项目目录的.env文件
    try:
        env_file = Path(".env")
        env_content = f"HF_TOKEN={token}\nHF_HUB_OFFLINE=0\n"
        env_file.write_text(env_content, encoding='utf-8')
        saved_locations.append(str(env_file))
        print(f"✅ 已保存到: {env_file}")
    except Exception as e:
        print(f"⚠️  位置3保存失败: {e}")
    
    # 设置环境变量
    os.environ['HF_TOKEN'] = token
    os.environ['HUGGING_FACE_HUB_TOKEN'] = token
    print(f"✅ 已设置环境变量")
    
    print()
    print("=" * 60)
    print("✅ Token已保存到以下位置:")
    for loc in saved_locations:
        print(f"   - {loc}")
    print("=" * 60)
    print()
    
    print("📝 PowerShell命令（备用）：")
    print(f'   $env:HF_TOKEN = "{token}"')
    print()
    
    print("🎉 配置完成！")
    print()
    print("下一步：")
    print("1. 如果网络正常，直接运行:")
    print("   python train_flux_lora_with_coat.py")
    print()
    print("2. 如果需要使用镜像站点:")
    print("   运行: python setup_hf_mirror.py")
    print()
    print("3. 如果需要手动下载模型:")
    print("   查看: SETUP_HUGGINGFACE_CN.md")
    print()
    
    return True

if __name__ == "__main__":
    try:
        setup_token_offline()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        import traceback
        traceback.print_exc()






