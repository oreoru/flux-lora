"""
配置HuggingFace镜像站点（解决网络连接问题）
"""

import os

def setup_mirror():
    """设置HuggingFace镜像环境变量"""
    
    print("=" * 60)
    print("🌏 HuggingFace 镜像配置")
    print("=" * 60)
    print()
    
    mirrors = {
        "1": {
            "name": "HF-Mirror（国内推荐）",
            "endpoint": "https://hf-mirror.com"
        },
        "2": {
            "name": "ModelScope（阿里云）",
            "endpoint": "https://www.modelscope.cn"
        },
        "3": {
            "name": "取消镜像（使用官方）",
            "endpoint": ""
        }
    }
    
    print("选择镜像站点：")
    for key, info in mirrors.items():
        print(f"{key}. {info['name']}")
    print()
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice not in mirrors:
        print("❌ 无效选择")
        return
    
    selected = mirrors[choice]
    endpoint = selected["endpoint"]
    
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"\n✅ 已设置镜像: {selected['name']}")
        print(f"   端点: {endpoint}")
        print()
        print("📝 PowerShell命令（永久设置）：")
        print(f'   $env:HF_ENDPOINT = "{endpoint}"')
        print(f'   [System.Environment]::SetEnvironmentVariable("HF_ENDPOINT", "{endpoint}", "User")')
    else:
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        print(f"\n✅ 已取消镜像设置")
    
    print()
    print("=" * 60)
    print("下一步：运行训练")
    print("=" * 60)
    print("python train_flux_lora_with_coat.py")
    print()

if __name__ == "__main__":
    try:
        setup_mirror()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n❌ 出错: {e}")






