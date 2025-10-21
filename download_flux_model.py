"""
手动下载 FLUX.1-dev 模型到本地
使用 HuggingFace Hub 的下载功能，支持断点续传
"""

import os
from huggingface_hub import snapshot_download

def download_flux_model():
    """下载 FLUX.1-dev 模型到本地"""
    
    # 设置镜像端点（如果需要）
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 获取 HF Token（如果需要）
    hf_token = os.environ.get('HF_TOKEN', None)
    
    # 模型 ID
    model_id = "black-forest-labs/FLUX.1-dev"
    
    # 本地缓存目录
    local_dir = "./models/FLUX.1-dev"
    
    print("=" * 60)
    print("🚀 开始下载 FLUX.1-dev 模型")
    print("=" * 60)
    print(f"模型 ID: {model_id}")
    print(f"下载到: {local_dir}")
    print(f"使用镜像: {os.environ.get('HF_ENDPOINT', '默认HuggingFace')}")
    print("=" * 60)
    print("\n⏳ 正在下载... 这可能需要较长时间，请耐心等待")
    print("💡 提示: 该脚本支持断点续传，如果中断可以重新运行\n")
    
    try:
        # 下载模型，支持断点续传
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接下载文件
            resume_download=True,  # 支持断点续传
            token=hf_token,
            max_workers=4,  # 使用4个并行下载线程
        )
        
        print("\n" + "=" * 60)
        print("✅ 模型下载完成！")
        print("=" * 60)
        print(f"📁 模型位置: {os.path.abspath(local_dir)}")
        print("\n📝 下一步操作：")
        print("1. 修改配置文件中的 name_or_path 为本地路径")
        print(f"   将 'black-forest-labs/FLUX.1-dev' 改为:")
        print(f"   '{os.path.abspath(local_dir)}'")
        print("\n2. 重新运行训练脚本")
        print("=" * 60)
        
        return os.path.abspath(local_dir)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 下载被用户中断")
        print("💡 下次运行此脚本时会从中断处继续下载")
        return None
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ 下载失败！")
        print("=" * 60)
        print(f"错误信息: {str(e)}")
        print("\n🔧 可能的解决方案：")
        print("1. 检查网络连接")
        print("2. 确认 HF_TOKEN 是否正确设置")
        print("3. 尝试使用 VPN")
        print("4. 或者从其他渠道下载模型文件")
        print("=" * 60)
        return None

if __name__ == "__main__":
    download_flux_model()





