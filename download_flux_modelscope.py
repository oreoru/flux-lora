"""
使用 ModelScope 下载 FLUX.1-dev 模型（国内推荐）
ModelScope 是阿里云提供的模型平台，国内访问速度快
"""

import os
import sys

def check_and_install_modelscope():
    """检查并安装 ModelScope"""
    try:
        import modelscope
        print("✅ ModelScope 已安装")
        return True
    except ImportError:
        print("📦 ModelScope 未安装，正在安装...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
            print("✅ ModelScope 安装成功")
            return True
        except Exception as e:
            print(f"❌ ModelScope 安装失败: {e}")
            print("请手动安装: pip install modelscope")
            return False

def download_flux_from_modelscope():
    """从 ModelScope 下载 FLUX.1-dev 模型"""
    
    if not check_and_install_modelscope():
        return None
    
    from modelscope import snapshot_download
    
    # ModelScope 上的 FLUX 模型 ID
    model_id = "AI-ModelScope/FLUX.1-dev"
    
    # 本地缓存目录
    cache_dir = "./models"
    
    print("=" * 60)
    print("🚀 开始从 ModelScope 下载 FLUX.1-dev 模型")
    print("=" * 60)
    print(f"模型 ID: {model_id}")
    print(f"下载到: {cache_dir}")
    print(f"数据源: ModelScope (阿里云)")
    print("=" * 60)
    print("\n⏳ 正在下载... 这可能需要较长时间，请耐心等待")
    print("💡 提示: 国内访问 ModelScope 速度通常较快\n")
    
    try:
        # 从 ModelScope 下载模型
        model_dir = snapshot_download(
            model_id=model_id,
            cache_dir=cache_dir,
            revision='master'
        )
        
        print("\n" + "=" * 60)
        print("✅ 模型下载完成！")
        print("=" * 60)
        print(f"📁 模型位置: {os.path.abspath(model_dir)}")
        print("\n📝 下一步操作：")
        print("1. 修改配置文件中的 name_or_path 为本地路径")
        print(f"   将 'black-forest-labs/FLUX.1-dev' 改为:")
        print(f"   '{os.path.abspath(model_dir)}'")
        print("\n2. 重新运行训练脚本")
        print("=" * 60)
        
        return os.path.abspath(model_dir)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 下载被用户中断")
        return None
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ 下载失败！")
        print("=" * 60)
        print(f"错误信息: {str(e)}")
        print("\n🔧 可能的解决方案：")
        print("1. 检查网络连接")
        print("2. 尝试使用 VPN")
        print("3. 或者尝试手动下载: python download_flux_model.py")
        print("=" * 60)
        return None

if __name__ == "__main__":
    print("\n🇨🇳 使用 ModelScope 下载模型（国内推荐）\n")
    download_flux_from_modelscope()





