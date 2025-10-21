"""
检查CUDA和GPU状态
"""

import torch

print("="*60)
print("GPU状态检查")
print("="*60)

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"FP8支持: {hasattr(torch, 'float8_e4m3fn')}")
    
    print("\n✅ CUDA已启用，可以使用GPU训练！")
else:
    print("\n❌ CUDA不可用，当前使用CPU")
    print("\n需要安装CUDA版本的PyTorch:")
    print("pip uninstall torch torchvision torchaudio -y")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("="*60)






