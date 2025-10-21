import torch

print("="*60)
print("  GPU 状态检查")
print("="*60)

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"\nCUDA 可用: {cuda_available}")

if cuda_available:
    # GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"GPU 数量: {gpu_count}")
    
    # 当前 GPU
    current_gpu = torch.cuda.current_device()
    print(f"当前 GPU: {current_gpu}")
    
    # GPU 名称
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU 名称: {gpu_name}")
    
    # 显存信息
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"总显存: {total_memory / 1024**3:.2f} GB")
    
    allocated = torch.cuda.memory_allocated(0)
    print(f"已分配: {allocated / 1024**3:.2f} GB")
    
    reserved = torch.cuda.memory_reserved(0)
    print(f"已预留: {reserved / 1024**3:.2f} GB")
    
    free = total_memory - reserved
    print(f"可用: {free / 1024**3:.2f} GB")
    
    # CUDA 版本
    print(f"\nCUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 检查 FP8 支持
    capability = torch.cuda.get_device_capability(0)
    print(f"计算能力: {capability[0]}.{capability[1]}")
    
    if capability[0] >= 9 or (capability[0] == 8 and capability[1] >= 9):
        print("✅ 支持硬件 FP8 加速")
    else:
        print("⚠️  不支持硬件 FP8 (使用软件模拟)")
else:
    print("❌ CUDA 不可用！请检查:")
    print("   1. 是否安装了 NVIDIA 驱动")
    print("   2. PyTorch 是否为 CUDA 版本")

print("\n" + "="*60)


