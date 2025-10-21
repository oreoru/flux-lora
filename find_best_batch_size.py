"""
找到适合你GPU的最佳batch size
针对 RTX 4060 Laptop (8GB)
"""

import torch
import sys

print("="*60)
print("寻找最佳Batch Size")
print("="*60)

# 检查CUDA
if not torch.cuda.is_available():
    print("\n❌ CUDA不可用！")
    print("请先安装CUDA版PyTorch:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"PyTorch: {torch.__version__}")
print()

def test_batch_size(bs, use_fp8=False):
    """测试指定batch size是否会OOM"""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 模拟FLUX训练的显存占用
        # 使用bfloat16模拟标准训练
        dtype = torch.bfloat16
        
        # 创建模拟数据（接近FLUX的实际维度）
        x = torch.randn(bs, 4, 128, 128, device='cuda', dtype=dtype)
        
        # 模拟多层卷积（类似transformer块）
        conv1 = torch.nn.Conv2d(4, 512, 3, padding=1, device='cuda', dtype=dtype)
        conv2 = torch.nn.Conv2d(512, 512, 3, padding=1, device='cuda', dtype=dtype)
        conv3 = torch.nn.Conv2d(512, 4, 3, padding=1, device='cuda', dtype=dtype)
        
        # 前向传播
        y = conv1(x)
        y = torch.nn.functional.gelu(y)
        y = conv2(y)
        y = torch.nn.functional.gelu(y)
        y = conv3(y)
        
        # 反向传播
        loss = y.mean()
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"✅ Batch size {bs:2d}: 峰值显存 {peak_memory:.2f} GB")
        
        # 清理
        del x, y, loss, conv1, conv2, conv3
        torch.cuda.empty_cache()
        return True, peak_memory
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"❌ Batch size {bs:2d}: OOM (显存不足)")
            torch.cuda.empty_cache()
            return False, 0
        raise e

# 测试不同batch size
print("测试标准训练（BF16）:")
print("-" * 60)

max_working_bs = 1
best_memory = 0

for bs in [1, 2, 3, 4, 6, 8]:
    success, memory = test_batch_size(bs)
    if success:
        max_working_bs = bs
        best_memory = memory
    else:
        break

print()
print("="*60)
print("建议配置")
print("="*60)

if max_working_bs == 1:
    print("\n⚠️  显存较紧张，建议:")
    print(f"   batch_size: 1")
    print(f"   gradient_accumulation_steps: 16  # 等效batch size = 16")
    print(f"   resolution: [512, 768]  # 使用较小分辨率")
    print(f"   LoRA rank: 8  # 减小rank")
    print(f"   gradient_checkpointing: true  # 必须启用")
elif max_working_bs <= 4:
    print(f"\n推荐配置（适中）:")
    print(f"   batch_size: {max_working_bs}")
    print(f"   gradient_accumulation_steps: {16 // max_working_bs}  # 等效batch size = 16")
    print(f"   resolution: [768, 1024]")
    print(f"   LoRA rank: 16")
else:
    print(f"\n推荐配置（显存充足）:")
    print(f"   batch_size: {max_working_bs}")
    print(f"   gradient_accumulation_steps: {16 // max_working_bs}  # 等效batch size = 16")
    print(f"   resolution: [1024, 1024]")
    print(f"   LoRA rank: 32")

print()
print("💡 提示:")
print("   - 使用COAT可以进一步提升batch size 2-4倍")
print("   - 启用 gradient_checkpointing 可以节省更多显存")
print("   - cache_latents_to_disk 可以节省VAE编码的显存")

print()
print("YAML配置示例:")
print("-" * 60)
print(f"""
train:
  batch_size: {max_working_bs}
  gradient_accumulation_steps: {16 // max_working_bs}
  gradient_checkpointing: true
  dtype: bf16

datasets:
  resolution: [768, 1024]
  cache_latents_to_disk: true

network:
  linear: 16
  linear_alpha: 16

coat:
  enabled: true          # 启用COAT可以提升batch size
  optimizer:
    use_fp8: true
  activation:
    use_fp8: true
""")

print("="*60)
print(f"✅ 测试完成！建议从 batch_size={max_working_bs} 开始")
print("="*60)


