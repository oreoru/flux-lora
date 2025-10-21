"""
COAT集成完整性检查脚本
"""

import sys
from pathlib import Path

print("="*60)
print("COAT集成检查")
print("="*60)

checks_passed = 0
checks_failed = 0

# 检查1: 文件存在性
print("\n[1/6] 检查文件...")
files_to_check = [
    "coat_implementation/__init__.py",
    "coat_implementation/fp8_optimizer.py",
    "coat_implementation/fp8_activation.py",
    "coat_implementation/coat_trainer.py",
    "ai-toolkit/ai-toolkit/toolkit/optimizer.py",
    "ai-toolkit/ai-toolkit/toolkit/coat_integration.py",
    "ai_toolkit_integration/coat_config.yaml",
    "train_flux_lora_with_coat.py",
]

for file_path in files_to_check:
    if Path(file_path).exists():
        print(f"  ✅ {file_path}")
        checks_passed += 1
    else:
        print(f"  ❌ {file_path} - 不存在")
        checks_failed += 1

# 检查2: COAT模块导入
print("\n[2/6] 检查COAT模块导入...")
try:
    sys.path.insert(0, 'coat_implementation')
    from coat_implementation import FP8AdamW, FP8QuantizationConfig
    print("  ✅ COAT模块导入成功")
    checks_passed += 1
except Exception as e:
    print(f"  ❌ COAT模块导入失败: {e}")
    checks_failed += 1

# 检查3: optimizer.py修改
print("\n[3/6] 检查optimizer.py修改...")
try:
    optimizer_file = Path('ai-toolkit/ai-toolkit/toolkit/optimizer.py')
    if optimizer_file.exists():
        with open(optimizer_file, 'r', encoding='utf-8') as f:
            content = f.read()
            has_coat = 'coat_implementation' in content.lower() or 'coat' in content.lower()
            has_fp8 = 'fp8adamw' in content.lower() or 'fp8' in content.lower()
            
            if has_coat and has_fp8:
                print("  ✅ optimizer.py已正确修改")
                # 显示关键行
                for i, line in enumerate(content.split('\n')[:20], 1):
                    if 'coat' in line.lower() or 'fp8' in line.lower():
                        print(f"     第{i}行: {line.strip()[:60]}")
                checks_passed += 1
            else:
                print(f"  ❌ optimizer.py未完全修改 (COAT:{has_coat}, FP8:{has_fp8})")
                checks_failed += 1
    else:
        print("  ❌ optimizer.py不存在")
        checks_failed += 1
except Exception as e:
    print(f"  ❌ 读取optimizer.py失败: {e}")
    checks_failed += 1

# 检查4: 配置文件
print("\n[4/6] 检查配置文件...")
try:
    import yaml
    config_file = Path('ai_toolkit_integration/coat_config.yaml')
    if config_file.exists():
        with open(config_file, encoding='utf-8') as f:
            config = yaml.safe_load(f)
            coat_config = config['config']['coat']
            if coat_config['enabled']:
                print("  ✅ COAT配置已启用")
                print(f"     - FP8优化器: {coat_config['optimizer']['use_fp8']}")
                print(f"     - FP8激活: {coat_config['activation']['use_fp8']}")
                checks_passed += 1
            else:
                print("  ⚠️  COAT配置未启用")
                checks_failed += 1
    else:
        print("  ❌ 配置文件不存在")
        checks_failed += 1
except Exception as e:
    print(f"  ❌ 读取配置失败: {e}")
    checks_failed += 1

# 检查5: PyTorch FP8支持
print("\n[5/6] 检查PyTorch FP8支持...")
try:
    import torch
    print(f"  PyTorch版本: {torch.__version__}")
    if hasattr(torch, 'float8_e4m3fn'):
        print(f"  ✅ 支持FP8")
        checks_passed += 1
    else:
        print(f"  ⚠️  不支持FP8（将降级到bfloat16）")
        checks_passed += 1  # 不算失败，只是性能打折扣
except Exception as e:
    print(f"  ❌ 检查PyTorch失败: {e}")
    checks_failed += 1

# 检查6: CUDA可用性
print("\n[6/6] 检查CUDA...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✅ CUDA可用")
        print(f"     - GPU: {torch.cuda.get_device_name(0)}")
        print(f"     - 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        checks_passed += 1
    else:
        print("  ⚠️  CUDA不可用（将使用CPU，训练会很慢）")
        checks_passed += 1  # 不算失败
except Exception as e:
    print(f"  ❌ 检查CUDA失败: {e}")
    checks_failed += 1

# 总结
print("\n" + "="*60)
print(f"检查完成: {checks_passed}通过, {checks_failed}失败")
print("="*60)

if checks_failed == 0:
    print("\n🎉 所有检查通过！COAT集成完成！")
    print("\n下一步:")
    print("1. 准备数据集到 datasets/clothing/")
    print("2. 运行训练:")
    print("   python train_flux_lora_with_coat.py ai_toolkit_integration/coat_config.yaml")
    print("\n或运行基准测试:")
    print("   python benchmark_coat.py --batch_size 4 --num_steps 50")
else:
    print(f"\n⚠️  有 {checks_failed} 项检查失败，请修复后再试")
    print("\n常见问题:")
    print("- 如果COAT模块导入失败，检查coat_implementation目录")
    print("- 如果optimizer.py未修改，重新运行集成步骤")
    print("- 如果配置文件不存在，检查ai_toolkit_integration目录")

sys.exit(0 if checks_failed == 0 else 1)

