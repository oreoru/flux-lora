
# COAT FP8优化器集成
import sys
from pathlib import Path

# 添加COAT路径
coat_path = Path(__file__).parent.parent / "coat_implementation"
sys.path.insert(0, str(coat_path))

from coat_implementation import FP8AdamW, FP8QuantizationConfig

def create_optimizer(optimizer_type, parameters, lr, **kwargs):
    """创建优化器，支持COAT FP8优化器"""
    
    if optimizer_type == "coat_fp8_adamw":
        # 从kwargs提取COAT配置
        fp8_config = FP8QuantizationConfig(
            use_fp8_m1=kwargs.get('use_fp8_m1', True),
            use_fp8_m2=kwargs.get('use_fp8_m2', True),
            m1_format=kwargs.get('m1_format', 'e4m3'),
            m2_format=kwargs.get('m2_format', 'e4m3'),
            block_size=kwargs.get('block_size', 128),
            use_dynamic_range_expansion=kwargs.get('use_dynamic_range_expansion', True)
        )
        
        return FP8AdamW(
            parameters,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.01),
            fp8_config=fp8_config
        )
    
    # 原有优化器逻辑...
    elif optimizer_type == "adamw":
        import torch
        return torch.optim.AdamW(parameters, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
