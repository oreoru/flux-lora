"""
COAT训练器集成模块
用于ai-toolkit的FLUX LoRA训练

使用方法:
1. 在ai-toolkit的训练配置中启用COAT
2. 自动应用FP8优化器和激活量化
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import time
from dataclasses import dataclass

from .fp8_optimizer import FP8AdamW, FP8QuantizationConfig
from .fp8_activation import (
    FP8ActivationQuantizer, 
    FP8PrecisionFlow,
    replace_linear_with_fp8
)


@dataclass
class COATConfig:
    """COAT训练配置"""
    
    # 优化器状态量化
    use_fp8_optimizer: bool = True
    optimizer_m1_format: str = 'e4m3'  # 一阶动量格式
    optimizer_m2_format: str = 'e4m3'  # 二阶动量格式
    optimizer_block_size: int = 128
    use_dynamic_range_expansion: bool = True
    
    # 激活量化
    use_fp8_activation: bool = True
    activation_linear_granularity: str = 'per_tensor'
    activation_nonlinear_granularity: str = 'per_group'
    activation_group_size: int = 128
    
    # 训练设置
    enable_memory_efficient_checkpoint: bool = True
    log_memory_stats: bool = True
    
    def to_fp8_quant_config(self) -> FP8QuantizationConfig:
        """转换为FP8量化配置"""
        return FP8QuantizationConfig(
            use_fp8_m1=self.use_fp8_optimizer,
            use_fp8_m2=self.use_fp8_optimizer,
            m1_format=self.optimizer_m1_format,
            m2_format=self.optimizer_m2_format,
            block_size=self.optimizer_block_size,
            use_dynamic_range_expansion=self.use_dynamic_range_expansion
        )


class COATTrainer:
    """
    COAT训练器
    封装了COAT的所有优化技术
    """
    
    def __init__(self, config: COATConfig):
        self.config = config
        self.precision_flow = FP8PrecisionFlow() if config.use_fp8_activation else None
        self.memory_stats = []
    
    def create_optimizer(self, 
                        model_params,
                        lr: float = 1e-4,
                        **optimizer_kwargs) -> torch.optim.Optimizer:
        """
        创建优化器
        
        Args:
            model_params: 模型参数
            lr: 学习率
            optimizer_kwargs: 其他优化器参数
        
        Returns:
            优化器实例
        """
        if self.config.use_fp8_optimizer:
            print("🚀 使用COAT FP8优化器")
            fp8_config = self.config.to_fp8_quant_config()
            return FP8AdamW(
                model_params,
                lr=lr,
                fp8_config=fp8_config,
                **optimizer_kwargs
            )
        else:
            print("使用标准AdamW优化器")
            return torch.optim.AdamW(model_params, lr=lr, **optimizer_kwargs)
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        准备模型以使用COAT
        
        Args:
            model: 要准备的模型
        
        Returns:
            修改后的模型
        """
        if self.config.use_fp8_activation:
            print("🔄 将模型的Linear层替换为FP8版本...")
            model = replace_linear_with_fp8(model, recursive=True)
        
        return model
    
    def log_memory(self, step: int, phase: str = ""):
        """记录内存使用情况"""
        if not self.config.log_memory_stats:
            return
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            
            self.memory_stats.append({
                'step': step,
                'phase': phase,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'timestamp': time.time()
            })
            
            print(f"📊 [{phase}] Step {step}: "
                  f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """获取内存使用报告"""
        if not self.memory_stats:
            return {}
        
        import numpy as np
        
        allocated = [s['allocated_gb'] for s in self.memory_stats]
        reserved = [s['reserved_gb'] for s in self.memory_stats]
        
        return {
            'peak_allocated_gb': max(allocated),
            'peak_reserved_gb': max(reserved),
            'avg_allocated_gb': np.mean(allocated),
            'avg_reserved_gb': np.mean(reserved),
            'total_samples': len(self.memory_stats)
        }
    
    def training_step(self,
                     model: nn.Module,
                     batch: Dict[str, torch.Tensor],
                     optimizer: torch.optim.Optimizer,
                     step: int) -> Dict[str, float]:
        """
        执行一步训练
        
        Args:
            model: 模型
            batch: 批次数据
            optimizer: 优化器
            step: 当前步数
        
        Returns:
            损失字典
        """
        self.log_memory(step, "before_forward")
        
        # 前向传播
        if self.precision_flow:
            self.precision_flow.clear_cache()
        
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        self.log_memory(step, "after_forward")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        self.log_memory(step, "after_backward")
        
        # 优化器步进
        optimizer.step()
        
        self.log_memory(step, "after_optimizer_step")
        
        return {'loss': loss.item()}


def create_coat_trainer_for_flux_lora(
    learning_rate: float = 1e-4,
    enable_fp8_optimizer: bool = True,
    enable_fp8_activation: bool = True,
    **kwargs
) -> COATTrainer:
    """
    为FLUX LoRA训练创建COAT训练器的便捷函数
    
    Args:
        learning_rate: 学习率
        enable_fp8_optimizer: 是否启用FP8优化器
        enable_fp8_activation: 是否启用FP8激活
        kwargs: 其他COAT配置参数
    
    Returns:
        COATTrainer实例
    
    示例:
        >>> trainer = create_coat_trainer_for_flux_lora(
        ...     learning_rate=1e-4,
        ...     enable_fp8_optimizer=True,
        ...     enable_fp8_activation=True
        ... )
    """
    config = COATConfig(
        use_fp8_optimizer=enable_fp8_optimizer,
        use_fp8_activation=enable_fp8_activation,
        **kwargs
    )
    
    return COATTrainer(config)

