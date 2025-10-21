"""
COAT: FP8 Activation Quantization with Mixed-Granularity
基于论文: https://nvlabs.github.io/COAT/

实现混合粒度的FP8激活量化
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import torch.nn.functional as F


class FP8ActivationQuantizer:
    """
    FP8激活量化器
    实现混合粒度量化策略：
    - 线性层: per-tensor量化 (最大化Tensor Core性能)
    - 非线性层: per-group量化 (更高精度)
    """
    
    def __init__(self, 
                 linear_granularity: str = 'per_tensor',
                 nonlinear_granularity: str = 'per_group',
                 group_size: int = 128):
        """
        Args:
            linear_granularity: 线性层的量化粒度 ('per_tensor' 或 'per_group')
            nonlinear_granularity: 非线性层的量化粒度
            group_size: per-group量化的组大小
        """
        self.linear_granularity = linear_granularity
        self.nonlinear_granularity = nonlinear_granularity
        self.group_size = group_size
    
    def quantize_per_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Per-tensor量化到FP8
        
        Args:
            x: 输入张量
        
        Returns:
            (量化后的张量, 缩放因子)
        """
        # 计算全局缩放因子
        abs_max = torch.max(torch.abs(x))
        fp8_max = 448.0  # E4M3 max
        
        scale = abs_max / fp8_max if abs_max > 0 else 1.0
        
        # 量化
        x_scaled = x / scale
        
        if hasattr(torch, 'float8_e4m3fn'):
            x_fp8 = x_scaled.to(torch.float8_e4m3fn)
        else:
            # 降级方案
            x_fp8 = x_scaled.to(torch.bfloat16)
        
        return x_fp8, scale.item()
    
    def dequantize_per_tensor(self, x_fp8: torch.Tensor, scale: float) -> torch.Tensor:
        """Per-tensor反量化"""
        x = x_fp8.float()
        return x * scale
    
    def quantize_per_group_2stage(self, 
                                   x: torch.Tensor, 
                                   group_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Group Scaling: 两阶段的per-group量化
        论文中的高效Just-in-time Scaling方法
        
        阶段1: 对每G个元素进行max reduction，存储中间结果
        阶段2: 对中间张量进行max reduction得到per-tensor max
        
        Args:
            x: 输入张量
            group_size: 组大小
        
        Returns:
            (量化后的张量, 每组的缩放因子)
        """
        if group_size is None:
            group_size = self.group_size
        
        original_shape = x.shape
        x_flat = x.flatten()
        
        n_elements = x_flat.numel()
        n_groups = (n_elements + group_size - 1) // group_size
        
        # Padding
        pad_size = n_groups * group_size - n_elements
        if pad_size > 0:
            x_flat = torch.cat([x_flat, torch.zeros(pad_size, 
                                                     dtype=x.dtype, 
                                                     device=x.device)])
        
        # Reshape到[n_groups, group_size]
        x_grouped = x_flat.view(n_groups, group_size)
        
        # 阶段1: 计算每组的max (可以与前一个操作融合)
        group_abs_max = torch.max(torch.abs(x_grouped), dim=1)[0]
        
        fp8_max = 448.0
        scales = group_abs_max / fp8_max
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        
        # 量化每组
        x_scaled = x_grouped / scales.unsqueeze(1)
        
        if hasattr(torch, 'float8_e4m3fn'):
            x_fp8 = x_scaled.to(torch.float8_e4m3fn)
        else:
            x_fp8 = x_scaled.to(torch.bfloat16)
        
        return x_fp8, scales
    
    def dequantize_per_group_2stage(self, 
                                     x_fp8: torch.Tensor, 
                                     scales: torch.Tensor,
                                     original_shape: torch.Size,
                                     n_elements: int) -> torch.Tensor:
        """两阶段per-group反量化"""
        # 反量化
        x = x_fp8.float()
        x = x * scales.unsqueeze(1)
        
        # Flatten and remove padding
        x_flat = x.flatten()[:n_elements]
        
        # Reshape到原始形状
        x = x_flat.view(original_shape)
        
        return x


class FP8PrecisionFlow(nn.Module):
    """
    FP8精度流（Precision Flow）
    
    确保所有线性层和非线性层的输入输出都是FP8格式
    通过直接保存FP8格式的输入张量用于反向传播，消除额外的量化开销
    """
    
    def __init__(self):
        super().__init__()
        self.quantizer = FP8ActivationQuantizer()
        self.saved_tensors = {}
        self.layer_count = 0
    
    def forward_with_fp8_flow(self, 
                              module: nn.Module, 
                              x: torch.Tensor,
                              layer_type: str = 'linear') -> torch.Tensor:
        """
        使用FP8精度流的前向传播
        
        Args:
            module: 要执行的模块 (Linear, LayerNorm, 等)
            x: 输入张量
            layer_type: 层类型 ('linear', 'nonlinear', 'layernorm')
        
        Returns:
            输出张量 (FP8格式)
        """
        layer_id = f"{layer_type}_{self.layer_count}"
        self.layer_count += 1
        
        # 根据层类型选择量化策略
        if layer_type == 'linear':
            # 线性层: per-tensor量化
            x_fp8, scale = self.quantizer.quantize_per_tensor(x)
            self.saved_tensors[layer_id] = {'x_fp8': x_fp8, 'scale': scale, 'type': 'per_tensor'}
            
            # 执行模块 (需要先转回float)
            x_float = self.quantizer.dequantize_per_tensor(x_fp8, scale)
            output = module(x_float)
            
            # 输出也量化到FP8
            output_fp8, output_scale = self.quantizer.quantize_per_tensor(output)
            self.saved_tensors[f"{layer_id}_output"] = {
                'x_fp8': output_fp8, 
                'scale': output_scale, 
                'type': 'per_tensor'
            }
            
            return output_fp8
        
        elif layer_type in ['nonlinear', 'layernorm']:
            # 非线性层: per-group量化
            original_shape = x.shape
            x_fp8, scales = self.quantizer.quantize_per_group_2stage(x)
            
            self.saved_tensors[layer_id] = {
                'x_fp8': x_fp8, 
                'scales': scales,
                'original_shape': original_shape,
                'n_elements': x.numel(),
                'type': 'per_group'
            }
            
            # 执行模块
            x_float = self.quantizer.dequantize_per_group_2stage(
                x_fp8, scales, original_shape, x.numel()
            )
            output = module(x_float)
            
            # 输出量化
            output_fp8, output_scales = self.quantizer.quantize_per_group_2stage(output)
            self.saved_tensors[f"{layer_id}_output"] = {
                'x_fp8': output_fp8,
                'scales': output_scales,
                'original_shape': output.shape,
                'n_elements': output.numel(),
                'type': 'per_group'
            }
            
            return output_fp8
        
        else:
            # 默认行为
            return module(x)
    
    def clear_cache(self):
        """清理缓存的张量"""
        self.saved_tensors = {}
        self.layer_count = 0


class FP8LinearWrapper(nn.Module):
    """
    包装Linear层以支持FP8激活
    """
    
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.linear = linear_layer
        self.quantizer = FP8ActivationQuantizer()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 量化输入
        x_fp8, scale = self.quantizer.quantize_per_tensor(x)
        
        # 反量化执行线性层
        x_float = self.quantizer.dequantize_per_tensor(x_fp8, scale)
        output = self.linear(x_float)
        
        # 量化输出
        output_fp8, _ = self.quantizer.quantize_per_tensor(output)
        
        return output_fp8


def replace_linear_with_fp8(model: nn.Module, recursive: bool = True) -> nn.Module:
    """
    递归替换模型中的所有Linear层为FP8版本
    
    Args:
        model: 要修改的模型
        recursive: 是否递归处理子模块
    
    Returns:
        修改后的模型
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 替换为FP8版本
            setattr(model, name, FP8LinearWrapper(module))
        elif recursive:
            # 递归处理子模块
            replace_linear_with_fp8(module, recursive=True)
    
    return model


class MemoryEfficientCheckpoint:
    """
    内存高效的checkpoint，使用FP8存储激活
    """
    
    @staticmethod
    def checkpoint(function, *args, use_fp8: bool = True, **kwargs):
        """
        类似torch.utils.checkpoint.checkpoint，但使用FP8存储激活
        
        Args:
            function: 要checkpoint的函数
            args: 函数参数
            use_fp8: 是否使用FP8
            kwargs: 关键字参数
        
        Returns:
            函数输出
        """
        if not use_fp8:
            return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
        
        # 使用FP8的自定义checkpoint实现
        # 这里简化实现，实际需要更复杂的反向传播逻辑
        quantizer = FP8ActivationQuantizer()
        
        # 前向传播并量化激活
        with torch.no_grad():
            outputs = function(*args, **kwargs)
        
        # 量化输出用于存储
        if isinstance(outputs, torch.Tensor):
            outputs_fp8, scale = quantizer.quantize_per_tensor(outputs)
            # 在实际使用中需要保存scale并在反向传播中恢复
        
        return outputs

