"""
COAT: FP8 Optimizer States with Dynamic Range Expansion
基于论文: https://nvlabs.github.io/COAT/

实现了动态范围扩展的FP8量化优化器状态
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import math


class DynamicRangeExpansion:
    """
    动态范围扩展函数，用于将optimizer states的动态范围对齐到FP8的表示范围
    
    对于E4M3格式:
    - 最小可表示值: 0.00195
    - 最大可表示值: 448
    - 动态范围: ~2e5
    """
    
    @staticmethod
    def calculate_optimal_k(x: torch.Tensor, 
                           eps: float = 1e-8,
                           fp8_format: str = 'e4m3') -> float:
        """
        计算最优的k参数来扩展动态范围
        
        Args:
            x: 输入张量
            eps: 防止除零的小值
            fp8_format: FP8格式 ('e4m3' 或 'e5m2')
        
        Returns:
            optimal k value
        """
        # FP8 E4M3的动态范围
        if fp8_format == 'e4m3':
            fp8_min = 0.00195
            fp8_max = 448.0
        else:  # e5m2
            fp8_min = 0.0000152587890625
            fp8_max = 57344.0
        
        fp8_dynamic_range = fp8_max / fp8_min
        
        # 计算输入张量的动态范围
        x_abs = torch.abs(x)
        x_max = torch.max(x_abs)
        x_min = torch.min(x_abs[x_abs > eps])
        
        if x_min == 0 or x_max == 0:
            return 1.0
        
        x_dynamic_range = x_max / (x_min + eps)
        
        # 计算k使得扩展后的动态范围接近FP8的动态范围
        # x_max^k / x_min^k = fp8_dynamic_range
        # k = log(fp8_dynamic_range) / log(x_max/x_min)
        k = math.log(fp8_dynamic_range) / (math.log(x_dynamic_range) + eps)
        
        # 限制k在合理范围内
        k = max(1.0, min(k, 3.0))
        
        return k
    
    @staticmethod
    def expand(x: torch.Tensor, k: float) -> torch.Tensor:
        """
        应用动态范围扩展函数
        f(x) = sign(x) * |x|^k
        
        Args:
            x: 输入张量
            k: 扩展参数
        
        Returns:
            扩展后的张量
        """
        return torch.sign(x) * torch.pow(torch.abs(x), k)
    
    @staticmethod
    def contract(x: torch.Tensor, k: float) -> torch.Tensor:
        """
        反向操作，从扩展后的值恢复原始值
        
        Args:
            x: 扩展后的张量
            k: 扩展参数
        
        Returns:
            原始张量
        """
        return torch.sign(x) * torch.pow(torch.abs(x), 1.0 / k)


class FP8QuantizationConfig:
    """FP8量化配置"""
    
    def __init__(self,
                 use_fp8_m1: bool = True,  # 一阶动量使用FP8
                 use_fp8_m2: bool = True,  # 二阶动量使用FP8
                 m1_format: str = 'e4m3',  # 一阶动量格式
                 m2_format: str = 'e4m3',  # 二阶动量格式
                 block_size: int = 128,     # 分组量化的块大小
                 use_dynamic_range_expansion: bool = True):
        self.use_fp8_m1 = use_fp8_m1
        self.use_fp8_m2 = use_fp8_m2
        self.m1_format = m1_format
        self.m2_format = m2_format
        self.block_size = block_size
        self.use_dynamic_range_expansion = use_dynamic_range_expansion


class FP8Quantizer:
    """FP8量化器，支持per-group量化"""
    
    def __init__(self, config: FP8QuantizationConfig):
        self.config = config
        self.dre = DynamicRangeExpansion()
    
    def quantize_per_group(self, 
                          tensor: torch.Tensor, 
                          fp8_format: str = 'e4m3',
                          use_dre: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Per-group量化到FP8
        
        Args:
            tensor: 输入张量
            fp8_format: FP8格式
            use_dre: 是否使用动态范围扩展
        
        Returns:
            量化后的张量和量化参数
        """
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        # 计算分组数量
        n_elements = tensor_flat.numel()
        n_groups = (n_elements + self.config.block_size - 1) // self.config.block_size
        
        # Pad到block_size的倍数
        pad_size = n_groups * self.config.block_size - n_elements
        if pad_size > 0:
            tensor_flat = torch.cat([tensor_flat, torch.zeros(pad_size, 
                                                               dtype=tensor.dtype, 
                                                               device=tensor.device)])
        
        # Reshape到[n_groups, block_size]
        tensor_grouped = tensor_flat.view(n_groups, self.config.block_size)
        
        # 存储量化参数
        quant_params = {
            'original_shape': original_shape,
            'n_elements': n_elements,
            'scales': [],
            'k_values': [] if use_dre else None
        }
        
        quantized_groups = []
        
        for i in range(n_groups):
            group = tensor_grouped[i]
            
            # 动态范围扩展
            if use_dre and self.config.use_dynamic_range_expansion:
                k = self.dre.calculate_optimal_k(group, fp8_format=fp8_format)
                group_expanded = self.dre.expand(group, k)
                quant_params['k_values'].append(k)
            else:
                group_expanded = group
                if use_dre:
                    quant_params['k_values'].append(1.0)
            
            # 计算缩放因子
            abs_max = torch.max(torch.abs(group_expanded))
            if fp8_format == 'e4m3':
                fp8_max = 448.0
            else:  # e5m2
                fp8_max = 57344.0
            
            scale = abs_max / fp8_max if abs_max > 0 else 1.0
            quant_params['scales'].append(scale.item())
            
            # 量化到FP8 (模拟使用float8_e4m3fn)
            group_scaled = group_expanded / scale
            
            # PyTorch 2.1+支持float8_e4m3fn
            if hasattr(torch, 'float8_e4m3fn') and fp8_format == 'e4m3':
                group_fp8 = group_scaled.to(torch.float8_e4m3fn)
            elif hasattr(torch, 'float8_e5m2') and fp8_format == 'e5m2':
                group_fp8 = group_scaled.to(torch.float8_e5m2)
            else:
                # 降级方案：使用bfloat16
                group_fp8 = group_scaled.to(torch.bfloat16)
            
            quantized_groups.append(group_fp8)
        
        quantized_tensor = torch.stack(quantized_groups)
        
        return quantized_tensor, quant_params
    
    def dequantize_per_group(self, 
                            quantized_tensor: torch.Tensor, 
                            quant_params: Dict) -> torch.Tensor:
        """
        从FP8反量化
        
        Args:
            quantized_tensor: 量化后的张量
            quant_params: 量化参数
        
        Returns:
            反量化后的张量
        """
        n_groups = quantized_tensor.shape[0]
        dequantized_groups = []
        
        for i in range(n_groups):
            group_fp8 = quantized_tensor[i]
            
            # 转回float32
            group = group_fp8.float()
            
            # 反缩放
            scale = quant_params['scales'][i]
            group = group * scale
            
            # 反向动态范围扩展
            if quant_params['k_values'] is not None:
                k = quant_params['k_values'][i]
                if k != 1.0:
                    group = self.dre.contract(group, k)
            
            dequantized_groups.append(group)
        
        # 重组
        dequantized = torch.cat(dequantized_groups)
        
        # 移除padding并reshape
        dequantized = dequantized[:quant_params['n_elements']]
        dequantized = dequantized.view(quant_params['original_shape'])
        
        return dequantized


class FP8AdamW(torch.optim.Optimizer):
    """
    COAT FP8 AdamW优化器
    使用FP8存储optimizer states以节省内存
    """
    
    def __init__(self, 
                 params, 
                 lr: float = 1e-4,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01,
                 fp8_config: Optional[FP8QuantizationConfig] = None):
        """
        Args:
            params: 模型参数
            lr: 学习率
            betas: Adam的beta参数
            eps: 数值稳定性参数
            weight_decay: 权重衰减
            fp8_config: FP8量化配置
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.fp8_config = fp8_config or FP8QuantizationConfig()
        self.quantizer = FP8Quantizer(self.fp8_config)
    
    @torch.no_grad()
    def step(self, closure=None):
        """执行一步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    # 初始化为零
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_fp8'] = None
                    state['exp_avg_sq_fp8'] = None
                    state['exp_avg_params'] = None
                    state['exp_avg_sq_params'] = None
                
                state['step'] += 1
                
                # 权重衰减
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # 从FP8恢复momentum (如果已量化)
                if state['exp_avg_fp8'] is not None and self.fp8_config.use_fp8_m1:
                    exp_avg = self.quantizer.dequantize_per_group(
                        state['exp_avg_fp8'], 
                        state['exp_avg_params']
                    )
                else:
                    exp_avg = state['exp_avg']
                
                if state['exp_avg_sq_fp8'] is not None and self.fp8_config.use_fp8_m2:
                    exp_avg_sq = self.quantizer.dequantize_per_group(
                        state['exp_avg_sq_fp8'],
                        state['exp_avg_sq_params']
                    )
                else:
                    exp_avg_sq = state['exp_avg_sq']
                
                # 更新momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                
                # 更新参数
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # 量化并存储到FP8
                if self.fp8_config.use_fp8_m1:
                    state['exp_avg_fp8'], state['exp_avg_params'] = \
                        self.quantizer.quantize_per_group(
                            exp_avg, 
                            fp8_format=self.fp8_config.m1_format,
                            use_dre=self.fp8_config.use_dynamic_range_expansion
                        )
                else:
                    state['exp_avg'] = exp_avg
                
                if self.fp8_config.use_fp8_m2:
                    state['exp_avg_sq_fp8'], state['exp_avg_sq_params'] = \
                        self.quantizer.quantize_per_group(
                            exp_avg_sq,
                            fp8_format=self.fp8_config.m2_format,
                            use_dre=self.fp8_config.use_dynamic_range_expansion
                        )
                else:
                    state['exp_avg_sq'] = exp_avg_sq
        
        return loss

