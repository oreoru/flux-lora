"""
COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training
用于FLUX LoRA训练的FP8加速实现

论文: https://nvlabs.github.io/COAT/
"""

from .fp8_optimizer import (
    FP8AdamW,
    FP8QuantizationConfig,
    FP8Quantizer,
    DynamicRangeExpansion
)

from .fp8_activation import (
    FP8ActivationQuantizer,
    FP8PrecisionFlow,
    FP8LinearWrapper,
    replace_linear_with_fp8,
    MemoryEfficientCheckpoint
)

from .coat_trainer import (
    COATConfig,
    COATTrainer,
    create_coat_trainer_for_flux_lora
)

__version__ = "0.1.0"

__all__ = [
    # Optimizer
    'FP8AdamW',
    'FP8QuantizationConfig',
    'FP8Quantizer',
    'DynamicRangeExpansion',
    
    # Activation
    'FP8ActivationQuantizer',
    'FP8PrecisionFlow',
    'FP8LinearWrapper',
    'replace_linear_with_fp8',
    'MemoryEfficientCheckpoint',
    
    # Trainer
    'COATConfig',
    'COATTrainer',
    'create_coat_trainer_for_flux_lora',
]




