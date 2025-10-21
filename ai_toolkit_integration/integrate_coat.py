"""
将COAT集成到ai-toolkit的脚本
修改ai-toolkit的训练流程以支持COAT优化

使用方法:
1. 将此文件放在ai-toolkit目录下
2. 运行: python integrate_coat.py
3. 使用coat_config.yaml进行训练
"""

import os
import sys
from pathlib import Path


def patch_ai_toolkit_optimizer():
    """
    修补ai-toolkit的优化器创建逻辑
    添加对COAT FP8优化器的支持
    """
    
    # 假设ai-toolkit的优化器在 toolkit/optimizers.py 或类似位置
    optimizer_patch = '''
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
'''
    
    return optimizer_patch


def patch_ai_toolkit_trainer():
    """
    修补ai-toolkit的训练器
    添加FP8激活量化支持
    """
    
    trainer_patch = '''
# COAT FP8激活量化集成
import sys
from pathlib import Path

coat_path = Path(__file__).parent.parent / "coat_implementation"
sys.path.insert(0, str(coat_path))

from coat_implementation import replace_linear_with_fp8, FP8PrecisionFlow

class COATEnhancedTrainer:
    """增强的训练器，集成COAT优化"""
    
    def __init__(self, config):
        self.config = config
        self.coat_enabled = config.get('coat', {}).get('enabled', False)
        self.precision_flow = None
        
        if self.coat_enabled:
            print("🚀 COAT优化已启用!")
            self.precision_flow = FP8PrecisionFlow()
    
    def prepare_model(self, model):
        """准备模型"""
        if self.coat_enabled:
            activation_config = self.config.get('coat', {}).get('activation', {})
            if activation_config.get('use_fp8', False):
                print("🔄 应用FP8激活量化...")
                model = replace_linear_with_fp8(model, recursive=True)
        
        return model
    
    def training_step(self, model, batch, optimizer, step):
        """训练步骤"""
        
        # 记录内存
        if self.coat_enabled and self.config.get('coat', {}).get('memory', {}).get('log_memory_stats', False):
            self._log_memory(step, "before_forward")
        
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        if self.coat_enabled:
            self._log_memory(step, "after_forward")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        if self.coat_enabled:
            self._log_memory(step, "after_backward")
        
        # 梯度裁剪
        if 'gradient_clipping' in self.config.get('train', {}):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.config['train']['gradient_clipping']
            )
        
        # 优化器步进
        optimizer.step()
        
        if self.coat_enabled:
            self._log_memory(step, "after_optimizer_step")
        
        return {'loss': loss.item()}
    
    def _log_memory(self, step, phase):
        """记录内存使用"""
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 [{phase}] Step {step}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
'''
    
    return trainer_patch


def create_integration_guide():
    """创建集成指南"""
    
    guide = """
# COAT + ai-toolkit 集成指南

## 1. 安装依赖

```bash
pip install torch>=2.1.0  # 需要支持FP8的PyTorch版本
pip install transformers diffusers accelerate
```

## 2. 目录结构

确保目录结构如下:
```
flux lora/
├── coat_implementation/          # COAT实现代码
│   ├── __init__.py
│   ├── fp8_optimizer.py
│   ├── fp8_activation.py
│   └── coat_trainer.py
├── ai-toolkit/                   # ai-toolkit仓库
│   └── ...
├── ai_toolkit_integration/       # 集成代码
│   ├── coat_config.yaml
│   └── integrate_coat.py
└── datasets/
    └── clothing/                 # 服装图片数据集
        ├── image1.jpg
        ├── image1.txt
        ├── image2.jpg
        ├── image2.txt
        └── ...
```

## 3. 修改ai-toolkit代码

### 方法A: 手动集成 (推荐)

#### 3.1 修改优化器创建代码

在 `ai-toolkit/toolkit/optimizers.py` (或相应文件) 中添加:

```python
# 在文件开头添加
import sys
from pathlib import Path
coat_path = Path(__file__).parent.parent.parent / "coat_implementation"
sys.path.insert(0, str(coat_path))

from coat_implementation import FP8AdamW, FP8QuantizationConfig

# 在create_optimizer函数中添加
def get_optimizer(config, parameters):
    optimizer_type = config.get('train', {}).get('optimizer', 'adamw')
    lr = config.get('train', {}).get('lr', 1e-4)
    
    if optimizer_type == "coat_fp8_adamw":
        coat_config = config.get('coat', {}).get('optimizer', {})
        fp8_config = FP8QuantizationConfig(
            use_fp8_m1=coat_config.get('use_fp8', True),
            use_fp8_m2=coat_config.get('use_fp8', True),
            m1_format=coat_config.get('m1_format', 'e4m3'),
            m2_format=coat_config.get('m2_format', 'e4m3'),
            block_size=coat_config.get('block_size', 128),
            use_dynamic_range_expansion=coat_config.get('use_dynamic_range_expansion', True)
        )
        
        return FP8AdamW(parameters, lr=lr, fp8_config=fp8_config)
    
    # 原有逻辑...
```

#### 3.2 修改训练器代码

在 `ai-toolkit/toolkit/trainer.py` (或相应文件) 中:

```python
# 导入COAT模块
from coat_implementation import replace_linear_with_fp8

class Trainer:
    def setup(self):
        # 在模型准备阶段
        if self.config.get('coat', {}).get('enabled', False):
            if self.config.get('coat', {}).get('activation', {}).get('use_fp8', False):
                print("🔄 应用COAT FP8激活量化...")
                self.model = replace_linear_with_fp8(self.model)
```

### 方法B: 使用Monkey Patching

创建 `train_with_coat.py`:

```python
import sys
sys.path.insert(0, 'coat_implementation')

# 导入ai-toolkit
from ai_toolkit import train

# 导入COAT
from coat_implementation import COATTrainer, COATConfig

# Monkey patch ai-toolkit的训练逻辑
original_create_optimizer = train.create_optimizer

def coat_create_optimizer(config, parameters):
    if config.get('train', {}).get('optimizer') == 'coat_fp8_adamw':
        coat_trainer = COATTrainer(COATConfig())
        return coat_trainer.create_optimizer(parameters, lr=config.get('train', {}).get('lr', 1e-4))
    return original_create_optimizer(config, parameters)

train.create_optimizer = coat_create_optimizer

# 运行训练
if __name__ == "__main__":
    train.main()
```

## 4. 准备数据集

### 4.1 数据集格式

服装图片数据集应该包含:
- 图片文件: `.jpg`, `.jpeg`, `.png`
- 标注文件: 与图片同名的 `.txt` 文件

示例:
```
datasets/clothing/
├── dress_001.jpg
├── dress_001.txt      # 内容: "a beautiful red [trigger] dress with floral patterns"
├── jacket_002.jpg
├── jacket_002.txt     # 内容: "a leather [trigger] jacket, black color"
└── ...
```

### 4.2 标注建议

- 使用描述性标注: "a [trigger] dress with floral patterns, high quality"
- 包含触发词: `[trigger]` 会被配置中的 `trigger_word` 替换
- 描述颜色、材质、风格等细节

## 5. 运行训练

```bash
cd ai-toolkit
python run.py ../ai_toolkit_integration/coat_config.yaml
```

或使用自定义脚本:
```bash
python train_with_coat.py --config ../ai_toolkit_integration/coat_config.yaml
```

## 6. 性能对比

COAT预期收益:
- ✅ 内存使用减少 ~1.54x
- ✅ 训练速度提升 ~1.43x
- ✅ 可以使用更大的batch size (提升到2-4倍)
- ✅ 模型精度无损失

### 监控内存使用

训练时会自动打印内存统计:
```
📊 [before_forward] Step 100: Allocated: 8.23GB, Reserved: 10.50GB
📊 [after_forward] Step 100: Allocated: 12.45GB, Reserved: 14.00GB
📊 [after_backward] Step 100: Allocated: 10.67GB, Reserved: 14.00GB
📊 [after_optimizer_step] Step 100: Allocated: 8.23GB, Reserved: 14.00GB
```

## 7. 常见问题

### Q: PyTorch不支持FP8怎么办?

A: 代码会自动降级到bfloat16。虽然内存节省效果会打折扣，但仍然可以运行。建议使用PyTorch 2.1+和CUDA 12+。

### Q: 训练速度没有提升?

A: 确保:
1. 使用支持FP8的GPU (如H100, L40S)
2. CUDA版本 >= 12.0
3. batch_size足够大以利用内存节省

### Q: 精度损失?

A: COAT设计为无损加速。如果观察到精度损失，尝试:
1. 调整 `block_size` (默认128)
2. 使用 `e5m2` 格式代替 `e4m3`
3. 禁用动态范围扩展 (不推荐)

## 8. 进阶配置

### 只训练特定层

```yaml
network:
  type: "lora"
  linear: 16
  linear_alpha: 16
  network_kwargs:
    only_if_contains:
      - "transformer.single_transformer_blocks.7.proj_out"
      - "transformer.single_transformer_blocks.20.proj_out"
```

### 调整FP8格式

```yaml
coat:
  optimizer:
    m1_format: "e4m3"  # 一阶动量: e4m3 (推荐) 或 e5m2
    m2_format: "e5m2"  # 二阶动量: e4m3 或 e5m2
```

### 自定义量化粒度

```yaml
coat:
  activation:
    linear_granularity: "per_tensor"    # per_tensor 或 per_group
    nonlinear_granularity: "per_group"  
    group_size: 256                     # 增大以减少开销
```

## 9. 引用

如果使用COAT方法，请引用原论文:
```bibtex
@article{xi2024coat,
  title={COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training},
  author={Xi, Haocheng and Cai, Han and Zhu, Ligeng and Lu, Yao and Keutzer, Kurt and Chen, Jianfei and Han, Song},
  journal={arXiv preprint arXiv:2410.19313},
  year={2024}
}
```
"""
    
    return guide


def main():
    """主函数"""
    
    print("=" * 60)
    print("COAT + ai-toolkit 集成工具")
    print("=" * 60)
    print()
    
    # 创建集成指南
    guide_path = Path("ai_toolkit_integration/INTEGRATION_GUIDE.md")
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(create_integration_guide())
    
    print(f"✅ 集成指南已创建: {guide_path}")
    print()
    
    # 生成补丁文件
    optimizer_patch_path = Path("ai_toolkit_integration/patches/optimizer_patch.py")
    optimizer_patch_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(optimizer_patch_path, 'w', encoding='utf-8') as f:
        f.write(patch_ai_toolkit_optimizer())
    
    print(f"✅ 优化器补丁已创建: {optimizer_patch_path}")
    
    trainer_patch_path = Path("ai_toolkit_integration/patches/trainer_patch.py")
    with open(trainer_patch_path, 'w', encoding='utf-8') as f:
        f.write(patch_ai_toolkit_trainer())
    
    print(f"✅ 训练器补丁已创建: {trainer_patch_path}")
    print()
    
    print("=" * 60)
    print("下一步:")
    print("1. 阅读集成指南: cat ai_toolkit_integration/INTEGRATION_GUIDE.md")
    print("2. 准备服装数据集到 datasets/clothing/")
    print("3. 根据指南修改ai-toolkit代码")
    print("4. 运行训练: cd ai-toolkit && python run.py ../ai_toolkit_integration/coat_config.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()

