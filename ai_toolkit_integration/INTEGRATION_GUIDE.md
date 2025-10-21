
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
