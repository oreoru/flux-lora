"""
简单的COAT训练示例
不依赖ai-toolkit的独立训练脚本

运行方法:
python examples/simple_train.py --dataset_path datasets/clothing --output_dir output/simple_coat
"""

import sys
sys.path.insert(0, '.')

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import json

from coat_implementation import (
    COATTrainer,
    COATConfig,
    create_coat_trainer_for_flux_lora
)


def load_dataset(dataset_path: str):
    """
    加载服装数据集
    
    Args:
        dataset_path: 数据集路径
    
    Returns:
        图片和标注列表
    """
    dataset_path = Path(dataset_path)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(dataset_path.glob(ext)))
    
    print(f"找到 {len(image_files)} 张图片")
    
    dataset = []
    for img_file in image_files:
        caption_file = img_file.with_suffix('.txt')
        
        if caption_file.exists():
            caption = caption_file.read_text(encoding='utf-8').strip()
            dataset.append({
                'image': str(img_file),
                'caption': caption
            })
    
    print(f"加载了 {len(dataset)} 个样本")
    return dataset


def create_simple_model():
    """
    创建一个简单的测试模型
    在实际使用中应该加载FLUX模型
    """
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 1024)
            self.act = nn.GELU()
            self.linear2 = nn.Linear(1024, 512)
            self.norm = nn.LayerNorm(512)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.act(x)
            x = self.linear2(x)
            x = self.norm(x)
            return x
    
    return SimpleModel()


def train_simple(args):
    """简单训练流程"""
    
    print("=" * 60)
    print("COAT简单训练示例")
    print("=" * 60)
    print()
    
    # 创建COAT配置
    coat_config = COATConfig(
        use_fp8_optimizer=args.use_fp8_optimizer,
        use_fp8_activation=args.use_fp8_activation,
        log_memory_stats=True
    )
    
    print("COAT配置:")
    print(f"  FP8优化器: {coat_config.use_fp8_optimizer}")
    print(f"  FP8激活:   {coat_config.use_fp8_activation}")
    print(f"  动态范围扩展: {coat_config.use_dynamic_range_expansion}")
    print()
    
    # 创建训练器
    trainer = COATTrainer(coat_config)
    
    # 加载数据集
    print("加载数据集...")
    dataset = load_dataset(args.dataset_path)
    
    if len(dataset) == 0:
        print("⚠️  数据集为空！请检查数据集路径。")
        return
    
    # 创建模型
    print("创建模型...")
    model = create_simple_model()
    
    # 应用COAT准备
    model = trainer.prepare_model(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"使用设备: {device}")
    print()
    
    # 创建优化器
    print("创建优化器...")
    optimizer = trainer.create_optimizer(
        model.parameters(),
        lr=args.lr
    )
    
    # 训练循环
    print("开始训练...")
    print()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        epoch_loss = 0.0
        progress_bar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch + 1}")
        
        for step in progress_bar:
            # 生成虚拟batch (实际应该从数据集采样)
            batch = {
                'input': torch.randn(args.batch_size, 512, device=device)
            }
            
            # 训练步骤
            metrics = trainer.training_step(
                model,
                batch,
                optimizer,
                step=epoch * args.steps_per_epoch + step
            )
            
            epoch_loss += metrics['loss']
            progress_bar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
        
        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
        print()
        
        # 保存checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"✅ 保存checkpoint: {checkpoint_path}")
    
    # 保存内存统计
    memory_report = trainer.get_memory_report()
    if memory_report:
        report_path = output_dir / 'memory_report.json'
        with open(report_path, 'w') as f:
            json.dump(memory_report, f, indent=2)
        
        print()
        print("📊 内存统计:")
        print(f"  峰值内存: {memory_report['peak_allocated_gb']:.2f} GB")
        print(f"  平均内存: {memory_report['avg_allocated_gb']:.2f} GB")
        print(f"  报告已保存: {report_path}")
    
    print()
    print("✅ 训练完成!")


def main():
    parser = argparse.ArgumentParser(description='COAT简单训练示例')
    
    # 数据集
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='数据集路径')
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                       help='每轮步数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    
    # COAT配置
    parser.add_argument('--use_fp8_optimizer', action='store_true',
                       help='使用FP8优化器')
    parser.add_argument('--use_fp8_activation', action='store_true',
                       help='使用FP8激活量化')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='output/simple_coat',
                       help='输出目录')
    parser.add_argument('--save_every', type=int, default=5,
                       help='保存checkpoint的频率')
    
    args = parser.parse_args()
    
    # 运行训练
    train_simple(args)


if __name__ == "__main__":
    main()







