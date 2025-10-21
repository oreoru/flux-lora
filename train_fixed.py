"""
FLUX LoRA训练启动脚本 - 修复版
禁用TensorFlow以避免numpy冲突
"""

import os
# 在导入任何库之前设置环境变量
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'  # 禁用transformers中的TensorFlow

import sys
from pathlib import Path

# 添加ai-toolkit路径
ai_toolkit_path = Path(__file__).parent / "ai-toolkit" / "ai-toolkit"
sys.path.insert(0, str(ai_toolkit_path))

# 添加COAT路径
coat_path = Path(__file__).parent / "coat_implementation"
sys.path.insert(0, str(coat_path))

import argparse
import yaml


def patch_ai_toolkit_for_coat():
    """为ai-toolkit打补丁以支持COAT"""
    try:
        from jobs.process import BaseSDTrainProcess
        from toolkit.coat_integration import COATTrainingWrapper
        
        original_setup_trainer = BaseSDTrainProcess.setup_trainer
        
        def coat_setup_trainer(self):
            original_setup_trainer(self)
            config = self.config if hasattr(self, 'config') else {}
            coat_config = config.get('coat', {})
            
            if coat_config.get('enabled', False):
                print("\n🚀 应用COAT优化...")
                coat_wrapper = COATTrainingWrapper(config)
                coat_wrapper.on_train_start()
                
                if hasattr(self, 'sd') and self.sd is not None:
                    if hasattr(self.sd, 'unet'):
                        print("  对UNet应用FP8激活量化...")
                        self.sd.unet = coat_wrapper.prepare_model(self.sd.unet)
                    if hasattr(self.sd, 'transformer'):
                        print("  对Transformer应用FP8激活量化...")
                        self.sd.transformer = coat_wrapper.prepare_model(self.sd.transformer)
                
                self.coat_wrapper = coat_wrapper
                print("✅ COAT优化应用完成\n")
        
        BaseSDTrainProcess.setup_trainer = coat_setup_trainer
        print("✅ COAT补丁已应用到ai-toolkit")
        return True
    except Exception as e:
        print(f"⚠️  应用COAT补丁失败: {e}")
        print("   将使用标准训练模式")
        return False


def load_and_validate_config(config_path):
    """加载并验证配置文件"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if 'coat' in config.get('config', {}):
        coat_config = config['config']['coat']
        if coat_config.get('enabled', False):
            print("\n" + "="*60)
            print("COAT配置检测:")
            print("="*60)
            opt_cfg = coat_config.get('optimizer', {})
            print(f"FP8优化器: {opt_cfg.get('use_fp8', False)}")
            print(f"  - 一阶动量格式: {opt_cfg.get('m1_format', 'e4m3')}")
            print(f"  - 二阶动量格式: {opt_cfg.get('m2_format', 'e4m3')}")
            print(f"  - 动态范围扩展: {opt_cfg.get('use_dynamic_range_expansion', True)}")
            act_cfg = coat_config.get('activation', {})
            print(f"FP8激活: {act_cfg.get('use_fp8', False)}")
            mem_cfg = coat_config.get('memory', {})
            print(f"内存统计: {mem_cfg.get('log_memory_stats', False)}")
            print("="*60 + "\n")
    
    return config


def main():
    parser = argparse.ArgumentParser(description='FLUX LoRA训练 - COAT加速版')
    parser.add_argument('config', type=str, help='配置文件路径')
    parser.add_argument('--no-coat', action='store_true', help='禁用COAT优化')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FLUX LoRA训练 - COAT FP8加速")
    print("="*60 + "\n")
    
    # 加载配置
    try:
        config = load_and_validate_config(args.config)
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        sys.exit(1)
    
    # 应用COAT补丁
    if not args.no_coat and config.get('config', {}).get('coat', {}).get('enabled', False):
        patch_ai_toolkit_for_coat()
    else:
        print("⚠️  COAT优化已禁用\n")
    
    # 导入并运行ai-toolkit
    try:
        print("启动ai-toolkit训练...")
        print("="*60 + "\n")
        
        sys.argv = ['run.py', args.config]
        from run import main as ai_toolkit_main
        ai_toolkit_main()
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()






