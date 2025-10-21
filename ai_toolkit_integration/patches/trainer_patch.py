
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
