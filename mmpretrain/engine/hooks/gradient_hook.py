from typing import Optional, Union

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class GradientLoggingHook(Hook):
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
    
    def after_train_iter(self, runner, batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None):
        if runner.iter % self.log_interval == 0:
            # Gradientの統計情報を計算
            gradient_norms = {}
            gradient_stats = {}
            
            for name, param in runner.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_norms[f'grad_norm/{name}'] = grad_norm
                    
                    # 追加の統計情報
                    gradient_stats[f'grad_mean/{name}'] = param.grad.mean().item()
                    gradient_stats[f'grad_std/{name}'] = param.grad.std().item()
            
            # WandBにロギング
            runner.visualizer.add_scalars(gradient_norms)
            runner.visualizer.add_scalars(gradient_stats)
