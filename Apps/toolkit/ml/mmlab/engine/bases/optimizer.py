from typing import Dict, Any, Optional, Union, List, ContextManager
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from mmengine.optim import OptimWrapper, AmpOptimWrapper, OptimWrapperDict
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyBaseOptimWrapper(OptimWrapper):
    def __init__(self, 
                 optimizer: Optimizer,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[Union[dict, float]] = None,
                 **kwargs: Any):
        super().__init__(optimizer, accumulative_counts, clip_grad, **kwargs)
        self.param_schedulers: Dict[str, _LRScheduler] = {}

    def update_params(self, loss: torch.Tensor) -> None:
        self.zero_grad()
        self.backward(loss)
        self.step()

    def set_param_scheduler(self, param_name: str, scheduler: _LRScheduler) -> None:
        self.param_schedulers[param_name] = scheduler

    def step_schedulers(self, metrics: Optional[Dict[str, float]] = None) -> None:
        for param_name, scheduler in self.param_schedulers.items():
            if metrics and param_name in metrics:
                scheduler.step(metrics[param_name])
            else:
                scheduler.step()

    def get_lr(self) -> Dict[str, List[float]]:
        return {param_name: [group['lr'] for group in self.optimizer.param_groups] 
                for param_name in self.param_schedulers.keys()}

    def get_momentum(self) -> Dict[str, List[float]]:
        return {param_name: [group.get('momentum', 0) for group in self.optimizer.param_groups] 
                for param_name in self.param_schedulers.keys()}

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = super().state_dict()
        state['param_schedulers'] = {name: scheduler.state_dict() 
                                     for name, scheduler in self.param_schedulers.items()}
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        scheduler_states = state_dict.pop('param_schedulers', {})
        super().load_state_dict(state_dict)
        for name, scheduler_state in scheduler_states.items():
            if name in self.param_schedulers:
                self.param_schedulers[name].load_state_dict(scheduler_state)

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyAmpOptimWrapper(AmpOptimWrapper, MyBaseOptimWrapper):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class MyOptimWrapperDict(OptimWrapperDict):
    def update_params(self, losses: Dict[str, torch.Tensor]) -> None:
        for name, optimizer in self.items():
            if isinstance(optimizer, OptimWrapper):
                optimizer.update_params(losses[name])
            elif isinstance(optimizer, OptimWrapperDict):
                optimizer.update_params({k: losses[f'{name}.{k}'] for k in optimizer.keys()})

    def get_lr(self) -> Dict[str, List[float]]:
        lr_dict: Dict[str, List[float]] = {}
        for name, optimizer in self.items():
            if isinstance(optimizer, OptimWrapper):
                lr_dict.update({f'{name}.{k}': v for k, v in optimizer.get_lr().items()})
            elif isinstance(optimizer, OptimWrapperDict):
                lr_dict.update({f'{name}.{k}': v for k, v in optimizer.get_lr().items()})
        return lr_dict

    def get_momentum(self) -> Dict[str, List[float]]:
        momentum_dict: Dict[str, List[float]] = {}
        for name, optimizer in self.items():
            if isinstance(optimizer, OptimWrapper):
                momentum_dict.update({f'{name}.{k}': v for k, v in optimizer.get_momentum().items()})
            elif isinstance(optimizer, OptimWrapperDict):
                momentum_dict.update({f'{name}.{k}': v for k, v in optimizer.get_momentum().items()})
        return momentum_dict

# Usage example
"""
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner

class MyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        return self.linear(inputs)

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optim_wrapper = MyBaseOptimWrapper(optimizer=optimizer, accumulative_counts=2, clip_grad=dict(max_norm=1.0))
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)

runner = Runner(
    model=model,
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=optim_wrapper),
    param_scheduler=param_scheduler,
    train_cfg=dict(by_epoch=True, max_epochs=12),
)
runner.train()
"""