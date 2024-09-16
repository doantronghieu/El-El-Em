# Optimizer and Scheduler

## OptimWrapper

In previous tutorials on runners and models, we have touched on the concept of OptimWrapper but haven't fully explored its need and advantages over PyTorch’s native optimizer. This tutorial aims to clarify the benefits of OptimWrapper and demonstrate its usage.

As the name suggests, OptimWrapper is a high-level abstraction of PyTorch’s native optimizer that offers a unified interface while introducing additional functionality. It supports various training strategies, including mixed precision training, gradient accumulation, and gradient clipping. This flexibility allows users to select the appropriate training strategy based on their requirements. OptimWrapper also establishes a standardized process for parameter updating, enabling users to switch between different training strategies with minimal code changes.

### OptimWrapper vs Optimizer

We will now compare the use of PyTorch’s native optimizer and OptimWrapper in MMEngine across different training scenarios: single-precision training, mixed-precision training, and gradient accumulation.

### Model Training

#### 1.1 Single-Precision Training with SGD in PyTorch

```python
import torch
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F

inputs = [torch.zeros(10, 1, 1)] * 10
targets = [torch.ones(10, 1, 1)] * 10
model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()

for input, target in zip(inputs, targets):
    output = model(input)
    loss = F.l1_loss(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### 1.2 Single-Precision Training with OptimWrapper in MMEngine

```python
from mmengine.optim import OptimWrapper

optim_wrapper = OptimWrapper(optimizer=optimizer)

for input, target in zip(inputs, targets):
    output = model(input)
    loss = F.l1_loss(output, target)
    optim_wrapper.update_params(loss)
```

The `OptimWrapper.update_params` method handles gradient computation, parameter updating, and gradient zeroing, directly updating the model parameters.

### 2.1 Mixed-Precision Training with SGD in PyTorch

```python
from torch.cuda.amp import autocast

model = model.cuda()
inputs = [torch.zeros(10, 1, 1, 1)] * 10
targets = [torch.ones(10, 1, 1, 1)] * 10

for input, target in zip(inputs, targets):
    with autocast():
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### 2.2 Mixed-Precision Training with OptimWrapper in MMEngine

```python
from mmengine.optim import AmpOptimWrapper

optim_wrapper = AmpOptimWrapper(optimizer=optimizer)

for input, target in zip(inputs, targets):
    with optim_wrapper.optim_context(model):
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    optim_wrapper.update_params(loss)
```

To enable mixed precision training, use `AmpOptimWrapper.optim_context`, similar to `autocast` in PyTorch. This context also accelerates gradient accumulation during distributed training.

### 3.1 Mixed-Precision Training and Gradient Accumulation with SGD in PyTorch

```python
for idx, (input, target) in enumerate(zip(inputs, targets)):
    with autocast():
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    loss.backward()
    if idx % 2 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 3.2 Mixed-Precision Training and Gradient Accumulation with OptimWrapper in MMEngine

```python
optim_wrapper = AmpOptimWrapper(optimizer=optimizer, accumulative_counts=2)

for input, target in zip(inputs, targets):
    with optim_wrapper.optim_context(model):
        output = model(input.cuda())
    loss = F.l1_loss(output, target.cuda())
    optim_wrapper.update_params(loss)
```

By configuring the `accumulative_counts` parameter, we achieve gradient accumulation. In distributed training, enabling the `optim_context` helps avoid unnecessary gradient synchronization during accumulation.

## Advanced Features of OptimWrapper

OptimWrapper offers more fine-grained control over parameter updates through methods like `backward`, `step`, and `zero_grad`, which replicate the functionality of PyTorch’s optimizer methods. 

### Gradient Clipping

To implement gradient clipping with OptimWrapper:

```python
# Based on torch.nn.utils.clip_grad_norm_ method
optim_wrapper = AmpOptimWrapper(
    optimizer=optimizer, clip_grad=dict(max_norm=1))

# Based on torch.nn.utils.clip_grad_value_ method
optim_wrapper = AmpOptimWrapper(
    optimizer=optimizer, clip_grad=dict(clip_value=0.2))
```

### Get Learning Rate and Momentum

Retrieve the learning rate and momentum using:

```python
import torch.nn as nn
from torch.optim import SGD
from mmengine.optim import OptimWrapper

model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)
optim_wrapper = OptimWrapper(optimizer)

print(optimizer.param_groups[0]['lr'])  # 0.01
print(optimizer.param_groups[0]['momentum'])  # 0
print(optim_wrapper.get_lr())  # {'lr': [0.01]}
print(optim_wrapper.get_momentum())  # {'momentum': [0]}
```

### Export/Load State Dicts

OptimWrapper supports exporting and loading state dictionaries:

```python
import torch.nn as nn
from torch.optim import SGD
from mmengine.optim import OptimWrapper, AmpOptimWrapper

model = nn.Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.01)

optim_wrapper = OptimWrapper(optimizer=optimizer)
amp_optim_wrapper = AmpOptimWrapper(optimizer=optimizer)

# Export state dicts
optim_state_dict = optim_wrapper.state_dict()
amp_optim_state_dict = amp_optim_wrapper.state_dict()

print(optim_state_dict)
print(amp_optim_state_dict)

# Load state dicts
optim_wrapper_new = OptimWrapper(optimizer=optimizer)
amp_optim_wrapper_new = AmpOptimWrapper(optimizer=optimizer)

amp_optim_wrapper_new.load_state_dict(amp_optim_state_dict)
optim_wrapper_new.load_state_dict(optim_state_dict)
```

### Use Multiple Optimizers

To handle multiple optimizers, such as in GANs, use `OptimWrapperDict`:

```python
from torch.optim import SGD
import torch.nn as nn
from mmengine.optim import OptimWrapper, OptimWrapperDict

gen = nn.Linear(1, 1)
disc = nn.Linear(1, 1)
optimizer_gen = SGD(gen.parameters(), lr=0.01)
optimizer_disc = SGD(disc.parameters(), lr=0.01)

optim_wapper_gen = OptimWrapper(optimizer=optimizer_gen)
optim_wapper_disc = OptimWrapper(optimizer=optimizer_disc)
optim_dict = OptimWrapperDict(gen=optim_wapper_gen, disc=optim_wapper_disc)

print(optim_dict.get_lr())  # {'gen.lr': [0.01], 'disc.lr': [0.01]}
print(optim_dict.get_momentum())  # {'gen.momentum': [0], 'disc.momentum': [0]}
```

### Configure OptimWrapper in Runner

To configure OptimWrapper in MMEngine, set up the optimizer in a dict format:

```python
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# For mixed-precision training and gradient accumulation
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer, accumulative_counts=2)
```

### Advanced Usages

The default optimizer wrapper constructor in MMEngine supports setting different hyperparameters for different model parameters. For instance, setting different learning rates and decay coefficients for various parts of a model can be configured using `paramwise_cfg`.

To implement a custom optimizer wrapper constructor, such as for layer decay, use:

```python
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.logging import print_log

@OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
class LayerDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    ...
```

For more advanced adjustments and customizations, refer to the MMEngine documentation on parameter schedulers and optimizer wrapper constructors.

---

## Parameter Scheduler

During neural network training, optimization hyperparameters such as learning rate are typically adjusted throughout the training process. A widely-used method for learning rate adjustment is multi-step learning rate decay, which periodically reduces the learning rate by a specified fraction. PyTorch offers the `LRScheduler` to implement various learning rate adjustment strategies. MMEngine extends this functionality with a more versatile `ParamScheduler`, which can adjust optimization hyperparameters like learning rate and momentum. Additionally, it supports combining multiple schedulers to create more complex scheduling strategies.

## Usage

### PyTorch’s Built-in Learning Rate Scheduler

To adjust the learning rate using PyTorch’s built-in scheduler, you can use `torch.optim.lr_scheduler`. The `mmengine.optim.scheduler` module supports most of PyTorch’s learning rate schedulers, including `ExponentialLR`, `LinearLR`, `StepLR`, and `MultiStepLR`. For details on supported schedulers, refer to the parameter scheduler API documentation.

MMEngine also allows for momentum adjustment with parameter schedulers. To use momentum schedulers, replace "LR" in the class name with "Momentum" (e.g., `ExponentialMomentum`, `LinearMomentum`). Furthermore, MMEngine provides a general parameter scheduler called `ParamScheduler` for adjusting specific hyperparameters in the optimizer, such as `weight_decay`. This feature facilitates complex hyperparameter tuning strategies.

Unlike manual training loop implementations, MMEngine’s `ParamSchedulerHook` automatically manages the training progress and controls the execution of the parameter scheduler.

### Using a Single LRScheduler

If only one scheduler is required for the entire training process, it is similar to using PyTorch’s learning rate scheduler. Here is an example of manually building a scheduler:

```python
from torch.optim import SGD
from mmengine.runner import Runner
from mmengine.optim.scheduler import MultiStepLR

optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
param_scheduler = MultiStepLR(optimizer, milestones=[8, 11], gamma=0.1)

runner = Runner(
    model=model,
    optim_wrapper=dict(
        optimizer=optimizer),
    param_scheduler=param_scheduler,
    ...
)
```

When using a configuration file, specify the scheduler in the `param_scheduler` field. The runner will build the parameter scheduler automatically:

```python
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
```

Here, `by_epoch` controls the frequency of learning rate adjustment. If `True`, adjustments are made per epoch; if `False`, adjustments are made per iteration. The default is `True`. For example, `[8, 11]` in milestones means the learning rate will be reduced by a factor of 0.1 at the end of the 8th and 11th epochs.

To adjust by iterations instead, use:

```python
param_scheduler = dict(type='MultiStepLR', by_epoch=False, milestones=[600, 800], gamma=0.1)
```

To convert epoch-based settings to iteration-based, use:

```python
epoch_length = len(train_dataloader)
param_scheduler = MultiStepLR.build_iter_from_epoch(optimizer, milestones=[8, 11], gamma=0.1, epoch_length=epoch_length)
```

Alternatively, set `convert_to_iter_based=True` in the configuration:

```python
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1, convert_to_iter_based=True)
```

For a Cosine Annealing learning rate scheduler updated by epoch:

```python
param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, T_max=12)
```

To convert this to iteration-based:

```python
param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, T_max=12, convert_to_iter_based=True)
```

### Combining Multiple LRSchedulers

In some algorithms, the learning rate is adjusted differently at various stages. For example, a linear warm-up might be used initially, followed by a different strategy. MMEngine supports combining multiple schedulers. You can specify a list of scheduler configurations in the `param_scheduler` field:

```python
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
]
```

In this example, the `LinearLR` scheduler applies for the first 50 iterations, followed by the `MultiStepLR` scheduler. Note that `begin` and `end` parameters define the scheduler’s active interval.

Another example combines linear warm-up and cosine annealing:

```python
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(type='CosineAnnealingLR', T_max=800, by_epoch=False, begin=100, end=900)
]
```

### Adjusting Other Hyperparameters

#### Momentum

Momentum, like learning rate, can be adjusted using a scheduler. Add the momentum scheduler config to the `param_scheduler` list:

```python
param_scheduler = [
    dict(type='LinearLR', ...),
    dict(type='LinearMomentum', start_factor=0.001, by_epoch=False, begin=0, end=1000)
]
```

#### Generic Parameter Scheduler

MMEngine provides generic parameter schedulers for other hyperparameters in the optimizer’s parameter groups. Replace "LR" in the class name with "Param", such as `LinearParamScheduler`, and set the `param_name` variable to the desired hyperparameter:

```python
param_scheduler = [
    dict(type='LinearParamScheduler', param_name='lr', start_factor=0.001, by_epoch=False, begin=0, end=1000)
]
```

For example, to adjust `weight_decay`:

```python
param_scheduler = [
    dict(type='LinearParamScheduler', param_name='weight_decay', start_factor=0.001, by_epoch=False, begin=0, end=1000)
]
```

---

## Better Performance Optimizers

This document provides some third-party optimizers supported by MMEngine, which may bring faster convergence speed or higher performance.

### D-Adaptation

D-Adaptation provides `DAdaptAdaGrad`, `DAdaptAdam`, and `DAdaptSGD` optimizers.

**Note**: If you use the optimizer provided by D-Adaptation, you need to upgrade mmengine to 0.6.0.

**Installation**:

```bash
pip install dadaptation
```

**Usage**:

Take the `DAdaptAdaGrad` as an example.

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # To view the input parameters for DAdaptAdaGrad, you can refer to
    # https://github.com/facebookresearch/dadaptation/blob/main/dadaptation/dadapt_adagrad.py
    optim_wrapper=dict(optimizer=dict(type='DAdaptAdaGrad', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

### Lion-Pytorch

`lion-pytorch` provides the Lion optimizer.

**Note**: If you use the optimizer provided by Lion-Pytorch, you need to upgrade mmengine to 0.6.0.

**Installation**:

```bash
pip install lion-pytorch
```

**Usage**:

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # To view the input parameters for Lion, you can refer to
    # https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/lion_pytorch.py
    optim_wrapper=dict(optimizer=dict(type='Lion', lr=1e-4, weight_decay=1e-2)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

### Sophia

Sophia provides `Sophia`, `SophiaG`, `DecoupledSophia`, and `Sophia2` optimizers.

**Note**: If you use the optimizer provided by Sophia, you need to upgrade mmengine to 0.7.4.

**Installation**:

```bash
pip install Sophia-Optimizer
```

**Usage**:

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # To view the input parameters for SophiaG, you can refer to
    # https://github.com/kyegomez/Sophia/blob/main/Sophia/Sophia.py
    optim_wrapper=dict(optimizer=dict(type='SophiaG', lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

### bitsandbytes

`bitsandbytes` provides `AdamW8bit`, `Adam8bit`, `Adagrad8bit`, `PagedAdam8bit`, `PagedAdamW8bit`, `LAMB8bit`, `LARS8bit`, `RMSprop8bit`, `Lion8bit`, `PagedLion8bit`, and `SGD8bit` optimizers.

**Note**: If you use the optimizer provided by `bitsandbytes`, you need to upgrade mmengine to 0.9.0.

**Installation**:

```bash
pip install bitsandbytes
```

**Usage**:

Take the `AdamW8bit` as an example.

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # To view the input parameters for AdamW8bit, you can refer to
    # https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/optim/adamw.py
    optim_wrapper=dict(optimizer=dict(type='AdamW8bit', lr=1e-4, weight_decay=1e-2)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

### transformers

`transformers` provides `Adafactor` optimizer.

**Note**: If you use the optimizer provided by `transformers`, you need to upgrade mmengine to 0.9.0.

**Installation**:

```bash
pip install transformers
```

**Usage**:

Take the `Adafactor` as an example.

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    # To view the input parameters for Adafactor, you can refer to
    # https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/optimization.py#L492
    optim_wrapper=dict(optimizer=dict(type='Adafactor', lr=1e-5, weight_decay=1e-2, scale_parameter=False, relative_step=False)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
)
runner.train()
```

---

## mmengine.optim

### Optimizer

#### AmpOptimWrapper
A subclass of `OptimWrapper` that supports automatic mixed precision training based on `torch.cuda.amp`.

#### ApexOptimWrapper
A subclass of `OptimWrapper` that supports automatic mixed precision training based on `apex.amp`.

#### OptimWrapper
Optimizer wrapper provides a common interface for updating parameters.

#### OptimWrapperDict
A dictionary container of `OptimWrapper`.

#### DefaultOptimWrapperConstructor
Default constructor for optimizers.

#### ZeroRedundancyOptimizer
A wrapper class of `ZeroRedundancyOptimizer` that gets an optimizer type as a string.

#### build_optim_wrapper
Build function of `OptimWrapper`.

### Scheduler

#### _ParamScheduler
Base class for parameter schedulers.

#### ConstantLR
Decays the learning rate value of each parameter group by a small constant factor until the number of epochs reaches a pre-defined milestone: end.

#### ConstantMomentum
Decays the momentum value of each parameter group by a small constant factor until the number of epochs reaches a pre-defined milestone: end.

#### ConstantParamScheduler
Decays the parameter value of each parameter group by a small constant factor until the number of epochs reaches a pre-defined milestone: end.

#### CosineAnnealingLR
Sets the learning rate of each parameter group using a cosine annealing schedule, where `eta_min` is set to the initial value and `T_i` is the number of epochs since the last restart in SGDR.

#### CosineAnnealingMomentum
Sets the momentum of each parameter group using a cosine annealing schedule, where `eta_min` is set to the initial value and `T_i` is the number of epochs since the last restart in SGDR.

#### CosineAnnealingParamScheduler
Sets the parameter value of each parameter group using a cosine annealing schedule, where `eta_min` is set to the initial value and `T_i` is the number of epochs since the last restart in SGDR.

#### ExponentialLR
Decays the learning rate of each parameter group by `gamma` every epoch.

#### ExponentialMomentum
Decays the momentum of each parameter group by `gamma` every epoch.

#### ExponentialParamScheduler
Decays the parameter value of each parameter group by `gamma` every epoch.

#### LinearLR
Decays the learning rate of each parameter group by linearly changing a small multiplicative factor until the number of epochs reaches a pre-defined milestone: end.

#### LinearMomentum
Decays the momentum of each parameter group by linearly changing a small multiplicative factor until the number of epochs reaches a pre-defined milestone: end.

#### LinearParamScheduler
Decays the parameter value of each parameter group by linearly changing a small multiplicative factor until the number of epochs reaches a pre-defined milestone: end.

#### MultiStepLR
Decays the specified learning rate in each parameter group by `gamma` once the number of epochs reaches one of the milestones.

#### MultiStepMomentum
Decays the specified momentum in each parameter group by `gamma` once the number of epochs reaches one of the milestones.

#### MultiStepParamScheduler
Decays the specified parameter in each parameter group by `gamma` once the number of epochs reaches one of the milestones.

#### OneCycleLR
Sets the learning rate of each parameter group according to the 1cycle learning rate policy.

#### OneCycleParamScheduler
Sets the parameters of each parameter group according to the 1cycle learning rate policy.

#### PolyLR
Decays the learning rate of each parameter group in a polynomial decay scheme.

#### PolyMomentum
Decays the momentum of each parameter group in a polynomial decay scheme.

#### PolyParamScheduler
Decays the parameter value of each parameter group in a polynomial decay scheme.

#### StepLR
Decays the learning rate of each parameter group by `gamma` every `step_size` epochs.

#### StepMomentum
Decays the momentum of each parameter group by `gamma` every `step_size` epochs.

#### StepParamScheduler
Decays the parameter value of each parameter group by `gamma` every `step_size` epochs.

#### ReduceOnPlateauLR
Reduces the learning rate of each parameter group when a metric has stopped improving.

#### ReduceOnPlateauMomentum
Reduces the momentum of each parameter group when a metric has stopped improving.

#### ReduceOnPlateauParamScheduler
Reduces the parameters of each parameter group when a metric has stopped improving.