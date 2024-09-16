# Hooks

## Hook

Hook programming is a programming pattern in which a mount point is set in one or more locations of a program. When the program runs to a mount point, all methods registered to it at runtime are automatically called. Hook programming can increase the flexibility and extensibility of the program, as users can register custom methods to the mount point to be called without modifying the code in the program.

## Built-in Hooks

MMEngine encapsulates many utilities as built-in hooks. These hooks are divided into two categories: default hooks and custom hooks. The former refers to those registered with the Runner by default, while the latter refers to those registered by the user on demand.

Each hook has a corresponding priority. At each mount point, hooks with higher priority are called earlier by the Runner. When sharing the same priority, the hooks are called in their registration order. The priority list is as follows:

- **HIGHEST (0)**
- **VERY_HIGH (10)**
- **HIGH (30)**
- **ABOVE_NORMAL (40)**
- **NORMAL (50)**
- **BELOW_NORMAL (60)**
- **LOW (70)**
- **VERY_LOW (90)**
- **LOWEST (100)**

### Default Hooks

| Name                | Function                                                          | Priority |
|---------------------|-------------------------------------------------------------------|----------|
| RuntimeInfoHook     | Update runtime information into the message hub                  | VERY_HIGH (10) |
| IterTimerHook       | Update the time spent during iteration into the message hub        | NORMAL (50) |
| DistSamplerSeedHook | Ensure distributed Sampler shuffle is active                      | NORMAL (50) |
| LoggerHook          | Collect logs from different components of Runner and write them to terminal, JSON file, TensorBoard, and wandb, etc. | BELOW_NORMAL (60) |
| ParamSchedulerHook  | Update some hyper-parameters of optimizer                          | LOW (70) |
| CheckpointHook      | Save checkpoints periodically                                      | VERY_LOW (90) |

### Custom Hooks

| Name              | Function                                                            | Priority |
|-------------------|---------------------------------------------------------------------|----------|
| EMAHook           | Apply Exponential Moving Average (EMA) on the model during training | NORMAL (50) |
| EmptyCacheHook    | Release all unoccupied cached GPU memory during the training process | NORMAL (50) |
| SyncBuffersHook   | Synchronize model buffers at the end of each epoch                  | NORMAL (50) |
| ProfilerHook      | Analyze the execution time and GPU memory usage of model operators  | VERY_LOW (90) |

**Note:** It is not recommended to modify the priority of the default hooks, as hooks with lower priority may depend on hooks with higher priority. For example, `CheckpointHook` needs to have a lower priority than `ParamSchedulerHook` so that the saved optimizer state is correct. Also, the priority of custom hooks defaults to `NORMAL (50)`.

The two types of hooks are set differently in the Runner, with the configuration of default hooks being passed to the `default_hooks` parameter of the Runner and the configuration of custom hooks being passed to the `custom_hooks` parameter, as follows:

```python
from mmengine.runner import Runner

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    logger=dict(type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
)

custom_hooks = [dict(type='EmptyCacheHook')]

runner = Runner(default_hooks=default_hooks, custom_hooks=custom_hooks, ...)
runner.train()
```

## LoggerHook

`LoggerHook` collects logs from different components of Runner and writes them to terminal, JSON file, TensorBoard, and wandb, etc.

## CheckpointHook

`CheckpointHook` saves the checkpoints at a given interval. In the case of distributed training, only the master process will save the checkpoints. The main features of `CheckpointHook` are as follows:

- Save checkpoints by interval, and support saving them by epoch or iteration
- Save the most recent checkpoints
- Save the best checkpoints
- Specify the path to save the checkpoints
- Make checkpoints for publish
- Control the epoch number or iteration number at which checkpoint saving begins

For more features, please read the `CheckpointHook` API documentation.

**Save checkpoints by interval, and support saving them by epoch or iteration**

Suppose we train a total of 20 epochs and want to save the checkpoints every 5 epochs. The following configuration will help us achieve this requirement:

```python
# The default value of by_epoch is True
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, by_epoch=True))
```

If you want to save checkpoints by iteration, you can set `by_epoch` to `False` and `interval=5` to save them every 5 iterations:

```python
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, by_epoch=False))
```

**Save the most recent checkpoints**

If you only want to keep a certain number of checkpoints, you can set the `max_keep_ckpts` parameter. When the number of checkpoints saved exceeds `max_keep_ckpts`, the previous checkpoints will be deleted:

```python
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=2))
```

The above configuration shows that if a total of 20 epochs are trained, the model will be saved at epochs 5, 10, 15, and 20, but the checkpoint `epoch_5.pth` will be deleted at epoch 15, and at epoch 20, the checkpoint `epoch_10.pth` will be deleted, so only `epoch_15.pth` and `epoch_20.pth` will be saved.

**Save the best checkpoints**

If you want to save the best checkpoints of the validation set for the training process, you can set the `save_best` parameter. If set to `'auto'`, the current checkpoint is judged to be the best based on the first evaluation metric of the validation set (the evaluation metrics returned by evaluator are an ordered dictionary):

```python
default_hooks = dict(checkpoint=dict(type='CheckpointHook', save_best='auto'))
```

You can also directly specify the value of `save_best` as the evaluation metric. For example, in a classification task, you can specify `save_best='top-1'`, then the current checkpoint will be judged as best based on the value of `'top-1'`.

In addition to the `save_best` parameter, other parameters related to saving the best checkpoint are `rule`, `greater_keys`, and `less_keys`, which are used to imply whether it is good to have a large value or not. For example, if you specify `save_best='top-1'`, you can specify `rule='greater'` to imply that the larger the value, the better the checkpoint.

**Specify the path to save the checkpoints**

The checkpoints are saved in `work_dir` by default, but the path can be changed by setting `out_dir`:

```python
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, out_dir='/path/of/directory'))
```

**Make checkpoints for publish**

If you want to automatically generate publishable checkpoints after training (remove unnecessary keys, such as optimizer state), you can set the `published_keys` parameter to choose which information to keep. Note: You need to set the `save_best` or `save_last` parameters accordingly so that the releasable checkpoints will be generated. Setting `save_best` will generate the releasable weights of the optimal checkpoint, and setting `save_last` will generate the releasable final checkpoint. These two parameters can also be set at the same time:

```python
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, save_best='accuracy', rule='less', published_keys=['meta', 'state_dict']))
```

**Control the epoch number or iteration number at which checkpoint saving begins**

If you want to set the number of epochs or iterations to control the start of saving weights, you can set the `save_begin` parameter, which defaults to 0, meaning saving checkpoints from the beginning of training. For example, if you train for a total of 10 epochs and `save_begin` is set to 5, then the checkpoints for epochs 5, 6, 7, 8, 9, and 10 will be saved. If `interval=2`, only checkpoints for epochs 5, 7, and 9 will be saved:

```python
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2, save_begin=5))
```

## ParamSchedulerHook

`ParamSchedulerHook` iterates through all optimizer parameter schedulers of the Runner and calls their step method to update the optimizer parameters in order. See Parameter Schedulers for more details about what parameter schedulers are. `ParamSchedulerHook` is registered to the Runner by default and has no configurable parameters, so there is no need to configure it.

## IterTimerHook

`IterTimerHook` is used to record the time taken to load data and iterate once. `IterTimerHook` is registered to the Runner by default and has no configurable parameters, so there is no need to configure it.

## DistSamplerSeedHook

`DistSamplerSeedHook` calls the step method of the Sampler during distributed training to ensure that the shuffle operation takes effect. `DistSamplerSeedHook` is registered to the Runner by default and has no configurable parameters, so there is no need to configure it.

## RuntimeInfoHook

`RuntimeInfoHook` updates the current runtime information (e.g., epoch, iter, max_epochs, max_iters, lr,

 metrics, etc.) to the message hub at different mount points in the Runner so that other modules without access to the Runner can obtain this information. `RuntimeInfoHook` is registered to the Runner by default and has no configurable parameters, so there is no need to configure it.

## EMAHook

`EMAHook` performs an exponential moving average operation on the model during training, aiming to improve the robustness of the model. Note that the model generated by exponential moving average is only used for validation and testing, and does not affect training.

```python
custom_hooks = [dict(type='EMAHook')]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

`EMAHook` uses `ExponentialMovingAverage` by default, with optional values of `StochasticWeightAverage` and `MomentumAnnealingEMA`. Other averaging strategies can be used by setting `ema_type`:

```python
custom_hooks = [dict(type='EMAHook', ema_type='StochasticWeightAverage')]
```

See `EMAHook` API Reference for more usage.

## EmptyCacheHook

`EmptyCacheHook` calls `torch.cuda.empty_cache()` to release all unoccupied cached GPU memory. The timing of releasing memory can be controlled by setting parameters like `before_epoch`, `after_iter`, and `after_epoch`, meaning before the start of each epoch, after each iteration, and after each epoch respectively:

```python
# The release operation is performed at the end of each epoch
custom_hooks = [dict(type='EmptyCacheHook', after_epoch=True)]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

## SyncBuffersHook

`SyncBuffersHook` synchronizes the buffer of the model at the end of each epoch during distributed training, e.g., running_mean and running_var of the BN layer:

```python
custom_hooks = [dict(type='SyncBuffersHook')]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

## ProfilerHook

`ProfilerHook` is used to analyze the execution time and GPU memory occupancy of model operators:

```python
custom_hooks = [dict(type='ProfilerHook', on_trace_ready=dict(type='tb_trace'))]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

The profiling results will be saved in the `tf_tracing_logs` directory under `work_dirs/{timestamp}`, and can be visualized using TensorBoard with the command `tensorboard --logdir work_dirs/{timestamp}/tf_tracing_logs`.

For more information on the usage of the `ProfilerHook`, please refer to the `ProfilerHook` documentation.

## Customize Your Hooks

If the built-in hooks provided by MMEngine do not cover your demands, you are encouraged to customize your own hooks by simply inheriting the base hook class and overriding the corresponding mount point methods.

For example, if you want to check whether the loss value is valid, i.e., not infinite, during training, you can simply override the `after_train_iter` method as below. The check will be performed after each training iteration:

```python
import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Defaults to 50.
    """
    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """All subclasses should override this method if they need any
        operations after each training iteration.
        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.every_n_train_iters(runner, self.interval):
            assert torch.isfinite(outputs['loss']),\
                runner.logger.info('loss become infinite or NaN!')
```

We simply pass the hook config to the `custom_hooks` parameter of the Runner, which will register the hooks when the Runner is initialized:

```python
from mmengine.runner import Runner

custom_hooks = [
    dict(type='CheckInvalidLossHook', interval=50)
]

runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()  # start training
```

Then the loss values are checked after each iteration.

**Note:** The priority of the custom hook is `NORMAL (50)` by default. If you want to change the priority of the hook, you can set the `priority` key in the config:

```python
custom_hooks = [
    dict(type='CheckInvalidLossHook', interval=50, priority='ABOVE_NORMAL')
]
```

You can also set priority when defining classes:

```python
@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    priority = 'ABOVE_NORMAL'
```

---

## Hook

Hook programming is a design pattern where a mount point is established within a program. When the program reaches these mount points during execution, all methods registered to them are automatically invoked. This pattern enhances the flexibility and extensibility of a program by allowing users to register custom methods to be executed at runtime without altering the core codebase.

## Examples

Here is a basic example demonstrating how hook programming functions:

```python
pre_hooks = [(print, 'hello')]
post_hooks = [(print, 'goodbye')]

def main():
    for func, arg in pre_hooks:
        func(arg)
    print('do something here')
    for func, arg in post_hooks:
        func(arg)

main()
```

Output:

```
hello
do something here
goodbye
```

In this example, the `main` function executes `print` statements defined in `pre_hooks` and `post_hooks` without modifying the core logic of the `main` function.

Hook programming is extensively utilized in PyTorch, particularly in the neural network module (`nn.Module`). For instance, the `register_forward_hook` method attaches a forward hook to a module, allowing users to access the forward inputs and outputs of the module. 

Here is an example demonstrating `register_forward_hook` usage:

```python
import torch
import torch.nn as nn

def forward_hook_fn(module, input, output):
    print(f'"forward_hook_fn" is invoked by {module.name}')
    print('weight:', module.weight.data)
    print('bias:', module.bias.data)
    print('input:', input)
    print('output:', output)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        y = self.fc(x)
        return y

model = Model()
# Register forward_hook_fn to each submodule of model
for module in model.children():
    module.register_forward_hook(forward_hook_fn)

x = torch.Tensor([[0.0, 1.0, 2.0]])
y = model(x)
```

Output:

```
"forward_hook_fn" is invoked by Linear(in_features=3, out_features=1, bias=True)
weight: tensor([[-0.4077,  0.0119, -0.3606]])
bias: tensor([-0.2943])
input: (tensor([[0., 1., 2.]]),)
output: tensor([[-1.0036]], grad_fn=<AddmmBackward>)
```

This example illustrates how `forward_hook_fn` prints the weights, biases, inputs, and outputs of the `nn.Linear` module during the forward pass.

## Design on MMEngine

Before delving into the design of hooks in MMEngine, let's review the basic steps for training a model using PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    pass

class Net(nn.Module):
    pass

def main():
    transform = transforms.ToTensor()
    train_dataset = CustomDataset(transform=transform, ...)
    val_dataset = CustomDataset(transform=transform, ...)
    test_dataset = CustomDataset(transform=transform, ...)
    train_dataloader = DataLoader(train_dataset, ...)
    val_dataloader = DataLoader(val_dataset, ...)
    test_dataloader = DataLoader(test_dataset, ...)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for i in range(max_epochs):
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = net(inputs)
            accuracy = ...
```

The pseudo-code above outlines the fundamental steps for model training. To integrate custom operations, you need to modify the `main` function continuously. By inserting mount points into the `main` function and calling hooks at these points, you can implement custom logic such as model weight loading or parameter updates.

Here is an updated `main` function utilizing hooks:

```python
def main():
    ...
    call_hooks('before_run', hooks)
    call_hooks('after_load_checkpoint', hooks)
    call_hooks('before_train', hooks)
    for i in range(max_epochs):
        call_hooks('before_train_epoch', hooks)
        for inputs, labels in train_dataloader:
            call_hooks('before_train_iter', hooks)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            call_hooks('after_train_iter', hooks)
            loss.backward()
            optimizer.step()
        call_hooks('after_train_epoch', hooks)

        call_hooks('before_val_epoch', hooks)
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                call_hooks('before_val_iter', hooks)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                call_hooks('after_val_iter', hooks)
        call_hooks('after_val_epoch', hooks)

        call_hooks('before_save_checkpoint', hooks)
    call_hooks('after_train', hooks)

    call_hooks('before_test_epoch', hooks)
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            call_hooks('before_test_iter', hooks)
            outputs = net(inputs)
            accuracy = ...
            call_hooks('after_test_iter', hooks)
    call_hooks('after_test_epoch', hooks)

    call_hooks('after_run', hooks)
```

In MMEngine, the training process is encapsulated in an executor (Runner). The Runner invokes hooks at specific mount points to execute custom logic. For more details on the Runner, refer to the Runner documentation.

MMEngine defines 22 mount points in the Base Hook, which are:

- before_run
- after_run
- before_train
- after_train
- before_train_epoch
- after_train_epoch
- before_train_iter
- after_train_iter
- before_val
- after_val
- before_test_epoch
- after_test_epoch
- before_val_iter
- after_val_iter
- before_test
- after_test
- before_test_epoch
- after_test_epoch
- before_test_iter
- after_test_iter
- before_save_checkpoint
- after_load_checkpoint

By inheriting from the base hook and implementing custom logic for these mount points, you can easily register and manage hooks in the Runner.

---

## mmengine.hooks

### Hook

Base hook class.

### CheckpointHook

Save checkpoints periodically.

### EMAHook

A Hook to apply Exponential Moving Average (EMA) on the model during training.

### LoggerHook

Collect logs from different components of Runner and write them to terminal, JSON file, TensorBoard, and wandb, etc.

### NaiveVisualizationHook

Show or write the predicted results during the process of testing.

### ParamSchedulerHook

A hook to update some hyper-parameters in optimizer, e.g., learning rate and momentum.

### RuntimeInfoHook

A hook that updates runtime information into message hub.

### DistSamplerSeedHook

Data-loading sampler for distributed training.

### IterTimerHook

A hook that logs the time spent during iteration.

### SyncBuffersHook

Synchronize model buffers such as running_mean and running_var in BN at the end of each epoch.

### EmptyCacheHook

Releases all unoccupied cached GPU memory during the process of training.

### ProfilerHook

A hook to analyze performance during training and inference.

### NPUProfilerHook

NPUProfiler to analyze performance during training.

### PrepareTTAHook

Wraps runner.model with a subclass of BaseTTAModel before_test.

### EarlyStoppingHook

Early stop the training when the monitored metric reaches a plateau.