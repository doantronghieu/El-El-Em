# Runner

## 15 minutes to get started with MMEngine

In this tutorial, we’ll take training a ResNet-50 model on the CIFAR-10 dataset as an example. We will build a complete and configurable pipeline for both training and validation in only 80 lines of code with MMEngine. The whole process includes the following steps:

- Build a Model
- Build a Dataset and DataLoader
- Build Evaluation Metrics
- Build a Runner and Run the Task

## Build a Model

First, we need to build a model. In MMEngine, the model should inherit from `BaseModel`. Aside from parameters representing inputs from the dataset, its `forward` method needs to accept an extra argument called `mode`:

- For training, the value of `mode` is `"loss"`, and the `forward` method should return a dict containing the key `"loss"`.
- For validation, the value of `mode` is `"predict"`, and the `forward` method should return results containing both predictions and labels.

```python
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel

class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels
```

## Build a Dataset and DataLoader

Next, we need to create `Dataset` and `DataLoader` for training and validation. For basic training and validation, we can simply use built-in datasets supported in TorchVision.

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))

val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))
```

## Build Evaluation Metrics

To validate and test the model, we need to define a Metric called `Accuracy` to evaluate the model. This metric needs to inherit from `BaseMetric` and implement the `process` and `compute_metrics` methods. The `process` method accepts the output of the dataset and other outputs when `mode="predict"`. The output data in this scenario is a batch of data. After processing this batch of data, we save the information to the `self.results` property. `compute_metrics` accepts a `results` parameter, which contains all the information saved in `process` (in the case of a distributed environment, `results` are the information collected from all processes). Use this information to calculate and return a dict that holds the results of the evaluation metrics.

```python
from mmengine.evaluator import BaseMetric

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)
```

## Build a Runner and Run the Task

Now we can build a Runner with the previously defined Model, DataLoader, and Metrics, and some other configs shown as follows:

```python
from torch.optim import SGD
from mmengine.runner import Runner

runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
)
runner.train()
```

Finally, let’s put all the code above together into a complete script that uses the MMEngine executor for training and validation:

```python
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import DataLoader

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner

class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(batch_size=32,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(**norm_cfg)
                                  ])))

val_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))

runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
)
runner.train()

---

The tutorial on the `runner` component of MMEngine provides a comprehensive overview of its role and functionalities. Here’s a summary and explanation of the key points covered:

### Overview

The `runner` in MMEngine serves as the central orchestrator, integrating various modules and managing their interactions. Its primary responsibilities include organizing and scheduling tasks, handling configurations, and ensuring smooth execution of the training and evaluation processes.

### Key Points

1. **Common Usage and Configuration**:
   - The `runner` simplifies the process of setting up training pipelines by providing a clear and organized interface.
   - Users can either configure the `runner` using its API documentation or modify existing configurations from examples like MMDet.

2. **Example Code**:
   - The tutorial provides a detailed example to illustrate how to use the `runner` for training a model.
   - Key components include model definition, dataset preparation, data loading, optimizer setup, and evaluation metrics.
   - The `Runner` class from `mmengine.runner` is used to initialize and start the training process.

3. **Best Practices**:
   - It’s recommended to use config files for managing `runner` configurations.
   - Config files use Python syntax, making it easy to adjust settings and manage experiments.
   - Example config files are provided to demonstrate how to convert code-based configurations into a more manageable format.

4. **Dataflow and Execution**:
   - The tutorial includes a diagram illustrating the basic dataflow within the `runner`, covering training and validation processes.
   - Key components include data preprocessors, models, and evaluators, with specific data formats and conventions.

5. **Advantages of Using `runner`**:
   - The `runner` modularizes the training pipeline, allowing easy modifications and scalability.
   - It provides a clear interface for configuring and managing experiments, with support for various features like mixed precision training and distributed computing.

6. **Philosophy**:
   - The `runner` aims to simplify the training process by abstracting complex interactions and providing a user-friendly interface.
   - It supports evolving practices and technologies, allowing users to focus on their core work without being bogged down by implementation details.

### Answers to Specific Questions

- **Data Item Types**: Each data item in the diagram represents different stages and formats, such as raw data, processed data, and output metrics. The specific types and formats are detailed in the tutorial.
  
- **Data Format Convention**: The convention ensures consistent data handling between different modules, such as how data is passed from the dataloader to the model and then to the evaluator.

- **Data Preprocessor**: It can handle various preprocessing tasks like cropping and resizing images before they are fed into the model.

- **Model Outputs**: The model can produce outputs in different formats depending on the mode: raw tensors, predictions, or loss values.

- **Lines in the Diagram**: Different colored lines represent various processes in the training and validation workflows, such as the training process (red), validation/testing (blue), and data flow (green).

- **Overriding Methods**: Customizing methods like `train_step` can alter the dataflow, but the diagram provides a general overview applicable to standard setups.

This tutorial emphasizes the flexibility and power of the `runner` while encouraging users to focus on high-level configuration and architecture rather than getting lost in implementation details.

---

## Runner

```python
class mmengine.runner.Runner:
    def __init__(self, model, work_dir, train_dataloader=None, val_dataloader=None, test_dataloader=None, train_cfg=None, val_cfg=None, test_cfg=None, auto_scale_lr=None, optim_wrapper=None, param_scheduler=None, val_evaluator=None, test_evaluator=None, default_hooks=None, custom_hooks=None, data_preprocessor=None, load_from=None, resume=False, launcher='none', env_cfg={'dist_cfg': {'backend': 'nccl'}}, log_processor=None, log_level='INFO', visualizer=None, default_scope='mmengine', randomness={'seed': None}, experiment_name=None, cfg=None):
        pass
```

A training helper for PyTorch.

The `Runner` object can be built from a config using `runner = Runner.from_cfg(cfg)`, where `cfg` usually contains training, validation, and test-related configurations to build corresponding components. Typically, the same config is used to launch training, testing, and validation tasks. However, only the necessary components are initialized at runtime. For example, testing a model does not require training or validation components.

To avoid repeatedly modifying the config, `Runner` adopts lazy initialization, initializing components only when needed. The model is initialized at the beginning, while training, validation, and testing components are initialized only when calling `runner.train()`, `runner.val()`, and `runner.test()`, respectively.

### Parameters

- **model (torch.nn.Module or dict)**: The model to be run. It can be a dict used to build a model.

- **work_dir (str)**: The working directory to save checkpoints. Logs will be saved in the subdirectory of `work_dir` named `timestamp`.

- **train_dataloader (Dataloader or dict, optional)**: A dataloader object or a dict to build a dataloader. Defaults to None, meaning training steps will be skipped if not specified.

- **val_dataloader (Dataloader or dict, optional)**: A dataloader object or a dict to build a dataloader. Defaults to None, meaning validation steps will be skipped if not specified.

- **test_dataloader (Dataloader or dict, optional)**: A dataloader object or a dict to build a dataloader. Defaults to None, meaning test steps will be skipped if not specified.

- **train_cfg (dict, optional)**: A dict to build a training loop. Should contain "by_epoch" to decide the type of training loop. Defaults to None.

- **val_cfg (dict, optional)**: A dict to build a validation loop. Defaults to None. If `fp16=True` is set, `runner.val()` will be performed under fp16 precision.

- **test_cfg (dict, optional)**: A dict to build a test loop. Defaults to None. If `fp16=True` is set, `runner.val()` will be performed under fp16 precision.

- **auto_scale_lr (dict, optional)**: Config to scale the learning rate automatically. Includes `base_batch_size` and `enable`.

- **optim_wrapper (OptimWrapper or dict, optional)**: For computing gradients of model parameters. Defaults to None.

- **param_scheduler (_ParamScheduler or dict or list, optional)**: Parameter scheduler for updating optimizer parameters. Defaults to None.

- **val_evaluator (Evaluator or dict or list, optional)**: An evaluator object used for computing metrics for validation. Defaults to None.

- **test_evaluator (Evaluator or dict or list, optional)**: An evaluator object used for computing metrics for test steps. Defaults to None.

- **default_hooks (dict[str, dict] or dict[str, Hook], optional)**: Hooks to execute default actions like updating model parameters and saving checkpoints. Defaults to None.

- **custom_hooks (list[dict] or list[Hook], optional)**: Hooks to execute custom actions like visualizing images processed by pipeline. Defaults to None.

- **data_preprocessor (dict, optional)**: Pre-process config of `BaseDataPreprocessor`. Defaults to None.

- **load_from (str, optional)**: The checkpoint file to load from. Defaults to None.

- **resume (bool)**: Whether to resume training. Defaults to False. If True and `load_from` is None, will find the latest checkpoint from `work_dir`.

- **launcher (str)**: Way to launch multi-process. Supported launchers are `pytorch`, `mpi`, `slurm`, and `none`.

- **env_cfg (dict)**: A dict used for setting environment. Defaults to `dict(dist_cfg=dict(backend='nccl'))`.

- **log_processor (dict, optional)**: A processor to format logs. Defaults to None.

- **log_level (int or str)**: The log level of MMLogger handlers. Defaults to 'INFO'.

- **visualizer (Visualizer or dict, optional)**: A Visualizer object or a dict to build a Visualizer object. Defaults to None.

- **default_scope (str)**: Used to reset registries location. Defaults to "mmengine".

- **randomness (dict)**: Settings to make the experiment reproducible, like seed and deterministic. Defaults to `dict(seed=None)`.

- **experiment_name (str, optional)**: Name of the current experiment. Defaults to None.

- **cfg (dict or Configdict or Config, optional)**: Full config. Defaults to None.

### Note

Since PyTorch 2.0.0, you can enable `torch.compile` by passing `cfg.compile = True`. For compile options, pass a dict, e.g., `cfg.compile = dict(backend='eager')`. Refer to PyTorch API Documentation for more valid options.

### Examples

```python
>>> from mmengine.runner import Runner
>>> cfg = dict(
>>>     model=dict(type='ToyModel'),
>>>     work_dir='path/of/work_dir',
>>>     train_dataloader=dict(
>>>         dataset=dict(type='ToyDataset'),
>>>         sampler=dict(type='DefaultSampler', shuffle=True),
>>>         batch_size=1,
>>>         num_workers=0),
>>>     val_dataloader=dict(
>>>         dataset=dict(type='ToyDataset'),
>>>         sampler=dict(type='DefaultSampler', shuffle=False),
>>>         batch_size=1,
>>>         num_workers=0),
>>>     test_dataloader=dict(
>>>         dataset=dict(type='ToyDataset'),
>>>         sampler=dict(type='DefaultSampler', shuffle=False),
>>>         batch_size=1,
>>>         num_workers=0),
>>>     auto_scale_lr=dict(base_batch_size=16, enable=False),
>>>     optim_wrapper=dict(type='OptimizerWrapper', optimizer=dict(
>>>         type='SGD', lr=0.01)),
>>>     param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
>>>     val_evaluator=dict(type='ToyEvaluator'),
>>>     test_evaluator=dict(type='ToyEvaluator'),
>>>     train_cfg=dict(by_epoch=True, max_epochs=3, val_interval=1),
>>>     val_cfg=dict(),
>>>     test_cfg=dict(),
>>>     custom_hooks=[],
>>>     default_hooks=dict(
>>>         timer=dict(type='IterTimerHook'),
>>>         checkpoint=dict(type='CheckpointHook', interval=1),
>>>         logger=dict(type='LoggerHook'),
>>>         optimizer=dict(type='OptimizerHook', grad_clip=False),
>>>         param_scheduler=dict(type='ParamSchedulerHook')),
>>>     launcher='none',
>>>     env_cfg=dict(dist_cfg=dict(backend='nccl')),
>>>     log_processor=dict(window_size=20),
>>>     visualizer=dict(type='Visualizer',
>>>         vis_backends=[dict(type='LocalVisBackend',
>>>                            save_dir='temp_dir')])
>>> )
>>> runner = Runner.from_cfg(cfg)
>>> runner.train()
>>> runner.test()
```

## static build_dataloader(dataloader, seed=None, diff_rank_seed=False)

Build dataloader.

The method builds three components:

1. Dataset
2. Sampler
3. Dataloader

An example of dataloader:

```python
dataloader = dict(
    dataset=dict(type='ToyDataset'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_size=1,
    num_workers=9
)
```

### Parameters

- **dataloader (DataLoader or dict)**: A Dataloader object or a dict to build Dataloader object.

- **seed (int, optional)**: Random seed. Defaults to None.

- **diff_rank_seed (bool)**: Whether to set different seeds for different ranks. Defaults to False.

### Returns

- **DataLoader**: Built from `dataloader_cfg`.

## build_evaluator(evaluator)

Build evaluator.

Examples of evaluator:

```python
# evaluator could be a built Evaluator instance
evaluator = Evaluator(metrics=[ToyMetric()])

# evaluator can also be a list of dict
evaluator = [
    dict(type='ToyMetric1'),
    dict(type='ToyEvaluator2')
]

# evaluator can also be a list of built metric
evaluator = [ToyMetric1(), ToyMetric2()]

# evaluator can also be a dict with key metrics
evaluator = dict(metrics=ToyMetric())
# metric is a list
evaluator = dict(metrics=[ToyMetric()])
```

### Parameters

- **evaluator (Evaluator or dict or list)**: An Evaluator object or a config dict or list of config dict used to build an Evaluator.

### Returns

- **Evaluator**: Built from `evaluator`.

## build_log_processor(log_processor)

Build test log_processor.

Examples of log_processor:

```python
# LogProcessor will be used
log_processor = dict()

# custom log_processor
log_processor = dict(type='CustomLogProcessor')
```

### Parameters

- **log_processor (LogProcessor or dict)**: A log processor or a dict to build log processor.

### Returns

- **LogProcessor**:

 Built from `log_processor`.

---

## Runner

Deep learning algorithms usually share similar pipelines for training, validation, and testing. Therefore, MMengine designed Runner to simplify the construction of these pipelines. In most cases, users can use our default Runner directly. If you find it not feasible to implement your ideas, you can also modify it or customize your own runner.

Before introducing the design of Runner, let’s walk through some examples to better understand why we should use runner. Below is a few lines of pseudo codes for training models in PyTorch:

```python
model = ResNet()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
train_dataset = ImageNetDataset(...)
train_dataloader = DataLoader(train_dataset, ...)

for i in range(max_epochs):
    for data_batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(data_batch)
        loss = loss_func(outputs, data_batch)
        loss.backward()
        optimizer.step()
```

Pseudo codes for model validation in PyTorch:

```python
model = ResNet()
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()

test_dataset = ImageNetDataset(...)
test_dataloader = DataLoader(test_dataset, ...)

for data_batch in test_dataloader:
    outputs = model(data_batch)
    acc = calculate_acc(outputs, data_batch)
```

Pseudo codes for model inference in PyTorch:

```python
model = ResNet()
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()

for img in imgs:
    prediction = model(img)
```

The observation from the above 3 pieces of codes is that they are similar. They can all be divided into some distinct steps, such as model construction, data loading, and loop iterations. Although the above examples are based on image classification tasks, the same holds for many other tasks as well, including object detection, image segmentation, etc. Based on the observation above, we propose runner, which structures the training, validation, and testing pipeline. With runner, the only thing you need to do is to prepare necessary components (models, data, etc.) of your pipeline and leave the schedule and execution to Runner. You are free of constructing similar pipelines one and another time. You are free of annoying details like the differences between distributed and non-distributed training. You can focus on your own awesome ideas. These are all achieved by runner and various practical modules in MMEngine.

## Runner

The Runner in MMEngine contains various modules required for training, testing, and validation, as well as loop controllers (Loop) and Hook, as shown in the figure above. It provides 3 APIs for users: `train`, `val`, and `test`, each corresponding to a specific Loop. You can use Runner either by providing a config file or by providing manually constructed modules. Once activated, the Runner will automatically set up the runtime environment, build/compose your modules, execute the loop iterations in Loop, and call registered hooks during iterations.

The execution order of Runner is as follows:

```text
runner_flow
```

A feature of Runner is that it will always lazily initialize modules managed by itself. To be specific, Runner won’t build every module on initialization, and it won’t build a module until it is needed in the current Loop. Therefore, if you are running only one of the train, val, or test pipelines, you only need to provide the relevant configs/modules.

## Loop

In MMEngine, we abstract the execution process of the task into Loop, based on the observation that most deep learning tasks can be summarized as a model iterating over datasets. We provide 4 built-in loops in MMEngine:

- `EpochBasedTrainLoop`
- `IterBasedTrainLoop`
- `ValLoop`
- `TestLoop`

The built-in runner and loops are capable of most deep learning tasks, but surely not all. Some tasks need extra modifications and refactorizations. Therefore, we make it possible for users to customize their own pipelines for model training, validation, and testing.

You can write your own pipeline by subclassing `BaseLoop`, which needs 2 arguments for initialization: 1) `runner` the Runner instance, and 2) `dataloader` the dataloader used in this loop. You are free to add more arguments to your own loop subclass. After defining your own loop subclass, you should register it to `LOOPS(mmengine.registry.LOOPS)`, and specify it in config files by the `type` field in `train_cfg`, `val_cfg`, and `test_cfg`. In fact, you can write any execution order, any hook position in your own loop. However, built-in hooks may not work if you change hook positions, which may lead to inconsistent behavior during training. Therefore, we strongly recommend you implement your subclass with a similar execution order illustrated in the figure above and with the same hook positions defined in hook documentation.

```python
from mmengine.registry import LOOPS, HOOKS
from mmengine.runner import BaseLoop
from mmengine.hooks import Hook

# Customized validation loop
@LOOPS.register_module()
class CustomValLoop(BaseLoop):
    def __init__(self, runner, dataloader, evaluator, dataloader2):
        super().__init__(runner, dataloader, evaluator)
        self.dataloader2 = runner.build_dataloader(dataloader2)

    def run(self):
        self.runner.call_hooks('before_val_epoch')
        for idx, data_batch in enumerate(self.dataloader):
            self.runner.call_hooks(
                'before_val_iter', batch_idx=idx, data_batch=data_batch)
            outputs = self.run_iter(idx, data_batch)
            self.runner.call_hooks(
                'after_val_iter', batch_idx=idx, data_batch=data_batch, outputs=outputs)
        metric = self.evaluator.evaluate()

        # add extra loop for validation purpose
        for idx, data_batch in enumerate(self.dataloader2):
            # add new hooks
            self.runner.call_hooks(
                'before_valloader2_iter', batch_idx=idx, data_batch=data_batch)
            self.run_iter(idx, data_batch)
            # add new hooks
            self.runner.call_hooks(
                'after_valloader2_iter', batch_idx=idx, data_batch=data_batch, outputs=outputs)
        metric2 = self.evaluator.evaluate()

        ...

        self.runner.call_hooks('after_val_epoch')
```

The example above shows how to implement a different validation loop. The new loop validates on two different validation datasets. It also defines a new hook position in the second validation. You can easily use it by setting `type='CustomValLoop'` in `val_cfg` in your config file.

```python
# Customized validation loop
val_cfg = dict(type='CustomValLoop', dataloader2=dict(dataset=dict(type='ValDataset2'), ...))
# Customized hook with extra hook position
custom_hooks = [dict(type='CustomValHook')]
```

## Customize Runner

Moreover, you can write your own runner by subclassing `Runner` if the built-in Runner is not feasible. The method is similar to writing other modules: write your subclass inherited from Runner, override some functions, register it to `mmengine.registry.RUNNERS`, and access it by assigning `runner_type` in your config file.

```python
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

@RUNNERS.register_module()
class CustomRunner(Runner):

    def setup_env(self):
        ...
```

The example above shows how to implement a customized runner which overrides the `setup_env` function and is registered to `RUNNERS`. Now `CustomRunner` is prepared to be used by setting `runner_type='CustomRunner'` in your config file.

---

## BaseLoop

### `class mmengine.runner.BaseLoop(runner, dataloader)`

Base loop class.

All subclasses inherited from `BaseLoop` should overwrite the `run()` method.

#### Parameters:

- **runner** (`Runner`) – A reference to the runner.
- **dataloader** (`Dataloader` or `dict`) – An iterator to generate one batch of dataset each iteration.

### `abstract run()`

Execute loop.

**Return type:** `Any`

---

## Logging

Runner will produce a lot of logs during the running process, such as loss, iteration time, learning rate, etc. MMEngine implements a flexible logging system that allows us to choose different types of log statistical methods when configuring the runner. It helps set/get the recorded log at any location in the code.

## Flexible Logging System

The logging system is configured by passing a `LogProcessor` to the runner. If no log processor is passed, the runner will use the default log processor, which is equivalent to:

```python
log_processor = dict(window_size=10, by_epoch=True, custom_cfg=None, num_digits=4)
```

The format of the output log is as follows:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmengine.runner import Runner
from mmengine.model import BaseModel

train_dataset = [(torch.ones(1, 1), torch.ones(1, 1))] * 50
train_dataloader = DataLoader(train_dataset, batch_size=2)

class ToyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, mode):
        feat = self.linear(img)
        loss1 = (feat - label).pow(2)
        loss2 = (feat - label).abs()
        return dict(loss1=loss1, loss2=loss2)

runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01))
)
runner.train()
```

Example log output:

```
08/21 02:58:41 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0019  data_time: 0.0004  loss1: 0.8381  loss2: 0.9007  loss: 1.7388
08/21 02:58:41 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0029  data_time: 0.0010  loss1: 0.1978  loss2: 0.4312  loss: 0.6290
```

`LogProcessor` will output the log in the following format:

- **Prefix of the log:**
  - `epoch mode(by_epoch=True)`: `Epoch(train) [{current_epoch}/{current_iteration}]/{dataloader_length}`
  - `iteration mode(by_epoch=False)`: `Iter(train) [{current_iteration}/{max_iteration}]`

- **Learning rate (lr):** The learning rate of the last iteration.

- **Time:**
  - `time:` The averaged time for inference of the last `window_size` iterations.
  - `data_time:` The averaged time for loading data of the last `window_size` iterations.
  - `eta:` The estimated time of arrival to finish the training.

- **Loss:** The averaged loss output by model of the last `window_size` iterations.

### Note

- `window_size=10` by default.
- The significant digits (`num_digits`) of the log is `4` by default.
- Output the value of all custom logs at the last iteration by default.

### Warning

`LogProcessor` outputs the epoch-based log by default (`by_epoch=True`). To get an expected log matched with the `train_cfg`, we should set the same value for `by_epoch` in `train_cfg` and `log_processor`.

Based on the rules above, the code snippet will count the average value of `loss1` and `loss2` every 10 iterations.

If we want to count the global average value of `loss1`, we can set `custom_cfg` like this:

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    log_processor=dict(
        custom_cfg=[
            dict(data_src='loss1',  # original loss name: loss1
                 method_name='mean',  # statistical method: mean
                 window_size='global')])  # window_size: global
)
runner.train()
```

Example log output:

```
08/21 02:58:49 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0026  data_time: 0.0007  loss1: 0.7381  loss2: 0.8446  loss: 1.5827
08/21 02:58:49 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0030  data_time: 0.0012  loss1: 0.4521  loss2: 0.3939  loss: 0.5600
```

`data_src` means the original loss name, `method_name` means the statistic method, `window_size` means the window size of the statistic method. Since we want to count the global average value of `loss1`, we set `window_size` to `global`.

Currently, MMEngine supports the following statistical methods:

| Statistic Method | Arguments       | Function                                              |
|------------------|-----------------|-------------------------------------------------------|
| mean             | window_size     | Statistic the average log of the last `window_size`  |
| min              | window_size     | Statistic the minimum log of the last `window_size`  |
| max              | window_size     | Statistic the maximum log of the last `window_size`  |
| current          | /               | Statistic the latest log                              |

`window_size` mentioned above could be:

- `int number`: The window size of the statistic method.
- `global`: Equivalent to `window_size=cur_iteration`.
- `epoch`: Equivalent to `window_size=len(dataloader)`.

If we want to statistic the average value of `loss1` of the last 10 iterations, and also want to statistic the global average value of `loss1`, we need to set `log_name` additionally:

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    log_processor=dict(
        custom_cfg=[
            # log_name means the second name of loss1
            dict(data_src='loss1', log_name='loss1_global', method_name='mean', window_size='global')])
)
runner.train()
```

Example log output:

```text
08/21 18:39:32 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0016  data_time: 0.0004  loss1: 0.1512  loss2: 0.3751  loss: 0.5264  loss1_global: 0.1512
08/21 18:39:32 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0051  data_time: 0.0036  loss1: 0.0113  loss2: 0.0856  loss: 0.0970  loss1_global: 0.0813
```

Similarly, we can also statistic the global/local maximum value of loss at the same time:

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    log_processor=dict(custom_cfg=[
        # Statistic loss1 with the local maximum value
        dict(data_src='loss1',
             log_name='loss1_local_max',
             window_size=10,
             method_name='max'),
        # Statistic loss1 with the global maximum value
        dict(
            data_src='loss1',
            log_name='loss1_global_max',
            method_name='max',
            window_size='global')
    ]))
runner.train()
```

Example log output:

```text
08/21 03:17:26 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0021  data_time: 0.0006  loss1: 1.8495  loss2: 1.3427  loss: 3.1922  loss1_local_max: 2.8872  loss1_global_max: 2.8872
08/21 03:17:26 -

 mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0024  data_time: 0.0010  loss1: 0.5464  loss2: 0.7251  loss: 1.2715  loss1_local_max: 2.8872  loss1_global_max: 2.8872
```

More examples can be found in `log_processor`.

## Customize Log

The logging system can not only log the loss, lr, etc., but also collect and output custom logs. For example, if we want to statistic the intermediate loss:

```python
from mmengine.logging import MessageHub

class ToyModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, mode):
        feat = self.linear(img)
        loss_tmp = (feat - label).abs()
        loss = loss_tmp.pow(2)

        message_hub = MessageHub.get_current_instance()
        # Update the intermediate `loss_tmp` in the message hub
        message_hub.update_scalar('train/loss_tmp', loss_tmp.sum())
        return dict(loss=loss)
```

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
    log_processor=dict(
        custom_cfg=[
            # Statistic the loss_tmp with the averaged value
            dict(
                data_src='loss_tmp',
                window_size=10,
                method_name='mean')
        ]
    )
)
runner.train()
```

Example log output:

```
08/21 03:40:31 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0026  data_time: 0.0008  loss_tmp: 0.0097  loss: 0.0000
08/21 03:40:31 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0028  data_time: 0.0013  loss_tmp: 0.0065  loss: 0.0000
```

The custom log will be recorded by updating the `MessageHub`:

- Call `MessageHub.get_current_instance()` to get the message of runner.
- Call `MessageHub.update_scalar` to update the custom log. The first argument means the log name with the mode prefix (`train/val/test`). The output log will only retain the log name without the mode prefix.

Configure the statistic method of `loss_tmp` in `log_processor`. If it is not configured, only the latest value of `loss_tmp` will be logged.

## Export the Debug Log

Set `log_level='DEBUG'` for the runner, and the debug log will be exported to the `work_dir`:

```python
runner = Runner(
    model=ToyModel(),
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    log_level='DEBUG',
    train_cfg=dict(by_epoch=True, max_epochs=1),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)))
runner.train()
```

Example debug log output:

```text
08/21 18:16:22 - mmengine - DEBUG - Get class `LocalVisBackend` from "vis_backend" registry in "mmengine"
08/21 18:16:22 - mmengine - DEBUG - An `LocalVisBackend` instance is built from registry, its implementation can be found in mmengine.visualization.vis_backend
08/21 18:16:22 - mmengine - DEBUG - Get class `RuntimeInfoHook` from "hook" registry in "mmengine"
08/21 18:16:22 - mmengine - DEBUG - An `RuntimeInfoHook` instance is built from registry, its implementation can be found in mmengine.hooks.runtime_info_hook
08/21 18:16:22 - mmengine - DEBUG - Get class `IterTimerHook` from "hook" registry in "mmengine"
...
```

Besides, logs of different ranks will be saved in debug mode if you are training your model with shared storage. The hierarchy of the log is as follows:

```text
./tmp
├── tmp.log
├── tmp_rank1.log
├── tmp_rank2.log
├── tmp_rank3.log
├── tmp_rank4.log
├── tmp_rank5.log
├── tmp_rank6.log
└── tmp_rank7.log
...
└── tmp_rank63.log
```

For multiple machines with independent storage:

```text
# device: 0:
work_dir/
└── exp_name_logs
    ├── exp_name.log
    ├── exp_name_rank1.log
    ├── exp_name_rank2.log
    ├── exp_name_rank3.log
    ...
    └── exp_name_rank7.log

# device: 7:
work_dir/
└── exp_name_logs
    ├── exp_name_rank56.log
    ├── exp_name_rank57.log
    ├── exp_name_rank58.log
    ...
    └── exp_name_rank63.log
```
