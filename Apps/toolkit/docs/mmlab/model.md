# 15 Minutes to Get Started with MMEngine

In this tutorial, we’ll take training a ResNet-50 model on the CIFAR-10 dataset as an example. We will build a complete and configurable pipeline for both training and validation in only 80 lines of code with MMEngine. The whole process includes the following steps:

1. Build a Model
2. Build a Dataset and DataLoader
3. Build Evaluation Metrics
4. Build a Runner and Run the Task

## Build a Model

First, we need to build a model. In MMEngine, the model should inherit from `BaseModel`. Aside from parameters representing inputs from the dataset, its `forward` method needs to accept an extra argument called `mode`:

- For training, the value of `mode` is `loss`, and the `forward` method should return a dict containing the key `loss`.
- For validation, the value of `mode` is `predict`, and the `forward` method should return results containing both predictions and labels.

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

To validate and test the model, we need to define a metric called `accuracy` to evaluate the model. This metric needs to inherit from `BaseMetric` and implement the `process` and `compute_metrics` methods where the `process` method accepts the output of the dataset and other outputs when `mode="predict"`. The output data at this scenario is a batch of data. After processing this batch of data, we save the information to the `self.results` property. `compute_metrics` accepts a `results` parameter. The input `results` of `compute_metrics` is all the information saved in `process` (In the case of a distributed environment, `results` are the information collected from all processes). Use this information to calculate and return a dict that holds the results of the evaluation metrics.

```python
from mmengine.evaluator import BaseMetric

class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # Save the middle result of a batch to `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # Return the dict containing the eval results
        return dict(accuracy=100 * total_correct / total_size)
```

## Build a Runner and Run the Task

Now we can build a `Runner` with the previously defined Model, DataLoader, and Metrics, and some other configs shown as follows:

```python
from torch.optim import SGD
from mmengine.runner import Runner

runner = Runner(
    # The model used for training and validation. Needs to meet specific interface requirements
    model=MMResNet50(),
    # Working directory which saves training logs and weight files
    work_dir='./work_dir',
    # Train dataloader needs to meet the PyTorch data loader protocol
    train_dataloader=train_dataloader,
    # Optimizer wrapper for optimization with additional features like AMP, gradient accumulation, etc
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    # Training configs for specifying training epochs, verification intervals, etc
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    # Validation dataloader also needs to meet the PyTorch data loader protocol
    val_dataloader=val_dataloader,
    # Validation configs for specifying additional parameters required for validation
    val_cfg=dict(),
    # Validation evaluator. The default one is used here
    val_evaluator=dict(type=Accuracy),
)

runner.train()
```

Finally, let’s put all the code together into a complete script that uses the MMEngine executor for training and validation:

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
```
---

# Model

## Runner and Model

As mentioned in the basic dataflow, the interaction between the `DataLoader`, `model`, and `evaluator` follows specific rules. Here’s a quick review:

### Training Process

```python
for data_batch in train_dataloader:
    data_batch = model.data_preprocessor(data_batch, training=True)
    if isinstance(data_batch, dict):
        losses = model(**data_batch, mode='loss')
    elif isinstance(data_batch, (list, tuple)):
        losses = model(*data_batch, mode='loss')
    else:
        raise TypeError()
```

### Validation Process

```python
for data_batch in val_dataloader:
    data_batch = model.data_preprocessor(data_batch, training=False)
    if isinstance(data_batch, dict):
        outputs = model(**data_batch, mode='predict')
    elif isinstance(data_batch, (list, tuple)):
        outputs = model(**data_batch, mode='predict')
    else:
        raise TypeError()
    evaluator.process(data_samples=outputs, data_batch=data_batch)
metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

In the runner tutorial, we touched on the relationships between `DataLoader`, `model`, and `evaluator`, and introduced the concept of `data_preprocessor`. You may have a basic understanding of the model, but the actual execution of the runner is more complex than the above pseudo-code.

To focus on the algorithm itself and simplify the complex relationships between the model, `DataLoader`, and evaluator, we designed `BaseModel`. Typically, you only need to inherit from `BaseModel` and implement the `forward` method to handle training, testing, and validation processes.

### Key Questions

1. When do we update the model parameters, and how can we do it with a custom optimization process?
2. Why is the `data_preprocessor` concept necessary, and what functions can it perform?

## Interface Introduction

Typically, we define a model to implement the body of the algorithm. In MMEngine, the model is managed by the `Runner` and needs to implement specific interfaces like `train_step`, `val_step`, and `test_step`. For high-level tasks such as detection, classification, and segmentation, these interfaces usually follow a standard workflow. For example, `train_step` calculates the loss and updates the model parameters, while `val_step` and `test_step` calculate metrics and return predictions. Thus, MMEngine abstracts `BaseModel` to implement this common workflow.

### Benefits of BaseModel

By inheriting from `BaseModel`, you only need to implement the `forward` function to handle training, testing, and validation processes.

**Note:**
`BaseModel` inherits from `BaseModule`, allowing dynamic initialization of model parameters.

- **`forward` Method:**
  The arguments of `forward` must match the data provided by the `DataLoader`. If the `DataLoader` samples a tuple, `forward` needs to accept unpacked `*data`. If the `DataLoader` returns a dictionary, `forward` should accept unpacked `**data`. The `forward` method also accepts a `mode` parameter to control the execution branch:
  - **`mode='loss'`:** Enables loss mode during training, returning a differentiable loss dictionary. This branch is called by `train_step`.
  - **`mode='predict'`:** Enables prediction mode during validation/testing, returning predictions matching the arguments of `process`. In OpenMMLab repositories, predictions must be a list with each element being a `BaseDataElement`. This branch is called by `val_step`.
  - **`mode='tensor'`:** In both tensor and predict modes, `forward` returns predictions. However, `tensor` mode returns a tensor or container of tensors without post-processing, such as non-maximum suppression (NMS). Post-processing can be customized after obtaining results in tensor mode.

- **`train_step`:**
  Retrieves the loss dictionary by calling `forward` with loss mode. `BaseModel` implements a standard optimization process as follows:

  ```python
  def train_step(self, data, optim_wrapper):
      data = self.data_preprocessor(data, training=True)
      loss = self(**data, mode='loss')
      parsed_losses, log_vars = self.parse_losses()
      optim_wrapper.update_params(parsed_losses)
      return log_vars
  ```

- **`val_step`:**
  Retrieves predictions by calling `forward` with predict mode.

  ```python
  def val_step(self, data, optim_wrapper):
      data = self.data_preprocessor(data, training=False)
      outputs = self(**data, mode='predict')
      return outputs
  ```

- **`test_step`:**
  Functions similarly to `val_step` in `BaseModel`, but can be customized in subclasses if needed.

### Understanding BaseModel

Here's a more complete pseudo-code based on the interfaces:

```python
# Training
for data_batch in train_dataloader:
    loss_dict = model.train_step(data_batch)

# Validation
for data_batch in val_dataloader:
    preds = model.test_step(data_batch)
    evaluator.process(data_samples=preds, data_batch=data_batch)
metrics = evaluator.evaluate(len(val_dataloader.dataset))
```

Great! Ignoring hooks, the pseudo-code above almost implements the main loop logic. Let's revisit the 15-minute tutorial to understand what MMResNet has done:

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

    def train_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        loss = self(*data, mode='loss')
        parsed_losses, log_vars = self.parse_losses()
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs

    def test_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs
```

Now, you should have a deeper understanding of the dataflow and be able to answer the initial questions in the runner and model context.

## DataPreprocessor

If your computer has a GPU (or other hardware for acceleration like MPS, IPU, etc.), you will see that the program runs on the GPU. When does MMEngine move data and the model from CPU to GPU?

The `Runner` moves the model to the specified device during construction, while data is moved to the device during `self.data_preprocessor(data)` in the previous code snippet. The moved data is then passed to the model.

### Addressing Questions

1. **Why does MMResNet50 not define `data_preprocessor`, but still accesses it?**
   `MMResNet50` inherits from `BaseModel`, and `super().__init__` builds a default `data_preprocessor`.

2. **Why doesn’t `BaseModel` move data using `data = data.to(device)` but uses `DataPreprocessor` instead?**
   `BaseDataPreprocessor` handles moving data to the specified device:

   ```python
   class BaseDataPreprocessor(nn.Module):
       def forward(self, data, training=True):
           return tuple(_data.cuda() for _data in data)
   ```

### Additional Considerations

- **Normalization:**
  It’s more efficient to perform normalization in the `DataPreprocessor` to reduce bandwidth usage and improve efficiency, especially since data transfer times are longer than normalization.

- **Data Augmentation:**
  Augmentations like MixUp and Mosaic can be implemented in `DataPreprocessor` because they involve multiple images and are easier to manage with batch data rather than individual transformations.

- **Data Matching:**
  Adaptations should be handled by `DataPreprocessor` rather than modifying the `DataLoader` or model interface.

By now, you should understand the rationale behind the `DataPreprocessor` and be able to address the questions about the optimization wrapper and evaluator connections. Further details are covered in the evaluation tutorial and optimizer wrapper tutorial.

---

# BaseModel

## Class Definition
```python
class BaseModel(data_preprocessor=None, init_cfg=None)[source]
```

Base class for all algorithmic models. `BaseModel` implements the basic functions of the algorithmic model, such as weight initialization, batch input preprocessing (see more information in `BaseDataPreprocessor`), loss parsing, and model parameter updates.

Subclasses inheriting from `BaseModel` only need to implement the `forward` method, which handles the logic to calculate loss and predictions, allowing the model to be trained in the runner.

### Example
```python
@MODELS.register_module()
class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential()
        self.backbone.add_module('conv1', nn.Conv2d(3, 6, 5))
        self.backbone.add_module('pool', nn.MaxPool2d(2, 2))
        self.backbone.add_module('conv2', nn.Conv2d(6, 16, 5))
        self.backbone.add_module('fc1', nn.Linear(16 * 5 * 5, 120))
        self.backbone.add_module('fc2', nn.Linear(120, 84))
        self.backbone.add_module('fc3', nn.Linear(84, 10))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch_inputs, data_samples, mode='tensor'):
        data_samples = torch.stack(data_samples)
        if mode == 'tensor':
            return self.backbone(batch_inputs)
        elif mode == 'predict':
            feats = self.backbone(batch_inputs)
            predictions = torch.argmax(feats, 1)
            return predictions
        elif mode == 'loss':
            feats = self.backbone(batch_inputs)
            loss = self.criterion(feats, data_samples)
            return dict(loss=loss)
```

### Parameters
- **data_preprocessor** (dict, optional): The pre-process config of `BaseDataPreprocessor`.
- **init_cfg** (dict, optional): The weight initialization config for `BaseModule`.

#### `data_preprocessor`
Used for pre-processing data sampled by dataloader to the format accepted by `forward()`.
- **Type**: `BaseDataPreprocessor`

#### `init_cfg`
Initialization config dict.
- **Type**: dict, optional

## Methods

### `cpu(*args, **kwargs)[source]`
Overrides this method to call `BaseDataPreprocessor.cpu()` additionally.
- **Returns**: The model itself.
- **Return type**: `nn.Module`

### `cuda(device=None)[source]`
Overrides this method to call `BaseDataPreprocessor.cuda()` additionally.
- **Returns**: The model itself.
- **Return type**: `nn.Module`
- **Parameters**: `device` (int | str | device | None)

### `abstract forward(inputs, data_samples=None, mode='tensor')[source]`
Returns losses or predictions for training, validation, testing, and simple inference process. This method is abstract and must be implemented by subclasses. It accepts `batch_inputs` and `data_samples` processed by `data_preprocessor`, returning results according to the `mode` argument.

- **Parameters**:
  - `inputs` (torch.Tensor): Batch input tensor collated by `data_preprocessor`.
  - `data_samples` (list, optional): Data samples collated by `data_preprocessor`.
  - `mode` (str): Should be one of 'loss', 'predict', and 'tensor'.
    - `loss`: Called by `train_step` and returns a loss dict used for logging.
    - `predict`: Called by `val_step` and `test_step`, returns a list of results used for computing metrics.
    - `tensor`: Called by custom use to get Tensor type results.

- **Returns**:
  - If `mode == 'loss'`, returns a dict of loss tensor used for backward and logging.
  - If `mode == 'predict'`, returns a list of inference results.
  - If `mode == 'tensor'`, returns a tensor or tuple of tensors or dict of tensors for custom use.
- **Return type**: dict or list

### `mlu(device=None)[source]`
Overrides this method to call `BaseDataPreprocessor.mlu()` additionally.
- **Returns**: The model itself.
- **Return type**: `nn.Module`
- **Parameters**: `device` (int | str | device | None)

### `musa(device=None)[source]`
Overrides this method to call `BaseDataPreprocessor.musa()` additionally.
- **Returns**: The model itself.
- **Return type**: `nn.Module`
- **Parameters**: `device` (int | str | device | None)

### `npu(device=None)[source]`
Overrides this method to call `BaseDataPreprocessor.npu()` additionally.
- **Returns**: The model itself.
- **Return type**: `nn.Module`
- **Parameters**: `device` (int | str | device | None)

**Note**: This generation of NPU (Ascend910) does not support the use of multiple cards in a single process, so the index here needs to be consistent with the default device.

### `parse_losses(losses)[source]`
Parses the raw outputs (losses) of the network.
- **Parameters**: `losses` (dict) – Raw output of the network, usually containing losses and other necessary information.
- **Returns**: A tuple with two elements: the loss tensor passed to `optim_wrapper`, which may be a weighted sum of all losses, and `log_vars` which will be sent to the logger.
- **Return type**: tuple[Tensor, dict]

### `test_step(data)[source]`
Implements `test_step` the same as `val_step`.
- **Parameters**: `data` (dict or tuple or list) – Data sampled from dataset.
- **Returns**: The predictions of the given data.
- **Return type**: list

### `to(*args, **kwargs)[source]`
Overrides this method to call `BaseDataPreprocessor.to()` additionally.
- **Returns**: The model itself.
- **Return type**: `nn.Module`

### `train_step(data, optim_wrapper)[source]`
Implements the default model training process, including preprocessing, model forward propagation, loss calculation, optimization, and back-propagation. During non-distributed training, if subclasses do not override `train_step()`, `EpochBasedTrainLoop` or `IterBasedTrainLoop` will call this method to update model parameters. The default parameter update process includes:
- Calls `self.data_processor(data, training=False)` to collect `batch_inputs` and corresponding `data_samples` (labels).
- Calls `self(batch_inputs, data_samples, mode='loss')` to get the raw loss.
- Calls `self.parse_losses` to get `parsed_losses` tensor used for backward and a dict of loss tensor used to log messages.
- Calls `optim_wrapper.update_params(loss)` to update the model.

- **Parameters**:
  - `data` (dict or tuple or list): Data sampled from the dataset.
  - `optim_wrapper` (OptimWrapper): `OptimWrapper` instance used to update model parameters.
- **Returns**: A dict of tensors for logging.
- **Return type**: Dict[str, torch.Tensor]

### `val_step(data)[source]`
Gets the predictions of given data. Calls `self.data_preprocessor(data, False)` and `self(inputs, data_sample, mode='predict')` in order, and returns the predictions which will be passed to the evaluator.
- **Parameters**: `data` (dict or tuple or list) – Data sampled from the dataset.
- **Returns**: The predictions of the given data.
- **Return type**: list

---

```markdown
## Weight Initialization

When customizing a module based on `nn.Module` in PyTorch, we often use `torch.nn.init` to initialize the model parameters. MMEngine simplifies this process through the `BaseModule` class, which helps in defining and initializing the model from a configuration file.

### Initialize the Model from Config

The core functionality of `BaseModule` is to facilitate model initialization from a configuration. Subclasses of `BaseModule` can define the `init_cfg` in the `__init__` method, allowing for various initialization methods to be selected through configuration.

#### Supported Initialization Methods

| Initializer              | Registered Name  | Function                                                                                       |
|--------------------------|------------------|------------------------------------------------------------------------------------------------|
| `ConstantInit`           | Constant         | Initialize weights and biases with a constant. Commonly used for Convolution layers.          |
| `XavierInit`             | Xavier           | Initialize weights using Xavier initialization and biases with a constant.                    |
| `NormalInit`             | Normal           | Initialize weights using a normal distribution and biases with a constant.                    |
| `TruncNormalInit`        | TruncNormal      | Initialize weights using a truncated normal distribution and biases with a constant. Commonly used for Transformer models. |
| `UniformInit`            | Uniform          | Initialize weights using a uniform distribution and biases with a constant. Commonly used for Convolution layers. |
| `KaimingInit`            | Kaiming          | Initialize weights using Kaiming initialization and biases with a constant. Commonly used for Convolution layers. |
| `Caffe2XavierInit`       | Caffe2Xavier     | Apply Xavier initialization from Caffe2 and Kaiming initialization in PyTorch with "fan_in" and "normal" mode. Commonly used for Convolution layers. |
| `PretrainedInit`         | Pretrained       | Initialize the model using a pre-trained model.                                               |

### Example: Initialize the Model with a Pretrained Model

Define the `ToyNet` class as follows:

```python
import torch
import torch.nn as nn
from mmengine.model import BaseModule

class ToyNet(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = nn.Linear(1, 1)
```

Save the checkpoint:

```python
toy_net = ToyNet()
torch.save(toy_net.state_dict(), './pretrained.pth')
pretrained = './pretrained.pth'
```

Load the model with pretrained weights:

```python
toy_net = ToyNet(init_cfg=dict(type='Pretrained', checkpoint=pretrained))
toy_net.init_weights()
```

Console output will indicate the loading process:

```
08/19 16:50:24 - mmengine - INFO - load model from: ./pretrained.pth
08/19 16:50:24 - mmengine - INFO - local loads checkpoint from path: ./pretrained.pth
```

If `init_cfg` is a dictionary, `type` specifies the initializer registered in `WEIGHT_INITIALIZERS`. The `Pretrained` type maps to `PretrainedInit`, used for loading the checkpoint. The `checkpoint` argument specifies the path of the checkpoint, which can be a local path or URL.

### Commonly Used Initialization Methods

`Kaiming` initialization can be used similarly to the `Pretrained` initializer. For instance, configuring `init_cfg=dict(type='Kaiming', layer='Conv2d')` initializes all `Conv2d` modules with Kaiming initialization.

For different initialization methods for different modules:

```python
import torch.nn as nn
from mmengine.model import BaseModule

class ToyNet(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Linear(1, 1)
        self.conv = nn.Conv2d(1, 1, 1)

toy_net = ToyNet(
    init_cfg=[
        dict(type='Kaiming', layer='Conv2d'),
        dict(type='Xavier', layer='Linear')
    ]
)
toy_net.init_weights()
```

Console output will show:

```
08/19 16:50:24 - mmengine - INFO -
linear.weight - torch.Size([1, 1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
linear.bias - torch.Size([1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv.weight - torch.Size([1, 1, 1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution=normal, bias=0
```

### More Fine-Grained Initialization

For different initialization methods for submodules, use `override` in `init_cfg`:

```python
import torch.nn as nn
from mmengine.model import BaseModule

class ToyNet(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

toy_net = ToyNet(
    init_cfg=[
        dict(
            type='Kaiming',
            layer=['Conv2d'],
            override=dict(name='conv2', type='Xavier')
        )
    ]
)
toy_net.init_weights()
```

Console output will indicate:

```
08/19 16:50:24 - mmengine - INFO -
conv1.weight - torch.Size([1, 1, 1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv1.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.weight - torch.Size([1, 1, 1, 1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution=normal, bias=0
```

The `override` configuration allows nested `init_cfg` settings. The `name` field specifies the submodule scope for applying different initialization methods.

### Customize the Initialization Method

For custom initialization, override the `init_weights` method:

```python
import torch
import torch.nn as nn
from mmengine.model import BaseModule

class ToyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_weight = nn.Parameter(torch.empty(1, 1, 1, 1))
        self.custom_bias = nn.Parameter(torch.empty(1))

    def init_weights(self):
        with torch.no_grad():
            self.custom_weight.fill_(1)
            self.custom_bias.fill_(0)

class ToyNet(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.custom_conv = ToyConv()

toy_net = ToyNet(
    init_cfg=[
        dict(
            type='Kaiming',
            layer=['Conv2d'],
            override=dict(name='conv2', type='Xavier')
        )
    ]
)

toy_net.init_weights()
```

Console output will confirm:

```
08/19 16:50:24 - mmengine - INFO -
conv1.weight - torch.Size([1, 1, 1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv1.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.weight - torch.Size([1, 1, 1, 1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
custom_conv.custom_weight - torch.Size([1, 1, 1, 1]):
Initialized by user-defined `init_weights` in ToyConv

08/19 16:50:24 - mmengine - INFO -
custom_conv.custom_bias - torch.Size([1]):
Initialized by user-defined `init_weights` in ToyConv
```

### Conclusion

1. **Configure `init_cfg` to Initialize Models**:
   - Commonly used for initializing `Conv2d`,

 `Linear`, and other modules.
   - All initialization methods are managed by `WEIGHT_INITIALIZERS`.
   - Dynamic initialization controlled by `init_cfg`.

2. **Customize `init_weights`**:
   - Simpler and does not require registration compared to `init_cfg`.
   - Less flexible and does not support dynamic module initialization.

### Initialize Module with Function

MMEngine offers module initialization functions to simplify the process:

```python
from torch.nn.init import normal_, constant_
import torch.nn as nn

model = nn.Conv2d(1, 1, 1)
normal_(model.weight, mean=0, std=0.01)
constant_(model.bias, val=0)
```

MMEngine simplifies this with functions like:

```python
from mmengine.model import normal_init

normal_init(model, mean=0, std=0.01, bias=0)
```

Similarly, for Kaiming and Xavier initialization:

```python
from mmengine.model import kaiming_init, xavier_init

kaiming_init(model)
xavier_init(model)
```

### Available Initialization Functions

| Initialization Function | Description                                                                                           |
|-------------------------|-------------------------------------------------------------------------------------------------------|
| `constant_init`         | Initialize weights and biases with a constant. Commonly used for Convolution layers.                |
| `xavier_init`           | Initialize weights using Xavier initialization and biases with a constant.                          |
| `normal_init`           | Initialize weights using a normal distribution and biases with a constant.                          |
| `trunc_normal_init`     | Initialize weights using a truncated normal distribution and biases with a constant. Commonly used for Transformer models. |
| `uniform_init`          | Initialize weights using a uniform distribution and biases with a constant. Commonly used for Convolution layers. |
| `kaiming_init`          | Initialize weights using Kaiming initialization and biases with a constant. Commonly used for Convolution layers. |
| `caffe2_xavier_init`    | Apply Xavier initialization from Caffe2 and Kaiming initialization in PyTorch with "fan_in" and "normal" mode. Commonly used for Convolution layers. |
| `bias_init_with_prob`   | Initialize biases with a probability.                                                                 |

---

# mmengine.model

## Module

### Model

#### EMA

- **BaseAveragedModel**: A base class for averaging model weights.
- **ExponentialMovingAverage**: Implements the exponential moving average (EMA) of the model.
- **MomentumAnnealingEMA**: Exponential moving average (EMA) with momentum annealing strategy.
- **StochasticWeightAverage**: Implements the stochastic weight averaging (SWA) of the model.

### Model Wrapper

- **MMDistributedDataParallel**: A distributed model wrapper used for training, testing, and validation in loop.
- **MMSeparateDistributedDataParallel**: A DistributedDataParallel wrapper for models in MMGeneration.
- **MMFullyShardedDataParallel**: A wrapper for sharding Module parameters across data parallel workers.
- **is_model_wrapper**: Check if a module is a model wrapper.

### Weight Initialization

- **BaseInit**
- **Caffe2XavierInit**
- **ConstantInit**: Initialize module parameters with constant values.
- **KaimingInit**: Initialize module parameters with the values according to the method described in the paper below.
- **NormalInit**: Initialize module parameters with the values drawn from the normal distribution.
- **PretrainedInit**: Initialize module by loading a pretrained model.
- **TruncNormalInit**: Initialize module parameters with the values drawn from the normal distribution with values outside.
- **UniformInit**: Initialize module parameters with values drawn from the uniform distribution.
- **XavierInit**: Initialize module parameters with values according to the method described in the paper below.
- **bias_init_with_prob**: Initialize conv/fc bias value according to a given probability value.
- **caffe2_xavier_init**
- **constant_init**: Initialize a module.
- **kaiming_init**
- **normal_init**
- **trunc_normal_init**
- **uniform_init**
- **update_init_info**: Update the _params_init_info in the module if the value of parameters are changed.
- **xavier_init**

### Utils

- **detect_anomalous_params**
- **merge_dict**: Merge all dictionaries into one dictionary.
- **stack_batch**: Stack multiple tensors to form a batch and pad the tensor to the max shape using the right bottom padding mode in these images.
- **revert_sync_batchnorm**: Helper function to convert all SyncBatchNorm (SyncBN) and mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers in the model to BatchNormXd layers.
- **convert_sync_batchnorm**: Helper function to convert all BatchNorm layers in the model to SyncBatchNorm (SyncBN) or mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers.