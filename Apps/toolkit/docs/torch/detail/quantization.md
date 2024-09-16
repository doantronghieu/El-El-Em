# Quantization

All imports

```python
import torch
import torch.nn as nn
import torchvision
from torch.ao.quantization import QConfigMapping, default_dynamic_qconfig, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from torch.quantization import per_channel_dynamic_qconfig, quantize_dynamic_jit
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms

from torch._export import capture_pre_autograd_graph, dynamic_dim
from torch.ao.quantization import default_eval_fn, quantize
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
```

Functions

```python
torch.ao.quantization.DeQuantStub
torch.ao.quantization.QuantStub
torch.ao.quantization.backend_config
torch.ao.quantization.default_dynamic_qconfig
torch.ao.quantization.get_default_qat_qconfig
torch.ao.quantization.get_default_qat_qconfig_mapping
torch.ao.quantization.get_default_qconfig
torch.ao.quantization.get_default_qconfig_mapping
torch.ao.quantization.move_exported_model_to_eval
torch.ao.quantization.move_exported_model_to_train
torch.backends.quantized.engine
torch.dequantize
torch.export.export
torch.export.load
torch.export.save
torch.fx
torch.jit.load
torch.jit.RecursiveScriptModule
torch.jit.save
torch.jit.script
torch.jit.trace
torch.quantization.convert
torch.quantization.fuse_modules
torch.quantization.get_default_config
torch.quantization.prepare
torch.quantization.prepare_qat
torch.quantization.quantize_dynamic
torch.quantize_per_channel
torch.quantize_per_tensor
torch.quantize_per_tensor_dynamic
```

## [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)

*Authors: Raghuraman Krishnamoorthi, James Reed, Min Ni, Chris Gottbrath, and Seth Weidman*

It's crucial to make efficient use of both server-side and on-device compute resources when developing machine learning applications. To support more efficient deployment on servers and edge devices, PyTorch added support for model quantization using the familiar eager mode Python API.

Quantization leverages 8-bit integer (int8) instructions to reduce the model size and run the inference faster (reduced latency). This can be the difference between a model achieving quality of service goals or even fitting into the resources available on a mobile device. Even when resources aren't quite so constrained, it may enable you to deploy a larger and more accurate model. Quantization is available in PyTorch starting in version 1.3 and with the release of PyTorch 1.4, we published quantized models for ResNet, ResNext, MobileNetV2, GoogleNet, InceptionV3, and ShuffleNetV2 in the PyTorch torchvision 0.5 library.

This blog post provides an overview of the quantization support on PyTorch and its incorporation with the TorchVision domain library.

### What is Quantization?

Quantization refers to techniques for doing both computations and memory accesses with lower precision data, usually int8 compared to floating point implementations. This enables performance gains in several important areas:

- 4x reduction in model size
- 2-4x reduction in memory bandwidth
- 2-4x faster inference due to savings in memory bandwidth and faster compute with int8 arithmetic (the exact speed up varies depending on the hardware, the runtime, and the model).

Quantization does not however come without additional cost. Fundamentally, quantization means introducing approximations and the resulting networks have slightly less accuracy. These techniques attempt to minimize the gap between the full floating point accuracy and the quantized accuracy.

### Designing Quantization for PyTorch

We designed quantization to fit into the PyTorch framework. This means that:

- PyTorch has data types corresponding to quantized tensors, which share many of the features of tensors.
- One can write kernels with quantized tensors, much like kernels for floating point tensors to customize their implementation. PyTorch supports quantized modules for common operations as part of the torch.nn.quantized and torch.nn.quantized.dynamic namespaces.
- Quantization is compatible with the rest of PyTorch: quantized models are traceable and scriptable. The quantization method is virtually identical for both server and mobile backends. One can easily mix quantized and floating point operations in a model.
- Mapping of floating point tensors to quantized tensors is customizable with user-defined observer/fake-quantization blocks. PyTorch provides default implementations that should work for most use cases.

### Three Techniques for Quantizing Neural Networks in PyTorch

We developed three techniques for quantizing neural networks in PyTorch as part of quantization tooling in the torch.quantization namespace.

#### Dynamic Quantization

The easiest method of quantization PyTorch supports is called dynamic quantization. This involves not just converting the weights to int8 - as happens in all quantization variants - but also converting the activations to int8 on the fly, just before doing the computation (hence “dynamic”). The computations will thus be performed using efficient int8 matrix multiplication and convolution implementations, resulting in faster compute. However, the activations are read and written to memory in floating point format.

PyTorch API: We have a simple API for dynamic quantization in PyTorch. torch.quantization.quantize_dynamic takes in a model, as well as a couple other arguments, and produces a quantized model! Our end-to-end tutorial illustrates this for a BERT model; while the tutorial is long and contains sections on loading pre-trained models and other concepts unrelated to quantization, the part that quantizes the BERT model is simply:

```python
import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

See the documentation for the function [here](https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic) and an end-to-end example in our tutorials [here](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html) and [here](https://pytorch.org/tutorials/intermediate/dynamic_quantization_tutorial.html).

#### Post-Training Static Quantization

One can further improve the performance (latency) by converting networks to use both integer arithmetic and int8 memory accesses. Static quantization performs the additional step of first feeding batches of data through the network and computing the resulting distributions of the different activations (specifically, this is done by inserting “observer” modules at different points that record these distributions). This information is used to determine how specifically the different activations should be quantized at inference time (a simple technique would be to simply divide the entire range of activations into 256 levels, but we support more sophisticated methods as well). Importantly, this additional step allows us to pass quantized values between operations instead of converting these values to floats - and then back to ints - between every operation, resulting in a significant speed-up.

With this release, we’re supporting several features that allow users to optimize their static quantization:

- Observers: you can customize observer modules which specify how statistics are collected prior to quantization to try out more advanced methods to quantize your data.
- Operator fusion: you can fuse multiple operations into a single operation, saving on memory access while also improving the operation’s numerical accuracy.
- Per-channel quantization: we can independently quantize weights for each output channel in a convolution/linear layer, which can lead to higher accuracy with almost the same speed.

PyTorch API:

- To fuse modules, we have torch.quantization.fuse_modules
- Observers are inserted using torch.quantization.prepare
- Finally, quantization itself is done using torch.quantization.convert

We have a tutorial with an end-to-end example of quantization (this same tutorial also covers our third quantization method, quantization-aware training), but because of our simple API, the three lines that perform post-training static quantization on the pre-trained model myModel are:

```python
# set quantization config for server (x86)
deploymentmyModel.qconfig = torch.quantization.get_default_config('fbgemm')

# insert observers
torch.quantization.prepare(myModel, inplace=True)
# Calibrate the model and collect statistics

# convert to quantized version
torch.quantization.convert(myModel, inplace=True)
```

#### Quantization Aware Training

Quantization-aware training(QAT) is the third method, and the one that typically results in highest accuracy of these three. With QAT, all weights and activations are “fake quantized” during both the forward and backward passes of training: that is, float values are rounded to mimic int8 values, but all computations are still done with floating point numbers. Thus, all the weight adjustments during training are made while “aware” of the fact that the model will ultimately be quantized; after quantizing, therefore, this method usually yields higher accuracy than the other two methods.

PyTorch API:

- torch.quantization.prepare_qat inserts fake quantization modules to model quantization.
- Mimicking the static quantization API, torch.quantization.convert actually quantizes the model once training is complete.

For example, in the end-to-end example, we load in a pre-trained model as qat_model, then we simply perform quantization-aware training using:

```python
# specify quantization config for QAT
qat_model.qconfig=torch.quantization.get_default_qat_qconfig('fbgemm')

# prepare QAT
torch.quantization.prepare_qat(qat_model, inplace=True)

# convert to quantized version, removing dropout, to check for accuracy on each
epochquantized_model=torch.quantization.convert(qat_model.eval(), inplace=False)
```

### Device and Operator Support

Quantization support is restricted to a subset of available operators, depending on the method being used. For a list of supported operators, please see the documentation at [https://pytorch.org/docs/stable/quantization.html](https://pytorch.org/docs/stable/quantization.html).

The set of available operators and the quantization numerics also depend on the backend being used to run quantized models. Currently, quantized operators are supported only for CPU inference in the following backends: x86 and ARM. Both the quantization configuration (how tensors should be quantized and the quantized kernels (arithmetic with quantized tensors) are backend dependent. One can specify the backend by doing:

```python
import torch
backend='fbgemm'
# 'fbgemm' for server, 'qnnpack' for mobile
my_model.qconfig = torch.quantization.get_default_qconfig(backend)
# prepare and convert model
# Set the backend on which the quantized kernels need to be run
torch.backends.quantized.engine=backend
```

However, quantization aware training occurs in full floating point and can run on either GPU or CPU. Quantization aware training is typically only used in CNN models when post training static or dynamic quantization doesn’t yield sufficient accuracy. This can occur with models that are highly optimized to achieve small size (such as Mobilenet).

### Integration in torchvision

We’ve also enabled quantization for some of the most popular models in torchvision: Googlenet, Inception, Resnet, ResNeXt, Mobilenet, and Shufflenet. We have upstreamed these changes to torchvision in three forms:

- Pre-trained quantized weights so that you can use them right away.
- Quantization ready model definitions so that you can do post-training quantization or quantization aware training.
- A script for doing quantization aware training — which is available for any of these model though, as you will learn below, we only found it necessary for achieving accuracy with Mobilenet.

We also have a tutorial showing how you can do transfer learning with quantization using one of the torchvision models.

### Choosing an approach

The choice of which scheme to use depends on multiple factors:

- Model/Target requirements: Some models might be sensitive to quantization, requiring quantization aware training.
- Operator/Backend support: Some backends require fully quantized operators.

Currently, operator coverage is limited and may restrict the choices listed in the table below:

| Model Type | Preferred scheme | Why |
| --- | --- | --- |
| LSTM/RNN | Dynamic Quantization | Throughput dominated by compute/memory bandwidth for weights |
| BERT/Transformer | Dynamic Quantization | Throughput dominated by compute/memory bandwidth for weights |
| CNN | Static Quantization | Throughput limited by memory bandwidth for activations |
| CNN | Quantization Aware Training | In the case where accuracy can't be achieved with static quantization |

### Performance Results

Quantization provides a 4x reduction in the model size and a speedup of 2x to 3x compared to floating point implementations depending on the hardware platform and the model being benchmarked. Some sample results are:

| Model | Float Latency (ms) | Quantized Latency (ms) | Inference Performance Gain | Device | Notes |
| --- | --- | --- | --- | --- | --- |
| BERT | 581 | 313 | 1.8x | Xeon-D2191 (1.6GHz) | Batch size = 1, Maximum sequence length= 128, Single thread, x86-64, Dynamic quantization |
| Resnet-50 | 214 | 103 | 2x | Xeon-D2191 (1.6GHz) | Single thread, x86-64, Static quantization |
| Mobilenet-v2 | 97 | 17 | 5.7x | Samsung S9 | Static quantization, Floating point numbers are based on Caffe2 run-time and are not optimized |

### Accuracy results

We also compared the accuracy of static quantized models with the floating point models on Imagenet. For dynamic quantization, we compared the F1 score of BERT on the GLUE benchmark for MRPC.

#### Computer Vision Model accuracy

| Model | Top-1 Accuracy (Float) | Top-1 Accuracy (Quantized) | Quantization scheme |
| --- | --- | --- | --- |
| Googlenet | 69.8 | 69.7 | Static post training quantization |
| Inception-v3 | 77.5 | 77.1 | Static post training quantization |
| ResNet-18 | 69.8 | 69.4 | Static post training quantization |
| Resnet-50 | 76.1 | 75.9 | Static post training quantization |
| ResNext-101 32x8d | 79.3 | 79 | Static post training quantization |
| Mobilenet-v2 | 71.9 | 71.6 | Quantization Aware Training |
| Shufflenet-v2 | 69.4 | 68.4 | Static post training quantization |

#### Speech and NLP Model accuracy

| Model | F1 (GLUEMRPC) Float | F1 (GLUEMRPC) Quantized | Quantization scheme |
| --- | --- | --- | --- |
| BERT | 0.902 | 0.895 | Dynamic quantization |

### Conclusion

To get started on quantizing your models in PyTorch, start with the tutorials on the PyTorch website. If you are working with sequence data start with dynamic quantization for LSTM, or BERT. If you are working with image data then we recommend starting with the transfer learning with quantization tutorial. Then you can explore static post training quantization. If you find that the accuracy drop with post training quantization is too high, then try quantization aware training.

If you run into issues you can get community help by posting in at discuss.pytorch.org, use the quantization category for quantization related issues.

This post is authored by Raghuraman Krishnamoorthi, James Reed, Min Ni, Chris Gottbrath and Seth Weidman. Special thanks to Jianyu Huang, Lingyi Liu and Haixin Liu for producing quantization metrics included in this post.

## [Quantization Recipe](https://pytorch.org/tutorials/recipes/quantization.html)

## [Accelerate PyTorch Models Using Quantization Techniques with Intel Extension for PyTorch](https://pytorch.org/blog/accelerate-pytorch-models/)

## [API-Quantization](https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic)

### Warning

Quantization is in beta and subject to change.

### Introduction to Quantization

Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than floating point precision. This allows for a more compact model representation and the use of high performance vectorized operations on many hardware platforms. PyTorch supports INT8 quantization compared to typical FP32 models allowing for a 4x reduction in the model size and a 4x reduction in memory bandwidth requirements. Hardware support for INT8 computations is typically 2 to 4 times faster compared to FP32 compute. Quantization is primarily a technique to speed up inference and only the forward pass is supported for quantized operators.

PyTorch supports multiple approaches to quantizing a deep learning model. In most cases the model is trained in FP32 and then the model is converted to INT8. In addition, PyTorch also supports quantization aware training, which models quantization errors in both the forward and backward passes using fake-quantization modules. Note that the entire computation is carried out in floating point. At the end of quantization aware training, PyTorch provides conversion functions to convert the trained model into lower precision.

At lower level, PyTorch provides a way to represent quantized tensors and perform operations with them. They can be used to directly construct models that perform all or part of the computation in lower precision. Higher-level APIs are provided that incorporate typical workflows of converting FP32 model to lower precision with minimal accuracy loss.

### Quantization API Summary

PyTorch provides three different modes of quantization: Eager Mode Quantization, FX Graph Mode Quantization (maintenance) and PyTorch 2 Export Quantization.

- **Eager Mode Quantization** is a beta feature. User needs to do fusion and specify where quantization and dequantization happens manually, also it only supports modules and not functionals.
- **FX Graph Mode Quantization** is an automated quantization workflow in PyTorch, and currently it’s a prototype feature, it is in maintenance mode since we have PyTorch 2 Export Quantization. It improves upon Eager Mode Quantization by adding support for functionals and automating the quantization process, although people might need to refactor the model to make the model compatible with FX Graph Mode Quantization (symbolically traceable with torch.fx).
- **PyTorch 2 Export Quantization** is the new full graph mode quantization workflow, released as prototype feature in PyTorch 2.1. With PyTorch 2, we are moving to a better solution for full program capture (torch.export) since it can capture a higher percentage (88.8% on 14K models) of models compared to torch.fx.symbolic_trace (72.7% on 14K models), the program capture solution used by FX Graph Mode Quantization.

### Quantization Types

There are three types of quantization supported:

1. **Dynamic Quantization**: Weights are quantized ahead of time but the activations are dynamically quantized during inference. This is used for situations where the model execution time is dominated by loading weights from memory rather than computing the matrix multiplications. This is true for LSTM and Transformer type models with small batch size.
2. **Static Quantization**: Both weights and activations are quantized. It fuses activations into preceding layers where possible. It requires calibration with a representative dataset to determine optimal quantization parameters for activations. Post Training Static Quantization is typically used when both memory bandwidth and compute savings are important with CNNs being a typical use case.
3. **Static Quantization Aware Training**: Weights and activations are quantized, and quantization numerics are modeled during training. This is commonly used with CNNs and yields a higher accuracy compared to static quantization.

### Operator Coverage

Operator coverage varies between dynamic and static quantization and is captured in the table below.

| Static Quantization | Dynamic Quantization |
| --- | --- |
| nn.Linear | nn.Conv1d/2d/3d |
| Y | Y |
| Y | N |
| nn.LSTM | nn.GRU |
| N | N |
| Y | Y |
| nn.RNNCell | nn.GRUCell |
| nn.LSTMCell | N |
| N | N |
| Y | Y |
| nn.EmbeddingBag | Y (activations are in fp32) |
| Y | nn.Embedding |
| Y | Y |
| nn.MultiheadAttention | Not Supported |
| Not supported | Activations |
| Broadly supported | Un-changed, computations stay in fp32 |

### Eager Mode Quantization

For a general introduction to the quantization flow, including different types of quantization, please take a look at General Quantization Flow.

#### Post Training Dynamic Quantization

This is the simplest to apply form of quantization where the weights are quantized ahead of time but the activations are dynamically quantized during inference. This is used for situations where the model execution time is dominated by loading weights from memory rather than computing the matrix multiplications. This is true for LSTM and Transformer type models with small batch size.

#### Post Training Static Quantization

Post Training Static Quantization (PTQ static) quantizes the weights and activations of the model. It fuses activations into preceding layers where possible. It requires calibration with a representative dataset to determine optimal quantization parameters for activations. Post Training Static Quantization is typically used when both memory bandwidth and compute savings are important with CNNs being a typical use case.

#### Quantization Aware Training for Static Quantization

Quantization Aware Training (QAT) models the effects of quantization during training allowing for higher accuracy compared to other quantization methods. We can do QAT for static, dynamic or weight only quantization. During training, all calculations are done in floating point, with fake_quant modules modeling the effects of quantization by clamping and rounding to simulate the effects of INT8. After model conversion, weights and activations are quantized, and activations are fused into the preceding layer where possible. It is commonly used with CNNs and yields a higher accuracy compared to static quantization.

### Model Preparation for Eager Mode Static Quantization

It is necessary to currently make some modifications to the model definition prior to Eager mode quantization. This is because currently quantization works on a module by module basis. Specifically, for all quantization techniques, the user needs to:

- Convert any operations that require output requantization (and thus have additional parameters) from functionals to module form (for example, using torch.nn.ReLU instead of torch.nn.functional.relu).
- Specify which parts of the model need to be quantized either by assigning .qconfig attributes on submodules or by specifying qconfig_mapping.
- For static quantization techniques which quantize activations, the user needs to do the following in addition:
  - Specify where activations are quantized and de-quantized. This is done using QuantStub and DeQuantStub modules.
  - Use FloatFunctional to wrap tensor operations that require special handling for quantization into modules. Examples are operations like add and cat which require special handling to determine output quantization parameters.
  - Fuse modules: combine operations/modules into a single module to obtain higher accuracy and performance. This is done using the fuse_modules() API, which takes in lists of modules to be fused. We currently support the following fusions: [Conv, Relu], [Conv, BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu]

### (Prototype - maintenance mode) FX Graph Mode Quantization

There are multiple quantization types in post training quantization (weight only, dynamic and static) and the configuration is done through qconfig_mapping (an argument of the prepare_fx function).

### (Prototype) PyTorch 2 Export Quantization

API Example:

```python
import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)

    def forward(self, x):
        return self.linear(x)

# initialize a floating point model
float_model = M().eval()

# define calibration function
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

# Step 1. program capture
# NOTE: this API will be updated to torch.export API in the future, but the captured
# result should mostly stay the same
m = capture_pre_autograd_graph(m, *example_inputs)
# we get a model with aten ops

# Step 2. quantization
# backend developer will write their own Quantizer and expose methods to allow
# users to express how they
# want the model to be quantized
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
# or prepare_qat_pt2e for Quantization Aware Training
m = prepare_pt2e(m, quantizer)

# run calibration
# calibrate(m, sample_inference_data)
m = convert_pt2e(m)

# Step 3. lowering
# lower to target backend
```

### Quantization Stack

Quantization is the process to convert a floating point model to a quantized model. So at high level the quantization stack can be split into two parts: 1). The building blocks or abstractions for a quantized model 2). The building blocks or abstractions for the quantization flow that converts a floating point model to a quantized model

### Quantized Model

#### Quantized Tensor

In order to do quantization in PyTorch, we need to be able to represent quantized data in Tensors. A Quantized Tensor allows for storing quantized data (represented as int8/uint8/int32) along with quantization parameters like scale and zero_point. Quantized Tensors allow for many useful operations making quantized arithmetic easy, in addition to allowing for serialization of data in a quantized format.

PyTorch supports both per tensor and per channel symmetric and asymmetric quantization. Per tensor means that all the values within the tensor are quantized the same way with the same quantization parameters. Per channel means that for each dimension, typically the channel dimension of a tensor, the values in the tensor are quantized with different quantization parameters. This allows for less error in converting tensors to quantized values since outlier values would only impact the channel it was in, instead of the entire Tensor.

The mapping is performed by converting the floating point tensors using the equation:

![Quantization Equation](https://pytorch.org/docs/stable/_images/math-quantizer-equation.png)

Note that, we ensure that zero in floating point is represented with no error after quantization, thereby ensuring that operations like padding do not cause additional quantization error.

#### Quantize and Dequantize

The input and output of a model are floating point Tensors, but activations in the quantized model are quantized, so we need operators to convert between floating point and quantized Tensors.

- Quantize (float -> quantized)
  - torch.quantize_per_tensor(x, scale, zero_point, dtype)
  - torch.quantize_per_channel(x, scales, zero_points, axis, dtype)
  - torch.quantize_per_tensor_dynamic(x, dtype, reduce_range)
  - to(torch.float16)
- Dequantize (quantized -> float)
  - quantized_tensor.dequantize() - calling dequantize on a torch.float16 Tensor will convert the Tensor back to torch.float
  - torch.dequantize(x)

#### Quantized Operators/Modules

Quantized Operator are the operators that takes quantized Tensor as inputs, and outputs a quantized Tensor. Quantized Modules are PyTorch Modules that performs quantized operations. They are typically defined for weighted operations like linear and conv.

#### Quantized Engine

When a quantized model is executed, the qengine (torch.backends.quantized.engine) specifies which backend is to be used for execution. It is important to ensure that the qengine is compatible with the quantized model in terms of value range of quantized activation and weights.

### Quantization Flow

#### Observer and FakeQuantize

Observer are PyTorch Modules used to:

- collect tensor statistics like min value and max value of the Tensor passing through the observer
- and calculate quantization parameters based on the collected tensor statistics

FakeQuantize are PyTorch Modules used to:

- simulate quantization (performing quantize/dequantize) for a Tensor in the network
- it can calculate quantization parameters based on the collected statistics from observer, or it can learn the quantization parameters as well

#### QConfig

QConfig is a namedtuple of Observer or FakeQuantize Module class that can are configurable with qscheme, dtype etc. it is used to configure how an operator should be observed.

#### General Quantization Flow

In general, the flow is the following:

1. prepare
   - insert Observer/FakeQuantize modules based on user specified qconfig
2. calibrate/train (depending on post training quantization or quantization aware training)
   - allow Observers to collect statistics or FakeQuantize modules to learn the quantization parameters
3. convert
   - convert a calibrated/trained model to a quantized model

There are different modes of quantization, they can be classified in two ways:

- In terms of where we apply the quantization flow, we have:
  - Post Training Quantization (apply quantization after training, quantization parameters are calculated based on sample calibration data)
  - Quantization Aware Training (simulate quantization during training so that the quantization parameters can be learned together with the model using training data)
- And in terms of how we quantize the operators, we can have:
  - Weight Only Quantization (only weight is statically quantized)
  - Dynamic Quantization (weight is statically quantized, activation is dynamically quantized)
  - Static Quantization (both weight and activations are statically quantized)

We can mix different ways of quantizing operators in the same quantization flow. For example, we can have post training quantization that has both statically and dynamically quantized operators.

### Quantization Support Matrix

#### Quantization Mode Support

| Quantization Mode | Dataset Requirement | Works Best For | Accuracy | Notes |
| --- | --- | --- | --- | --- |
| Post Training Quantization | None | LSTM, MLP, Embedding, Transformer | good | Easy to use, close to static quantization when performance is compute or memory bound due to weights |
| Dynamic Quantization | - | - | - | - |
| Static Quantization | calibration dataset | CNN | good | Provides best perf, may have big impact on accuracy, good for hardwares that only support int8 computation |
| Quantization Aware Training | fine-tuning dataset | CNN, MLP, Embedding | best | Typically used when static quantization leads to bad accuracy, and used to close the accuracy gap |
| Dynamic Quantization | activation dynamically quantized (fp16, int8) or not quantized, weight statically quantized (fp16, int8, in4) | - | - | - |
| Static Quantization | activation and weights statically quantized (int8) | - | - | - |

#### Quantization Flow Support

PyTorch provides two modes of quantization: Eager Mode Quantization and FX Graph Mode Quantization.

- Eager Mode Quantization is a beta feature. User needs to do fusion and specify where quantization and dequantization happens manually, also it only supports modules and not functionals.
- FX Graph Mode Quantization is an automated quantization framework in PyTorch, and currently it’s a prototype feature. It improves upon Eager Mode Quantization by adding support for functionals and automating the quantization process, although people might need to refactor the model to make the model compatible with FX Graph Mode Quantization (symbolically traceable with torch.fx).

#### Backend/Hardware Support

Today, PyTorch supports the following backends for running quantized operators efficiently:

- x86 CPUs with AVX2 support or higher (without AVX2 some operations have inefficient implementations), via x86 optimized by fbgemm and onednn (see the details at RFC)
- ARM CPUs (typically found in mobile/embedded devices), via qnnpack
- (early prototype) support for NVidia GPU via TensorRT through fx2trt (to be open sourced)

#### Operator Support

Operator coverage varies between dynamic and static quantization and is captured in the table below. Note that for FX Graph Mode Quantization, the corresponding functionals are also supported.

### Quantization API Reference

The Quantization API Reference contains documentation of quantization APIs, such as quantization passes, quantized tensor operations, and supported quantized modules and functions.

### Quantization Backend Configuration

The Quantization Backend Configuration contains documentation on how to configure the quantization workflows for various backends.

### Quantization Accuracy Debugging

The Quantization Accuracy Debugging contains documentation on how to debug quantization accuracy.

### Quantization Customizations

While default implementations of observers to select the scale factor and bias based on observed tensor data are provided, developers can provide their own quantization functions. Quantization can be applied selectively to different parts of the model or configured differently for different parts of the model.

We also provide support for per channel quantization for conv1d(), conv2d(), conv3d() and linear().

Quantization workflows work by adding (e.g. adding observers as .observer submodule) or replacing (e.g. converting nn.Conv2d to nn.quantized.Conv2d) submodules in the model’s module hierarchy. It means that the model stays a regular nn.Module-based instance throughout the process and thus can work with the rest of PyTorch APIs.

### Quantization Custom Module API

Both Eager mode and FX graph mode quantization APIs provide a hook for the user to specify module quantized in a custom way, with user defined logic for observation and quantization. The user needs to specify:

- The Python type of the source fp32 module (existing in the model)
- The Python type of the observed module (provided by user). This module needs to define a from_float function which defines how the observed module is created from the original fp32 module.
- The Python type of the quantized module (provided by user). This module needs to define a from_observed function which defines how the quantized module is created from the observed module.
- A configuration describing (1), (2), (3) above, passed to the quantization APIs.

The framework will then do the following:

- during the prepare module swaps, it will convert every module of type specified in (1) to the type specified in (2), using the from_float function of the class in (2).
- during the convert module swaps, it will convert every module of type specified in (2) to the type specified in (3), using the from_observed function of the class in (3).

### Best Practices

1. If you are using the x86 backend, we need to use 7 bits instead of 8 bits. Make sure you reduce the range for the quant\_min, quant\_max, e.g. if dtype is torch.quint8, make sure to set a custom quant_min to be 0 and quant_max to be 127 (255 / 2) if dtype is torch.qint8, make sure to set a custom quant_min to be -64 (-128 / 2) and quant_max to be 63 (127 / 2), we already set this correctly if you call the torch.ao.quantization.get_default_qconfig(backend) or torch.ao.quantization.get_default_qat_qconfig(backend) function to get the default qconfig for x86 or qnnpack backend
2. If onednn backend is selected, 8 bits for activation will be used in the default qconfig mapping torch.ao.quantization.get_default_qconfig_mapping('onednn') and default qconfig torch.ao.quantization.get_default_qconfig('onednn'). It is recommended to be used on CPUs with Vector Neural Network Instruction (VNNI) support. Otherwise, setting reduce_range to True of the activation’s observer to get better accuracy on CPUs without VNNI support.

### Frequently Asked Questions

- How can I do quantized inference on GPU?: We don’t have official GPU support yet, but this is an area of active development, you can find more information here
- Where can I get ONNX support for my quantized model?
  - If you get errors exporting the model (using APIs under torch.onnx), you may open an issue in the PyTorch repository. Prefix the issue title with [ONNX] and tag the issue as module: onnx.
  - If you encounter issues with ONNX Runtime, open an issue at GitHub - microsoft/onnxruntime.
- How can I use quantization with LSTM’s?: LSTM is supported through our custom module api in both eager mode and fx graph mode quantization. Examples can be found at Eager Mode: pytorch/test\_quantized\_op.py TestQuantizedOps.test\_custom\_module\_lstm FX Graph Mode: pytorch/test\_quantize\_fx.py TestQuantizeFx.test\_static\_lstm

### Common Errors

- Passing a non-quantized Tensor into a quantized kernel
  - If you see an error similar to: RuntimeError: Could not run 'quantized::some\_operator' with arguments from the 'CPU' backend... This means that you are trying to pass a non-quantized Tensor to a quantized kernel. A common workaround is to use torch.ao.quantization.QuantStub to quantize the tensor. This needs to be done manually in Eager mode quantization. An e2e example:
- Passing a quantized Tensor into a non-quantized kernel
  - If you see an error similar to: RuntimeError: Could not run 'aten::thnn\_conv2d\_forward' with arguments from the 'QuantizedCPU' backend. This means that you are trying to pass a quantized Tensor to a non-quantized kernel. A common workaround is to use torch.ao.quantization.DeQuantStub to dequantize the tensor. This needs to be done manually in Eager mode quantization. An e2e example:

### Saving and Loading Quantized models

When calling torch.load on a quantized model, if you see an error like: AttributeError: 'LinearPackedParams' object has no attribute '\_modules' This is because directly saving and loading a quantized model using torch.save and torch.load is not supported. To save/load quantized models, the following ways can be used:

- Saving/Loading the quantized model state\_dict
  - An example:
- Saving/Loading scripted quantized models using torch.jit.save and torch.jit.load
  - An example:

### Symbolic Trace Error when using FX Graph Mode Quantization

Symbolic traceability is a requirement for (Prototype - maintenance mode) FX Graph Mode Quantization, so if you pass a PyTorch Model that is not symbolically traceable to torch.ao.quantization.prepare\_fx or torch.ao.quantization.prepare\_qat\_fx, we might see an error like the following: torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow Please take a look at Limitations of Symbolic Tracing and use - User Guide on Using FX Graph Mode Quantization to workaround the problem.

## [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)

## [PyTorch Numeric Suite Tutorial](https://pytorch.org/tutorials/prototype/numeric_suite_tutorial.html)

### Introduction

Quantization is good when it works, but it’s difficult to know what’s wrong when it doesn’t satisfy the accuracy we expect. Debugging the accuracy issue of quantization is not easy and time consuming. One important step of debugging is to measure the statistics of the float model and its corresponding quantized model to know where are they differ most. We built a suite of numeric tools called PyTorch Numeric Suite in PyTorch quantization to enable the measurement of the statistics between quantized module and float module to support quantization debugging efforts. Even for the quantized model with good accuracy, PyTorch Numeric Suite can still be used as the profiling tool to better understand the quantization error within the model and provide the guidance for further optimization.

PyTorch Numeric Suite currently supports models quantized through both static quantization and dynamic quantization with unified APIs.

In this tutorial we will first use ResNet18 as an example to show how to use PyTorch Numeric Suite to measure the statistics between static quantized model and float model in eager mode. Then we will use LSTM based sequence model as an example to show the usage of PyTorch Numeric Suite for dynamic quantized model.

### Numeric Suite for Static Quantization

#### Setup

We’ll start by doing the necessary imports:

```python
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models, datasets
import torchvision.transforms as transforms
import os
import torch.quantization
import torch.quantization._numeric_suite as ns
from torch.quantization import (
    default_eval_fn,
    default_qconfig,
    quantize,
)
```

Then we load the pretrained float ResNet18 model, and quantize it into qmodel. We cannot compare two arbitrary models, only a float model and the quantized model derived from it can be compared.

```python
float_model = torchvision.models.quantization.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1, quantize=False)
float_model.to('cpu')
float_model.eval()
float_model.fuse_model()
float_model.qconfig = torch.quantization.default_qconfig
img_data = [(torch.rand(2, 3, 10, 10, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long)) for _ in range(2)]
qmodel = quantize(float_model, default_eval_fn, [img_data], inplace=False)
```

#### 1. Compare the weights of float and quantized models

The first thing we usually want to compare are the weights of quantized model and float model. We can call `compare_weights()` from PyTorch Numeric Suite to get a dictionary `wt_compare_dict` with key corresponding to module names and each entry is a dictionary with two keys ‘float’ and ‘quantized’, containing the float and quantized weights. `compare_weights()` takes in floating point and quantized state dict and returns a dict, with keys corresponding to the floating point weights and values being a dictionary of floating point and quantized weights.

```python
wt_compare_dict = ns.compare_weights(float_model.state_dict(), qmodel.state_dict())

print('keys of wt_compare_dict:')
print(wt_compare_dict.keys())

print("\nkeys of wt_compare_dict entry for conv1's weight:")
print(wt_compare_dict['conv1.weight'].keys())
print(wt_compare_dict['conv1.weight']['float'].shape)
print(wt_compare_dict['conv1.weight']['quantized'].shape)
```

Once get `wt_compare_dict`, users can process this dictionary in whatever way they want. Here as an example we compute the quantization error of the weights of float and quantized models as following. Compute the Signal-to-Quantization-Noise Ratio (SQNR) of the quantized tensor y. The SQNR reflects the relationship between the maximum nominal signal strength and the quantization error introduced in the quantization. Higher SQNR corresponds to lower quantization error.

```python
def compute_error(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x-y)
    return 20*torch.log10(Ps/Pn)

for key in wt_compare_dict:
    print(key, compute_error(wt_compare_dict[key]['float'], wt_compare_dict[key]['quantized'].dequantize()))
```

As another example `wt_compare_dict` can also be used to plot the histogram of the weights of floating point and quantized models.

```python
import matplotlib.pyplot as plt

f = wt_compare_dict['conv1.weight']['float'].flatten()
plt.hist(f, bins = 100)
plt.title("Floating point model weights of conv1")
plt.show()

q = wt_compare_dict['conv1.weight']['quantized'].flatten().dequantize()
plt.hist(q, bins = 100)
plt.title("Quantized model weights of conv1")
plt.show()
```

#### 2. Compare float point and quantized models at corresponding locations

The second tool allows for comparison of weights and activations between float and quantized models at corresponding locations for the same input as shown in the figure below. Red arrows indicate the locations of the comparison.

![compare_output.png](../_images/compare_output.png)

We call `compare_model_outputs()` from PyTorch Numeric Suite to get the activations in float model and quantized model at corresponding locations for the given input data. This API returns a dict with module names being keys. Each entry is itself a dict with two keys ‘float’ and ‘quantized’ containing the activations.

```python
data = img_data[0][0]

# Take in floating point and quantized model as well as input data, and returns a dict, with keys
# corresponding to the quantized module names and each entry being a dictionary with two keys 'float' and
# 'quantized', containing the activations of floating point and quantized model at matching locations.
act_compare_dict = ns.compare_model_outputs(float_model, qmodel, data)

print('keys of act_compare_dict:')
print(act_compare_dict.keys())

print("\nkeys of act_compare_dict entry for conv1's output:")
print(act_compare_dict['conv1.stats'].keys())
print(act_compare_dict['conv1.stats']['float'][0].shape)
print(act_compare_dict['conv1.stats']['quantized'][0].shape)
```

This dict can be used to compare and compute the quantization error of the activations of float and quantized models as following.

```python
for key in act_compare_dict:
    print(key, compute_error(act_compare_dict[key]['float'][0], act_compare_dict[key]['quantized'][0].dequantize()))
```

If we want to do the comparison for more than one input data, we can do the following. Prepare the model by attaching the logger to both floating point module and quantized module if they are in the white_list. Default logger is `OutputLogger`, and default white_list is `DEFAULT_NUMERIC_SUITE_COMPARE_MODEL_OUTPUT_WHITE_LIST`.

```python
ns.prepare_model_outputs(float_model, qmodel)

for data in img_data:
    float_model(data[0])
    qmodel(data[0])

# Find the matching activation between floating point and quantized modules, and return a dict with key
# corresponding to quantized module names and each entry being a dictionary with two keys 'float'
# and 'quantized', containing the matching floating point and quantized activations logged by the logger
act_compare_dict = ns.get_matching_activations(float_model, qmodel)
```

The default logger used in above APIs is `OutputLogger`, which is used to log the outputs of the modules. We can inherit from base `Logger` class and create our own logger to perform different functionalities. For example we can make a new `MyOutputLogger` class as below.

```python
class MyOutputLogger(ns.Logger):
    r"""Customized logger class
    """

    def __init__(self):
        super(MyOutputLogger, self).__init__()

    def forward(self, x):
        # Custom functionalities
        # ...
        return x
```

And then we can pass this logger into above APIs such as:

```python
data = img_data[0][0]
act_compare_dict = ns.compare_model_outputs(float_model, qmodel, data, logger_cls=MyOutputLogger)
```

or:

```python
ns.prepare_model_outputs(float_model, qmodel, MyOutputLogger)
for data in img_data:
    float_model(data[0])
    qmodel(data[0])
act_compare_dict = ns.get_matching_activations(float_model, qmodel)
```

#### 3. Compare a module in a quantized model with its float point equivalent, with the same input data

The third tool allows for comparing a quantized module in a model with its float point counterpart, feeding both of them the same input and comparing their outputs as shown below.

In practice we call `prepare_model_with_stubs()` to swap the quantized module that we want to compare with the Shadow module, which is illustrated as below:

The Shadow module takes quantized module, float module and logger as input, and creates a forward path inside to make the float module to shadow quantized module sharing the same input tensor.

The logger can be customizable, default logger is `ShadowLogger` and it will save the outputs of the quantized module and float module that can be used to compute the module level quantization error.

Notice before each call of `compare_model_outputs()` and `compare_model_stub()` we need to have clean float and quantized model. This is because `compare_model_outputs()` and `compare_model_stub()` modify float and quantized model inplace, and it will cause unexpected results if call one right after another.

```python
float_model = torchvision.models.quantization.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1, quantize=False)
float_model.to('cpu')
float_model.eval()
float_model.fuse_model()
float_model.qconfig = torch.quantization.default_qconfig
img_data = [(torch.rand(2, 3, 10, 10, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long)) for _ in range(2)]
qmodel = quantize(float_model, default_eval_fn, [img_data], inplace=False)
```

In the following example we call `compare_model_stub()` from PyTorch Numeric Suite to compare `QuantizableBasicBlock` module with its float point equivalent. This API returns a dict with key corresponding to module names and each entry being a dictionary with two keys ‘float’ and ‘quantized’, containing the output tensors of quantized and its matching float shadow module.

```python
data = img_data[0][0]
module_swap_list = [torchvision.models.quantization.resnet.QuantizableBasicBlock]

# Takes in floating point and quantized model as well as input data, and returns a dict with key
# corresponding to module names and each entry being a dictionary with two keys 'float' and
# 'quantized', containing the output tensors of quantized module and its matching floating point shadow module.
ob_dict = ns.compare_model_stub(float_model, qmodel, module_swap_list, data)

print('keys of ob_dict:')
print(ob_dict.keys())

print("\nkeys of ob_dict entry for layer1.0's output:")
print(ob_dict['layer1.0.stats'].keys())
print(ob_dict['layer1.0.stats']['float'][0].shape)
print(ob_dict['layer1.0.stats']['quantized'][0].shape)
```

This dict can be then used to compare and compute the module level quantization error.

```python
for key in ob_dict:
    print(key, compute_error(ob_dict[key]['float'][0], ob_dict[key]['quantized'][0].dequantize()))
```

If we want to do the comparison for more than one input data, we can do the following.

```python
ns.prepare_model_with_stubs(float_model, qmodel, module_swap_list, ns.ShadowLogger)
for data in img_data:
    qmodel(data[0])
ob_dict = ns.get_logger_dict(qmodel)
```

The default logger used in above APIs is `ShadowLogger`, which is used to log the outputs of the quantized module and its matching float shadow module. We can inherit from base `Logger` class and create our own logger to perform different functionalities. For example we can make a new `MyShadowLogger` class as below.

```python
class MyShadowLogger(ns.Logger):
    r"""Customized logger class
    """

    def __init__(self):
        super(MyShadowLogger, self).__init__()

    def forward(self, x, y):
        # Custom functionalities
        # ...
        return x
```

And then we can pass this logger into above APIs such as:

```python
data = img_data[0][0]
ob_dict = ns.compare_model_stub(float_model, qmodel, module_swap_list, data, logger_cls=MyShadowLogger)
```

or:

```python
ns.prepare_model_with_stubs(float_model, qmodel, module_swap_list, MyShadowLogger)
for data in img_data:
    qmodel(data[0])
ob_dict = ns.get_logger_dict(qmodel)
```

### Numeric Suite for Dynamic Quantization

Numeric Suite APIs are designed in such as way that they work for both dynamic quantized model and static quantized model. We will use a model with both LSTM and Linear modules to demonstrate the usage of Numeric Suite on dynamic quantized model. This model is the same one used in the tutorial of dynamic quantization on LSTM word language model [1].

#### Setup

First we define the model as below. Notice that within this model only `nn.LSTM` and `nn.Linear` modules will be quantized dynamically and `nn.Embedding` will remain as floating point module after quantization.

```python
class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
```

Then we create the `float_model` and quantize it into `qmodel`.

```python
ntokens = 10

float_model = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)

float_model.eval()

qmodel = torch.quantization.quantize_dynamic(
    float_model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
```

#### 1. Compare the weights of float and quantized models

We first call `compare_weights()` from PyTorch Numeric Suite to get a dictionary `wt_compare_dict` with key corresponding to module names and each entry is a dictionary with two keys ‘float’ and ‘quantized’, containing the float and quantized weights.

```python
wt_compare_dict = ns.compare_weights(float_model.state_dict(), qmodel.state_dict())
```

Once we get `wt_compare_dict`, it can be used to compare and compute the quantization error of the weights of float and quantized models as following.

```python
for key in wt_compare_dict:
    if wt_compare_dict[key]['quantized'].is_quantized:
        print(key, compute_error(wt_compare_dict[key]['float'], wt_compare_dict[key]['quantized'].dequantize()))
    else:
        print(key, compute_error(wt_compare_dict[key]['float'], wt_compare_dict[key]['quantized']))
```

#### 2. Compare float point and quantized models at corresponding locations

Then we call `compare_model_outputs()` from PyTorch Numeric Suite to get the activations in float model and quantized model at corresponding locations for the given input data. This API returns a dict with module names being keys. Each entry is itself a dict with two keys ‘float’ and ‘quantized’ containing the activations. Notice that this sequence model has two inputs, and we can pass both inputs into `compare_model_outputs()` and `compare_model_stub()`.

```python
input_ = torch.randint(ntokens, (1, 1), dtype=torch.long)
hidden = float_model.init_hidden(1)

act_compare_dict = ns.compare_model_outputs(float_model, qmodel, input_, hidden)
print(act_compare_dict.keys())
```

This dict can be used to compare and compute the quantization error of the activations of float and quantized models as following. The LSTM module in this model has two outputs, in this example we compute the error of the first output.

```python
for key in act_compare_dict:
    print(key, compute_error(act_compare_dict[key]['float'][0][0], act_compare_dict[key]['quantized'][0][0]))
```

#### 3. Compare a module in a quantized model with its float point equivalent, with the same input data

Next we call `compare_model_stub()` from PyTorch Numeric Suite to compare LSTM and Linear module with its float point equivalent. This API returns a dict with key corresponding to module names and each entry being a dictionary with two keys ‘float’ and ‘quantized’, containing the output tensors of quantized and its matching float shadow module.

We reset the model first.

```python
float_model = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)
float_model.eval()

qmodel = torch.quantization.quantize_dynamic(
    float_model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
```

Next we call `compare_model_stub()` from PyTorch Numeric Suite to compare LSTM and Linear module with its float point equivalent. This API returns a dict with key corresponding to module names and each entry being a dictionary with two keys ‘float’ and ‘quantized’, containing the output tensors of quantized and its matching float shadow module.

```python
module_swap_list = [nn.Linear, nn.LSTM]
ob_dict = ns.compare_model_stub(float_model, qmodel, module_swap_list, input_, hidden)
print(ob_dict.keys())
```

This dict can be then used to compare and compute the module level quantization error.

```python
for key in ob_dict:
    print(key, compute_error(ob_dict[key]['float'][0], ob_dict[key]['quantized'][0]))
```

SQNR of 40 dB is high and this is a situation where we have very good numerical alignment between the floating point and quantized model.

### Conclusion

In this tutorial, we demonstrated how to use PyTorch Numeric Suite to measure and compare the statistics between quantized model and float model in eager mode with unified APIs for both static quantization and dynamic quantization.

## [PyTorch BackendConfig](https://pytorch.org/tutorials/prototype/backend_config_tutorial.html?highlight=backend)


The BackendConfig API enables developers to integrate their backends with PyTorch quantization. It is currently only supported in FX graph mode quantization, but support may be extended to other modes of quantization in the future. In this tutorial, we will demonstrate how to use this API to customize quantization support for specific backends. For more information on the motivation and implementation details behind BackendConfig, please refer to [this README](README.md).

### 1. Derive reference pattern for each quantized operator

Suppose we are a backend developer and we wish to integrate our backend with PyTorch’s quantization APIs. Our backend consists of two ops only: quantized linear and quantized conv-relu. In this section, we will walk through how to achieve this by quantizing an example model using a custom BackendConfig through prepare_fx and convert_fx.

For quantized linear, suppose our backend expects the reference pattern [dequant - fp32_linear - quant] and lowers it into a single quantized linear op. The way to achieve this is to first insert quant-dequant ops before and after the float linear op, such that we produce the following reference model:

quant1 - [dequant1 - fp32_linear - quant2] - dequant2

Similarly, for quantized conv-relu, we wish to produce the following reference model, where the reference pattern in the square brackets will be lowered into a single quantized conv-relu op:

quant1 - [dequant1 - fp32_conv_relu - quant2] - dequant2

### 2. Set DTypeConfigs with backend constraints

In the reference patterns above, the input dtype specified in the DTypeConfig will be passed as the dtype argument to quant1, while the output dtype will be passed as the dtype argument to quant2. If the output dtype is fp32, as in the case of dynamic quantization, then the output quant-dequant pair will not be inserted. This example also shows how to specify restrictions on quantization and scale ranges on a particular dtype.

### 3. Set up fusion for conv-relu

Note that the original user model contains separate conv and relu ops, so we need to first fuse the conv and relu ops into a single conv-relu op (fp32_conv_relu), and then quantize this op similar to how the linear op is quantized. We can set up fusion by defining a function that accepts 3 arguments, where the first is whether or not this is for QAT, and the remaining arguments refer to the individual items of the fused pattern.

### 4. Define the BackendConfig

Now we have all the necessary pieces, so we go ahead and define our BackendConfig. Here we use different observers (will be renamed) for the input and output for the linear op, so the quantization params passed to the two quantize ops (quant1 and quant2) will be different. This is commonly the case for weighted ops like linear and conv.

For the conv-relu op, the observation type is the same. However, we need two BackendPatternConfigs to support this op, one for fusion and one for quantization. For both conv-relu and linear, we use the DTypeConfig defined above.

### 5. Set up QConfigMapping that satisfies the backend constraints

In order to use the ops defined above, the user must define a QConfig that satisfies the constraints specified in the DTypeConfig. For more detail, see the documentation for DTypeConfig. We will then use this QConfig for all the modules used in the patterns we wish to quantize.

### 6. Quantize the model through prepare and convert

Finally, we quantize the model by passing the BackendConfig we defined into prepare and convert. This produces a quantized linear module and a fused quantized conv-relu module.

### 7. Experiment with faulty BackendConfig setups

As an experiment, here we modify the model to use conv-bn-relu instead of conv-relu, but use the same BackendConfig, which doesn’t know how to quantize conv-bn-relu. As a result, only linear is quantized, but conv-bn-relu is neither fused nor quantized.

### Built-in BackendConfigs

PyTorch quantization supports a few built-in native BackendConfigs under the torch.ao.quantization.backend_config namespace:

- get_fbgemm_backend_config: for server target settings
- get_qnnpack_backend_config: for mobile and edge device target settings, also supports XNNPACK quantized ops
- get_native_backend_config (default): a BackendConfig that supports a union of the operator patterns supported in the FBGEMM and QNNPACK BackendConfigs

There are also other BackendConfigs under development (e.g. for TensorRT and x86), but these are still mostly experimental at the moment. If the user wishes to integrate a new, custom backend with PyTorch’s quantization API, they may define their own BackendConfigs using the same set of APIs used to define the natively supported ones as in the example above.

## [Dynamic Quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)

In this recipe, you will learn how to leverage Dynamic Quantization to accelerate inference on an LSTM-style recurrent neural network. This technique reduces the size of the model weights and speeds up model execution.

### Introduction

When designing neural networks, there are various trade-offs to consider. During model development and training, you can adjust the number of layers and parameters in a recurrent neural network, trading off accuracy against model size and/or latency or throughput. Quantization provides a way to make a similar trade-off between performance and model accuracy after training is completed.

With dynamic quantization, you can significantly reduce your model size and potentially achieve a significant latency reduction without losing much accuracy. This technique allows you to test the trade-off between performance and model accuracy in a single session.

### What is dynamic quantization?

Quantizing a network means converting it to use a reduced precision integer representation for the weights and/or activations. This saves on model size and allows the use of higher throughput math operations on your CPU or GPU.

Dynamic quantization determines the scale factor for activations dynamically based on the data range observed at runtime. This ensures that the scale factor is "tuned" so that as much signal as possible about each observed dataset is preserved. The model parameters, on the other hand, are converted ahead of time and stored in INT8 form.

Arithmetic in the quantized model is done using vectorized INT8 instructions. Accumulation is typically done with INT16 or INT32 to avoid overflow. This higher precision value is scaled back to INT8 if the next layer is quantized or converted to FP32 for output.

Dynamic quantization is relatively free of tuning parameters, which makes it well-suited to be added into production pipelines as a standard part of converting LSTM models to deployment.

**Note:**

Limitations of the approach taken here:

- This recipe provides a quick introduction to the dynamic quantization features in PyTorch and the workflow for using it.
- The focus is on explaining the specific functions used to convert the model.
- Several significant simplifications are made in the interest of brevity and clarity.
- The network used is a minimal LSTM network.
- The network is initialized with a random hidden state.
- The network is tested with random inputs.
- The network is not trained in this tutorial.
- The output values of the quantized network are generally in the same ballpark as the output of the FP32 network, but the expected accuracy loss on a real trained network is not demonstrated here.

### Steps

This recipe has 5 steps.

#### 1: Set Up

This is a straightforward bit of code to set up for the rest of the recipe.

The unique module we are importing here is `torch.quantization`, which includes PyTorch’s quantized operators and conversion functions. We also define a very simple LSTM model and set up some inputs.

```python
# import the modules used here in this recipe
import torch
import torch.quantization
import torch.nn as nn
import copy
import os
import time

# define a very, very simple LSTM for demonstration purposes
# in this case, we are wrapping ``nn.LSTM``, one layer, no preprocessing or postprocessing
# inspired by
# `Sequence Models and Long Short-Term Memory Networks tutorial <https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html>`_, by Robert Guthrie
# and `Dynamic Quanitzation tutorial <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>`__.
class lstm_for_demonstration(nn.Module):
  """Elementary Long Short Term Memory style model which simply wraps ``nn.LSTM``
     Not to be used for anything other than demonstration.
  """
  def __init__(self,in_dim,out_dim,depth):
     super(lstm_for_demonstration,self).__init__()
     self.lstm = nn.LSTM(in_dim,out_dim,depth)

  def forward(self,inputs,hidden):
     out,hidden = self.lstm(inputs,hidden)
     return out, hidden

torch.manual_seed(29592)  # set the seed for reproducibility

# shape parameters
model_dimension = 8
sequence_length = 20
batch_size = 1
lstm_depth = 1

# random data for input
inputs = torch.randn(sequence_length, batch_size, model_dimension)
# hidden is actually is a tuple of the initial hidden state and the initial cell state
hidden = (torch.randn(lstm_depth, batch_size, model_dimension), torch.randn(lstm_depth, batch_size, model_dimension))
```

#### 2: Do the Quantization

Now we get to the fun part. First, we create an instance of the model called `float_lstm`, then we are going to quantize it. We’re going to use the `torch.quantization.quantize_dynamic` function, which takes the model, then a list of the submodules which we want to have quantized if they appear, then the datatype we are targeting. This function returns a quantized version of the original model as a new module.

```python
# here is our floating point instance
float_lstm = lstm_for_demonstration(model_dimension, model_dimension, lstm_depth)

# this is the call that does the work
quantized_lstm = torch.quantization.quantize_dynamic(
    float_lstm, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)

# show the changes that were made
print('Here is the floating point version of this module:')
print(float_lstm)
print('')
print('and now the quantized version:')
print(quantized_lstm)
```

#### 3: Look at Model Size

We’ve quantized the model. What does that get us? Well, the first benefit is that we’ve replaced the FP32 model parameters with INT8 values (and some recorded scale factors). This means about 75% less data to store and move around. With the default values, the reduction shown below will be less than 75% but if you increase the model size above (for example, you can set model dimension to something like 80), this will converge towards 4x smaller as the stored model size is dominated more and more by the parameter values.

```python
def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, '\t', 'Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# compare the sizes
f = print_size_of_model(float_lstm, "fp32")
q = print_size_of_model(quantized_lstm, "int8")
print("{0:.2f} times smaller".format(f/q))
```

#### 4: Look at Latency

The second benefit is that the quantized model will typically run faster. This is due to a combination of effects including at least:

- Less time spent moving parameter data in
- Faster INT8 operations

As you will see, the quantized version of this super-simple network runs faster. This will generally be true of more complex networks, but as they say, "your mileage may vary" depending on a number of factors including the structure of the model and the hardware you are running on.

```python
# compare the performance
print("Floating point FP32")
%timeit float_lstm.forward(inputs, hidden)
print("Quantized INT8")
%timeit quantized_lstm.forward(inputs, hidden)
```

#### 5: Look at Accuracy

We are not going to do a careful look at accuracy here because we are working with a randomly initialized network rather than a properly trained one. However, I think it is worth quickly showing that the quantized network does produce output tensors that are "in the same ballpark" as the original one.

For a more detailed analysis, please see the more advanced tutorials referenced at the end of this recipe.

```python
# run the float model
out1, hidden1 = float_lstm(inputs, hidden)
mag1 = torch.mean(abs(out1)).item()
print('mean absolute value of output tensor values in the FP32 model is {0:.5f} '.format(mag1))

# run the quantized model
out2, hidden2 = quantized_lstm(inputs, hidden)
mag2 = torch.mean(abs(out2)).item()
print('mean absolute value of output tensor values in the INT8 model is {0:.5f}'.format(mag2))

# compare them
mag3 = torch.mean(abs(out1-out2)).item()
print('mean absolute value of the difference between the output tensors is {0:.5f} or {1:.2f} percent'.format(mag3, mag3/mag1*100))
```

## [Dynamic Quantization on an LSTM Word Language Model](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html)

### Introduction

Quantization involves converting the weights and activations of your model from float to int, which can result in smaller model size and faster inference with only a small hit to accuracy.

In this tutorial, we will apply the easiest form of quantization - dynamic quantization - to an LSTM-based next word-prediction model, closely following the word language model from the PyTorch examples.

### Imports

```python
import os
from io import open
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
```

### 1. Define the model

Here we define the LSTM model architecture, following the model from the word language model example.

```python
class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
```

### 2. Load in the text data

Next, we load the Wikitext-2 dataset into a Corpus, again following the preprocessing from the word language model example.

```python
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

model_data_filepath = 'data/'

corpus = Corpus(model_data_filepath + 'wikitext-2')
```

### 3. Load the pretrained model

This is a tutorial on dynamic quantization, a quantization technique that is applied after a model has been trained. Therefore, we’ll simply load some pretrained weights into this model architecture; these weights were obtained by training for five epochs using the default settings in the word language model example.

```python
ntokens = len(corpus.dictionary)

model = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)

model.load_state_dict(
    torch.load(
        model_data_filepath + 'word_language_model_quantize.pth',
        map_location=torch.device('cpu')
        )
    )

model.eval()
print(model)
```

Now let’s generate some text to ensure that the pretrained model is working properly - similarly to before, we follow here

```python
input_ = torch.randint(ntokens, (1, 1), dtype=torch.long)
hidden = model.init_hidden(1)
temperature = 1.0
num_words = 1000

with open(model_data_filepath + 'out.txt', 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(num_words):
            output, hidden = model(input_, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input_.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(str(word.encode('utf-8')) + ('\n' if i % 20 == 19 else ' '))

            if i % 100 == 0:
                print('| Generated {}/{} words'.format(i, 1000))

with open(model_data_filepath + 'out.txt', 'r') as outf:
    all_output = outf.read()
    print(all_output)
```

It’s no GPT-2, but it looks like the model has started to learn the structure of language!

### 4. Test dynamic quantization

Finally, we can call torch.quantization.quantize_dynamic on the model! Specifically,

- We specify that we want the nn.LSTM and nn.Linear modules in our model to be quantized
- We specify that we want weights to be converted to int8 values

```python
import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
print(quantized_model)
```

The model looks the same; how has this benefited us? First, we see a significant reduction in model size:

```python
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)
```

Second, we see faster inference time, with no difference in evaluation loss:

Note: we set the number of threads to one for single threaded comparison, since quantized models run single threaded.

```python
torch.set_num_threads(1)

def time_model_evaluation(model, test_data):
    s = time.time()
    loss = evaluate(model, test_data)
    elapsed = time.time() - s
    print('''loss: {0:.3f}\nelapsed time (seconds): {1:.1f}'''.format(loss, elapsed))

time_model_evaluation(model, test_data)
time_model_evaluation(quantized_model, test_data)
```

## [Dynamic Quantization on BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)

### Introduction

In this tutorial, we will apply the dynamic quantization on a BERT model, closely following the BERT model from the HuggingFace Transformers examples. With this step-by-step journey, we would like to demonstrate how to convert a well-known state-of-the-art model like BERT into dynamic quantized model.

BERT, or Bidirectional Embedding Representations from Transformers, is a new method of pre-training language representations which achieves the state-of-the-art accuracy results on many popular Natural Language Processing (NLP) tasks, such as question answering, text classification, and others. The original paper can be found [here](https://arxiv.org/abs/1810.04805).

Dynamic quantization support in PyTorch converts a float model to a quantized model with static int8 or float16 data types for the weights and dynamic quantization for the activations. The activations are quantized dynamically (per batch) to int8 when the weights are quantized to int8. In PyTorch, we have torch.quantization.quantize_dynamic API, which replaces specified modules with dynamic weight-only quantized versions and output the quantized model.

We demonstrate the accuracy and inference performance results on the Microsoft Research Paraphrase Corpus (MRPC) task in the General Language Understanding Evaluation benchmark (GLUE). The MRPC (Dolan and Brockett, 2005) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations of whether the sentences in the pair are semantically equivalent. As the classes are imbalanced (68% positive, 32% negative), we follow the common practice and report F1 score. MRPC is a common NLP task for language pair classification, as shown below.

### 1. Setup

#### 1.1 Install PyTorch and HuggingFace Transformers

To start this tutorial, let’s first follow the installation instructions in PyTorch [here](https://pytorch.org/get-started/locally/) and HuggingFace Github Repo [here](https://github.com/huggingface/transformers). In addition, we also install scikit-learn package, as we will reuse its built-in F1 score calculation helper function.

```bash
pip install sklearn
pip install transformers==4.29.2
```

#### 1.2 Import the necessary modules

In this step we import the necessary Python modules for the tutorial.

```python
import logging
import numpy as np
import os
import random
import sys
import time
import torch

from argparse import Namespace
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

logging.getLogger("transformers.modeling_utils").setLevel(
   logging.WARN)  # Reduce logging

print(torch.__version__)
```

We set the number of threads to compare the single thread performance between FP32 and INT8 performance. In the end of the tutorial, the user can set other number of threads by building PyTorch with right parallel backend.

```python
torch.set_num_threads(1)
print(torch.__config__.parallel_info())
```

#### 1.3 Learn about helper functions

The helper functions are built-in in transformers library. We mainly use the following helper functions: one for converting the text examples into the feature vectors; The other one for measuring the F1 score of the predicted result.

The glue_convert_examples_to_features function converts the texts into input features:

- Tokenize the input sequences;
- Insert [CLS] in the beginning;
- Insert [SEP] between the first sentence and the second sentence, and in the end;
- Generate token type ids to indicate whether a token belongs to the first sequence or the second sequence.

The glue_compute_metrics function has the compute metrics with the F1 score, which can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.

The equation for the F1 score is:

F1 = 2 * (precision * recall) / (precision + recall)

#### 1.4 Download the dataset

Before running MRPC tasks we download the GLUE data by running this script and unpack it to a directory glue_data.

```bash
python download_glue_data.py --data_dir='glue_data' --tasks='MRPC'
```

### 2. Fine-tune the BERT model

The spirit of BERT is to pre-train the language representations and then to fine-tune the deep bi-directional representations on a wide range of tasks with minimal task-dependent parameters, and achieves state-of-the-art results. In this tutorial, we will focus on fine-tuning with the pre-trained BERT model to classify semantically equivalent sentence pairs on MRPC task.

To fine-tune the pre-trained BERT model (bert-base-uncased model in HuggingFace transformers) for the MRPC task, you can follow the command in examples:

```bash
export GLUE_DIR=./glue_data
export TASK_NAME=MRPC
export OUT_DIR=./$TASK_NAME/
python ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --save_steps 100000 \
    --output_dir $OUT_DIR
```

We provide the fined-tuned BERT model for MRPC task [here](https://download.pytorch.org/tutorial/bert_mrpc_quantization/bert_mrpc_quantization.zip). To save time, you can download the model file (~400 MB) directly into your local folder $OUT_DIR.

#### 2.1 Set global configurations

Here we set the global configurations for evaluating the fine-tuned BERT model before and after the dynamic quantization.

```python
configs = Namespace()

# The output directory for the fine-tuned model, $OUT_DIR.
configs.output_dir = "./MRPC/"

# The data directory for the MRPC task in the GLUE benchmark, $GLUE_DIR/$TASK_NAME.
configs.data_dir = "./glue_data/MRPC"

# The model name or path for the pre-trained model.
configs.model_name_or_path = "bert-base-uncased"
# The maximum length of an input sequence
configs.max_seq_length = 128

# Prepare GLUE task.
configs.task_name = "MRPC".lower()
configs.processor = processors[configs.task_name]()
configs.output_mode = output_modes[configs.task_name]
configs.label_list = configs.processor.get_labels()
configs.model_type = "bert".lower()
configs.do_lower_case = True

# Set the device, batch size, topology, and caching flags.
configs.device = "cpu"
configs.per_gpu_eval_batch_size = 8
configs.n_gpu = 0
configs.local_rank = -1
configs.overwrite_cache = False

# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)
```

#### 2.2 Load the fine-tuned BERT model

We load the tokenizer and fine-tuned BERT sequence classifier model (FP32) from the configs.output_dir.

```python
tokenizer = BertTokenizer.from_pretrained(
    configs.output_dir, do_lower_case=configs.do_lower_case)

model = BertForSequenceClassification.from_pretrained(configs.output_dir)
model.to(configs.device)
```

#### 2.3 Define the tokenize and evaluation function

We reuse the tokenize and evaluation function from Huggingface.

```python
def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
```

### 3. Apply the dynamic quantization

We call torch.quantization.quantize_dynamic on the model to apply the dynamic quantization on the HuggingFace BERT model. Specifically,

- We specify that we want the torch.nn.Linear modules in our model to be quantized;
- We specify that we want weights to be converted to quantized int8 values.

```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print(quantized_model)
```

#### 3.1 Check the model size

Let’s first check the model size. We can observe a significant reduction in model size (FP32 total size: 438 MB; INT8 total size: 181 MB):

```python
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)
```

The BERT model used in this tutorial (bert-base-uncased) has a vocabulary size V of 30522. With the embedding size of 768, the total size of the word embedding table is ~ 4 (Bytes/FP32) * 30522 * 768 = 90 MB. So with the help of quantization, the model size of the non-embedding table part is reduced from 350 MB (FP32 model) to 90 MB (INT8 model).

#### 3.2 Evaluate the inference accuracy and time

Next, let’s compare the inference time as well as the evaluation accuracy between the original FP32 model and the INT8 model after the dynamic quantization.

```python
def time_model_evaluation(model, configs, tokenizer):
    eval_start_time = time.time()
    result = evaluate(configs, model, tokenizer, prefix="")
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print(result)
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))

# Evaluate the original FP32 BERT model
time_model_evaluation(model, configs, tokenizer)

# Evaluate the INT8 BERT model after the dynamic quantization
time_model_evaluation(quantized_model, configs, tokenizer)
```

Running this locally on a MacBook Pro, without quantization, inference (for all 408 examples in MRPC dataset) takes about 160 seconds, and with quantization it takes just about 90 seconds. We summarize the results for running the quantized BERT model inference on a Macbook Pro as the follows:

| Prec | F1 score | Model Size | 1 thread | 4 threads |
|------|----------|------------|----------|-----------|
| FP32 |  0.9019  |   438 MB   | 160 sec  | 85 sec    |
| INT8 |  0.902   |   181 MB   |  90 sec  | 46 sec    |

We have 0.6% lower F1 score accuracy after applying the post-training dynamic quantization on the fine-tuned BERT model on the MRPC task. As a comparison, in a recent paper (Table 1), it achieved 0.8788 by applying the post-training dynamic quantization and 0.8956 by applying the quantization-aware training. The main difference is that we support the asymmetric quantization in PyTorch while that paper supports the symmetric quantization only.

Note that we set the number of threads to 1 for the single-thread comparison in this tutorial. We also support the intra-op parallelization for these quantized INT8 operators. The users can now set multi-thread by torch.set_num_threads(N) (N is the number of intra-op parallelization threads). One preliminary requirement to enable the intra-op parallelization support is to build PyTorch with the right backend such as OpenMP, Native or TBB. You can use torch.__config__.parallel_info() to check the parallelization settings. On the same MacBook Pro using PyTorch with Native backend for parallelization, we can get about 46 seconds for processing the evaluation of MRPC dataset.

#### 3.3 Serialize the quantized model

We can serialize and save the quantized model for the future use using torch.jit.save after tracing the model.

```python
def ids_tensor(shape, vocab_size):
    #  Creates a random int32 tensor of the shape within the vocab size
    return torch.randint(0, vocab_size, shape=shape, dtype=torch.int, device='cpu')

input_ids = ids_tensor([8, 128], 2)
token_type_ids = ids_tensor([8, 128], 2)
attention_mask = ids_tensor([8, 128], vocab_size=2)
dummy_input = (input_ids, attention_mask, token_type_ids)
traced_model = torch.jit.trace(quantized_model, dummy_input)
torch.jit.save(traced_model, "bert_traced_eager_quant.pt")
```

To load the quantized model, we can use torch.jit.load

```python
loaded_quantized_model = torch.jit.load("bert_traced_eager_quant.pt")
```

### Conclusion

In this tutorial, we demonstrated how to convert a well-known state-of-the-art NLP model like BERT into dynamic quantized model. Dynamic quantization can reduce the size of the model while only having a limited implication on accuracy.

## [FX Graph Mode Quantization User Guide](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)

### Introduction

FX Graph Mode Quantization requires a symbolically traceable model. We use the FX framework to convert a symbolically traceable nn.Module instance to IR, and we operate on the IR to execute the quantization passes. If you have questions about symbolically tracing your model in PyTorch, please post them in the PyTorch Discussion Forum.

Quantization will only work on the symbolically traceable parts of your model. The data dependent control flow-if statements / for loops, and so on using symbolically traced values-are one common pattern which is not supported.

### Options for Non-Traceable Code

#### Non-Traceable Code Doesn't Need to be Quantized

1. **Symbolically trace only the code that needs to be quantized**
   - When the whole model is not symbolically traceable but the submodule we want to quantize is symbolically traceable, we can run quantization only on that submodule.

   **Before:**
   ```python
   class M(nn.Module):
       def forward(self, x):
           x = non_traceable_code_1(x)
           x = traceable_code(x)
           x = non_traceable_code_2(x)
           return x
   ```

   **After:**
   ```python
   class FP32Traceable(nn.Module):
       def forward(self, x):
           x = traceable_code(x)
           return x

   class M(nn.Module):
       def __init__(self):
           self.traceable_submodule = FP32Traceable(...)
       def forward(self, x):
           x = self.traceable_code_1(x)
           # We'll only symbolic trace/quantize this submodule
           x = self.traceable_submodule(x)
           x = self.traceable_code_2(x)
           return x
   ```

   **Quantization code:**
   ```python
   qconfig_mapping = QConfigMapping().set_global(qconfig)
   model_fp32.traceable_submodule = prepare_fx(model_fp32.traceable_submodule, qconfig_mapping, example_inputs)
   ```

   Note: If the original model needs to be preserved, you will have to copy it yourself before calling the quantization APIs.

2. **Skip symbolic tracing the non-traceable code**
   - When we have some non-traceable code in the module, and this part of code doesn’t need to be quantized, we can factor out this part of the code into a submodule and skip symbolically trace that submodule.

   **Before:**
   ```python
   class M(nn.Module):
       def forward(self, x):
           x = self.traceable_code_1(x)
           x = non_traceable_code(x)
           x = self.traceable_code_2(x)
           return x
   ```

   **After:**
   ```python
   class FP32NonTraceable(nn.Module):
       def forward(self, x):
           x = non_traceable_code(x)
           return x

   class M(nn.Module):
       def __init__(self):
           ...
           self.non_traceable_submodule = FP32NonTraceable(...)

       def forward(self, x):
           x = self.traceable_code_1(x)
           # we will configure the quantization call to not trace through this submodule
           x = self.non_traceable_submodule(x)
           x = self.traceable_code_2(x)
           return x
   ```

   **Quantization code:**
   ```python
   qconfig_mapping = QConfigMapping.set_global(qconfig)
   prepare_custom_config_dict = {
       "non_traceable_module_name": "non_traceable_submodule",
       # or
       "non_traceable_module_class": [MNonTraceable],
   }
   model_prepared = prepare_fx(model_fp32, qconfig_mapping, example_inputs, prepare_custom_config_dict=prepare_custom_config_dict)
   ```

#### Non-Traceable Code Needs to be Quantized

1. **Refactor your code to make it symbolically traceable**
   - If it is easy to refactor the code and make the code symbolically traceable, we can refactor the code and remove the use of non-traceable constructs in python.

   **Before:**
   ```python
   def transpose_for_scores(self, x):
       new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
       x = x.view(*new_x_shape)
       return x.permute(0, 2, 1, 3)
   ```

   **After:**
   ```python
   def transpose_for_scores(self, x):
       new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
       x = x.view(new_x_shape)
       return x.permute(0, 2, 1, 3)
   ```

   This can be combined with other approaches and the quantization code depends on the model.

2. **Write your own observed and quantized submodule**
   - If the non-traceable code can’t be refactored to be symbolically traceable, for example it has some loops that can’t be eliminated, like nn.LSTM, we’ll need to factor out the non-traceable code to a submodule (we call it CustomModule in fx graph mode quantization) and define the observed and quantized version of the submodule (in post training static quantization or quantization aware training for static quantization) or define the quantized version (in post training dynamic and weight only quantization).

   **Before:**
   ```python
   class M(nn.Module):
       def forward(self, x):
           x = traceable_code_1(x)
           x = non_traceable_code(x)
           x = traceable_code_1(x)
           return x
   ```

   **After:**
   - Factor out non_traceable_code to FP32NonTraceable non-traceable logic, wrapped in a module
   - Define observed version of FP32NonTraceable
   - Define statically quantized version of FP32NonTraceable and a class method “from_observed” to convert from ObservedNonTraceable to StaticQuantNonTraceable

   **Quantization code:**
   - Post training static quantization or quantization aware training (that produces a statically quantized module)
   ```python
   prepare_custom_config_dict = {
       "float_to_observed_custom_module_class": {
           "static": {
               FP32NonTraceable: ObservedNonTraceable,
           }
       },
   }
   model_prepared = prepare_fx(model_fp32, qconfig_mapping, example_inputs, prepare_custom_config_dict=prepare_custom_config_dict)
   # calibrate / train (not shown)
   convert_custom_config_dict = {
       "observed_to_quantized_custom_module_class": {
           "static": {
               ObservedNonTraceable: StaticQuantNonTraceable,
           }
       },
   }
   model_quantized = convert_fx(model_prepared, convert_custom_config_dict)
   ```

   - Post training dynamic/weight only quantization in these two modes we don’t need to observe the original model, so we only need to define thee quantized model
   ```python
   class DynamicQuantNonTraceable: # or WeightOnlyQuantMNonTraceable
      ...
      @classmethod
      def from_observed(cls, ...):
          ...

   prepare_custom_config_dict = {
       "non_traceable_module_class": [
           FP32NonTraceable
       ]
   }
   # The example is for post training quantization
   model_fp32.eval()
   model_prepared = prepare_fx(model_fp32, qconfig_mapping, example_inputs, prepare_custom_config_dict=prepare_custom_config_dict)
   convert_custom_config_dict = {
       "observed_to_quantized_custom_module_class": {
           "dynamic": {
               FP32NonTraceable: DynamicQuantNonTraceable,
           }
       },
   }
   model_quantized = convert_fx(model_prepared, convert_custom_config_dict)
   ```

   You can also find examples for custom modules in test test_custom_module_class in torch/test/quantization/test_quantize_fx.py.

## [FX Graph Mode Post Training Static Quantization](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)

This tutorial introduces the steps to perform post training static quantization in graph mode based on torch.fx. The advantage of FX graph mode quantization is that it allows for fully automatic quantization of the model. Although there might be some effort required to make the model compatible with FX Graph Mode Quantization (symbolically traceable with torch.fx), we’ll have a separate tutorial to show how to make the part of the model we want to quantize compatible with FX Graph Mode Quantization. We also have a tutorial for FX Graph Mode Post Training Dynamic Quantization.

The FX Graph Mode API looks like the following:

```python
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

float_model.eval()
qconfig = get_default_qconfig("x86")
qconfig_mapping = QConfigMapping().set_global(qconfig)

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

example_inputs = (next(iter(data_loader))[0])
prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)
calibrate(prepared_model, data_loader_test)
quantized_model = convert_fx(prepared_model)
```

### 1. Motivation of FX Graph Mode Quantization

Currently, PyTorch only has eager mode quantization as an alternative: [Static Quantization with Eager Mode in PyTorch](link-to-eager-mode-tutorial).

We can see there are multiple manual steps involved in the eager mode quantization process, including:

- Explicitly quantizing and dequantizing activations
- Explicitly fusing modules
- Special handling for pytorch tensor operations (like add, concat, etc.)
- Functionals did not have first class support (functional.conv2d and functional.linear would not get quantized)

Most of these required modifications come from the underlying limitations of eager mode quantization. Eager mode works in module level since it cannot inspect the code that is actually run (in the forward function). Quantization is achieved by module swapping, and we don’t know how the modules are used in the forward function in eager mode, so it requires users to insert QuantStub and DeQuantStub manually to mark the points they want to quantize or dequantize.

In graph mode, we can inspect the actual code that’s been executed in the forward function (e.g. aten function calls) and quantization is achieved by module and graph manipulations. Since graph mode has full visibility of the code that is run, our tool is able to automatically figure out things like which modules to fuse and where to insert observer calls, quantize/dequantize functions, etc., we are able to automate the whole quantization process.

Advantages of FX Graph Mode Quantization are:

- Simple quantization flow, minimal manual steps
- Unlocks the possibility of doing higher level optimizations like automatic precision selection

### 2. Define Helper Functions and Prepare Dataset

We’ll start by doing the necessary imports, defining some helper functions, and preparing the data. These steps are identical to [Static Quantization with Eager Mode in PyTorch](link-to-eager-mode-tutorial).

To run the code in this tutorial using the entire ImageNet dataset, first download imagenet by following the instructions at [ImageNet Data](link-to-imagenet-data). Unzip the downloaded file into the ‘data_path’ folder.

Download the torchvision resnet18 model and rename it to data/resnet18_pretrained_float.pth.

```python
import os
import sys
import time
import numpy as np

import torch
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms

# ... (other imports and warnings filtering)

# Helper functions and data preparation code here
```

### 3. Set model to eval mode

For post training quantization, we’ll need to set the model to eval mode.

```python
model_to_quantize.eval()
```

### 4. Specify how to quantize the model with QConfigMapping

```python
qconfig_mapping = QConfigMapping.set_global(default_qconfig)
```

We use the same qconfig used in eager mode quantization, qconfig is just a named tuple of the observers for activation and weight. QConfigMapping contains mapping information from ops to qconfigs:

```python
qconfig_mapping = (QConfigMapping()
    .set_global(qconfig_opt)
    .set_object_type(torch.nn.Conv2d, qconfig_opt)
    .set_object_type("reshape", qconfig_opt)
    .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig_opt)
    .set_module_name("foo.bar", qconfig_opt)
    .set_module_name_object_type_order()
)
```

Priority (in increasing order): global, object_type, module_name_regex, module_name
qconfig == None means fusion and quantization should be skipped for anything matching the rule (unless a higher priority match is found)

Utility functions related to qconfig can be found in the qconfig file while those for QConfigMapping can be found in the [qconfig_mapping](https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/qconfig_mapping.py) file.

### 5. Prepare the Model for Post Training Static Quantization

```python
prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
```

The `prepare_fx` function folds BatchNorm modules into previous Conv2d modules, and inserts observers in appropriate places in the model.

### 6. Calibration

The calibration function is run after the observers are inserted in the model. The purpose of calibration is to run through some sample examples that are representative of the workload (for example, a sample of the training dataset) so that the observers in the model are able to observe the statistics of the Tensors and we can later use this information to calculate quantization parameters.

```python
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

calibrate(prepared_model, data_loader_test)
```

### 7. Convert the Model to a Quantized Model

```python
quantized_model = convert_fx(prepared_model)
```

### 8. Evaluation

We can now print the size and accuracy of the quantized model.

```python
print("Size of model before quantization")
print_size_of_model(float_model)
print("Size of model after quantization")
print_size_of_model(quantized_model)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("[before serialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

# Save and load the quantized model
torch.jit.save(torch.jit.script(quantized_model), fx_graph_mode_model_file_path)
loaded_quantized_model = torch.jit.load(fx_graph_mode_model_file_path)

top1, top5 = evaluate(loaded_quantized_model, criterion, data_loader_test)
print("[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))
```

### 9. Debugging Quantized Model

We can also print the weight for quantized and non-quantized convolution ops to see the difference. First, call `fuse_fx` explicitly to fuse the convolution and batch norm in the model. Note that `fuse_fx` only works in eval mode.

```python
fused = fuse_fx(float_model)

conv1_weight_after_fuse = fused.conv1[0].weight[0]
conv1_weight_after_quant = quantized_model.conv1.weight().dequantize()[0]

print(torch.max(abs(conv1_weight_after_fuse - conv1_weight_after_quant)))
```

### 10. Comparison with Baseline Float Model and Eager Mode Quantization

```python
print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test)
print("Baseline Float Model Evaluation accuracy: %2.2f, %2.2f"%(top1.avg, top5.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

print("Size of Fx graph mode quantized model")
print_size_of_model(quantized_model)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("FX graph mode quantized model Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

from torchvision.models.quantization.resnet import resnet18
eager_quantized_model = resnet18(pretrained=True, quantize=True).eval()
print("Size of eager mode quantized model")
eager_quantized_model = torch.jit.script(eager_quantized_model)
print_size_of_model(eager_quantized_model)
top1, top5 = evaluate(eager_quantized_model, criterion, data_loader_test)
print("eager mode quantized model Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))
torch.jit.save(eager_quantized_model, saved_model_dir + eager_mode_model_file)
```

We can see that the model size and accuracy of FX graph mode and eager mode quantized model are pretty similar.

Running the model in AIBench (with single threading) gives the following result:

- Scripted Float Model: Self CPU time total: 192.48ms
- Scripted Eager Mode Quantized Model: Self CPU time total: 50.76ms
- Scripted FX Graph Mode Quantized Model: Self CPU time total: 50.63ms

As we can see for resnet18, both FX graph mode and eager mode quantized model get similar speedup over the floating point model, which is around 2-4x faster than the floating point model. However, the actual speedup over the floating point model may vary depending on the model, device, build, input batch sizes, threading, etc.

## [FX Graph Mode Post Training Dynamic Quantization](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html)

This tutorial introduces the steps to do post training dynamic quantization in graph mode based on torch.fx. We have a separate tutorial for FX Graph Mode Post Training Static Quantization. Comparison between FX Graph Mode Quantization and Eager Mode Quantization can be found in the quantization docs.

**TL;DR:** The FX Graph Mode API for dynamic quantization looks like the following:

```python
import torch
from torch.ao.quantization import default_dynamic_qconfig, QConfigMapping
# Note that this is temporary, we'll expose these functions to torch.ao.quantization after official release
from torch.quantization.quantize_fx import prepare_fx, convert_fx

float_model.eval()
# The old 'fbgemm' is still available but 'x86' is the recommended default.
qconfig = get_default_qconfig("x86")
qconfig_mapping = QConfigMapping().set_global(qconfig)
prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)  # fuse modules and insert observers
# no calibration is required for dynamic quantization
quantized_model = convert_fx(prepared_model)  # convert the model to a dynamically quantized model
```

In this tutorial, we’ll apply dynamic quantization to an LSTM-based next word-prediction model, closely following the word language model from the PyTorch examples. We will copy the code from Dynamic Quantization on an LSTM Word Language Model and omit the descriptions.

### 1. Define the Model, Download Data and Model

Download the data and unzip to data folder:

```bash
mkdir data
cd data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip wikitext-2-v1.zip
```

Download model to the data folder:

```bash
wget https://s3.amazonaws.com/pytorch-tutorial-assets/word_language_model_quantize.pth
```

Define the model:

```python
# imports
import os
from io import open
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# Model Definition
class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

def init_hidden(lstm_model, bsz):
    # get the weight tensor and create hidden layer in the same device
    weight = lstm_model.encoder.weight
    # get weight from quantized model
    if not isinstance(weight, torch.Tensor):
        weight = weight()
    device = weight.device
    nlayers = lstm_model.rnn.num_layers
    nhid = lstm_model.rnn.hidden_size
    return (torch.zeros(nlayers, bsz, nhid, device=device),
            torch.zeros(nlayers, bsz, nhid, device=device))

# Load Text Data
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

model_data_filepath = 'data/'

corpus = Corpus(model_data_filepath + 'wikitext-2')

ntokens = len(corpus.dictionary)

# Load Pretrained Model
model = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)

model.load_state_dict(
    torch.load(
        model_data_filepath + 'word_language_model_quantize.pth',
        map_location=torch.device('cpu')
        )
    )

model.eval()
print(model)

bptt = 25
criterion = nn.CrossEntropyLoss()
eval_batch_size = 1

# create test data set
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    return data.view(bsz, -1).t().contiguous()

test_data = batchify(corpus.test, eval_batch_size)
example_inputs = (next(iter(test_data))[0])

# Evaluation functions
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def repackage_hidden(h):
  """Wraps hidden states in new Tensors, to detach them from their history."""

  if isinstance(h, torch.Tensor):
      return h.detach()
  else:
      return tuple(repackage_hidden(v) for v in h)

def evaluate(model_, data_source):
    # Turn on evaluation mode which disables dropout.
    model_.eval()
    total_loss = 0.
    hidden = init_hidden(model_, eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model_(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)
```

### 2. Post Training Dynamic Quantization

Now we can dynamically quantize the model. We can use the same function as post training static quantization but with a dynamic qconfig.

```python
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, QConfigMapping

# Full docs for supported qconfig for floating point modules/ops can be found in `quantization docs <https://pytorch.org/docs/stable/quantization.html#module-torch.quantization>`_
# Full docs for `QConfigMapping <https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping>`_
qconfig_mapping = (QConfigMapping()
    .set_object_type(nn.Embedding, float_qparams_weight_only_qconfig)
    .set_object_type(nn.LSTM, default_dynamic_qconfig)
    .set_object_type(nn.Linear, default_dynamic_qconfig)
)
# Load model to create the original model because quantization api changes the model inplace and we want
# to keep the original model for future comparison

model_to_quantize = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)

model_to_quantize.load_state_dict(
    torch.load(
        model_data_filepath + 'word_language_model_quantize.pth',
        map_location=torch.device('cpu')
        )
    )

model_to_quantize.eval()

prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
print("prepared model:", prepared_model)
quantized_model = convert_fx(prepared_model)
print("quantized model", quantized_model)
```

For dynamically quantized objects, we didn’t do anything in prepare_fx for modules, but will insert observers for weight for dynamically quantizable forunctionals and torch ops. We also fuse the modules like Conv + Bn, Linear + ReLU.

In convert we’ll convert the float modules to dynamically quantized modules and convert float ops to dynamically quantized ops. We can see in the example model, nn.Embedding, nn.Linear and nn.LSTM are dynamically quantized.

Now we can compare the size and runtime of the quantized model.

```python
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)
```

There is a 4x size reduction because we quantized all the weights in the model (nn.Embedding, nn.Linear and nn.LSTM) from float (4 bytes) to quantized int (1 byte).

```python
torch.set_num_threads(1)

def time_model_evaluation(model, test_data):
    s = time.time()
    loss = evaluate(model, test_data)
    elapsed = time.time() - s
    print('''loss: {0:.3f}\nelapsed time (seconds): {1:.1f}'''.format(loss, elapsed))

time_model_evaluation(model, test_data)
time_model_evaluation(quantized_model, test_data)
```

There is a roughly 2x speedup for this model. Also note that the speedup may vary depending on model, device, build, input batch sizes, threading etc.

### 3. Conclusion

This tutorial introduces the api for post training dynamic quantization in FX Graph Mode, which dynamically quantizes the same modules as Eager Mode Quantization.

## [Graph Mode Dynamic Quantization on BERT](https://pytorch.org/tutorials/prototype/graph_mode_dynamic_bert_tutorial.html)

### Introduction

This tutorial introduces the steps to do post training Dynamic Quantization with Graph Mode Quantization. Dynamic quantization converts a float model to a quantized model with static int8 data types for the weights and dynamic quantization for the activations. The activations are quantized dynamically (per batch) to int8 while the weights are statically quantized to int8. Graph Mode Quantization flow operates on the model graph and requires minimal user intervention to quantize the model. To be able to use graph mode, the float model needs to be either traced or scripted first.

#### Advantages of graph mode quantization are:

- In graph mode, we can inspect the code that is executed in forward function (e.g. aten function calls) and quantization is achieved by module and graph manipulations.
- Simple quantization flow, minimal manual steps.
- Unlocks the possibility of doing higher level optimizations like automatic precision selection.

For additional details on Graph Mode Quantization please refer to the [Graph Mode Static Quantization Tutorial](#).

#### tl;dr The Graph Mode Dynamic Quantization API:

```python
import torch
from torch.quantization import per_channel_dynamic_qconfig
from torch.quantization import quantize_dynamic_jit

ts_model = torch.jit.script(float_model) # or torch.jit.trace(float_model, input)

quantized = quantize_dynamic_jit(ts_model, {'': per_channel_dynamic_qconfig})
```

### 1. Quantizing BERT Model

The installation steps and details about the model are identical to the steps in the Eager Mode Tutorial. Please refer to the tutorial [here](#) for more details.

#### 1.1 Setup

Once all the necessary packages are downloaded and installed, we setup the code. We first start with the necessary imports and setup for the model.

```python
import logging
import numpy as np
import os
import random
import sys
import time
import torch

from argparse import Namespace
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from torch.quantization import per_channel_dynamic_qconfig
from torch.quantization import quantize_dynamic_jit

def ids_tensor(shape, vocab_size):
    #  Creates a random int32 tensor of the shape within the vocab size
    return torch.randint(0, vocab_size, shape=shape, dtype=torch.int, device='cpu')

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

logging.getLogger("transformers.modeling_utils").setLevel(
   logging.WARN)  # Reduce logging

print(torch.__version__)

torch.set_num_threads(1)
print(torch.__config__.parallel_info())
```

#### 1.2 Download GLUE dataset

Before running MRPC tasks, we download the GLUE data by running this script and unpack it to a directory `glue_data`.

```bash
python download_glue_data.py --data_dir='glue_data' --tasks='MRPC'
```

#### 1.3 Set global BERT configurations

To run this experiment, we first need a fine-tuned BERT model. We provide the fine-tuned BERT model for MRPC task [here](#). To save time, you can download the model file (~400 MB) directly into your local folder `$OUT_DIR`.

```python
configs = Namespace()

# The output directory for the fine-tuned model, $OUT_DIR.
configs.output_dir = "./MRPC/"

# The data directory for the MRPC task in the GLUE benchmark, $GLUE_DIR/$TASK_NAME.
configs.data_dir = "./glue_data/MRPC"

# The model name or path for the pre-trained model.
configs.model_name_or_path = "bert-base-uncased"
# The maximum length of an input sequence
configs.max_seq_length = 128

# Prepare GLUE task.
configs.task_name = "MRPC".lower()
configs.processor = processors[configs.task_name]()
configs.output_mode = output_modes[configs.task_name]
configs.label_list = configs.processor.get_labels()
configs.model_type = "bert".lower()
configs.do_lower_case = True

# Set the device, batch size, topology, and caching flags.
configs.device = "cpu"
configs.per_gpu_eval_batch_size = 8
configs.n_gpu = 0
configs.local_rank = -1
configs.overwrite_cache = False

# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

tokenizer = BertTokenizer.from_pretrained(
    configs.output_dir, do_lower_case=configs.do_lower_case)

model = BertForSequenceClassification.from_pretrained(configs.output_dir, torchscript=True)
model.to(configs.device)
```

#### 1.4 Quantizing BERT model with Graph Mode Quantization

##### 1.4.1 Script/Trace the model

The input for graph mode quantization is a TorchScript model, so you’ll need to either script or trace the model first. Currently, scripting the BERT model is not supported, so we trace the model here.

We first identify the inputs to be passed to the model. Here, we trace the model with the largest possible input size that will be passed during the evaluation. We choose a batch size of 8 and sequence length of 128 based on the input sizes passed in during the evaluation step below. Using the max possible shape during inference while tracing is a limitation of the HuggingFace BERT model as mentioned [here](#).

We trace the model using `torch.jit.trace`.

```python
input_ids = ids_tensor([8, 128], 2)
token_type_ids = ids_tensor([8, 128], 2)
attention_mask = ids_tensor([8, 128], vocab_size=2)
dummy_input = (input_ids, attention_mask, token_type_ids)
traced_model = torch.jit.trace(model, dummy_input)
```

##### 1.4.2 Specify qconfig_dict

`qconfig_dict` is a dictionary with names of submodules as keys and `qconfig` for that module as values. Empty key means the `qconfig` will be applied to the whole model unless it’s overwritten by more specific configurations. The `qconfig` for each module is either found in the dictionary or falls back to the `qconfig` of the parent module.

Right now, `qconfig_dict` is the only way to configure how the model is quantized, and it is done in the granularity of module. That is, we only support one type of `qconfig` for each module, and the `qconfig` for submodule will override the `qconfig` for the parent module. For example, if we have:

```python
qconfig = {
    '' : qconfig_global,
    'sub' : qconfig_sub,
    'sub.fc1' : qconfig_fc,
    'sub.fc2': None
}
```

Module `sub.fc1` will be configured with `qconfig_fc`, and all other child modules in `sub` will be configured with `qconfig_sub`. Module `sub.fc2` will not be quantized. All other modules in the model will be quantized with `qconfig_global`.

```python
qconfig_dict = {'': per_channel_dynamic_qconfig}
```

##### 1.4.3 Quantize the model (one-line API)

We call the one-line API (similar to eager mode) to perform quantization as follows.

```python
quantized_model = quantize_dynamic_jit(traced_model, qconfig_dict)
```

### 2. Evaluation

We reuse the tokenize and evaluation function from HuggingFace.

```python
def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1]}
                labels = batch[3]
                if args.model_type != 'distilbert':
                    inputs['input'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                logits = outputs[0]
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    # ... (code omitted for brevity)

def time_model_evaluation(model, configs, tokenizer):
    eval_start_time = time.time()
    result = evaluate(configs, model, tokenizer, prefix="")
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print(result)
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))
```

#### 2.1 Check Model Size

We print the model size to account for wins from quantization.

```python
def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print("Size of model before quantization")
print_size_of_model(traced_model)
print("Size of model after quantization")
print_size_of_model(quantized_model)
```

#### 2.2 Run the evaluation

We evaluate the FP32 and quantized model and compare the F1 score. Note that the performance numbers below are on a dev machine and they would likely improve on a production server.

```python
time_model_evaluation(traced_model, configs, tokenizer)
time_model_evaluation(quantized_model, configs, tokenizer)
```

FP32 model results -
'f1': 0.901
Time taken - 188.0s

INT8 model results -
'f1': 0.902
Time taken - 157.4s

### 3. Debugging the Quantized Model

We can debug the quantized model by passing in the debug option.

```python
quantized_model = quantize_dynamic_jit(traced_model, qconfig_dict, debug=True)
```

If debug is set to True:

- We can access the attributes of the quantized model the same way as in a TorchScript model, e.g., `model.fc1.weight`.
- The arithmetic operations all occur in floating point with the numerics being identical to the final quantized model, allowing for debugging.

```python
quantized_model_debug = quantize_dynamic_jit(traced_model, qconfig_dict, debug=True)
```

Calling `quantize_dynamic_jit` is equivalent to calling `prepare_dynamic_jit` followed by `convert_dynamic_jit`. Using the one-line API is recommended. However, if you wish to debug or analyze the model after each step, the multi-line API comes into use.

#### 3.1. Evaluate the Debug Model

```python
# Evaluate the debug model
time_model_evaluation(quantized_model_debug, configs, tokenizer)
```

Size (MB): 438.406429

INT8 (debug=True) model results -
'f1': 0.897

Note that the accuracy of the debug version is close to, but not exactly the same as the non-debug version as the debug version uses floating-point ops to emulate quantized ops and the numerics match is approximate. This is the case only for per-channel quantization (we are working on improving this). Per-tensor quantization (using `default_dynamic_qconfig`) has exact numerics match between debug and non-debug versions.

```python
print(str(quantized_model_debug.graph))
```

Snippet of the graph printed -

```
%111 : Tensor = prim::GetAttr[name="bias"](%110)
%112 : Tensor = prim::GetAttr[name="weight"](%110)
%113 : Float(768:1) = prim::GetAttr[name="4_scale_0"](%110)
%114 : Int(768:1) = prim::GetAttr[name="4_zero_point_0"](%110)
%115 : int = prim::GetAttr[name="4_axis_0"](%110)
%116 : int = prim::GetAttr[name="4_scalar_type_0"](%110)
%4.quant.6 : Tensor = aten::quantize_per_channel(%112, %113, %114, %115, %116)
%4.dequant.6 : Tensor = aten::dequantize(%4.quant.6)
%1640 : bool = prim::Constant[value=1]()
%input.5.scale.1 : float, %input.5.zero_point.1 : int = aten::_choose_qparams_per_tensor(%input.5, %1640)
%input.5.quant.1 : Tensor = aten::quantize_per_tensor(%input.5, %input.5.scale.1, %input.5.zero_point.1, %74)
%input.5.dequant.1 : Float(8:98304, 128:768, 768:1) = aten::dequantize(%input.5.quant.1)
%119 : Tensor = aten::linear(%input.5.dequant.1, %4.dequant.6, %111)
```

We can see that there is no `quantized::linear_dynamic` in the model, but the numerically equivalent pattern of `aten::_choose_qparams_per_tensor` - `aten::quantize_per_tensor` - `aten::dequantize` - `aten::linear`.

```python
# Get the size of the debug model
print_size_of_model(quantized_model_debug)
```

Size (MB): 438.406429

Size of the debug model is close to the floating-point model because all the weights are in float and not yet quantized and frozen, this allows people to inspect the weight. You may access the weight attributes directly in the TorchScript model. Accessing the weight in the debug model is the same as accessing the weight in a TorchScript model:

```python
print(quantized_model.bert.encoder.layer._c.getattr('0').attention.self.query.weight)
```

Accessing the scale and zero_point for the corresponding weight can be done as follows -

```python
print(quantized_model.bert.encoder.layer._c.getattr('0').attention.self.query.getattr('4_scale_0'))
print(quantized_model.bert.encoder.layer._c.getattr('0').attention.self.query.getattr('4_zero_point_0'))
```

Since we use per-channel quantization, we get per-channel scales tensor.

### 4. Comparing Results with Eager Mode

Following results show the F1 score and model size for Eager Mode Quantization of the same model by following the steps mentioned in the tutorial. Results show that Eager and Graph Mode Quantization on the model produce identical results.

FP32 model results -
Size (MB): 438.016605
'f1': 0.901

INT8 model results -
Size (MB): 182.878029
'f1': 0.902

### 5. Benchmarking the Model

We benchmark the model with dummy input and compare the Float model with Eager and Graph Mode Quantized Model on a production server machine.

```python
def benchmark(model):
    model = torch.jit.load(model)
    model.eval()
    torch.set_num_threads(1)
    input_ids = ids_tensor([8, 128], 2)
    token_type_ids = ids_tensor([8, 128], 2)
    attention_mask = ids_tensor([8, 128], vocab_size=2)
    elapsed = 0
    for _i in range(50):
        start = time.time()
        output = model(input_ids, token_type_ids, attention_mask)
        end = time.time()
        elapsed = elapsed + (end - start)
    print('Elapsed time: ', (elapsed / 50), ' s')
    return

print("Running benchmark for Float model")
benchmark(args.jit_model_path_float)
print("Running benchmark for Eager Mode Quantized model")
benchmark(args.jit_model_path_eager)
print("Running benchmark for Graph Mode Quantized model")
benchmark(args.jit_model_path_graph)
```

## [Static Quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

This tutorial shows how to do post-training static quantization, as well as illustrating two more advanced techniques - per-channel quantization and quantization-aware training - to further improve the model’s accuracy. Note that quantization is currently only supported for CPUs, so we will not be utilizing GPUs / CUDA in this tutorial. By the end of this tutorial, you will see how quantization in PyTorch can result in significant decreases in model size while increasing speed. Furthermore, you’ll see how to easily apply some advanced quantization techniques shown here so that your quantized models take much less of an accuracy hit than they would otherwise.

Warning: we use a lot of boilerplate code from other PyTorch repos to, for example, define the MobileNetV2 model architecture, define data loaders, and so on. We of course encourage you to read it; but if you want to get to the quantization features, feel free to skip to the “4. Post-training static quantization” section.

### 1. Model architecture

We first define the MobileNetV2 model architecture, with several notable modifications to enable quantization:

- Replacing addition with `nn.quantized.FloatFunctional`
- Insert `QuantStub` and `DeQuantStub` at the beginning and end of the network.
- Replace ReLU6 with ReLU

Note: this code is taken from [here](https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/mobilenet_example.py).

### 2. Helper functions

We next define several helper functions to help with model evaluation. These mostly come from [here](https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/mobilenet_example.py).

### 3. Define dataset and data loaders

As our last major setup step, we define our dataloaders for our training and testing set.

### 4. Post-training static quantization

Post-training static quantization involves not just converting the weights from float to int, as in dynamic quantization, but also performing the additional step of first feeding batches of data through the network and computing the resulting distributions of the different activations (specifically, this is done by inserting observer modules at different points that record this data). These distributions are then used to determine how the specifically the different activations should be quantized at inference time (a simple technique would be to simply divide the entire range of activations into 256 levels, but we support more sophisticated methods as well). Importantly, this additional step allows us to pass quantized values between operations instead of converting these values to floats - and then back to ints - between every operation, resulting in a significant speed-up.

### 5. Quantization-aware training

Quantization-aware training (QAT) is the quantization method that typically results in the highest accuracy. With QAT, all weights and activations are “fake quantized” during both the forward and backward passes of training: that is, float values are rounded to mimic int8 values, but all computations are still done with floating point numbers. Thus, all the weight adjustments during training are made while “aware” of the fact that the model will ultimately be quantized; after quantizing, therefore, this method will usually yield higher accuracy than either dynamic quantization or post-training static quantization.

The overall workflow for actually performing QAT is very similar to before:

- We can use the same model as before: there is no additional preparation needed for quantization-aware training.
- We need to use a qconfig specifying what kind of fake-quantization is to be inserted after weights and activations, instead of specifying observers
- We first define a training function:
- We fuse modules as before
- Finally, `prepare_qat` performs the “fake quantization”, preparing the model for quantization-aware training

### Speedup from quantization

Finally, let’s confirm something we alluded to above: do our quantized models actually perform inference faster? Let’s test:

```python
def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)

run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)
```

Running this locally on a MacBook pro yielded 61 ms for the regular model, and just 20 ms for the quantized model, illustrating the typical 2-4x speedup we see for quantized models compared to floating point ones.

### Conclusion

In this tutorial, we showed two quantization methods - post-training static quantization, and quantization-aware training - describing what they do “under the hood” and how to use them in PyTorch.

Thanks for reading! As always, we welcome any feedback, so please create an issue [here](https://github.com/pytorch/pytorch/issues) if you have any.

## [Quantization in PyTorch 2.0 Export Tutorial](https://pytorch.org/tutorials/prototype/quantization_in_pytorch_2_0_export_tutorial.html)

## [PyTorch 2 Export Post Training Quantization](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html)

This tutorial introduces the steps to do post training static quantization in graph mode based on torch._export.export. Compared to FX Graph Mode Quantization, this flow is expected to have significantly higher model coverage (88% on 14K models), better programmability, and a simplified UX.

### Exportable by torch.export.export is a Prerequisite

To use the flow, the model needs to be exportable by torch.export.export. You can find what are the constructs that’s supported in Export DB.

### High Level Architecture of Quantization 2 with Quantizer

The high level architecture of quantization 2 with quantizer could look like this:

```
float_model(Python)                          Example Input
    \                                              /
     \                                            /
---------------------------------------------------------
|                        export                        |
---------------------------------------------------------
                            |
                    FX Graph in ATen     Backend Specific Quantizer
                            |                       /
----------------------------------------------------------
|                     prepare_pt2e                      |
----------------------------------------------------------
                            |
                     Calibrate/Train
                            |
----------------------------------------------------------
|                    convert_pt2e                       |
----------------------------------------------------------
                            |
                    Quantized Model
                            |
----------------------------------------------------------
|                       Lowering                        |
----------------------------------------------------------
                            |
        Executorch, Inductor or <Other Backends>
```

### PyTorch 2 Export Quantization API

The PyTorch 2 export quantization API looks like this:

```python
import torch
from torch._export import capture_pre_autograd_graph

class M(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(5, 10)

   def forward(self, x):
      return self.linear(x)

example_inputs = (torch.randn(1, 5),)
m = M().eval()

# Step 1. program capture
m = capture_pre_autograd_graph(m, *example_inputs)

# Step 2. quantization
from torch.ao.quantization.quantize_pt2e import (
  prepare_pt2e,
  convert_pt2e,
)

from torch.ao.quantization.quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)

quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
m = prepare_pt2e(m, quantizer)

# calibration omitted

m = convert_pt2e(m)
```

### Motivation of PyTorch 2 Export Quantization

In PyTorch versions prior to 2, we have FX Graph Mode Quantization that uses QConfigMapping and BackendConfig for customizations. The new API addresses two main limitations for the current API:

1. Limited support on how user can express their intention of how they want their model to be quantized.
2. Limitations around expressing quantization intentions for complicated operator patterns using existing objects: QConfig and QConfigMapping.

The new API provides a single instance with which both backend and users interact, simplifying the UX and making it less error prone.

### Define Helper Functions and Prepare Dataset

We’ll start by doing the necessary imports, defining some helper functions and prepare the data. These steps are identical to Static Quantization with Eager Mode in PyTorch.

To run the code in this tutorial using the entire ImageNet dataset, first download Imagenet by following the instructions at [here](https://pytorch.org/vision/stable/datasets.html#imagenet). Unzip the downloaded file into the data_path folder.

Download the torchvision resnet18 model and rename it to data/resnet18_pretrained_float.pth.

```python
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)

# Helper functions and data preparation code here...
```

### Set the model to eval mode

For post training quantization, we’ll need to set the model to the eval mode.

```python
model_to_quantize.eval()
```

### Export the model with torch.export

Here is how you can use torch.export to export the model:

```python
from torch._export import capture_pre_autograd_graph

example_inputs = (torch.rand(2, 3, 224, 224),)
exported_model = capture_pre_autograd_graph(model_to_quantize, example_inputs)
```

### Import the Backend Specific Quantizer and Configure how to Quantize the Model

The following code snippets describes how to quantize the model:

```python
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)

quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())
```

Quantizer is backend specific, and each Quantizer will provide their own way to allow users to configure their model. Just as an example, here is the different configuration APIs supported by XNNPackQuantizer.

### Prepare the Model for Post Training Quantization

The `prepare_pt2e` function folds BatchNorm operators into preceding Conv2d operators, and inserts observers in appropriate places in the model.

```python
prepared_model = prepare_pt2e(exported_model, quantizer)
print(prepared_model.graph)
```

### Calibration

The calibration function is run after the observers are inserted in the model. The purpose for calibration is to run through some sample examples that is representative of the workload (for example a sample of the training data set) so that the observers in themodel are able to observe the statistics of the Tensors and we can later use this information to calculate quantization parameters.

```python
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

calibrate(prepared_model, data_loader_test)  # run calibration on sample data
```

### Convert the Calibrated Model to a Quantized Model

The `convert_pt2e` function takes a calibrated model and produces a quantized model.

```python
quantized_model = convert_pt2e(prepared_model)
print(quantized_model)
```

At this step, we currently have two representations that you can choose from, but exact representation we offer in the long term might change based on feedback from PyTorch users.

### Checking Model Size and Accuracy Evaluation

Now we can compare the size and model accuracy with baseline model.

```python
# Baseline model size and accuracy
scripted_float_model_file = "resnet18_scripted.pth"

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test)
print("Baseline Float Model Evaluation accuracy: %2.2f, %2.2f"%(top1.avg, top5.avg))

# Quantized model size and accuracy
print("Size of model after quantization")
print_size_of_model(quantized_model)

top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("[before serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))
```

### Save and Load Quantized Model

We’ll show how to save and load the quantized model.

```python
# 0. Store reference output, for example, inputs, and check evaluation accuracy:
example_inputs = (next(iter(data_loader))[0],)
ref = quantized_model(*example_inputs)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("[before serialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

# 1. Export the model and Save ExportedProgram
pt2e_quantized_model_file_path = saved_model_dir + "resnet18_pt2e_quantized.pth"
# capture the model to get an ExportedProgram
quantized_ep = torch.export.export(quantized_model, example_inputs)
# use torch.export.save to save an ExportedProgram
torch.export.save(quantized_ep, pt2e_quantized_model_file_path)

# 2. Load the saved ExportedProgram
loaded_quantized_ep = torch.export.load(pt2e_quantized_model_file_path)
loaded_quantized_model = loaded_quantized_ep.module()

# 3. Check results for example inputs and check evaluation accuracy again:
res = loaded_quantized_model(*example_inputs)
print("diff:", ref - res)

top1, top5 = evaluate(loaded_quantized_model, criterion, data_loader_test)
print("[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))
```

### Debugging the Quantized Model

You can use Numeric Suite that can help with debugging in eager mode and FX graph mode. The new version of Numeric Suite working with PyTorch 2 Export models is still in development.

### Lowering and Performance Evaluation

The model produced at this point is not the final model that runs on the device, it is a reference quantized model that captures the intended quantized computation from the user, expressed as ATen operators and some additional quantize/dequantize operators, to get a model that runs on real devices, we’ll need to lower the model. For example, for the models that run on edge devices, we can lower with delegation and ExecuTorch runtime operators.

### Conclusion

In this tutorial, we went through the overall quantization flow in PyTorch 2 Export Quantization using XNNPACKQuantizer and got a quantized model that could be further lowered to a backend that supports inference with XNNPACK backend. To use this for your own backend, please first follow the tutorial and implement a Quantizer for your backend, and then quantize the model with that Quantizer.

## [PyTorch 2.0 Export Post Training Static Quantization](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_static.html)

## [PyTorch 2 Export Quantization-Aware Training (QAT)](https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html)

This tutorial shows how to perform quantization-aware training (QAT) in graph mode based on `torch.export.export`. For more details about PyTorch 2 Export Quantization in general, refer to the post training quantization tutorial.

### PyTorch 2 Export QAT Flow

The PyTorch 2 Export QAT flow looks like the following—it is similar to the post training quantization (PTQ) flow for the most part:

```python
import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
  prepare_qat_pt2e,
  convert_pt2e,
)
from torch.ao.quantization.quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)

class M(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(5, 10)

   def forward(self, x):
      return self.linear(x)

example_inputs = (torch.randn(1, 5),)
m = M()

# Step 1. program capture
m = capture_pre_autograd_graph(m, *example_inputs)

# Step 2. quantization-aware training
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
m = prepare_qat_pt2e(m, quantizer)

# train omitted

m = convert_pt2e(m)

# move the quantized model to eval mode, equivalent to `m.eval()`
torch.ao.quantization.move_exported_model_to_eval(m)
```

Note that calling `model.eval()` or `model.train()` after program capture is not allowed, because these methods no longer correctly change the behavior of certain ops like dropout and batch normalization. Instead, please use `torch.ao.quantization.move_exported_model_to_eval()` and `torch.ao.quantization.move_exported_model_to_train()` (coming soon) respectively.

### Define Helper Functions and Prepare the Dataset

To run the code in this tutorial using the entire ImageNet dataset, first download ImageNet by following the instructions in [ImageNet Data](https://pytorch.org/vision/stable/datasets.html#imagenet). Unzip the downloaded file into the data_path folder.

Next, download the torchvision resnet18 model and rename it to `data/resnet18_pretrained_float.pth`.

We’ll start by doing the necessary imports, defining some helper functions and prepare the data. These steps are very similar to the ones defined in the static eager mode post training quantization tutorial:

```python
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)

# Helper functions and data preparation code here...
```

### Export the model with `torch.export`

Here is how you can use `torch.export` to export the model:

```python
from torch._export import capture_pre_autograd_graph

example_inputs = (torch.rand(2, 3, 224, 224),)
exported_model = capture_pre_autograd_graph(float_model, example_inputs)
# or, to capture with dynamic dimensions:
from torch._export import dynamic_dim

example_inputs = (torch.rand(2, 3, 224, 224),)
exported_model = capture_pre_autograd_graph(
    float_model,
    example_inputs,
    constraints=[dynamic_dim(example_inputs[0], 0)],
)
```

Note: `capture_pre_autograd_graph` is a short term API, it will be updated to use the official `torch.export` API when that is ready.

### Import the Backend Specific Quantizer and Configure how to Quantize the Model

The following code snippets describe how to quantize the model:

```python
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config(is_qat=True))
```

Quantizer is backend specific, and each Quantizer will provide their own way to allow users to configure their model.

Note: Check out our tutorial that describes how to write a new Quantizer.

### Prepare the Model for Quantization-Aware Training

`prepare_qat_pt2e` inserts fake quantizes in appropriate places in the model and performs the appropriate QAT “fusions”, such as Conv2d + BatchNorm2d, for better training accuracies. The fused operations are represented as a subgraph of ATen ops in the prepared graph.

```python
prepared_model = prepare_qat_pt2e(exported_model, quantizer)
print(prepared_model)
```

Note: If your model contains batch normalization, the actual ATen ops you get in the graph depend on the model’s device when you export the model. If the model is on CPU, then you’ll get `torch.ops.aten._native_batch_norm_legit`. If the model is on CUDA, then you’ll get `torch.ops.aten.cudnn_batch_norm`. However, this is not fundamental and may be subject to change in the future.

Between these two ops, it has been shown that `torch.ops.aten.cudnn_batch_norm` provides better numerics on models like MobileNetV2. To get this op, either call `model.cuda()` before export, or run the following after prepare to manually swap the ops:

```python
for n in prepared_model.graph.nodes:
    if n.target == torch.ops.aten._native_batch_norm_legit.default:
        n.target = torch.ops.aten.cudnn_batch_norm.default
prepared_model.recompile()
```

In the future, we plan to consolidate the batch normalization ops such that the above will no longer be necessary.

### Training Loop

The training loop is similar to the ones in previous versions of QAT. To achieve better accuracies, you may optionally disable observers and updating batch normalization statistics after a certain number of epochs, or evaluate the QAT or the quantized model trained so far every N epochs.

### Saving and Loading Model Checkpoints

Model checkpoints for the PyTorch 2 Export QAT flow are the same as in any other training flow. They are useful for pausing training and resuming it later, recovering from failed training runs, and performing inference on different machines at a later time. You can save model checkpoints during or after training as follows:

```python
checkpoint_path = "/path/to/my/checkpoint_%s.pth" % nepoch
torch.save(prepared_model.state_dict(), "checkpoint_path")
```

To load the checkpoints, you must export and prepare the model the exact same way it was initially exported and prepared.

### Convert the Trained Model to a Quantized Model

`convert_pt2e` takes a calibrated model and produces a quantized model. Note that, before inference, you must first call `torch.ao.quantization.move_exported_model_to_eval()` to ensure certain ops like dropout behave correctly in the eval graph. Otherwise, we would continue to incorrectly apply dropout in the forward pass during inference, for example.

```python
quantized_model = convert_pt2e(prepared_model)

# move certain ops like dropout to eval mode, equivalent to `m.eval()`
torch.ao.quantization.move_exported_model_to_eval(m)

print(quantized_model)

top1, top5 = evaluate(quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Final evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))
```

### Conclusion

In this tutorial, we demonstrated how to run Quantization-Aware Training (QAT) flow in PyTorch 2 Export Quantization. After convert, the rest of the flow is the same as Post-Training Quantization (PTQ); the user can serialize/deserialize the model and further lower it to a backend that supports inference with XNNPACK backend. For more detail, follow the PTQ tutorial.

## [How to Write a Quantizer for PyTorch 2 Export Quantization](https://pytorch.org/tutorials/prototype/pt2e_quantizer.html)

## [Quantization-Aware Training for Large Language Models with PyTorch](https://pytorch.org/blog/quantization-aware-training/)
