# Registry

## Registry

OpenMMLab supports a rich collection of algorithms and datasets, which means many modules with similar functionality are implemented. For instance, the implementations of ResNet and SE-ResNet are based on the classes ResNet and SEResNet, respectively. These classes have similar functions and interfaces and belong to the model components of the algorithm library. To manage these functionally similar modules, MMEngine implements the registry. Most of the algorithm libraries in OpenMMLab use registry to manage their modules, including MMDetection, MMDetection3D, MMPretrain, and MMagic, among others.

### What is a registry

The registry in MMEngine can be considered as a union of a mapping table and a build function of modules. The mapping table maintains a mapping from strings to classes or functions, allowing the user to find the corresponding class or function with its name/notation. For example, the mapping from the string "ResNet" to the ResNet class. The module build function defines how to find the corresponding class or function based on a string and how to instantiate the class or call the function. For example, finding nn.BatchNorm2d and instantiating the BatchNorm2d module by the string "bn", or finding the build_batchnorm2d function by the string "build_batchnorm2d" and then returning the result. The registries in MMEngine use the build_from_cfg function by default to find and instantiate the class or function corresponding to the string.

The classes or functions managed by a registry usually have similar interfaces and functionality, so the registry can be treated as an abstraction of those classes or functions. For example, the registry MODELS can be treated as an abstraction of all models, which manages classes such as ResNet, SEResNet, and RegNetX and constructors such as build_ResNet, build_SEResNet, and build_RegNetX.

### Getting started

There are three steps required to use the registry to manage modules in the codebase.

1. **Create a registry.**
2. **Create a build method for instantiating the class (optional because in most cases you can just use the default method).**
3. **Add the module to the registry**

Suppose we want to implement a series of activation modules and want to be able to switch to different modules by just modifying the configuration without modifying the code.

Let’s create a registry first.

```python
from mmengine import Registry
# `scope` represents the domain of the registry. If not set, the default value is the package name.
# e.g. in mmdetection, the scope is mmdet
# `locations` indicates the location where the modules in this registry are defined.
# The Registry will automatically import the modules when building them according to these predefined locations.
ACTIVATION = Registry('activation', scope='mmengine', locations=['mmengine.models.activations'])
```

The module mmengine.models.activations specified by locations corresponds to the mmengine/models/activations.py file. When building modules with registry, the ACTIVATION registry will automatically import implemented modules from this file. Therefore, we can implement different activation layers in the mmengine/models/activations.py file, such as Sigmoid, ReLU, and Softmax.

```python
import torch.nn as nn

# use the register_module
@ACTIVATION.register_module()
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Sigmoid.forward')
        return x

@ACTIVATION.register_module()
class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        print('call ReLU.forward')
        return x

@ACTIVATION.register_module()
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Softmax.forward')
        return x
```

The key of using the registry module is to register the implemented modules into the ACTIVATION registry. With the @ACTIVATION.register_module() decorator added before the implemented module, the mapping between strings and classes or functions can be built and maintained by ACTIVATION. We can achieve the same functionality with ACTIVATION.register_module(module=ReLU) as well.

By registering, we can create a mapping between strings and classes or functions via ACTIVATION.

```python
print(ACTIVATION.module_dict)
# {
#     'Sigmoid': __main__.Sigmoid,
#     'ReLU': __main__.ReLU,
#     'Softmax': __main__.Softmax
# }
```

Note: The key to trigger the registry mechanism is to make the module imported. There are three ways to register a module into the registry:

1. Implement the module in the locations. The registry will automatically import modules in the predefined locations. This is to ease the usage of algorithm libraries so that users can directly use REGISTRY.build(cfg).
2. Import the file manually. This is common when developers implement a new module in/out side the algorithm library.
3. Use custom_imports field in config. Please refer to Importing custom Python modules for more details.

Once the implemented module is successfully registered, we can use the activation module in the configuration file.

```python
import torch

input = torch.randn(2)

act_cfg = dict(type='Sigmoid')
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# call Sigmoid.forward
print(output)
```

We can switch to ReLU by just changing this configuration.

```python
act_cfg = dict(type='ReLU', inplace=True)
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# call ReLU.forward
print(output)
```

If we want to check the type of input parameters (or any other operations) before creating an instance, we can implement a build method and pass it to the registry to implement a custom build process.

Create a build_activation function.

```python
def build_activation(cfg, registry, *args, **kwargs):
    cfg_ = cfg.copy()
    act_type = cfg_.pop('type')
    print(f'build activation: {act_type}')
    act_cls = registry.get(act_type)
    act = act_cls(*args, **kwargs, **cfg_)
    return act
```

Pass the buid_activation to build_func.

```python
ACTIVATION = Registry('activation', build_func=build_activation, scope='mmengine', locations=['mmengine.models.activations'])

@ACTIVATION.register_module()
class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Tanh.forward')
        return x

act_cfg = dict(type='Tanh')
activation = ACTIVATION.build(act_cfg)
output = activation(input)
# build activation: Tanh
# call Tanh.forward
print(output)
```

Note: In the above example, we demonstrate how to customize the method of building an instance of a class using the build_func. This is similar to the default build_from_cfg method. In most cases, using the default method will be fine.

MMEngine’s registry can register classes as well as functions.

```python
FUNCTION = Registry('function', scope='mmengine')

@FUNCTION.register_module()
def print_args(**kwargs):
    print(kwargs)

func_cfg = dict(type='print_args', a=1, b=2)
func_res = FUNCTION.build(func_cfg)
```

### Advanced usage

The registry in MMEngine supports hierarchical registration, which enables cross-project calls, meaning that modules from one project can be used in another project. Though there are other ways to implement this, the registry provides a much easier solution.

To easily make cross-library calls, MMEngine provides twenty two root registries, including:

- RUNNERS: the registry for Runner.
- RUNNER_CONSTRUCTORS: the constructors for Runner.
- LOOPS: manages training, validation and testing processes, such as EpochBasedTrainLoop.
- HOOKS: the hooks, such as CheckpointHook, and ParamSchedulerHook.
- DATASETS: the datasets.
- DATA_SAMPLERS: Sampler of DataLoader, used to sample the data.
- TRANSFORMS: various data preprocessing methods, such as Resize, and Reshape.
- MODELS: various modules of the model.
- MODEL_WRAPPERS: model wrappers for parallelizing distributed data, such as MMDistributedDataParallel.
- WEIGHT_INITIALIZERS: the tools for weight initialization.
- OPTIMIZERS: registers all Optimizers and custom Optimizers in PyTorch.
- OPTIM_WRAPPER: the wrapper for Optimizer-related operations such as OptimWrapper, and AmpOptimWrapper.
- OPTIM_WRAPPER_CONSTRUCTORS: the constructors for optimizer wrappers.
- PARAM_SCHEDULERS: various parameter schedulers, such as MultiStepLR.
- METRICS: the evaluation metrics for computing model accuracy, such as Accuracy.
- EVALUATOR: one or more evaluation metrics used to calculate the model accuracy.
- TASK_UTILS: the task-intensive components, such as AnchorGenerator, and BboxCoder.
- VISUALIZERS: the management drawing module that draws prediction boxes on images, such as DetVisualizer.
- VISBACKENDS: the backend for storing training logs, such as LocalVisBackend, and TensorboardVisBackend.
- LOG_PROCESSORS: controls the log statistics window and statistics methods, by default we use LogProcessor. You may customize LogProcessor if you have special needs.
- FUNCTIONS: registers various functions, such as collate_fn in DataLoader.
- INFERENCERS: registers inferencers of different tasks, such as DetInferencer, which is used to perform inference on the detection task.

#### Use the module of the parent node

Let’s define a RReLU module in MMEngine and register it to the MODELS root registry.

```python
import torch.nn as nn
from mmengine import Registry, MODELS

@MODELS.register_module()
class RReLU(nn.Module):
    def __init__(self, lower=0.125, upper=0.333, inplace=False):
        super().__init__()

    def forward(self, x):
        print('call RReLU.forward')
        return x
```

Now suppose there is a project called MMAlpha, which also defines a MODELS and sets its parent node to the MODELS of MMEngine, which creates a hierarchical structure.

```python
from mmengine import Registry, MODELS as MMENGINE_MODELS

MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmalpha', locations=['mmalpha.models'])
```

The following figure shows the hierarchy of MMEngine and MMAlpha.

The count_registered_modules function can be used to print the modules that have been registered to MMEngine and their hierarchy.

```python
from mmengine.registry import count_registered_modules

count_registered_modules()
```

We define a customized LogSoftmax module in MMAlpha and register it to the MODELS in MMAlpha.

```python
@MODELS.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call LogSoftmax.forward')
        return x
```

Here we use the LogSoftmax in the configuration of MMAlpha.

```python
model = MODELS.build(cfg=dict(type='LogSoftmax'))
```

We can also use the modules of the parent node MMEngine here in the MMAlpha.

```python
model = MODELS.build(cfg=dict(type='RReLU', lower=0.2))
# scope is optional
model = MODELS.build(cfg=dict(type='mmengine.RReLU'))
```

If no prefix is added, the build method will first find out if the module exists in the current node and return it if there is one. Otherwise, it will continue to look up the parent nodes or even the ancestor node until it finds the module. If the same module exists in both the current node and the parent nodes, we need to specify the scope prefix to indicate that we want to use the module of the parent nodes.

```python
import torch

input = torch.randn(2)
output = model(input)
# call RReLU.forward
print(output)
```

#### How does the parent node know about child registry?

When working in our MMAlpha it might be necessary to use the Runner class defined in MMENGINE. This class is in charge of building most of the objects. If these objects are added to the child registry (MMAlpha), how is MMEngine able to find them? It cannot, MMEngine needs to switch to the Registry from MMEngine to MMAlpha according to the scope which is defined in default_runtime.py for searching the target class.

We can also init the scope accordingly, see example below:

```python
from mmalpha.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import init_default_scope
import torch.nn as nn

@MODELS.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call LogSoftmax.forward')
        return x

# Works because we are using mmalpha registry
MODELS.build(dict(type="LogSoftmax"))

# Fails because mmengine registry does not know about stuff registered in mmalpha
MMENGINE_MODELS.build(dict(type="LogSoftmax"))

# Works because we are using mmalpha registry
init_default_scope('mmalpha')
MMENGINE_MODELS.build(dict(type="LogSoftmax"))
```

#### Use the module of a sibling node

In addition to using the module of the parent nodes, users can also call the module of a sibling node.

Suppose there is another project called MMBeta, which, like MMAlpha, defines MODELS and set its parent node to MMEngine.

```python
from mmengine import Registry, MODELS as MMENGINE_MODELS

MODELS = Registry('model', parent=MMENGINE_MODELS, scope='mmbeta')
```

The following figure shows the registry structure of MMAlpha and MMBeta.

Now we call the modules of MMAlpha in MMBeta.

```python
model = MODELS.build(cfg=dict(type='mmalpha.LogSoftmax'))
output = model(input)
# call LogSoftmax.forward
print(output)
```

Calling a module of a sibling node requires the scope prefix to be specified in type, so the above configuration requires the prefix mmalpha.

However, if you need to call several modules of a sibling node, each with a prefix, this requires a lot of modification. Therefore, MMEngine introduces the DefaultScope, with which Registry can easily support temporary switching of the current node to the specified node.

If you need to switch the current node to the specified node temporarily, just set _scope_ to the scope of the specified node in cfg.

```python
model = MODELS.build(cfg=dict(type='LogSoftmax', _scope_='mmalpha'))
output = model(input)
# call LogSoftmax.forward
print(output)
```

## Registry

### classmmengine.registry.Registry

A registry to map strings to classes or functions. Registered object could be built from registry. Meanwhile, registered functions could be called from registry.

#### Parameters:
- **name** (str): Registry name.
- **build_func** (callable, optional): A function to construct instance from Registry. Defaults to None.
- **parent** (Registry, optional): Parent registry. Defaults to None.
- **scope** (str, optional): The scope of registry. Defaults to None.
- **locations** (list): The locations to import the modules registered in this registry. Defaults to [].

#### Methods:

##### build(cfg, *args, **kwargs)
Build an instance.

###### Parameters:
- **cfg** (dict): Config dict needs to be built.

###### Returns:
The constructed object.

##### get(key)
Get the registry record.

###### Parameters:
- **key** (str): Name of the registered item.

###### Returns:
Return the corresponding class if key exists, otherwise return None.

##### import_from_location()
Import modules from the pre-defined locations in self._location.

###### Returns:
None

##### infer_scope()
Infer the scope of registry.

###### Returns:
The inferred scope name.

##### register_module(name=None, force=False, module=None)
Register a module.

###### Parameters:
- **name** (str or list of str, optional): The module name to be registered.
- **force** (bool): Whether to override an existing class with the same name. Defaults to False.
- **module** (type, optional): Module class or function to be registered. Defaults to None.

###### Returns:
The registered module.

##### split_scope_key(key)
Split scope and key.

###### Parameters:
- **key** (str): The key to split.

###### Returns:
The former element is the first scope of the key, which can be None. The latter is the remaining key.

##### switch_scope_and_registry(scope)
Temporarily switch default scope to the target scope, and get the corresponding registry.

###### Parameters:
- **scope** (str, optional): The target scope.

###### Returns:
The corresponding registry.

---

## DefaultScope

The `DefaultScope` class in `mmengine.registry` is used to manage the scope of the current task. It allows for the resetting of the current registry and can be accessed globally.

### Class Methods

#### `get_current_instance()`

This method returns the latest created default scope. If there is no `DefaultScope` created yet, it returns `None`.

#### `overwrite_default_scope(scope_name)`

This method overwrites the current default scope with the provided `scope_name`.

### Properties

#### `scope_name`

This property returns the current scope.

### Initialization

```python
classmmengine.registry.DefaultScope(name, scope_name)
```

- `name` (str): Name of default scope for global access.
- `scope_name` (str): Scope of current task.

### Examples

```python
>>> from mmengine.model import MODELS
>>> # Define default scope in runner.
>>> DefaultScope.get_instance('task', scope_name='mmdet')
>>> # Get default scope globally.
>>> scope_name = DefaultScope.get_instance('task').scope_name
```

### Notes

In cases where the internal module cannot access the runner directly, it can be difficult to get the default scope defined in the Runner. However, if the Runner creates a `DefaultScope` instance with the given default scope, the internal module can access it using `DefaultScope.get_current_instance()` anywhere.

---

## mmengine.registry.build_from_cfg

The `build_from_cfg` function in `mmengine.registry` is used to construct a module or call a function based on a configuration dictionary. This function is versatile and can be used in various scenarios.

### Function Signature

```python
mmengine.registry.build_from_cfg(cfg, registry, default_args=None)
```

### Description

This function builds a module from a config dictionary when it is a class configuration, or calls a function from a config dictionary when it is a function configuration.

If the global variable `default_scope` exists, `build_from_cfg` will first retrieve the corresponding registry and then call its own `build` method.

The `cfg` and `default_args` dictionaries should contain the key "type", which can be either a string or a class. If both `cfg` and `default_args` contain the "type" key, the value from `cfg` will be used as it has a higher priority. The remaining keys in the merged dictionary will be used as initialization arguments.

### Examples

```python
from mmengine import Registry, build_from_cfg

MODELS = Registry('models')

@MODELS.register_module()
class ResNet:
    def __init__(self, depth, stages=4):
        self.depth = depth
        self.stages = stages

cfg = dict(type='ResNet', depth=50)
model = build_from_cfg(cfg, MODELS)
# Returns an instantiated object of ResNet with depth=50 and stages=4

@MODELS.register_module()
def resnet50():
    pass

resnet = build_from_cfg(dict(type='resnet50'), MODELS)
# Returns the result of calling the resnet50 function
```

### Parameters

- `cfg` (dict or ConfigDict or Config): A configuration dictionary. It should at least contain the key "type".
- `registry` (Registry): The registry to search the type from.
- `default_args` (dict or ConfigDict or Config, optional): Default initialization arguments. Defaults to None.

### Returns

- The constructed object or the result of the called function.
- Return type: object

---

## mmengine.registry.build_model_from_cfg

The `mmengine.registry.build_model_from_cfg` function is used to build a PyTorch model from a configuration dictionary or a list of configuration dictionaries. If the input `cfg` is a list, the built modules will be wrapped with `nn.Sequential`.

### Parameters:
- `cfg` (dict, list[dict]): The configuration of modules. It can be a single configuration dictionary or a list of configuration dictionaries.
- `registry` (Registry): A registry that the module belongs to.
- `default_args` (dict, optional): Default arguments to build the module. Defaults to None.

### Returns:
- `nn.Module`: A built PyTorch module.

### Usage:
```python
model = mmengine.registry.build_model_from_cfg(cfg, registry, default_args=None)
```

---

## mmengine.registry.build_runner_from_cfg

The `build_runner_from_cfg` function is used to build a Runner object from a configuration dictionary.

### Function Signature

```python
mmengine.registry.build_runner_from_cfg(cfg, registry)
```

### Description

This function constructs a Runner object based on the provided configuration dictionary (`cfg`). If the `cfg` dictionary contains a key named "runner_type", the function will use this type to build a custom runner. If not, it will build a default runner. The `registry` parameter is used to search for the type of the runner.

### Examples

```python
from mmengine.registry import Registry, build_runner_from_cfg

RUNNERS = Registry('runners', build_func=build_runner_from_cfg)

@RUNNERS.register_module()
class CustomRunner(Runner):
    def setup_env(env_cfg):
        pass

cfg = dict(runner_type='CustomRunner', ...)
custom_runner = RUNNERS.build(cfg)
```

### Parameters

- `cfg` (dict or ConfigDict or Config): The configuration dictionary. It can be a dictionary, a ConfigDict object, or a Config object. If it contains a "runner_type" key, a custom runner of that type will be built. Otherwise, a default runner will be built.
- `registry` (Registry): The registry to search the type of the runner from.

### Returns

- The constructed runner object.

### Return Type

- object

---

## mmengine.registry.build_scheduler_from_cfg

The `mmengine.registry.build_scheduler_from_cfg` function is used to build a ParamScheduler instance from a configuration dictionary. This function supports building instances by either the constructor or the `build_iter_from_epoch` method, making it versatile for different use cases.

### Function Signature

```python
mmengine.registry.build_scheduler_from_cfg(cfg, registry, default_args=None)
```

### Parameters

- `cfg` (dict or ConfigDict or Config): The configuration dictionary. If it contains the key `convert_to_iter_based`, the instance will be built by the `convert_to_iter_based` method. Otherwise, the instance will be built by its constructor.
- `registry` (Registry): The PARAM_SCHEDULERS registry.
- `default_args` (dict or ConfigDict or Config, optional): Default initialization arguments. If `convert_to_iter_based` is defined in `cfg`, it must additionally contain the key `epoch_length`. Defaults to None.

### Returns

- The constructed ParamScheduler instance.
- Return type: object

### Usage

This function is used to build a ParamScheduler instance based on the provided configuration. It can be used as follows:

```python
scheduler = mmengine.registry.build_scheduler_from_cfg(cfg, registry, default_args)
```

---

## mmengine.registry.count_registered_modules

The `mmengine.registry.count_registered_modules` function scans all modules in MMEngine’s root and child registries and dumps the results to a JSON file.

### Parameters:
- `save_path` (str, optional): Path to save the JSON file.
- `verbose` (bool): Whether to print log. Defaults to True.

### Returns:
- Statistic results of all registered modules.

### Return Type:
- dict

---

## mmengine.registry.traverse_registry_tree

The `mmengine.registry.traverse_registry_tree` function is used to traverse the entire registry tree from a given node and collect information about all registered modules in this registry tree.

### Parameters:
- `registry` (Registry): A registry node in the registry tree.
- `verbose` (bool): Whether to print log. Defaults to True.

### Returns:
- Statistic results of all modules in each node of the registry tree.

### Return Type:
- list

---

## mmengine.registry.init_default_scope

### mmengine.registry.init_default_scope(scope)[source]

Initialize the given default scope.

#### Parameters:
- scope (str) – The name of the default scope.

#### Return type:
- None