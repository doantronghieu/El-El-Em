# Data

## Dataset and DataLoader

### Hint

If you are new to PyTorch’s `Dataset` and `DataLoader` classes, it is recommended to review the official PyTorch tutorial to become familiar with the basic concepts.

Datasets and DataLoaders are essential components in MMEngine’s training pipeline, conceptually aligned with PyTorch. Typically, a dataset defines the quantity, parsing, and pre-processing of data, while a DataLoader iteratively loads data based on parameters like `batch_size`, `shuffle`, and `num_workers`. Together, datasets and DataLoaders constitute the data source.

This tutorial will guide you through using these components in MMEngine, starting with the DataLoader and progressing to the Dataset. By the end, you will be able to:

- Master the configuration of DataLoaders in MMEngine
- Utilize existing datasets (e.g., from torchvision) via config files
- Build and use your own datasets

### Details on DataLoader

DataLoaders in MMEngine’s Runner can be configured with three arguments:

- `train_dataloader`: Used in `Runner.train()` to provide training data for models
- `val_dataloader`: Used in `Runner.val()` or in `Runner.train()` at intervals for model evaluation
- `test_dataloader`: Used in `Runner.test()` for final testing

MMEngine fully supports PyTorch’s native `DataLoader` objects, allowing you to pass pre-built DataLoaders directly to the runner. Thanks to MMEngine's Registry Mechanism, these arguments also accept dictionaries as inputs. For example:

```python
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=torchvision.datasets.CIFAR10(...),
        collate_fn=dict(type='default_collate')
    )
)
```

When passed as a dictionary, the DataLoader will be lazily built when needed.

**Note**: For more configurable arguments of the `DataLoader`, refer to the PyTorch API documentation.

**Note**: For details on the building procedure, refer to `build_dataloader`.

### Sampler and Shuffle

A key difference is the addition of a `sampler` argument in the dictionary configuration. This is required because `sampler` must be explicitly specified when using a dictionary for a DataLoader. The `shuffle` parameter is removed from the DataLoader arguments because it conflicts with the `sampler`, as noted in the PyTorch DataLoader API documentation.

**Note**: `shuffle` is a convenience notation in PyTorch. Setting `shuffle=True` automatically switches the DataLoader to use `RandomSampler`.

With the `sampler` argument, the configuration in the dictionary is nearly equivalent to:

```python
from mmengine.dataset import DefaultSampler

dataset = torchvision.datasets.CIFAR10(...)
sampler = DefaultSampler(dataset, shuffle=True)

runner = Runner(
    train_dataloader=DataLoader(
        batch_size=32,
        sampler=sampler,
        dataset=dataset,
        collate_fn=default_collate
    )
)
```

**Warning**: The equivalence holds only if you are training with a single process and no randomness argument is passed to the runner. The runner ensures correct order and proper random seed through lazy initialization techniques, which is only possible for dictionary inputs. Manually building a sampler requires extra work and is error-prone, so using a dictionary is strongly recommended to avoid potential issues.

### DefaultSampler

You might wonder about `DefaultSampler` and why it is used. `DefaultSampler` is a built-in sampler in MMEngine that simplifies the transition between distributed and non-distributed training. If you have used `DistributedDataParallel` in PyTorch, you know that changing the sampler is necessary for correctness. MMEngine handles this internally with `DefaultSampler`.

`DefaultSampler` accepts the following arguments:

- `shuffle`: Set to `True` to load data in random order
- `seed`: Random seed for shuffling the dataset; typically managed by the runner
- `round_up`: If set to `True`, behaves like `drop_last=False` in PyTorch’s DataLoader

**Note**: For more details on `DefaultSampler`, refer to its API documentation.

`DefaultSampler` covers most use cases and manages error-prone details like random seeds. Apart from `DefaultSampler`, `InfiniteSampler` might be useful for iteration-based training pipelines. For advanced needs, refer to the code for these built-in samplers to create and register your own.

```python
@DATA_SAMPLERS.register_module()
class MySampler(Sampler):
    pass

runner = Runner(
    train_dataloader=dict(
        sampler=dict(type='MySampler'),
        ...
    )
)
```

### The Obscure collate_fn

The `collate_fn` argument in PyTorch’s `DataLoader` is often overlooked, but in MMEngine, special attention is required. When using the dictionary configuration, MMEngine defaults to `pseudo_collate`, which differs significantly from PyTorch’s `default_collate`. To maintain consistency when migrating from PyTorch, you must explicitly specify the `collate_fn` in the config files.

**Note**: `pseudo_collate` is used as the default value for historical compatibility reasons. You do not need to delve deeply into it; just be aware to avoid potential errors.

MMEngine provides two built-in `collate_fn` options:

- `pseudo_collate`: Default in MMEngine, does not concatenate data through the batch index. More details are in the `pseudo_collate` API doc.
- `default_collate`: Similar to PyTorch’s `default_collate`, transfers data into tensors and concatenates them through the batch index. Details and differences from PyTorch are in the `default_collate` API doc.

To use a custom `collate_fn`, register it with the `FUNCTIONS` registry:

```python
@FUNCTIONS.register_module()
def my_collate_func(data_batch: Sequence) -> Any:
    pass

runner = Runner(
    train_dataloader=dict(
        ...
        collate_fn=dict(type='my_collate_func')
    )
)
```

### Details on Dataset

Datasets define the quantity, parsing, and pre-processing of data, and are encapsulated in a DataLoader for batch loading. MMEngine supports PyTorch’s native `Dataset` class, and with the registry mechanism, a dataset argument in a DataLoader dictionary can also be specified as a dictionary for lazy initialization.

#### Use torchvision datasets

Torchvision offers various open datasets that can be used directly in MMEngine. For example, the CIFAR10 dataset is utilized with torchvision’s built-in data transforms. To use a dataset in config files, registration is necessary, and additional registration might be needed for data transforms. Here’s an example:

```python
import torchvision.transforms as tvt
from mmengine.registry import DATASETS, TRANSFORMS
from mmengine.dataset.base_dataset import Compose

@DATASETS.register_module(name='Cifar10', force=False)
def build_torchvision_cifar10(transform=None, **kwargs):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = Compose(transform)
    return torchvision.datasets.CIFAR10(**kwargs, transform=transform)

DATA_TRANSFORMS.register_module('RandomCrop', module=tvt.RandomCrop)
DATA_TRANSFORMS.register_module('RandomHorizontalFlip', module=tvt.RandomHorizontalFlip)
DATA_TRANSFORMS.register_module('ToTensor', module=tvt.ToTensor)
DATA_TRANSFORMS.register_module('Normalize', module=tvt.Normalize)

runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=dict(type='Cifar10',
            root='data/cifar10',
            train=True,
            download=True,
            transform=[
                dict(type='RandomCrop', size=32, padding=4),
                dict(type='RandomHorizontalFlip'),
                dict(type='ToTensor'),
                dict(type='Normalize', **norm_cfg)])
    )
)
```

**Note**: This example extensively uses the registry mechanism and MMEngine’s `Compose` module. While it is possible to adapt torchvision datasets to your config files, using datasets from downstream repos like MMDet or MMPretrain is recommended for a better experience.

#### Customize Your Dataset

You can customize your own datasets similarly to PyTorch. Existing datasets from previous PyTorch projects can be adapted. For guidance on customizing datasets, refer to PyTorch’s official tutorials.

#### Use MMEngine BaseDataset

In addition to using PyTorch’s native `Dataset` class, MMEngine offers the `BaseDataset` class for customization. `BaseDataset` provides conventions for annotation files, unifying data interfaces and facilitating multi-task training. It also integrates seamlessly with MMEngine’s built-in data transforms.

`BaseDataset` is widely used in downstream OpenMMLab 2.0 projects.

---

## BaseDataset

`class mmengine.dataset.BaseDataset(ann_file='', metainfo=None, data_root='', data_prefix={'img_path': ''}, filter_cfg=None, indices=None, serialize_data=True, pipeline=[], test_mode=False, lazy_init=False, max_refetch=1000)[source]`

BaseDataset for open source projects in OpenMMLab.

### Annotation Format

The annotation format is shown as follows:

```json
{
    "metainfo": {
      "dataset_type": "test_dataset",
      "task_name": "test_task"
    },
    "data_list": [
      {
        "img_path": "test_img.jpg",
        "height": 604,
        "width": 640,
        "instances": [
          {
            "bbox": [0, 0, 10, 20],
            "bbox_label": 1,
            "mask": [[0,0],[0,10],[10,20],[20,0]],
            "extra_anns": [1,2,3]
          },
          {
            "bbox": [10, 10, 110, 120],
            "bbox_label": 2,
            "mask": [[10,10],[10,110],[110,120],[120,10]],
            "extra_anns": [4,5,6]
          }
        ]
      }
    ]
}
```

### Parameters

- **ann_file (str, optional)** – Annotation file path. Defaults to ‘’.
- **metainfo (Mapping or Config, optional)** – Meta information for dataset, such as class information. Defaults to None.
- **data_root (str, optional)** – The root directory for data_prefix and ann_file. Defaults to ‘’.
- **data_prefix (dict)** – Prefix for training data. Defaults to `dict(img_path='')`.
- **filter_cfg (dict, optional)** – Config for filter data. Defaults to None.
- **indices (int or Sequence[int], optional)** – Support using first few data in annotation file to facilitate training/testing on a smaller dataset.
- **serialize_data (bool, optional)** – Whether to hold memory using serialized objects. When enabled, data loader workers can use shared RAM from the master process instead of making a copy. Defaults to True.
- **pipeline (list, optional)** – Processing pipeline. Defaults to [].
- **test_mode (bool, optional)** – `test_mode=True` means in test phase. Defaults to False.
- **lazy_init (bool, optional)** – Whether to load annotation during instantiation. In some cases, such as visualization, only the meta information of the dataset is needed, which is not necessary to load the annotation file. BaseDataset can skip loading annotations to save time by setting `lazy_init=True`. Defaults to False.
- **max_refetch (int, optional)** – If BaseDataset.prepare_data gets a None image, the maximum extra number of cycles to get a valid image. Defaults to 1000.

### Notes

- BaseDataset collects meta information from annotation file (the lowest priority), `BaseDataset.METAINFO` (medium), and `metainfo` parameter (highest) passed to constructors. The lower priority meta information will be overwritten by higher ones.
- Dataset wrappers such as `ConcatDataset`, `RepeatDataset`, etc., should not inherit from BaseDataset since `get_subset` and `get_subset_` could produce ambiguous sub-datasets which conflict with the original dataset.

### Examples

```python
# Assume the annotation file is given above.
class CustomDataset(BaseDataset):
    METAINFO: dict = dict(task_name='custom_task', dataset_type='custom_type')

metainfo = dict(task_name='custom_task_name')
custom_dataset = CustomDataset('path/to/ann_file', metainfo=metainfo)
# Meta information of annotation file will be overwritten by
# `CustomDataset.METAINFO`. The merged meta information will
# further be overwritten by argument `metainfo`.
print(custom_dataset.metainfo)
# Output: {'task_name': 'custom_task_name', 'dataset_type': 'custom_type'}
```

### Methods

#### `filter_data()[source]`

Filter annotations according to `filter_cfg`. Defaults return all `data_list`.

If some `data_list` could be filtered according to specific logic, the subclass should override this method.

**Returns:** Filtered results.

**Return type:** `list[int]`

#### `full_init()[source]`

Load annotation file and set `BaseDataset._fully_initialized` to True.

If `lazy_init=False`, `full_init` will be called during the instantiation, and `self._fully_initialized` will be set to True. If `obj._fully_initialized=False`, the class method decorated by `force_full_init` will call `full_init` automatically.

**Steps to Initialize Annotation:**

1. **load_data_list**: Load annotations from annotation file.
2. **filter data information**: Filter annotations according to `filter_cfg`.
3. **slice_data**: Slice dataset according to `self._indices`.
4. **serialize_data**: Serialize `self.data_list` if `self.serialize_data` is True.

#### `get_cat_ids(idx)[source]`

Get category ids by index. Dataset wrapped by `ClassBalancedDataset` must implement this method.

**Parameters:**

- `idx (int)` – The index of data.

**Returns:** All categories in the image of specified index.

**Return type:** `list[int]`

#### `get_data_info(idx)[source]`

Get annotation by index and automatically call `full_init` if the dataset has not been fully initialized.

**Parameters:**

- `idx (int)` – The index of data.

**Returns:** The `idx`-th annotation of the dataset.

**Return type:** `dict`

#### `get_subset(indices)[source]`

Return a subset of dataset.

This method will return a subset of the original dataset. If the type of indices is int, `get_subset_` will return a subdataset which contains the first or last few data information according to indices being positive or negative. If the type of indices is a sequence of int, the subdataset will extract the information according to the index given in indices.

**Examples:**

```python
dataset = BaseDataset('path/to/ann_file')
print(len(dataset))  # Output: 100
subdataset = dataset.get_subset(90)
print(len(subdataset))  # Output: 90
# if type of indices is list, extract the corresponding
# index data information
subdataset = dataset.get_subset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(len(subdataset))  # Output: 10
subdataset = dataset.get_subset(-3)
print(len(subdataset))  # Output: 3
```

**Parameters:**

- `indices (int or Sequence[int])` – If type of indices is int, indices represents the first or last few data of dataset according to indices being positive or negative. If type of indices is Sequence, indices represents the target data information index of dataset.

**Returns:** A subset of dataset.

**Return type:** `BaseDataset`

#### `get_subset_(indices)[source]`

The in-place version of `get_subset` to convert dataset to a subset of original dataset.

This method will convert the original dataset to a subset of the dataset. If the type of indices is int, `get_subset_` will return a subdataset which contains the first or last few data information according to indices being positive or negative. If the type of indices is a sequence of int, the subdataset will extract the data information according to the index given in indices.

**Examples:**

```python
dataset = BaseDataset('path/to/ann_file')
print(len(dataset))  # Output: 100
dataset.get_subset_(90)
print(len(dataset))  # Output: 90
# if type of indices is sequence, extract the corresponding
# index data information
dataset.get_subset_([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(len(dataset))  # Output: 10
dataset.get_subset_(-3)
print(len(dataset))  # Output: 3
```

**Parameters:**

- `indices (int or Sequence[int])` – If type of indices is int, indices represents the first or last few data of dataset according to indices being positive or negative. If type of indices is Sequence, indices represents the target data information index of dataset.

**Return type:** None

#### `load_data_list()[source]`

Load annotations from an annotation file named as `self.ann_file`.

If the annotation file does not follow OpenMMLab 2.0 format dataset, the subclass must override this method to load annotations. The meta information of the annotation file will be overwritten by `METAINFO` and `metainfo` arguments of the constructor.

**Returns:** A list of annotation.

**Return type:** `list[dict]`

#### `property metainfo: dict`

Get meta information of dataset.

**Returns:** Meta information collected from `BaseDataset.METAINFO`, annotation file, and `metainfo` argument during instantiation.

**Return type:** `dict`

#### `parse_data_info(raw_data_info)[source]`

Parse raw annotation to target format.

This method should return `dict` or `list` of `dict`. Each `dict` or `list` contains the data information of a training sample. If the protocol of the sample annotations is changed, this function can be overridden to update the parsing logic while keeping compatibility.

**Parameters:**

- `raw_data_info (dict)` – Raw data information loaded from `ann_file`.

**Returns:** Parsed annotation.

**Return type:** `list` or `list[dict

]`

#### `prepare_data(idx)[source]`

Get data processed by `self.pipeline`.

**Parameters:**

- `idx (int)` – The index of `data_info`.

**Returns:** Depends on `self.pipeline`.

**Return type:** Any

---

## BaseDataset

### Introduction

The Dataset class in the algorithm toolbox is responsible for providing input data for the model during the training/testing process. Each algorithm toolbox under OpenMMLab projects features a Dataset class with common characteristics and requirements, such as efficient internal data storage format, support for concatenation of different datasets, and dataset repeated sampling.

MMEngine implements `BaseDataset`, which offers basic interfaces and implements `DatasetWrappers` with the same interfaces. Most of the Dataset Classes in the OpenMMLab algorithm toolbox adhere to the `BaseDataset` interface and utilize the same `DatasetWrappers`.

The core function of `BaseDataset` is to load dataset information, categorized into meta information and data information. Meta information represents details related to the dataset itself, such as category information in image classification tasks. Data information includes file paths and corresponding label information for specific data. Additionally, `BaseDataset` continuously sends data into the data pipeline for preprocessing.

### The Standard Data Annotation File

To standardize dataset interfaces for various tasks and facilitate multi-task training in one model, OpenMMLab has defined the OpenMMLab 2.0 dataset format specification. Annotation files should conform to this specification, which `BaseDataset` reads and parses. If a provided data annotation file does not conform, users can convert it to the specified format for use with OpenMMLab’s algorithm toolbox.

The OpenMMLab 2.0 dataset format specifies that annotation files must be in JSON, YAML, YML, or Pickle formats. The dictionary in the annotation file must contain two fields: `metainfo` and `data_list`. `Metainfo` includes meta information about the dataset, while `data_list` is a list of dictionaries, each defining a raw data info, which contains one or more training/test samples.

Example JSON annotation file:
```json
{
    "metainfo": {
        "classes": ["cat", "dog"]
    },
    "data_list": [
        {
            "img_path": "xxx/xxx_0.jpg",
            "img_label": 0
        },
        {
            "img_path": "xxx/xxx_1.jpg",
            "img_label": 1
        }
    ]
}
```
Assumed data storage path:
```
data
├── annotations
│   ├── train.json
├── train
│   ├── xxx/xxx_0.jpg
│   ├── xxx/xxx_1.jpg
│   ├── ...
```

### The Initialization Process of BaseDataset

The initialization of `BaseDataset` involves several steps:

1. **Load Metainfo**: Obtain meta information from:
   - The `metainfo` dictionary passed by the user in `__init__()`.
   - The `BaseDataset.METAINFO` class attribute.
   - The `metainfo` in the annotation file.

   Priority: User-provided > Class attribute > Annotation file.

2. **Join Path**: Process paths for data info and annotation files.

3. **Build Pipeline**: Construct data pipeline for preprocessing and preparation.

4. **Full Init**: Fully initialize `BaseDataset`, including:
   - **Load Data List**: Read and parse annotation files using `parse_data_info()`.
   - **Filter Data (Optional)**: Filter out unnecessary data based on `filter_cfg`.
   - **Get Subset (Optional)**: Sample a subset of the dataset.
   - **Serialize Data (Optional)**: Serialize data samples to save memory.

   The `parse_data_info()` method processes raw data info into training/test samples. Users must implement this method to customize dataset classes.

### The Interface of BaseDataset

Once initialized, `BaseDataset` supports the following methods:

- **metainfo**: Returns meta information as a dictionary.
- **get_data_info(idx)**: Returns full data information for the given index.
- **__getitem__(idx)**: Returns the result of the data pipeline for the given index.
- **__len__()**: Returns the length of the dataset.
- **get_subset_(indices)**: Modifies the dataset class in place according to indices.
- **get_subset(indices)**: Returns a new sub-dataset class based on indices.

### Customize Dataset Class Based on BaseDataset

To customize a dataset class, understand the `BaseDataset` initialization and interface:

```python
import os.path as osp
from mmengine.dataset import BaseDataset

class ToyDataset(BaseDataset):
    def parse_data_info(self, raw_data_info):
        data_info = raw_data_info
        img_prefix = self.data_prefix.get('img_path', None)
        if img_prefix is not None:
            data_info['img_path'] = osp.join(img_prefix, data_info['img_path'])
        return data_info
```

### Using Customized Dataset Class

Instantiate the customized dataset and use it as follows:

```python
class LoadImage:
    def __call__(self, results):
        results['img'] = cv2.imread(results['img_path'])
        return results

class ParseImage:
    def __call__(self, results):
        results['img_shape'] = results['img'].shape
        return results

pipeline = [
    LoadImage(),
    ParseImage(),
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img_path='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline
)

# Access data
print(toy_dataset.metainfo)  # {'classes': ('cat', 'dog')}
print(toy_dataset.get_data_info(0))  # {'img_path': "data/train/xxx/xxx_0.jpg", 'img_label': 0}
print(len(toy_dataset))  # 2
print(toy_dataset[0])  # {'img_path': "data/train/xxx/xxx_0.jpg", 'img_label': 0, 'img': ndarray, 'img_shape': (H, W, 3)}
```

### Customize Dataset for Videos

For video datasets, ensure `parse_data_info()` returns a list of dictionaries:

```python
from mmengine.dataset import BaseDataset

class ToyVideoDataset(BaseDataset):
    def parse_data_info(self, raw_data_info):
        data_list = []
        for ...:
            data_info = {}
            ...
            data_list.append(data_info)
        return data_list
```

### Annotation Files Not Meeting OpenMMLab 2.0 Specification

- **Convert Annotation Files**: Convert to the required format and use `BaseDataset`.
- **Implement New Dataset Class**: Inherit from `BaseDataset` and override `load_data_list(self)` to handle non-conforming files.

### Other Features of BaseDataset

- **Lazy Init**: To speed up instantiation, use `lazy_init=True`. Initialize fully later as needed.

  ```python
  toy_dataset = ToyDataset(
      data_root='data/',
      data_prefix=dict(img_path='train/'),
      ann_file='annotations/train.json',
      pipeline=pipeline,
      lazy_init=True
  )
  ```

- **Save Memory**: By default, data_list is serialized to save memory. Control serialization with the `serialize_data` argument.

  ```python
  toy_dataset = ToyDataset(
      data_root='data/',
      data_prefix=dict(img_path='train/'),
      ann_file='annotations/train.json',
      pipeline=pipeline,
      serialize_data=False
  )
  ```

### DatasetWrappers

MMEngine provides several `DatasetWrappers`:

- **ConcatDataset**: Concatenates datasets.

  ```python
  from mmengine.dataset import ConcatDataset

  toy_dataset_1 = ToyDataset(...)
  toy_dataset_2 = ToyDataset(...)
  toy_dataset_12 = ConcatDataset(datasets=[toy_dataset_1, toy_dataset_2])
  ```

- **RepeatDataset**: Repeats a dataset.

  ```python
  from mmengine.dataset import RepeatDataset

  toy_dataset = ToyDataset(...)
  toy_dataset_repeat = RepeatDataset(dataset=toy_dataset, times=5)
  ```

- **ClassBalancedDataset**: Repeatedly samples based on category frequency.

  ```python
  from mmengine.dataset import ClassBalancedDataset

  class ToyDataset(BaseDataset):
      def get_cat_ids(self, idx):
          data_info = self.get_data_info(idx)
          return [int(data_info['img_label'])]

  toy_dataset = ToyDataset(...)
  toy_dataset_repeat = ClassBalancedDataset(dataset=toy_dataset, oversample_thr=1e-3)
  ```

### Customize DatasetWrapper

Customize a `DatasetWrapper` while supporting lazy initialization:

```python
from mmengine.dataset import BaseDataset
from mmengine.registry import DATASETS

@DATASETS.register_module()
class ExampleDatasetWrapper:
    def __init__(self, dataset, lazy_init=False, ...):
        ...
        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    def full_init(self):
        ...
        self._fully_initialized = True

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int):
        ...

    @force_full_init
    def get_data_info(self, idx):
        ...

    def __getitem__(self, idx):
        ...

    @force_full_init
    def __len__(self):
        ...

    @property
    def metainfo(self):
        return copy.deepcopy(self._metainfo)
```

---

## mmengine.dataset

### Dataset

**BaseDataset**  
BaseDataset for open source projects in OpenMMLab.

**Compose**  
Compose multiple transforms sequentially.

### Dataset Wrapper

**ClassBalancedDataset**  
A wrapper of class balanced dataset.

**ConcatDataset**  
A wrapper of concatenated dataset.

**RepeatDataset**  
A wrapper of repeated dataset.

### Sampler

**DefaultSampler**  
The default data sampler for both distributed and non-distributed environment.

**InfiniteSampler**  
Designed for iteration-based runners and yields mini-batch indices each time.

### Utils

**default_collate**  
Convert a list of data sampled from the dataset into a batch of data, with types consistent with each data item in the batch.

**pseudo_collate**  
Convert a list of data sampled from the dataset into a batch of data, with types consistent with each data item in the batch.

**worker_init_fn**  
This function will be called on each worker subprocess after seeding and before data loading.

---

## BaseDataElement

### Description

`class mmengine.structures.BaseDataElement(*, metainfo=None, **kwargs)[source]`

A base data interface that supports Tensor-like and dict-like operations.

A typical data element refers to predicted results or ground truth labels on a task, such as predicted bboxes, instance masks, semantic segmentation masks, etc. Because ground truth labels and predicted results often have similar properties (for example, the predicted bboxes and the ground truth bboxes), MMEngine uses the same abstract data interface to encapsulate predicted results and ground truth labels. It is recommended to use different name conventions to distinguish them, such as using `gt_instances` and `pred_instances` to differentiate between labels and predicted results. Additionally, data elements are distinguished at the instance level, pixel level, and label level, each having its own characteristics. Therefore, MMEngine defines the base class `BaseDataElement` and implements `InstanceData`, `PixelData`, and `LabelData` inheriting from `BaseDataElement` to represent different types of ground truth labels or predictions.

Another common data element is sample data, which consists of input data (such as an image) and its annotations and predictions. An image can have multiple types of annotations and/or predictions simultaneously (e.g., pixel-level semantic segmentation annotations and instance-level detection bboxes annotations). All labels and predictions of a training sample are often passed between Dataset, Model, Visualizer, and Evaluator components. To simplify the interface between components, these can be treated as a large data element and encapsulated. Such data elements are generally called `XXDataSample` in the OpenMMLab. Similar to `nn.Module`, `BaseDataElement` allows `BaseDataElement` as its attribute, encapsulating all the data of a sample in the algorithm library. For example, MMDetection uses `BaseDataElement` to encapsulate all the data elements of sample labeling and prediction.

### Attributes

The attributes in `BaseDataElement` are divided into two parts:

- **metainfo**: Contains information about the image, such as filename, image_shape, pad_shape, etc. The attributes can be accessed or modified by dict-like or object-like operations, such as `.`, `in`, `del`, `pop(str)`, `get(str)`, `metainfo_keys()`, `metainfo_values()`, `metainfo_items()`, and `set_metainfo()`.

- **data**: Stores annotations or model predictions. The attributes can be accessed or modified by dict-like or object-like operations, such as `.`, `in`, `del`, `pop(str)`, `get(str)`, `keys()`, `values()`, and `items()`. Users can also apply tensor-like methods to all `torch.Tensor` in the data fields, such as `.cuda()`, `.cpu()`, `.numpy()`, `.to()`, `to_tensor()`, and `.detach()`.

### Parameters

- **metainfo** (dict, optional): A dictionary containing the meta information of a single image, such as `dict(img_shape=(512, 512, 3), scale_factor=(1, 1, 1, 1))`. Defaults to `None`.
- **kwargs** (dict, optional): A dictionary containing annotations of a single image or model predictions. Defaults to `None`.

### Examples

```python
>>> import torch
>>> from mmengine.structures import BaseDataElement
>>> gt_instances = BaseDataElement()
>>> bboxes = torch.rand((5, 4))
>>> scores = torch.rand((5,))
>>> img_id = 0
>>> img_shape = (800, 1333)
>>> gt_instances = BaseDataElement(
...     metainfo=dict(img_id=img_id, img_shape=img_shape),
...     bboxes=bboxes, scores=scores)
>>> gt_instances = BaseDataElement(
...     metainfo=dict(img_id=img_id, img_shape=(640, 640)))
>>> # new
>>> gt_instances1 = gt_instances.new(
...     metainfo=dict(img_id=1, img_shape=(640, 640)),
...                   bboxes=torch.rand((5, 4)),
...                   scores=torch.rand((5,)))
>>> gt_instances2 = gt_instances1.new()
>>> # add and process property
>>> gt_instances = BaseDataElement()
>>> gt_instances.set_metainfo(dict(img_id=9, img_shape=(100, 100)))
>>> assert 'img_shape' in gt_instances.metainfo_keys()
>>> assert 'img_shape' in gt_instances
>>> assert 'img_shape' not in gt_instances.keys()
>>> assert 'img_shape' in gt_instances.all_keys()
>>> print(gt_instances.img_shape)
(100, 100)
>>> gt_instances.scores = torch.rand((5,))
>>> assert 'scores' in gt_instances.keys()
>>> assert 'scores' in gt_instances
>>> assert 'scores' in gt_instances.all_keys()
>>> assert 'scores' not in gt_instances.metainfo_keys()
>>> print(gt_instances.scores)
tensor([0.5230, 0.7885, 0.2426, 0.3911, 0.4876])
>>> gt_instances.bboxes = torch.rand((5, 4))
>>> assert 'bboxes' in gt_instances.keys()
>>> assert 'bboxes' in gt_instances
>>> assert 'bboxes' in gt_instances.all_keys()
>>> assert 'bboxes' not in gt_instances.metainfo_keys()
>>> print(gt_instances.bboxes)
tensor([[0.0900, 0.0424, 0.1755, 0.4469],
        [0.8648, 0.0592, 0.3484, 0.0913],
        [0.5808, 0.1909, 0.6165, 0.7088],
        [0.5490, 0.4209, 0.9416, 0.2374],
        [0.3652, 0.1218, 0.8805, 0.7523]])
>>> # delete and change property
>>> gt_instances = BaseDataElement(
...     metainfo=dict(img_id=0, img_shape=(640, 640)),
...     bboxes=torch.rand((6, 4)), scores=torch.rand((6,)))
>>> gt_instances.set_metainfo(dict(img_shape=(1280, 1280)))
>>> gt_instances.img_shape  # (1280, 1280)
>>> gt_instances.bboxes = gt_instances.bboxes * 2
>>> gt_instances.get('img_shape', None)  # (1280, 1280)
>>> gt_instances.get('bboxes', None)  # 6x4 tensor
>>> del gt_instances.img_shape
>>> del gt_instances.bboxes
>>> assert 'img_shape' not in gt_instances
>>> assert 'bboxes' not in gt_instances
>>> gt_instances.pop('img_shape', None)  # None
>>> gt_instances.pop('bboxes', None)  # None
>>> # Tensor-like
>>> cuda_instances = gt_instances.cuda()
>>> cuda_instances = gt_instances.to('cuda:0')
>>> cpu_instances = cuda_instances.cpu()
>>> cpu_instances = cuda_instances.to('cpu')
>>> fp16_instances = cuda_instances.to(
...     device=None, dtype=torch.float16, non_blocking=False,
...     copy=False, memory_format=torch.preserve_format)
>>> cpu_instances = cuda_instances.detach()
>>> np_instances = cpu_instances.numpy()
>>> # print
>>> metainfo = dict(img_shape=(800, 1196, 3))
>>> gt_instances = BaseDataElement(
...     metainfo=metainfo, det_labels=torch.LongTensor([0, 1, 2, 3]))
>>> sample = BaseDataElement(metainfo=metainfo,
...                          gt_instances=gt_instances)
>>> print(sample)
<BaseDataElement(
    META INFORMATION
    img_shape: (800, 1196, 3)
    DATA FIELDS
    gt_instances: <BaseDataElement(
            META INFORMATION
            img_shape: (800, 1196, 3)
            DATA FIELDS
            det_labels: tensor([0, 1, 2, 3])
        ) at 0x7f0ec5eadc70>
) at 0x7f0fea49e130>
>>> # inheritance
>>> class DetDataSample(BaseDataElement):
...     @property
...     def proposals(self):
...         return self._proposals
...     @proposals.setter
...     def proposals(self, value):
...         self.set_field(value, '_proposals', dtype=BaseDataElement)
...     @proposals.deleter
...     def proposals(self):
...         del self._proposals
...     @property
...     def gt_instances(self):
...         return self._gt_instances
...     @gt_instances.setter
...     def gt_instances(self, value):
...         self.set_field(value, '_gt_instances',
...                        dtype=BaseDataElement)
...     @gt_instances.deleter
...     def gt_instances(self):
...         del self._gt_instances
...     @property
...     def pred_instances(self):
...         return self._pred_instances
...     @pred_instances.setter
...     def pred_instances(self, value):
...         self.set_field(value, '_pred_instances',
...                        dtype=BaseDataElement)
...     @pred_instances.deleter
...     def pred_instances(self):
...         del self._pred_instances
>>> det_sample = DetDataSample()
>>> proposals = BaseDataElement(bboxes=torch.rand((5, 4)))
>>> det_sample.proposals = proposals
>>> assert 'proposals' in det_sample
>>> assert det_sample.proposals == proposals
>>> del det_sample.proposals
>>> assert 'proposals'

 not in det_sample
>>> with self.assertRaises(AssertionError):
...     det_sample.proposals = torch.rand((5, 4))
```

### Methods

- **all_items()**
  - Returns an iterator object whose elements are `(key, value)` tuple pairs for metainfo and data.
  - Return type: `iterator`

- **all_keys()**
  - Returns all keys in metainfo and data.
  - Return type: `list`

- **all_values()**
  - Returns all values in metainfo and data.
  - Return type: `list`

- **clone()**
  - Deep copies the current data element.
  - Return type: `BaseDataElement`

- **cpu()**
  - Converts all tensors to CPU in data.
  - Return type: `BaseDataElement`

- **cuda()**
  - Converts all tensors to GPU in data.
  - Return type: `BaseDataElement`

- **detach()**
  - Detaches all tensors in data.
  - Return type: `BaseDataElement`

- **get(key, default=None)**
  - Gets property in data and metainfo similarly to Python's `dict.get()`.
  - Return type: `Any`

- **items()**
  - Returns an iterator object whose elements are `(key, value)` tuple pairs for data.
  - Return type: `iterator`

- **keys()**
  - Returns all keys in data_fields.
  - Return type: `list`

- **metainfo** (property)
  - A dictionary containing metainfo of the current data element.
  - Type: `dict`

- **metainfo_items()**
  - Returns an iterator object whose elements are `(key, value)` tuple pairs for metainfo.
  - Return type: `iterator`

- **metainfo_keys()**
  - Returns all keys in metainfo_fields.
  - Return type: `list`

- **metainfo_values()**
  - Returns all values in metainfo.
  - Return type: `list`

- **mlu()**
  - Converts all tensors to MLU in data.
  - Return type: `BaseDataElement`

- **musa()**
  - Converts all tensors to MUSA in data.
  - Return type: `BaseDataElement`

- **new(*, metainfo=None, **kwargs)**
  - Returns a new data element of the same type. If `metainfo` and `data` are `None`, the new data element will have the same metainfo and data. If `metainfo` or `data` is not `None`, the new result will overwrite it with the input value.
  - Parameters:
    - **metainfo** (dict, optional): A dictionary containing the meta information of image, such as `img_shape`, `scale_factor`, etc. Defaults to `None`.
    - **kwargs** (dict): A dictionary containing annotations of image or model predictions.
  - Return type: `BaseDataElement`

- **npu()**
  - Converts all tensors to NPU in data.
  - Return type: `BaseDataElement`

- **numpy()**
  - Converts all tensors to `np.ndarray` in data.
  - Return type: `BaseDataElement`

- **pop(*args)**
  - Pops property in data and metainfo similarly to Python's `dict.pop()`.
  - Return type: `Any`

- **set_data(data)**
  - Sets or changes key-value pairs in data_field by parameter `data`.
  - Parameters:
    - **data** (dict): A dictionary containing annotations of image or model predictions.
  - Return type: `None`

- **set_field(value, name, dtype=None, field_type='data')**
  - Special method for set union field, used as `property.setter` functions.
  - Parameters:
    - **value** (Any)
    - **name** (str)
    - **dtype** (Type | Tuple[Type, ...] | None)
    - **field_type** (str)
  - Return type: `None`

- **set_metainfo(metainfo)**
  - Sets or changes key-value pairs in `metainfo_field` by parameter `metainfo`.
  - Parameters:
    - **metainfo** (dict): A dictionary containing the meta information of image, such as `img_shape`, `scale_factor`, etc.
  - Return type: `None`

- **to(*args, **kwargs)**
  - Applies the same name function to all tensors in data_fields.
  - Return type: `BaseDataElement`

- **to_dict()**
  - Converts `BaseDataElement` to a dictionary.
  - Return type: `dict`

- **to_tensor()**
  - Converts all `np.ndarray` to tensor in data.
  - Return type: `BaseDataElement`

- **update(instance)**
  - Updates the `BaseDataElement` with the elements from another `BaseDataElement` object.
  - Parameters:
    - **instance** (BaseDataElement): Another `BaseDataElement` object for updating the current object.
  - Return type: `None`

- **values()**
  - Returns all values in data.
  - Return type: `list`

---

## Data Transform

In the OpenMMLab repositories, dataset construction and data preparation are decoupled from each other. Usually, the dataset construction only parses the dataset and records the basic information of each sample, while the data preparation is performed by a series of data transforms, such as data loading, preprocessing, and formatting based on the basic information of the samples.

## To Use Data Transforms

In MMEngine, we use various callable data transforms classes to perform data manipulation. These data transformation classes can accept several configuration parameters for instantiation and then process the input data dictionary by calling. Also, all data transforms accept a dictionary as input and output the processed data as a dictionary. A simple example is as follows:

**Note:** In MMEngine, we don’t have the implementations of data transforms. You can find the base data transform class and many other data transforms in MMCV. So you need to install MMCV before learning this tutorial. See the [MMCV installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

```python
import numpy as np
from mmcv.transforms import Resize

transform = Resize(scale=(224, 224))
data_dict = {'img': np.random.rand(256, 256, 3)}
data_dict = transform(data_dict)
print(data_dict['img'].shape)
```

## To Use in Config Files

In config files, we can compose multiple data transforms as a list, called a data pipeline. The data pipeline is an argument of the dataset.

Usually, a data pipeline consists of the following parts:

1. **Data loading:** Use `LoadImageFromFile` to load image files.
2. **Label loading:** Use `LoadAnnotations` to load the bboxes, semantic segmentation, and keypoint annotations.
3. **Data processing and augmentation:** For example, `RandomResize`.
4. **Data formatting:** Different data transforms are used for different tasks. For example, the data formatting transform for image classification tasks is `PackClsInputs`, and it’s in MMPretrain.

Here, taking the classification task as an example, we show a typical data pipeline in the figure below. For each sample, the basic information stored in the dataset is a dictionary as shown on the far left side of the figure. Each blue block represents a data transform, and in every data transform, we add some new fields (marked in green) or update some existing fields (marked in orange) in the data dictionary.

To use the above data pipeline in our config file, use the below settings:

```python
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=256, keep_ratio=True),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs'),
        ]
    )
)
```

## Common Data Transforms

According to their functionality, data transform classes can be divided into data loading, data pre-processing & augmentation, and data formatting.

### Data Loading

To support loading large-scale datasets, usually, we won’t load all dense data during dataset construction, but only load the file path of these data. Therefore, we need to load these data in the data pipeline.

| Data Transforms         | Functionality                              |
|-------------------------|--------------------------------------------|
| `LoadImageFromFile`     | Load images according to the path.         |
| `LoadAnnotations`       | Load and format annotations information, including bbox, segmentation map, and others. |

### Data Pre-processing & Augmentation

Data transforms for pre-processing and augmentation usually manipulate the image and annotation data, such as cropping, padding, resizing, and others.

| Data Transforms         | Functionality                              |
|-------------------------|--------------------------------------------|
| `Pad`                   | Pad the margin of images.                  |
| `CenterCrop`            | Crop the image and keep the center part.   |
| `Normalize`             | Normalize the image pixels.                |
| `Resize`                | Resize images to the specified scale or ratio. |
| `RandomResize`          | Resize images to a random scale in the specified range. |
| `RandomChoiceResize`    | Resize images to a random scale from several specified scales. |
| `RandomGrayscale`       | Randomly grayscale images.                 |
| `RandomFlip`            | Randomly flip images.                     |

### Data Formatting

Data formatting transforms will convert the data to some specified type.

| Data Transforms         | Functionality                              |
|-------------------------|--------------------------------------------|
| `ToTensor`              | Convert the data of the specified field to `torch.Tensor`. |
| `ImageToTensor`         | Convert images to `torch.Tensor` in PyTorch format. |

## Custom Data Transform Classes

To implement a new data transform class, the class needs to inherit `BaseTransform` and implement the `transform` method. Here, we use a simple flip transform (`MyFlip`) as an example:

```python
import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class MyFlip(BaseTransform):
    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = mmcv.imflip(img, direction=self.direction)
        return results
```

Then, we can instantiate a `MyFlip` object and use it to process our data dictionary:

```python
import numpy as np

transform = MyFlip(direction='horizontal')
data_dict = {'img': np.random.rand(224, 224, 3)}
data_dict = transform(data_dict)
processed_img = data_dict['img']
```

Or, use it in the data pipeline by modifying our config file:

```python
pipeline = [
    ...
    dict(type='MyFlip', direction='horizontal'),
    ...
]
```

Please note that to use the class in our config file, we need to confirm the `MyFlip` class will be imported during runtime.

---