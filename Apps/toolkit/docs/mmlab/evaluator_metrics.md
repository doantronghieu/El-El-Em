# Evaluator and Metrics

## Evaluation

In model validation and testing, it is often necessary to make a quantitative evaluation of model accuracy. We can achieve this by specifying the metrics in the configuration file.

## Evaluation in Model Training or Testing

### Using a Single Evaluation Metric

When training or testing a model based on MMEngine, users only need to specify the evaluation metrics for the validation and testing stages through the `val_evaluator` and `test_evaluator` fields in the configuration file. For example, when using MMPretrain to train a classification model, if the user wants to evaluate the top-1 and top-5 classification accuracy during the model validation stage, they can configure it as follows:

```python
# using classification accuracy evaluation metric
val_evaluator = dict(type='Accuracy', top_k=(1, 5))
```

For specific parameter settings of evaluation metrics, users can refer to the documentation of the relevant algorithm libraries, such as the Accuracy documentation in the above example.

### Using Multiple Evaluation Metrics

If multiple evaluation metrics need to be evaluated simultaneously, `val_evaluator` or `test_evaluator` can be set as a list, with each item being the configuration information for an evaluation metric. For example, when using MMDetection to train a panoptic segmentation model, if the user wants to evaluate both the object detection (COCO AP/AR) and panoptic segmentation accuracy during the model testing stage, they can configure it as follows:

```python
test_evaluator = [
    # object detection metric
    dict(
        type='CocoMetric',
        metric=['bbox', 'segm'],
        ann_file='annotations/instances_val2017.json',
    ),
    # panoramic segmentation metric
    dict(
        type='CocoPanopticMetric',
        ann_file='annotations/panoptic_val2017.json',
        seg_prefix='annotations/panoptic_val2017',
    )
]
```

### Customizing Evaluation Metrics

If the common evaluation metrics provided in the algorithm library cannot meet the needs, users can also add custom evaluation metrics. As an example, we present the implementation of custom metrics with the simplified classification accuracy:

1. When defining a new evaluation metric class, you need to inherit the base class `BaseMetric` (for an introduction to this base class, you can refer to the design document). In addition, the evaluation metric class needs to be registered with the registrar `METRICS` (for a description of the registrar, please refer to the Registry documentation).

2. Implement the `process()` method. This method has two input parameters, which are a batch of test data samples, `data_batch`, and model prediction results, `data_samples`. We extract the sample category labels and the classification prediction results from them and store them in `self.results` respectively.

3. Implement the `compute_metrics()` method. This method has one input parameter `results`, which holds the results of all batches of test data processed by the `process()` method. The sample category labels and classification predictions are extracted from the results to calculate the classification accuracy (`acc`). Finally, the calculated evaluation metrics are returned in the form of a dictionary.

4. (Optional) You can assign a value to the class attribute `default_prefix`. This attribute is automatically prefixed to the output metric name (e.g., `default_prefix='my_metric'`, then the actual output metric name is `'my_metric/acc'`) to further distinguish the different metrics. This prefix can also be rewritten in the configuration file via the `prefix` parameter. We recommend describing the `default_prefix` value for the metric class and the names of all returned metrics in the docstring.

The specific implementation is as follows:

```python
from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

import numpy as np

@METRICS.register_module()  # register the Accuracy class to the METRICS registry
class SimpleAccuracy(BaseMetric):
    """ Accuracy Evaluator

    Default prefix: ACC

    Metrics:
        - accuracy (float): classification accuracy
    """

    default_prefix = 'ACC'  # set default_prefix

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        """Process one batch of data and predictions. The processed
        results should be stored in `self.results`, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Tuple[Any, dict]]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

        # fetch classification prediction results and category labels
        result = {
            'pred': data_samples['pred_label'],
            'gt': data_samples['data_sample']['gt_label']
        }

        # store the results of the current batch into self.results
        self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        # aggregate the classification prediction results and category labels for all samples
        preds = np.concatenate([res['pred'] for res in results])
        gts = np.concatenate([res['gt'] for res in results])

        # calculate the classification accuracy
        acc = (preds == gts).sum() / preds.size

        # return evaluation metric results
        return {'accuracy': acc}
```

### Using Offline Results for Evaluation

Another common way of model evaluation is to perform offline evaluation using model prediction results saved in files in advance. In this case, the user needs to manually build `Evaluator` and call the corresponding interface of the evaluator to complete the evaluation. For more details about offline evaluation and the relationship between the evaluator and the metric, please refer to the design document. We only give an example of offline evaluation here:

```python
from mmengine.evaluator import Evaluator
from mmengine.fileio import load

# Build the evaluator. The parameter `metrics` is the configuration of the evaluation metric
evaluator = Evaluator(metrics=dict(type='Accuracy', top_k=(1, 5)))

# Reads the test data from a file. The data format needs to refer to the metric used.
data = load('test_data.pkl')

# The model prediction result is read from the file. The result is inferred by the algorithm to be evaluated on the test dataset.
# The data format needs to refer to the metric used.
data_samples = load('prediction.pkl')

# Call the evaluator offline evaluation interface and get the evaluation results
# chunk_size indicates the number of samples processed at a time, which can be adjusted according to the memory size
results = evaluator.offline_evaluate(data, data_samples, chunk_size=128)
```

---

## BaseMetric

`class mmengine.evaluator.BaseMetric(collect_device='cpu', prefix=None, collect_dir=None)[source]`

Base class for a metric.

The metric first processes each batch of `data_samples` and `predictions`, and appends the processed results to the `results` list. Then it collects all results together from all ranks if distributed training is used. Finally, it computes the metrics of the entire dataset.

A subclass of `BaseMetric` should assign a meaningful value to the class attribute `default_prefix`. See the argument `prefix` for details.

### Parameters

- **collect_device (str)**: Device name used for collecting results from different ranks during distributed training. Must be ‘cpu’ or ‘gpu’. Defaults to ‘cpu’.
- **prefix (str, optional)**: The prefix that will be added in the metric names to disambiguate homonymous metrics of different evaluators. If `prefix` is not provided in the argument, `self.default_prefix` will be used instead. Default: `None`.
- **collect_dir (str | None, optional)**: Synchronize directory for collecting data from different ranks. This argument should only be configured when `collect_device` is ‘cpu’. Defaults to `None`. New in version 0.7.3.

### Methods

#### abstract compute_metrics(results)[source]

Compute the metrics from processed results.

**Parameters:**

- **results (list)**: The processed results of each batch.

**Returns:**

The computed metrics. The keys are the names of the metrics, and the values are corresponding results.

**Return type:**

`dict`

#### property dataset_meta: dict | None

Meta info of the dataset.

**Type:**

`Optional[dict]`

#### evaluate(size)[source]

Evaluate the model performance of the whole dataset after processing all batches.

**Parameters:**

- **size (int)**: Length of the entire validation dataset. When batch size > 1, the dataloader may pad some data samples to make sure all ranks have the same length of dataset slice. The `collect_results` function will drop the padded data based on this size.

**Returns:**

Evaluation metrics dict on the validation dataset. The keys are the names of the metrics, and the values are corresponding results.

**Return type:**

`dict`

#### abstract process(data_batch, data_samples)[source]

Process one batch of data samples and predictions. The processed results should be stored in `self.results`, which will be used to compute the metrics when all batches have been processed.

**Parameters:**

- **data_batch (Any)**: A batch of data from the dataloader.
- **data_samples (Sequence[dict])**: A batch of outputs from the model.

**Return type:**

`None`

---

## Evaluation

### Evaluation Metrics and Evaluators

In model validation and testing, it is often necessary to quantitatively assess the model’s performance. In MMEngine, `Metric` and `Evaluator` are implemented to facilitate this process.

`Metric` computes specific model metrics based on test data and model predictions. Each OpenMMLab algorithm library provides common metrics for different tasks. For instance, `Accuracy` in MMPreTrain calculates the Top-k classification accuracy for classification models, while `COCOMetric` in MMDetection calculates AP, AR, and other metrics for object detection models. The evaluation metrics are decoupled from the dataset; for example, `COCOMetric` can be used with non-COCO object detection datasets as well.

`Evaluator` is a higher-level module for `Metric`, typically encompassing one or more metrics. The evaluator performs necessary data format conversions during model evaluation and invokes evaluation metrics to calculate model accuracy. Evaluators are generally constructed from `Runner` or test scripts for online and offline evaluations, respectively.

### BaseMetric

`BaseMetric` is an abstract class with the following initialization parameters:

- `collect_device`: Device name used for synchronizing results in distributed evaluation, such as 'cpu' or 'gpu'.
- `prefix`: Prefix of the metric name to distinguish multiple metrics with the same name. If not provided, the class attribute `default_prefix` is used as the prefix.

```python
class BaseMetric(metaclass=ABCMeta):

    default_prefix: Optional[str] = None

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        ...
```

`BaseMetric` includes two essential methods that must be overridden in subclasses:

- `process()`: Processes test data and model prediction results for each batch. The processed results are stored in the `self.results` list, used to compute metrics after processing all test data. This method has the following parameters:
  - `data_batch`: A sample of test data from a batch, typically directly from the dataloader.
  - `data_samples`: Corresponding model prediction results. This method does not return a value. The function interface is defined as follows:

    ```python
    @abstractmethod
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
    ```

- `compute_metrics()`: Calculates the metrics and returns them in a dictionary. This method has one parameter:
  - `results`: A list holding the results of all batches processed by `process()`. This method returns a dictionary with metric names as keys and corresponding values. The function interface is defined as follows:

    ```python
    @abstractmethod
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
    ```

In this case, `compute_metrics()` is invoked in the `evaluate()` method; it collects and aggregates intermediate processing results of different ranks during distributed testing before calculating the metrics.

Note that the content of `self.results` depends on the subclass implementation. For large amounts of test samples or model output data (e.g., semantic segmentation, image generation), storing all data in memory may be impractical. In such cases, metrics computed per batch can be stored in `self.results` and collected in `compute_metrics()`, or intermediate results can be saved in temporary files with paths stored in `self.results`, then read and processed by `compute_metrics()`.

### Model Evaluation Process

The model accuracy evaluation process generally follows these steps:

- **Online Evaluation**: Test data is typically divided into batches. Each batch is fed into the model sequentially, producing corresponding predictions. The test data and model predictions are passed to the evaluator, which calls the `process()` method of the `Metric` to handle the data and prediction results. After processing all batches, the evaluator calls the `evaluate()` method of the metrics to calculate model accuracy.

- **Offline Evaluation**: Similar to online evaluation, but pre-saved model predictions are used directly for evaluation. The evaluator provides the `offline_evaluate` interface to call the metrics for offline accuracy calculation. To avoid memory overflow from processing large amounts of data at once, offline evaluation divides test data and predictions into chunks, akin to batches in online evaluation.

### Customize Evaluation Metrics

Each OpenMMLab algorithm library includes common evaluation metrics for their respective tasks. For example, COCO metrics are provided in MMDetection, while Accuracy, F1Score, etc., are provided in MMPreTrain.

Users can also add custom metrics. For details, refer to the examples in the tutorial documentation.

---

## mmengine.evaluator

### Evaluator

#### Description
Wrapper class to compose multiple `BaseMetric` instances.

### Metric

#### BaseMetric

##### Description
Base class for a metric.

#### DumpResults

##### Description
Dump model predictions to a pickle file for offline evaluation.

### Utils

#### get_metric_value

##### Description
Get the metric value specified by an indicator, which can be either a metric name or a full name with evaluator prefix.

## Evaluator Class

### `class mmengine.evaluator.Evaluator(metrics)[source]`

#### Description
Wrapper class to compose multiple `BaseMetric` instances.

#### Parameters
- **metrics** (dict or BaseMetric or Sequence): The config of metrics.

#### Properties
- **dataset_meta**: `dict | None`  
  Meta info of the dataset.  
  **Type**: Optional[dict]

#### Methods

- **evaluate(size)[source]**  
  Invoke the evaluate method of each metric and collect the metrics dictionary.
  
  **Parameters**:  
  - **size** (int): Length of the entire validation dataset. When batch size > 1, the dataloader may pad some data samples to make sure all ranks have the same length of dataset slice. The collect_results function will drop the padded data based on this size.

  **Returns**:  
  Evaluation results of all metrics. The keys are the names of the metrics, and the values are corresponding results.  
  **Return type**: dict

- **offline_evaluate(data_samples, data=None, chunk_size=1)[source]**  
  Offline evaluate the dumped predictions on the given data.

  **Parameters**:  
  - **data_samples** (Sequence): All predictions and ground truth of the model and the validation set.  
  - **data** (Sequence, optional): All data of the validation set.  
  - **chunk_size** (int): The number of data samples and predictions to be processed in a batch.

- **process(data_samples, data_batch=None)[source]**  
  Convert `BaseDataSample` to dict and invoke the process method of each metric.

  **Parameters**:  
  - **data_samples** (Sequence[BaseDataElement]): Predictions of the model and the ground truth of the validation set.  
  - **data_batch** (Any, optional): A batch of data from the dataloader.

## BaseMetric Class

### `class mmengine.evaluator.BaseMetric(collect_device='cpu', prefix=None, collect_dir=None)[source]`

#### Description
Base class for a metric. The metric first processes each batch of data_samples and predictions, and appends the processed results to the results list. Then it collects all results together from all ranks if distributed training is used. Finally, it computes the metrics of the entire dataset.

A subclass of `BaseMetric` should assign a meaningful value to the class attribute `default_prefix`. See the argument prefix for details.

#### Parameters
- **collect_device** (str): Device name used for collecting results from different ranks during distributed training. Must be ‘cpu’ or ‘gpu’. Defaults to ‘cpu’.  
- **prefix** (str, optional): The prefix that will be added in the metric names to disambiguate homonymous metrics of different evaluators. If prefix is not provided in the argument, `self.default_prefix` will be used instead. Default: None  
- **collect_dir** (str | None): Synchronize directory for collecting data from different ranks. This argument should only be configured when collect_device is ‘cpu’. Defaults to None. New in version 0.7.3.

#### Methods

- **abstract compute_metrics(results)[source]**  
  Compute the metrics from processed results.
  
  **Parameters**:  
  - **results** (list): The processed results of each batch.

  **Returns**:  
  The computed metrics. The keys are the names of the metrics, and the values are corresponding results.  
  **Return type**: dict

- **property dataset_meta**: `dict | None`  
  Meta info of the dataset.  
  **Type**: Optional[dict]

- **evaluate(size)[source]**  
  Evaluate the model performance of the whole dataset after processing all batches.

  **Parameters**:  
  - **size** (int): Length of the entire validation dataset. When batch size > 1, the dataloader may pad some data samples to make sure all ranks have the same length of dataset slice. The collect_results function will drop the padded data based on this size.

  **Returns**:  
  Evaluation metrics dict on the val dataset. The keys are the names of the metrics, and the values are corresponding results.  
  **Return type**: dict

- **abstract process(data_batch, data_samples)[source]**  
  Process one batch of data samples and predictions. The processed results should be stored in `self.results`, which will be used to compute the metrics when all batches have been processed.

  **Parameters**:  
  - **data_batch** (Any): A batch of data from the dataloader.  
  - **data_samples** (Sequence[dict]): A batch of outputs from the model.

  **Return type**: None

## Utils

### `mmengine.evaluator.get_metric_value(indicator, metrics)[source]`

#### Description
Get the metric value specified by an indicator, which can be either a metric name or a full name with evaluator prefix.

#### Parameters
- **indicator** (str): The metric indicator, which can be the metric name (e.g. ‘AP’) or the full name with prefix (e.g. ‘COCO/AP’).  
- **metrics** (dict): The evaluation results output by the evaluator.

#### Returns
The specified metric value  
**Return type**: Any