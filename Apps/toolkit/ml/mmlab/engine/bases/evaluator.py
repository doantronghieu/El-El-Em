from typing import Any, Dict, List, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmengine.evaluator import Evaluator

@METRICS.register_module()
class MyBaseMetric(BaseMetric):
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)

    def process(self, data_batch: Any, data_samples: Sequence[Dict[str, Any]]) -> None:
        """Process one batch of data and predictions.
        
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples: A batch of outputs from the model.
        """
        raise NotImplementedError

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results: The processed results of each batch.
        
        Returns:
            A dictionary of computed metrics.
        """
        raise NotImplementedError

class MyEvaluator(Evaluator):
    def __init__(self, metrics: Dict[str, Dict]):
        super().__init__(metrics)

    def process(self, data_batch: Any, data_samples: Sequence[Dict[str, Any]]) -> None:
        """Process one batch of data samples and predictions.
        
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples: A batch of outputs from the model.
        """
        for metric in self.metrics:
            metric.process(data_batch, data_samples)

    def evaluate(self, size: int) -> Dict[str, float]:
        """Evaluate all metrics on the whole dataset.
        
        Args:
            size: Length of the entire dataset.
        
        Returns:
            A dictionary of all evaluation metrics.
        """
        results = {}
        for metric in self.metrics:
            metric_results = metric.evaluate(size)
            results.update({f"{metric.default_prefix}/{k}": v for k, v in metric_results.items()})
        return results

# Example usage:
@METRICS.register_module()
class AccuracyMetric(MyBaseMetric):
    default_prefix = 'ACC'

    def process(self, data_batch: Any, data_samples: Sequence[Dict[str, Any]]) -> None:
        for gt, pred in zip(data_batch['gt_label'], data_samples):
            self.results.append({'gt': gt, 'pred': pred['pred_label']})

    def compute_metrics(self, results: List) -> Dict[str, float]:
        correct = sum(1 for item in results if item['gt'] == item['pred'])
        total = len(results)
        return {'accuracy': correct / total if total > 0 else 0.0}

"""
# Usage example:
accuracy_metric = dict(type='AccuracyMetric')
evaluator = MyEvaluator(metrics={'accuracy': accuracy_metric})

# In your training/evaluation loop:
for data_batch in dataloader:
    predictions = model(data_batch)
    evaluator.process(data_batch, predictions)

# After processing all batches:
final_metrics = evaluator.evaluate(len(dataset))
print(final_metrics)  # {'accuracy/accuracy': 0.85}
"""