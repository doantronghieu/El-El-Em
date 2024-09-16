import add_packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.registry import MODELS, METRICS, DATASETS
import yaml
from toolkit.mmlab.engine.bases.runner import RunnerManager

# Define a simple model
@MODELS.register_module()
class SimpleModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x, labels, mode):
        x = self.linear(x)
        if mode == 'loss':
            return {'loss': nn.MSELoss()(x, labels)}
        elif mode == 'predict':
            return x, labels

# Define a simple dataset
@DATASETS.register_module()
class SimpleDataset(Dataset):
    def __init__(self, num_samples=100):
        self.data = torch.randn(num_samples, 10)
        self.labels = torch.randn(num_samples, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define a simple metric
@METRICS.register_module()
class SimpleMSE(BaseMetric):
    def __init__(self, collect_device='cpu', prefix=''):
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch, data_samples):
        pred, labels = data_samples
        self.results.append({
            'batch_size': len(labels),
            'mse': nn.MSELoss()(pred, labels).item(),
        })

    def compute_metrics(self, results):
        total_mse = sum(item['mse'] for item in results)
        total_samples = sum(item['batch_size'] for item in results)
        return dict(mse=total_mse / total_samples)

# Create RunnerManager instance
runner_manager = RunnerManager('example_config.py')

# Run the entire pipeline
runner_manager.run()