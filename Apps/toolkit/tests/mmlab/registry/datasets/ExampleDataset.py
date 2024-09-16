import add_packages

import torch
import numpy as np
from typing import List, Dict, Union, Any, Optional
from mmengine.dataset import BaseDataset 
from mmengine.structures import BaseDataElement
from mmengine.registry import DATASETS
from toolkit.ml.mmlab.engine.bases.data import DataManager

@DATASETS.register_module()
class LinearRegressionDataset(BaseDataset):
    def __init__(self, num_samples=1000, num_features=1, noise=0.1, pipeline=None, **kwargs):
        self.num_samples = num_samples
        self.num_features = num_features
        self.noise = noise
        
        # Generate synthetic data
        self.X = np.random.randn(num_samples, num_features)
        true_coefficients = np.random.randn(num_features)
        self.y = np.dot(self.X, true_coefficients) + np.random.randn(num_samples) * noise
        
        super().__init__(
            ann_file=None,
            metainfo=dict(num_features=num_features),
            data_root=None,
            pipeline=pipeline or [],
            **kwargs
        )

    def load_data_list(self):
        return [
            dict(
                features=self.X[i],
                target=self.y[i],
            )
            for i in range(self.num_samples)
        ]

    def parse_data_info(self, raw_data_info):
        return BaseDataElement(
            metainfo=self.metainfo,
            features=torch.FloatTensor(raw_data_info['features']),
            target=torch.FloatTensor([raw_data_info['target']])
        )

class LinearRegressionDataManager(DataManager):
    def collate_fn(self, batch: List[Union[BaseDataElement, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        features = []
        targets = []
        
        for item in batch:
            if isinstance(item, BaseDataElement):
                features.append(item.features)
                targets.append(item.target)
            elif isinstance(item, dict):
                features.append(torch.FloatTensor(item['features']))
                targets.append(torch.FloatTensor([item['target']]))
            else:
                raise TypeError(f"Unexpected type in batch: {type(item)}")
        
        feature_tensor = torch.stack(features)
        target_tensor = torch.cat(targets)
        
        return {
            'features': feature_tensor,
            'targets': target_tensor
        }

# Configuration for the dataset
dataset_config = {
    'type': 'LinearRegressionDataset',
    'num_samples': 1000,
    'num_features': 3,
    'noise': 0.1
}

# Configuration for the dataloader
dataloader_config = {
    'batch_size': 32,
    'num_workers': 0  # Set to 0 for debugging
}

# Create the data manager
linear_regression_dm = LinearRegressionDataManager(
    dataset_configs=[dataset_config],
    dataloader_config=dataloader_config
)

# Prepare the data
linear_regression_dm.prepare_data()

# Now you can use the data manager to get batches
for batch in linear_regression_dm:
    features: torch.Tensor = batch['features']
    targets: torch.Tensor = batch['targets']
    print(f"Features shape: {features.shape}, Targets shape: {targets.shape}")

# You can also get a single batch like this:
single_batch = linear_regression_dm.get_batch()
print(f"Single batch features shape: {single_batch['features'].shape}")

# TODO: Write Pytest