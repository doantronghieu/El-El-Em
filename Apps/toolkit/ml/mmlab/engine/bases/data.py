# ml/mmlab/engine/bases/data.py

from typing import Any, Dict, List, Union, Optional, Tuple
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from mmengine.dataset import BaseDataset, DefaultSampler
from mmengine.registry import DATASETS, TRANSFORMS
from mmengine.dataset.base_dataset import Compose
from mmengine.dataset import ConcatDataset, RepeatDataset, ClassBalancedDataset
from mmengine.structures import BaseDataElement

class DataManager(ABC):
    def __init__(
        self,
        dataset_configs: Union[Dict[str, Any], List[Dict[str, Any]]],
        dataloader_config: Dict[str, Any],
        transform_configs: Optional[List[Dict[str, Any]]] = None,
        dataset_wrapper: Optional[str] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
        lazy_init: bool = False,
        serialize_data: bool = True,
    ):
        self.dataset_configs = dataset_configs if isinstance(dataset_configs, list) else [dataset_configs]
        self.dataloader_config = dataloader_config
        self.transform_configs = transform_configs or []
        self.dataset_wrapper = dataset_wrapper
        self.sampler_config = sampler_config or {'type': 'DefaultSampler', 'shuffle': True}
        self.lazy_init = lazy_init
        self.serialize_data = serialize_data
        
        self.datasets: List[BaseDataset] = []
        self.dataloader: Optional[DataLoader] = None
        self.transforms: Optional[Compose] = None

        if not self.lazy_init:
            self.prepare_data()

    def build_transforms(self) -> Optional[Compose]:
        if not self.transform_configs:
            return None
        built_transforms = []
        for transform_config in self.transform_configs:
            transform_type = transform_config.pop('type')
            transform = TRANSFORMS.build(dict(type=transform_type, **transform_config))
            built_transforms.append(transform)
        return Compose(built_transforms)

    def build_datasets(self) -> List[BaseDataset]:
        datasets = []
        for config in self.dataset_configs:
            config_copy = config.copy()
            dataset_type = config_copy.pop('type')
            if self.transforms:
                config_copy['pipeline'] = self.transforms
            dataset = DATASETS.build(dict(type=dataset_type, serialize_data=self.serialize_data, **config_copy))
            datasets.append(dataset)
        return datasets

    def apply_dataset_wrapper(self, datasets: List[BaseDataset]) -> BaseDataset:
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        if self.dataset_wrapper:
            if self.dataset_wrapper == 'RepeatDataset':
                dataset = RepeatDataset(dataset, times=self.dataloader_config.get('times', 1))
            elif self.dataset_wrapper == 'ClassBalancedDataset':
                dataset = ClassBalancedDataset(dataset, oversample_thr=self.dataloader_config.get('oversample_thr', 1e-3))
            else:
                raise ValueError(f"Unsupported dataset wrapper: {self.dataset_wrapper}")
        
        return dataset

    def build_sampler(self, dataset: BaseDataset) -> Any:
        sampler_type = self.sampler_config.get('type', 'DefaultSampler')
        sampler_config = {k: v for k, v in self.sampler_config.items() if k != 'type'}
        if sampler_type == 'DefaultSampler':
            return DefaultSampler(dataset, **sampler_config)
        else:
            return DATASETS.build(dict(type=sampler_type, dataset=dataset, **sampler_config))

    def prepare_data(self) -> None:
        self.transforms = self.build_transforms()
        self.datasets = self.build_datasets()
        dataset = self.apply_dataset_wrapper(self.datasets)
        
        sampler = self.build_sampler(dataset)
        
        collate_fn = self.dataloader_config.get('collate_fn', self.collate_fn)
        if isinstance(collate_fn, dict):
            collate_fn = DATASETS.build(collate_fn)
        elif collate_fn is None:
            collate_fn = self.collate_fn
        
        self.dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.dataloader_config.get('batch_size', 1),
            num_workers=self.dataloader_config.get('num_workers', 0),
            collate_fn=collate_fn,
            **{k: v for k, v in self.dataloader_config.items() if k not in ['batch_size', 'num_workers', 'collate_fn']}
        )

    def get_data_element(self, idx: int) -> BaseDataElement:
        if not self.datasets:
            self.prepare_data()
        if len(self.datasets) == 1:
            return self.datasets[0][idx]
        else:
            for dataset in self.datasets:
                if idx < len(dataset):
                    return dataset[idx]
                idx -= len(dataset)
            raise IndexError("Index out of range")

    def get_batch(self) -> Any:
        if self.dataloader is None:
            self.prepare_data()
        return next(iter(self.dataloader))

    def get_subset(self, indices: Union[int, List[int]]) -> 'DataManager':
        new_manager = self.__class__(
            dataset_configs=self.dataset_configs,
            dataloader_config=self.dataloader_config,
            transform_configs=self.transform_configs,
            dataset_wrapper=self.dataset_wrapper,
            sampler_config=self.sampler_config,
            lazy_init=True,
            serialize_data=self.serialize_data
        )
        new_manager.datasets = [dataset.get_subset(indices) for dataset in self.datasets]
        new_manager.prepare_data()
        return new_manager

    @abstractmethod
    def collate_fn(self, batch: List[Union[BaseDataElement, Dict[str, Any]]]) -> Any:
        pass

    def __iter__(self):
        if self.dataloader is None:
            self.prepare_data()
        return iter(self.dataloader)

    def __len__(self):
        if self.dataloader is None:
            self.prepare_data()
        return len(self.dataloader)
