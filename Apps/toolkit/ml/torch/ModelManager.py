import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchmetrics import MetricCollection
from pathlib import Path
from typing import Tuple, List, Dict, Union, Callable, Optional, Any, TypeVar, Type
from loguru import logger
from tqdm import tqdm
import torch.cuda.amp as amp
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.utils.prune as prune
import torch.multiprocessing as mp
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch.profiler as profiler
import torch.quantization
import torch.onnx
from dataclasses import dataclass, field
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import random
import os
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from torchinfo import summary
from torch.cuda.amp import autocast, GradScaler
import tritonclient.http as triton_http
import asyncio
from contextlib import contextmanager
from hydra.core.config_store import ConfigStore
from hydra import compose, initialize
from hydra.utils import instantiate
from abc import ABC, abstractmethod

T = TypeVar('T')

# Use Hydra for configuration management
@dataclass
class ModelConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    seed: int = 42
    clip_grad_norm: Optional[float] = None
    use_amp: bool = False
    gradient_accumulation_steps: int = 1
    output_dim: int = 10
    metrics: List[str] = field(default_factory=lambda: ['accuracy'])
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    distributed: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    checkpoint_dir: str = './checkpoints'
    checkpoint_frequency: int = 1
    max_checkpoints: int = 5
    use_tensorboard: bool = True
    tensorboard_log_dir: str = './tensorboard_logs'
    use_wandb: bool = False
    wandb_project: str = 'default_project'
    wandb_entity: str = 'default_entity'
    lr_scheduler: str = 'cosine_warm_restarts'
    warmup_steps: int = 1000
    max_lr: float = 0.01
    weight_decay: float = 0.01
    loss_fn: nn.Module = field(default_factory=lambda: nn.CrossEntropyLoss())

cs = ConfigStore.instance()
cs.store(name="model_config", node=ModelConfig)

# Custom exceptions
class ModelError(Exception):
    pass

class DataError(Exception):
    pass

class TrainingError(Exception):
    pass

class BaseDataset(Dataset, ABC):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

# Implement Registry pattern for callbacks and metrics
class Registry:
    def __init__(self):
        self._items = {}

    def register(self, name: str):
        def decorator(cls):
            self._items[name] = cls
            return cls
        return decorator

    def get(self, name: str):
        return self._items.get(name)

callback_registry = Registry()
metric_registry = Registry()

class BaseModel(nn.Module, ABC):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._set_random_seeds()

    def _set_random_seeds(self) -> None:
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.config.loss_fn(logits, y)

    def save_model(self, path: Union[str, Path]) -> None:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': OmegaConf.to_container(self.config)
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise ModelError(f"Failed to save model: {e}")

    def load_model(self, path: Union[str, Path]) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.config.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.config = OmegaConf.create(checkpoint['config'])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelError(f"Failed to load model: {e}")

    @staticmethod
    def compute_predictions(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)

# Implement Factory pattern for model creation
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, config: ModelConfig) -> BaseModel:
        if model_type == "simple_cnn":
            return SimpleCNN(config)
        elif model_type == "resnet":
            return ResNet(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Example model implementations
class SimpleCNN(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNet(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Callback(ABC):
    @abstractmethod
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        pass

    @abstractmethod
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        pass

    @abstractmethod
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        pass

    @abstractmethod
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        pass

@callback_registry.register("early_stopping")
class EarlyStopping(Callback):
    def __init__(self, patience: int = 10, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        score = logs.get('val_loss') if logs else None
        if score is None:
            return
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    # Implement other abstract methods
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        pass

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        pass

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

    def save_checkpoint(self, checkpoint: Dict, filename: str) -> None:
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
        self._manage_checkpoints()

    def _manage_checkpoints(self) -> None:
        checkpoints = sorted(self.checkpoint_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
        for checkpoint in checkpoints[self.max_checkpoints:]:
            checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint}")

    def load_latest_checkpoint(self) -> Optional[Dict]:
        checkpoints = sorted(self.checkpoint_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
        if checkpoints:
            latest_checkpoint = checkpoints[0]
            return torch.load(latest_checkpoint)
        return None

class ModelMetricsTracker:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.metric_collection = self._initialize_metrics()

    def _initialize_metrics(self) -> MetricCollection:
        metrics = {metric: metric_registry.get(metric)().to(self.config.device) for metric in self.config.metrics}
        return MetricCollection(metrics)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        self.metric_collection.update(predictions, targets)

    def compute(self) -> Dict[str, float]:
        return {k: v.item() for k, v in self.metric_collection.compute().items()}

    def reset(self) -> None:
        self.metric_collection.reset()

    def add_metric(self, name: str, metric: torch.nn.Module) -> None:
        self.metric_collection.add_metrics({name: metric.to(self.config.device)})

class ModelExperimentTracker:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.run: Optional[Any] = None
        if self.config.use_wandb:
            self.run = wandb.init(project=self.config.wandb_project, entity=self.config.wandb_entity)
            wandb.config.update(OmegaConf.to_container(self.config))
        if self.config.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.config.tensorboard_log_dir)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.config.use_wandb and self.run:
            self.run.log(metrics, step=step)
        if self.config.use_tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def finish(self) -> None:
        if self.config.use_wandb and self.run:
            self.run.finish()
        if self.config.use_tensorboard:
            self.writer.close()

class ModelTrainer:
    def __init__(self, model: BaseModel, optimizer: torch.optim.Optimizer, 
                 train_dataset: BaseDataset, val_dataset: BaseDataset, test_dataset: BaseDataset, config: ModelConfig):
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.history: Dict[str, List[float]] = self._initialize_history()
        self.metrics_tracker = ModelMetricsTracker(config)
        self.experiment_tracker = ModelExperimentTracker(config)
        self.callbacks: List[Callback] = []
        self.scaler = GradScaler(enabled=config.use_amp)
        self.lr_scheduler = self._initialize_lr_scheduler()
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_checkpoints=config.max_checkpoints
        )

    def _initialize_history(self) -> Dict[str, List[float]]:
        return {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            **{f'train_{metric}': [] for metric in self.config.metrics},
            **{f'val_{metric}': [] for metric in self.config.metrics},
            **{f'test_{metric}': [] for metric in self.config.metrics}
        }

    def _initialize_lr_scheduler(self):
        if self.config.lr_scheduler == 'cosine_warm_restarts':
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        elif self.config.lr_scheduler == 'one_cycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.max_lr,
                steps_per_epoch=len(self.train_dataset) // self.config.batch_size,
                epochs=self.config.epochs
            )
        else:
            logger.warning(f"Unknown lr_scheduler: {self.config.lr_scheduler}. No scheduler will be used.")
            return None
    
    @torch.jit.script
    def _train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(x_batch)
        loss = self.model.compute_loss(logits, y_batch)
        return loss, logits

    def _train_epoch(self, epoch: int, total_epochs: int, train_loader: DataLoader) -> None:
        self.model.train()
        train_losses = []
        self.metrics_tracker.reset()

        for i, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")):
            x_batch = x_batch.to(self.config.device, non_blocking=True)
            y_batch = y_batch.to(self.config.device, non_blocking=True)
            
            for callback in self.callbacks:
                callback.on_batch_begin(i)
            
            with autocast(enabled=self.config.use_amp):
                loss, logits = self._train_step(x_batch, y_batch)

            self.scaler.scale(loss).backward()

            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.lr_scheduler:
                    self.lr_scheduler.step()

            train_losses.append(loss.item())
            
            predictions = self.model.compute_predictions(logits)
            self.metrics_tracker.update(predictions, y_batch)

            for callback in self.callbacks:
                callback.on_batch_end(i, logs={'loss': loss.item()})

        train_metric_values = self.metrics_tracker.compute()
        self.history['train_loss'].append(sum(train_losses) / len(train_losses))
        for metric, value in train_metric_values.items():
            self.history[f'train_{metric}'].append(value)

        self.experiment_tracker.log({
            'train_loss': sum(train_losses) / len(train_losses),
            **{f'train_{k}': v for k, v in train_metric_values.items()},
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }, step=epoch)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, prefix: str = 'val') -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        self.metrics_tracker.reset()

        for x, y in tqdm(data_loader, desc=f"Evaluating {prefix}"):
            try:
                x = x.to(self.config.device, non_blocking=True)
                y = y.to(self.config.device, non_blocking=True)
                logits = self.model(x)
                loss = self.model.compute_loss(logits, y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                
                predictions = self.model.compute_predictions(logits)
                self.metrics_tracker.update(predictions, y)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("Out of memory error during evaluation. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                raise

        avg_loss = total_loss / total_samples
        avg_metric_values = self.metrics_tracker.compute()
        
        return avg_loss, avg_metric_values

    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.history['val_loss'][-1],
            'history': self.history,
            'config': OmegaConf.to_container(self.config),
        }
        if self.lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        self.checkpoint_manager.save_checkpoint(checkpoint, f"{self.model.__class__.__name__}_epoch_{epoch}.pth")

    def train(self, epochs: int) -> None:
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory,
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory,
        )
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory,
        )

        for callback in self.callbacks:
            callback.on_train_begin()

        try:
            for epoch in range(epochs):
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch)

                self._train_epoch(epoch, epochs, train_loader)
                val_loss, val_metrics = self.evaluate(val_loader, prefix='val')
                
                self.history['val_loss'].append(val_loss)
                for metric, value in val_metrics.items():
                    self.history[f'val_{metric}'].append(value)

                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(epoch)

                self.experiment_tracker.log({
                    'val_loss': val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }, step=epoch)

                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, logs={'val_loss': val_loss})

                if any(isinstance(callback, EarlyStopping) and callback.early_stop for callback in self.callbacks):
                    logger.info("Early stopping triggered")
                    break

            # Final evaluation on test set
            test_loss, test_metrics = self.evaluate(test_loader, prefix='test')
            self.history['test_loss'].append(test_loss)
            for metric, value in test_metrics.items():
                self.history[f'test_{metric}'].append(value)

            self.experiment_tracker.log({
                'test_loss': test_loss,
                **{f'test_{k}': v for k, v in test_metrics.items()}
            })

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise TrainingError(f"Training failed: {e}")

        finally:
            self.experiment_tracker.finish()

            for callback in self.callbacks:
                callback.on_train_end()

    def add_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callback) -> None:
        self.callbacks.remove(callback)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

class ModelAnalyzer:
    def __init__(self, model: BaseModel, train_dataset: BaseDataset, val_dataset: BaseDataset, test_dataset: BaseDataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    @torch.no_grad()
    def get_layer_output(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        try:
            for name, layer in self.model.named_modules():
                x = layer(x)
                if name == layer_name:
                    return x
            raise ValueError(f"Layer {layer_name} not found in the model.")
        except Exception as e:
            raise ModelError(f"Error getting layer output: {e}")

    def profile_model(self, num_steps: int = 100):
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.model.config.batch_size, 
            shuffle=True, 
            num_workers=self.model.config.num_workers, 
            pin_memory=self.model.config.pin_memory,
        )

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config.learning_rate)

        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            schedule=profiler.schedule(wait=1, warmup=1, active=num_steps),
            on_trace_ready=profiler.tensorboard_trace_handler(self.model.config.tensorboard_log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for step, (data, target) in enumerate(train_loader):
                if step >= num_steps:
                    break
                data, target = data.to(self.model.config.device), target.to(self.model.config.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.model.compute_loss(output, target)
                loss.backward()
                optimizer.step()
                
                prof.step()

        logger.info(f"Profiling results saved to {self.model.config.tensorboard_log_dir}")
        logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    def interpret_prediction(self, input_tensor: torch.Tensor, target_class: int):
        try:
            from captum.attr import IntegratedGradients
        except ImportError:
            raise ImportError("captum is not installed. Please install it to use this feature.")

        self.model.eval()
        ig = IntegratedGradients(self.model)
        
        input_tensor = input_tensor.unsqueeze(0).to(self.model.config.device)
        baseline = torch.zeros_like(input_tensor)
        
        attributions, approximation_error = ig.attribute(input_tensor, baseline, target=target_class, return_convergence_delta=True)
        
        return attributions.squeeze().cpu().detach().numpy(), approximation_error

    def print_model_summary(self, input_size: Tuple[int, ...]):
        """
        Print a summary of the model using torchinfo.
        
        Args:
            input_size (Tuple[int, ...]): The input size for the model, e.g., (1, 3, 224, 224)
        """
        model_summary = summary(self.model, input_size=input_size, device=self.model.config.device)
        logger.info(f"Model Summary:\n{model_summary}")

class HyperparameterTuner:
    def __init__(self, model_class: Type[BaseModel], dataset: BaseDataset, config: ModelConfig):
        self.model_class = model_class
        self.dataset = dataset
        self.base_config = config

    def tune_hyperparameters(self, param_space: Dict, num_samples: int = 10, max_num_epochs: int = 10, gpus_per_trial: float = 0.5) -> Dict:
        def train_model(config):
            model = self.model_class(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            trainer = ModelTrainer(model, optimizer, self.dataset, self.dataset, self.dataset, config)

            for epoch in range(max_num_epochs):
                trainer._train_epoch(epoch, max_num_epochs, DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True))
                val_loss, _ = trainer.evaluate(DataLoader(self.dataset, batch_size=config.batch_size))
                
                tune.report(loss=val_loss)

        config = {
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([32, 64, 128, 256]),
        }
        config.update(param_space)

        search_alg = tune.suggest.optuna.OptunaSearch()
        
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2
        )

        result = tune.run(
            train_model,
            resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            progress_reporter=CLIReporter(
                parameter_columns=["learning_rate", "batch_size"],
                metric_columns=["loss", "training_iteration"]
            )
        )

        best_trial = result.get_best_trial("loss", "min", "last")
        logger.info(f"Best trial config: {best_trial.config}")
        logger.info(f"Best trial final validation loss: {best_trial.last_result['loss']}")

        return best_trial.config

class DistributedTrainer:
    @staticmethod
    def setup(rank: int, world_size: int):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    @staticmethod
    def train(rank: int, world_size: int, model: BaseModel, dataset: BaseDataset, config: ModelConfig, num_epochs: int):
        try:
            DistributedTrainer.setup(rank, world_size)

            model = model.to(rank)
            model = DistributedDataParallel(model, device_ids=[rank])

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank
            )

            train_loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=train_sampler,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            scaler = GradScaler(enabled=config.use_amp)

            for epoch in range(num_epochs):
                train_sampler.set_epoch(epoch)
                model.train()
                
                total_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(rank), target.to(rank)
                    
                    optimizer.zero_grad()
                    
                    with autocast(enabled=config.use_amp):
                        output = model(data)
                        loss = model.module.compute_loss(output, target)
                    
                    scaler.scale(loss).backward()
                    
                    if config.clip_grad_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += loss.item()
                    
                    if batch_idx % 100 == 0 and rank == 0:
                        logger.info(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
                if rank == 0:
                    logger.info(f'Epoch: {epoch}\tAverage Loss: {total_loss / len(train_loader):.6f}')
        
        except Exception as e:
            logger.error(f"Error in distributed training: {e}")
            raise ModelError(f"Error in distributed training: {e}")
        finally:
            DistributedTrainer.cleanup()

    @staticmethod
    def run_distributed(model_fn: Callable[[ModelConfig], BaseModel], dataset: BaseDataset, config: ModelConfig, num_epochs: int):
        world_size = torch.cuda.device_count()
        if world_size > 1:
            mp.spawn(
                DistributedTrainer.train,
                args=(world_size, model_fn, dataset, config, num_epochs),
                nprocs=world_size,
                join=True
            )
        else:
            logger.warning("No multiple GPUs available. Running on a single GPU.")
            model = model_fn(config)
            DistributedTrainer.train(0, 1, model, dataset, config, num_epochs)

class ModelPostTrainingOptimizer:
    def __init__(self, model: BaseModel):
        self.model = model

    def quantize_model(self, path: Union[str, Path]) -> None:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            quantized_model = torch.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
            torch.save(quantized_model.state_dict(), path)
            logger.info(f"Model quantized and saved to {path}")
        except Exception as e:
            raise ModelError(f"Error quantizing model: {e}")

    def prune_model(self, pruning_method: str = 'l1_unstructured', amount: float = 0.2) -> None:
        try:
            if pruning_method == 'l1_unstructured':
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear):
                        prune.l1_unstructured(module, name='weight', amount=amount)
            elif pruning_method == 'random_unstructured':
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear):
                        prune.random_unstructured(module, name='weight', amount=amount)
            else:
                raise ValueError(f"Unsupported pruning method: {pruning_method}")
            logger.info(f"Model pruned using {pruning_method} method with amount {amount}")
        except Exception as e:
            raise ModelError(f"Error pruning model: {e}")

    def convert_to_torchscript(self, path: Union[str, Path]) -> None:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            script_model = torch.jit.script(self.model)
            script_model.save(str(path))
            logger.info(f"Model converted to TorchScript and saved to {path}")
        except Exception as e:
            raise ModelError(f"Error converting model to TorchScript: {e}")

    def export_to_onnx(self, path: Union[str, Path], input_sample: torch.Tensor) -> None:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(self.model, input_sample, path, export_params=True, opset_version=11, 
                              do_constant_folding=True, input_names=['input'], output_names=['output'], 
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
            logger.info(f"Model exported to ONNX and saved to {path}")
        except Exception as e:
            raise ModelError(f"Error exporting model to ONNX: {e}")

class ModelServer:
    def __init__(self, model: BaseModel, config: ModelConfig):
        self.model = model
        self.config = config
        self.triton_client: Optional[triton_http.InferenceServerClient] = None

    def _prepare_model_repository(self, repository_path: str):
        os.makedirs(repository_path, exist_ok=True)
        model_path = os.path.join(repository_path, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        config_path = os.path.join(repository_path, "config.pbtxt")
        with open(config_path, "w") as f:
            f.write(f"""
                name: "model"
                platform: "pytorch_libtorch"
                max_batch_size: {self.config.batch_size}
                input [
                {{
                    name: "input"
                    data_type: TYPE_FP32
                    dims: [-1, 3, 224, 224]  # Adjust according to your model's input shape
                }}
                ]
                output [
                {{
                    name: "output"
                    data_type: TYPE_FP32
                    dims: [-1, {self.config.output_dim}]
                }}
                ]
            """)

    async def start_server(self, repository_path: str, url: str = "localhost:8000"):
        self._prepare_model_repository(repository_path)
        
        cmd = f"tritonserver --model-repository {repository_path} --http-port 8000"
        process = await asyncio.create_subprocess_shell(cmd)
        
        self.triton_client = triton_http.InferenceServerClient(url=url)
        
        while not self.triton_client.is_server_ready():
            await asyncio.sleep(1)
        
        logger.info("Triton Inference Server is ready.")
        return process

    async def predict(self, input_data: np.ndarray) -> np.ndarray:
        if self.triton_client is None:
            raise RuntimeError("Triton Inference Server is not started.")

        input_tensor = triton_http.InferInput("input", input_data.shape, "FP32")
        input_tensor.set_data_from_numpy(input_data)

        output = triton_http.InferRequestedOutput("output")
        
        response = await self.triton_client.infer("model", [input_tensor], outputs=[output])
        return response.as_numpy("output")

    async def stop_server(self, process):
        if self.triton_client:
            self.triton_client = None
            process.terminate()
            await process.wait()
            logger.info("Triton Inference Server stopped.")

    @contextmanager
    def model_server_context(self, repository_path: str):
        loop = asyncio.get_event_loop()
        process = loop.run_until_complete(self.start_server(repository_path))
        try:
            yield self
        finally:
            loop.run_until_complete(self.stop_server(process))

