# My code starts from here
import os
import time
from packaging import version as LooseVersion
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Tuple, Optional, Dict, Any, Callable
from torch.distributed.algorithms.join import Join
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import enable_wrap, wrap, size_based_auto_wrap_policy
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, MixedPrecision, BackwardPrefetch, 
    ShardingStrategy, FullStateDictConfig, StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy, enable_wrap, wrap,
)
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed.tensor.parallel as tp
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel,
)
from torch.distributed.optim import ZeroRedundancyOptimizer


@dataclass
class TrainingConfig:
    # General settings
    world_size: int = 1
    rank: int = 0
    backend: str = 'nccl'
    parallel_strategy: str = 'data_parallel'
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model settings
    split_size: int = 20
    enable_checkpointing: bool = True

    # Optimization settings
    enable_amp: bool = False
    amp_dtype: torch.dtype = torch.float16
    enable_gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    enable_gradient_accumulation: bool = False
    gradient_accumulation_steps: int = 1
    enable_gradient_penalty: bool = False
    gradient_penalty_lambda: float = 10.0

    # DDP settings
    enable_ddp: bool = False
    enable_join: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False
    delay_all_reduce_named_params: Optional[List[Tuple[str, nn.Parameter]]] = None
    param_to_hook_all_reduce: Optional[nn.Parameter] = None
    enable_ddp_optimizer: bool = False

    # FSDP settings
    enable_fsdp: bool = False
    fsdp_cpu_offload: bool = False
    fsdp_auto_wrap_policy: Optional[Callable] = None
    fsdp_backward_prefetch: Optional[BackwardPrefetch] = None
    fsdp_sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    use_enhanced_fsdp: bool = False
    fsdp_mixed_precision: bool = False
    fsdp_forward_prefetch: bool = False
    fsdp_state_dict_type: str = "full"

    # Device mesh settings
    use_device_mesh: bool = False

    # Tensor parallel settings
    use_tensor_parallel: bool = False
    tp_mesh_shape: Tuple[int, ...] = (1,)

    # ZeRO settings
    enable_zero: bool = False
    zero_optimizer_class: torch.optim.Optimizer = torch.optim.Adam
    zero_overlap_comm: bool = True

    # Additional settings (you might want to add or remove based on your needs)
    seed: int = 42
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 1000
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    def __post_init__(self):
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

class ParallelStrategy:
    DATA_PARALLEL: str = "data_parallel"
    MODEL_PARALLEL: str = "model_parallel"
    PIPELINE_PARALLEL: str = "pipeline_parallel"
    DDP: str = "ddp"
    TORCH_DATA_PARALLEL: str = "torch_data_parallel" 
    FSDP: str = "fsdp"
    HSDP: str = "hsdp"
    TP_COLWISE: str = "tp_colwise"
    TP_ROWWISE: str = "tp_rowwise"
    TP_SEQUENCE: str = "tp_sequence"
    ZERO: str = "zero"

class BaseStrategy(ABC):
    @abstractmethod
    def prepare_model(self, model: nn.Module) -> nn.Module:
        pass

    @abstractmethod
    def prepare_dataloader(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        pass

    @abstractmethod
    def backward_loss(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, step: int) -> None:
        pass

    @abstractmethod
    def should_step(self, step: int) -> bool:
        pass

    @abstractmethod
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        pass

    @abstractmethod
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                        epoch: int, loss: float, path: str) -> None:
        pass


class DistributedTrainingManager:
    def __init__(
        self,
        model: nn.Module,
        world_size: int = 1,
        rank: int = 0,
        backend: str = 'nccl',
        parallel_strategy: str = ParallelStrategy.DATA_PARALLEL,
        enable_amp=False,
        amp_dtype: torch.dtype = torch.float16,
        enable_gradient_clipping: bool = False,
        gradient_clipping_threshold: float = 1.0,
        enable_gradient_accumulation: bool = False,
        gradient_accumulation_steps: int = 1,
        enable_gradient_penalty: bool = False,
        gradient_penalty_lambda: float = 10.0,
        
        split_size: int = 20,
        enable_checkpointing: bool = True,
        enable_ddp: bool = False,
        enable_join: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
        delay_all_reduce_named_params: Optional[List[Tuple[str, nn.Parameter]]] = None,
        param_to_hook_all_reduce: Optional[nn.Parameter] = None,
        enable_ddp_optimizer: bool = False,
        enable_fsdp: bool = False,
        fsdp_cpu_offload: bool = False,
        fsdp_auto_wrap_policy: Optional[Callable] = None,
        fsdp_backward_prefetch: Optional[BackwardPrefetch] = None,
        fsdp_sharding_strategy: str = "FULL_SHARD",
        use_enhanced_fsdp: bool = False,
        fsdp_mixed_precision: bool = False,
        fsdp_forward_prefetch: bool = False,
        fsdp_state_dict_type: str = "full",
        use_device_mesh: bool = False,
        use_tensor_parallel: bool = False,
        tp_mesh_shape: Tuple[int, ...] = (1,),
        enable_zero: bool = False,  
        zero_optimizer_class: torch.optim.Optimizer = torch.optim.Adam,  
        zero_overlap_comm: bool = True,
    ) -> None:
        self.model: nn.Module = model
        self.world_size: int = world_size
        self.rank: int = rank
        self.backend: str = backend
        self.parallel_strategy: str = parallel_strategy
        self.split_size: int = split_size
        self.device: torch.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.enable_checkpointing: bool = enable_checkpointing
        
        self.enable_amp = enable_amp
        self.scaler = GradScaler(enabled=self.enable_amp)
        self.amp_dtype = amp_dtype
        self.enable_gradient_clipping = enable_gradient_clipping
        self.gradient_clipping_threshold = gradient_clipping_threshold
        self.enable_gradient_accumulation = enable_gradient_accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_gradient_penalty = enable_gradient_penalty
        self.gradient_penalty_lambda = gradient_penalty_lambda
        
        self.enable_ddp: bool = enable_ddp
        self.enable_join: bool = enable_join
        self.enable_ddp_optimizer: bool = enable_ddp_optimizer
        
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.static_graph = static_graph
        self.delay_all_reduce_named_params = delay_all_reduce_named_params
        self.param_to_hook_all_reduce = param_to_hook_all_reduce

        self.enable_fsdp = enable_fsdp
        self.fsdp_cpu_offload = fsdp_cpu_offload
        self.fsdp_auto_wrap_policy = fsdp_auto_wrap_policy
        self.fsdp_sharding_strategy = fsdp_sharding_strategy
        self.use_enhanced_fsdp = use_enhanced_fsdp
        self.fsdp_mixed_precision = fsdp_mixed_precision
        self.fsdp_forward_prefetch = fsdp_forward_prefetch
        self.fsdp_backward_prefetch = fsdp_backward_prefetch
        self.fsdp_state_dict_type = fsdp_state_dict_type
        
        self.use_device_mesh = use_device_mesh
        self.device_mesh = None
        
        self.use_tensor_parallel = use_tensor_parallel
        self.tp_mesh_shape = tp_mesh_shape
        
        self.enable_zero = enable_zero
        self.zero_optimizer_class = zero_optimizer_class
        self.zero_overlap_comm = zero_overlap_comm
        
        if self.use_device_mesh:
            self.setup_device_mesh()
        
        if self.use_tensor_parallel:
            self.setup_tensor_parallel()
        
        if self.parallel_strategy == ParallelStrategy.ZERO and self.enable_zero:
            self.setup_distributed()
            self.model = self.model.to(self.device)
            self.setup_zero_redundancy_optimizer()

        if self.parallel_strategy == ParallelStrategy.DDP and self.enable_ddp:
            self.setup_distributed()
            self.model = self.model.to(self.device)
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                gradient_as_bucket_view=self.gradient_as_bucket_view,
                static_graph=self.static_graph,
                delay_all_reduce_named_params=self.delay_all_reduce_named_params,
                param_to_hook_all_reduce=self.param_to_hook_all_reduce,
            )
            self.register_ddp_comm_hook()
            
            if self.enable_ddp_optimizer:
                self.setup_ddp_optimizer()
        
        if self.parallel_strategy == ParallelStrategy.FSDP and self.enable_fsdp:
            self.setup_distributed()
            self.model = self.model.to(self.device)
            self.model = self.wrap_fsdp_model()
        
        if self.parallel_strategy == ParallelStrategy.MODEL_PARALLEL:
            self.model = ModelParallelModule(self.model, self.world_size)
        elif self.parallel_strategy == ParallelStrategy.PIPELINE_PARALLEL:
            self.model = PipelineParallelModule(self.model, self.world_size, self.split_size)
        elif self.parallel_strategy == ParallelStrategy.DDP and self.enable_ddp:
            self.setup_distributed()
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.rank])
        elif self.parallel_strategy == ParallelStrategy.TORCH_DATA_PARALLEL:
            self.model = nn.DataParallel(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)
        
        if self.world_size > 1 and not (self.parallel_strategy == ParallelStrategy.DDP and self.enable_ddp):
            self.setup_distributed()
    
    def setup_device_mesh(self) -> None:
        if self.parallel_strategy in [ParallelStrategy.DDP, ParallelStrategy.FSDP, ParallelStrategy.HSDP]:
            self.device_mesh = init_device_mesh("cuda", (self.world_size,))
        elif self.parallel_strategy == ParallelStrategy.MODEL_PARALLEL:
            self.device_mesh = init_device_mesh("cuda", (self.world_size,))
        elif self.parallel_strategy == ParallelStrategy.PIPELINE_PARALLEL:
            self.device_mesh = init_device_mesh("cuda", (self.world_size,))

    def setup_tensor_parallel(self) -> None:
        if not self.use_device_mesh:
            raise ValueError("Device mesh must be enabled for Tensor Parallelism")
        
        tp_mesh = init_device_mesh("cuda", self.tp_mesh_shape)
        
        if self.parallel_strategy == ParallelStrategy.TP_COLWISE:
            self.model = parallelize_module(self.model, tp_mesh, ColwiseParallel())
        elif self.parallel_strategy == ParallelStrategy.TP_ROWWISE:
            self.model = parallelize_module(self.model, tp_mesh, RowwiseParallel())
        elif self.parallel_strategy == ParallelStrategy.TP_SEQUENCE:
            self.model = parallelize_module(self.model, tp_mesh, SequenceParallel())
    
    def setup_zero_redundancy_optimizer(self) -> None:
        self.optimizer = ZeroRedundancyOptimizer(
            self.model.parameters(),
            optimizer_class=self.zero_optimizer_class,
            overlap_comm=self.zero_overlap_comm,
            lr=0.01  # You might want to make this configurable
        )
    
    def setup_ddp_optimizer(self):
        # Initialize TorchDynamo DDPOptimizer
        torch._dynamo.config.optimize_ddp = True
        torch._dynamo.config.log_level = "INFO"  # Set to "DEBUG" for more detailed logs
        self.model = torch.compile(self.model)

    def setup_ddp_with_device_mesh(self) -> None:
        if not self.use_device_mesh:
            return
        
        self.model = self.model.to(self.device)
        self.model = DDP(
            self.model,
            device_ids=[self.rank],
            process_group=self.device_mesh.get_group(),
            gradient_as_bucket_view=self.gradient_as_bucket_view,
            static_graph=self.static_graph,
            delay_all_reduce_named_params=self.delay_all_reduce_named_params,
            param_to_hook_all_reduce=self.param_to_hook_all_reduce,
        )

    def setup_hsdp(self) -> None:
        if not self.use_device_mesh:
            return
        
        mesh_2d = init_device_mesh("cuda", (2, self.world_size // 2), mesh_dim_names=("replicate", "shard"))
        self.model = FSDP(
            self.model,
            device_mesh=mesh_2d,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD
        )

    def register_ddp_comm_hook(self):
        def ddp_comm_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            fut = torch.distributed.all_reduce(bucket.buffer()).get_future()
            return fut

        self.model.register_comm_hook(state=None, hook=ddp_comm_hook)

    def wrap_fsdp_model(self) -> nn.Module:
        if self.use_enhanced_fsdp:
            if self.use_device_mesh:
                fsdp_config = {
                    "cpu_offload": CPUOffload(offload_params=self.fsdp_cpu_offload),
                    "sharding_strategy": getattr(ShardingStrategy, self.fsdp_sharding_strategy),
                    "forward_prefetch": self.fsdp_forward_prefetch,
                    "backward_prefetch": self.fsdp_backward_prefetch,
                    "device_mesh": self.device_mesh,
                }
            else:
                fsdp_config = {
                    "cpu_offload": CPUOffload(offload_params=self.fsdp_cpu_offload),
                    "sharding_strategy": getattr(ShardingStrategy, self.fsdp_sharding_strategy),
                    "forward_prefetch": self.fsdp_forward_prefetch,
                    "backward_prefetch": self.fsdp_backward_prefetch,
                }
            
            if self.fsdp_mixed_precision:
                fsdp_config["mixed_precision"] = MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )
            
            if self.fsdp_auto_wrap_policy:
                auto_wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls={nn.TransformerEncoderLayer, nn.TransformerDecoderLayer})
                with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
                    wrapped_model = wrap(self.model, auto_wrap_policy=auto_wrap_policy)
            else:
                wrapped_model = FSDP(self.model, **fsdp_config)
        else:
            # Original FSDP implementation
            fsdp_config = {
                "cpu_offload": CPUOffload(offload_params=self.fsdp_cpu_offload),
            }
            
            if self.fsdp_auto_wrap_policy:
                with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
                    wrapped_model = wrap(self.model)
            else:
                wrapped_model = FSDP(self.model, **fsdp_config)
        
        return wrapped_model

    def save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer, loss: float, path: str) -> None:
        if not self.enable_checkpointing:
            return
        
        if self.rank == 0:  # Only save checkpoint on the main process
            if self.parallel_strategy == ParallelStrategy.FSDP:
                if self.use_enhanced_fsdp:
                    # Enhanced FSDP checkpointing
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                        state_dict = self.model.state_dict()
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, path)
                else:
                    # Original FSDP checkpointing
                    FSDP.save_model_checkpoint(self.model, path)
            elif self.parallel_strategy == ParallelStrategy.ZERO:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }
                torch.save(checkpoint, path)
            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'parallel_strategy': self.parallel_strategy,
                    'world_size': self.world_size,
                    'split_size': self.split_size,
                    'enable_ddp_optimizer': self.enable_ddp_optimizer,
                    'enable_fsdp': self.enable_fsdp,
                    'fsdp_cpu_offload': self.fsdp_cpu_offload,
                    'enable_amp': self.enable_amp,
                    'scaler_state_dict': self.scaler.state_dict(),
                }
                torch.save(checkpoint, path)
            print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, float]:
        if not self.enable_checkpointing:
            return 0, 0.0
        
        if self.parallel_strategy == ParallelStrategy.FSDP:
            if self.use_enhanced_fsdp:
                # Enhanced FSDP checkpoint loading
                checkpoint = torch.load(path, map_location=self.device)
                load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, load_policy):
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                if optimizer:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                return checkpoint['epoch'], checkpoint['loss']
            else:
                # Original FSDP checkpoint loading
                FSDP.load_model_checkpoint(self.model, path)
                return 0, 0.0  # Return default values for now
        elif self.parallel_strategy == ParallelStrategy.ZERO:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            checkpoint = torch.load(path, map_location=map_location)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Checkpoint loaded from {path}")
            return checkpoint['epoch'], checkpoint['loss']
        
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(path, map_location=map_location)
        
        if checkpoint['parallel_strategy'] != self.parallel_strategy:
            raise ValueError("Checkpoint parallel strategy does not match current strategy")
        
        if checkpoint['world_size'] != self.world_size:
            raise ValueError("Checkpoint world size does not match current world size")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('enable_amp', False) and self.enable_amp:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore DDPOptimizer state
        self.enable_ddp_optimizer = checkpoint.get('enable_ddp_optimizer', False)
        if self.enable_ddp_optimizer:
            self.setup_ddp_optimizer()
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['loss']

    def export_model(self, path: str) -> None:
        """Export the trained model in TorchScript format."""
        if self.rank == 0:  # Only export from the main process
            # If using DDPOptimizer, we need to handle the compiled model differently
            if self.enable_ddp_optimizer:
                # Export the original model, not the compiled one
                original_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
                model_scripted = torch.jit.script(original_model)
            else:
                model_scripted = torch.jit.script(self.model)
            model_scripted.save(path)
            print(f"Model exported to {path}")

    def setup_distributed(self) -> None:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size)

    def partition_dataset(self, dataset: Dataset, uneven: bool = False) -> DataLoader:
        if uneven:
            # Create uneven partitions for testing Join context manager
            partition_sizes: List[float] = [1.0 / self.world_size * (1 + 0.1 * i) for i in range(self.world_size)]
            partition_sizes = [size / sum(partition_sizes) for size in partition_sizes]  # Normalize
        else:
            partition_sizes: List[float] = [1.0 / self.world_size] * self.world_size
        
        partition: DataPartitioner = DataPartitioner(dataset, partition_sizes)
        partition = partition.use(self.rank)
        batch_size: int = 128 // self.world_size  # Adjust batch size based on number of processes
        return DataLoader(partition, batch_size=batch_size, shuffle=True)

    def average_gradients(self) -> None:
        size: float = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

    def train(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        if self.enable_amp:
            self.train_amp(dataset, num_epochs, optimizer, criterion, checkpoint_path)
        elif self.use_tensor_parallel:
            self.train_tensor_parallel(dataset, num_epochs, optimizer, criterion, checkpoint_path)
        elif self.parallel_strategy == ParallelStrategy.HSDP and self.use_device_mesh:
            self.train_hsdp(dataset, num_epochs, optimizer, criterion, checkpoint_path)
        elif self.parallel_strategy == ParallelStrategy.FSDP:
            self.train_fsdp(dataset, num_epochs, optimizer, criterion, checkpoint_path)
        elif self.parallel_strategy == ParallelStrategy.ZERO:
            self.train_zero(dataset, num_epochs, optimizer, criterion, checkpoint_path)
        elif self.enable_join:
            self.train_with_join(dataset, num_epochs, optimizer, criterion, checkpoint_path)
        else:
            self.train_without_join(dataset, num_epochs, optimizer, criterion, checkpoint_path)

    def train_amp(self, dataset, num_epochs, optimizer, criterion, checkpoint_path=None):
        train_set = self.partition_dataset(dataset)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss, num_samples = self._train_epoch_amp(train_set, optimizer, criterion)
            
            avg_loss = epoch_loss / num_samples
            print(f'Rank {self.rank}, epoch {epoch}: {avg_loss}')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")

    def _train_epoch_amp(self, train_set, optimizer, criterion):
        epoch_loss = 0.0
        num_samples = 0

        for i, (data, target) in enumerate(train_set):
            data, target = data.to(self.device), target.to(self.device)
            
            if self.enable_gradient_accumulation and i % self.gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.enable_amp):
                output = self.model(data)
                loss = criterion(output, target)
                
                if self.enable_gradient_penalty:
                    gradient_penalty = self.calculate_gradient_penalty(data, output)
                    loss += self.gradient_penalty_lambda * gradient_penalty
            
            self.scaler.scale(loss).backward()
            
            if self.enable_gradient_accumulation and (i + 1) % self.gradient_accumulation_steps == 0:
                if self.enable_gradient_clipping:
                    self.clip_gradients(optimizer)
                self.scaler.step(optimizer)
                self.scaler.update()
            elif not self.enable_gradient_accumulation:
                if self.enable_gradient_clipping:
                    self.clip_gradients(optimizer)
                self.scaler.step(optimizer)
                self.scaler.update()
            
            epoch_loss += loss.item()
            num_samples += len(data)

        return epoch_loss, num_samples

    def clip_gradients(self, optimizer):
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping_threshold)

    def calculate_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.enable_amp):
            disc_interpolates = self.model(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_hsdp(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        train_set: DataLoader = self.partition_dataset(dataset)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1  # Start from the next epoch
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss, num_samples = self._train_epoch_hsdp(train_set, optimizer, criterion)
            
            avg_loss = epoch_loss / num_samples
            print(f'Rank {self.rank}, epoch {epoch}: {avg_loss}')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")

    def _train_epoch_hsdp(self, train_set: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, int]:
        epoch_loss: float = 0.0
        num_samples: int = 0

        for data, target in train_set:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_samples += len(data)

        return epoch_loss, num_samples

    def _train_epoch(self, train_set: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, int]:
        epoch_loss: float = 0.0
        num_samples: int = 0

        for i, (data, target) in enumerate(train_set):
            if i % 100 == 0 and isinstance(self.model, DDP):
                # Use no_sync every 100 iterations to accumulate gradients locally
                with self.model.no_sync():
                    loss = self.train_step(data, target, optimizer, criterion)
            else:
                loss = self.train_step(data, target, optimizer, criterion)
            
            epoch_loss += loss
            num_samples += len(data)

        return epoch_loss, num_samples

    def train_without_join(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        train_set: DataLoader = self.partition_dataset(dataset)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1  # Start from the next epoch
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss, num_samples = self._train_epoch(train_set, optimizer, criterion)
            
            avg_loss = epoch_loss / num_samples
            print(f'Rank {self.rank}, epoch {epoch}: {avg_loss}')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")

    def train_with_join(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        if self.parallel_strategy not in [ParallelStrategy.DATA_PARALLEL, ParallelStrategy.DDP]:
            raise ValueError("Join context manager is only supported for data parallel and DDP strategies.")

        train_set: DataLoader = self.partition_dataset(dataset, uneven=True)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1  # Start from the next epoch
        
        for epoch in range(start_epoch, num_epochs):
            with Join([self.model]):
                epoch_loss, num_samples = self._train_epoch(train_set, optimizer, criterion)
            
            # Synchronize loss and num_samples across all ranks
            epoch_loss = torch.tensor([epoch_loss], device=self.device)
            num_samples = torch.tensor([num_samples], device=self.device)
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
            
            avg_loss = epoch_loss.item() / num_samples.item()
            print(f'Rank {self.rank}, epoch {epoch}: {avg_loss}')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")

    def train_step(self, data: torch.Tensor, target: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        data, target = data.to(self.device), target.to(self.device)
        optimizer.zero_grad()
        output: torch.Tensor = self.model(data)
        loss: torch.Tensor = criterion(output, target)
        loss.backward()
        
        if self.world_size > 1 and self.parallel_strategy == ParallelStrategy.DATA_PARALLEL and not self.enable_ddp:
            self.average_gradients()
        
        optimizer.step()
        return loss.item()

    def train_data_parallel(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1  # Start from the next epoch
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss: float = 0.0
            start_time = time.time()
            for data, target in train_loader:
                loss: float = self.train_step(data, target, optimizer, criterion)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(train_loader)
            end_time = time.time()
            print(f'Epoch {epoch}: loss={avg_loss:.4f}, time={end_time - start_time:.2f}s')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")
    
    def train_pipeline_parallel(self, data: torch.Tensor, target: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        splits: iter = iter(data.split(self.split_size, dim=0))
        split_targets: iter = iter(target.split(self.split_size, dim=0))
        
        total_loss: float = 0
        optimizer.zero_grad()
        
        for split, split_target in zip(splits, split_targets):
            split, split_target = split.to(self.device), split_target.to(self.device)
            output: torch.Tensor = self.model(split)
            loss: torch.Tensor = criterion(output, split_target)
            total_loss += loss.item()
            loss.backward()
        
        if self.world_size > 1:
            self.average_gradients()
        
        optimizer.step()
        return total_loss
    
    def check_bf16_support(self) -> bool:
        bf16_ready = (
            torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and LooseVersion(torch.version.cuda) >= "11.0"
            and dist.is_nccl_available()
            and LooseVersion(torch.distributed.nccl.version()) >= (2, 10)
        )
        return bf16_ready
    
    def train_fsdp(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        train_set: DataLoader = self.partition_dataset(dataset)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1  # Start from the next epoch
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss, num_samples = self._train_epoch_fsdp(train_set, optimizer, criterion)
            
            avg_loss = epoch_loss / num_samples
            print(f'Rank {self.rank}, epoch {epoch}: {avg_loss}')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")

    def _train_epoch_fsdp(self, train_set: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, int]:
        epoch_loss: float = 0.0
        num_samples: int = 0

        for data, target in train_set:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_samples += len(data)

        return epoch_loss, num_samples

    def train_tensor_parallel(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        train_set: DataLoader = self.partition_dataset(dataset)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss, num_samples = self._train_epoch_tensor_parallel(train_set, optimizer, criterion)
            
            avg_loss = epoch_loss / num_samples
            print(f'Rank {self.rank}, epoch {epoch}: {avg_loss}')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")

    def _train_epoch_tensor_parallel(self, train_set: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, int]:
        epoch_loss: float = 0.0
        num_samples: int = 0

        for data, target in train_set:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_samples += len(data)

        return epoch_loss, num_samples

    def train_zero(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, checkpoint_path: Optional[str] = None) -> None:
        train_set: DataLoader = self.partition_dataset(dataset)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path) and self.enable_checkpointing:
            start_epoch, _ = self.load_checkpoint(checkpoint_path, optimizer)
            start_epoch += 1
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss, num_samples = self._train_epoch_zero(train_set, optimizer, criterion)
            
            avg_loss = epoch_loss / num_samples
            print(f'Rank {self.rank}, epoch {epoch}: {avg_loss}')
            
            if self.enable_checkpointing:
                self.save_checkpoint(epoch, optimizer, avg_loss, f"{checkpoint_path}_epoch_{epoch}")

    def _train_epoch_zero(self, train_set: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, int]:
        epoch_loss: float = 0.0
        num_samples: int = 0

        for data, target in train_set:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_samples += len(data)

        return epoch_loss, num_samples
    
    @staticmethod
    def init_process(rank: int, size: int, fn: callable, backend: str = 'gloo') -> None:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)

    @classmethod
    def run_distributed(cls, world_size: int, fn: callable) -> None:
        mp.spawn(cls.init_process, args=(world_size, fn), nprocs=world_size, join=True)

    def benchmark(self, dataset: Dataset, num_epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, Any]:
        start_time = time.time()
        self.train(dataset, num_epochs, optimizer, criterion)
        end_time = time.time()
        
        total_samples = len(dataset) * num_epochs
        throughput = total_samples / (end_time - start_time)
        
        device_mesh_logging_data = {}
        if self.use_device_mesh:
            device_mesh_logging_data = {
                "mesh_shape": self.device_mesh.mesh.shape if self.device_mesh else None,
                "mesh_dim_names": self.device_mesh.mesh_dim_names if self.device_mesh else None,
            }
        
        fsdp_logging_data = {}
        if isinstance(self.model, FSDP):
            fsdp_logging_data = {
                "full_params_size": self.model.module.numel() * 4 / 1e9,  # in GB
                "sharded_params_size": sum(p.numel() for p in self.model.parameters()) * 4 / 1e9,  # in GB
                "cpu_offload": self.fsdp_cpu_offload,
                "sharding_strategy": self.fsdp_sharding_strategy,
                "backward_prefetch": self.fsdp_backward_prefetch is not None,
                "forward_prefetch": self.fsdp_forward_prefetch,
                "mixed_precision": self.fsdp_mixed_precision,
            }
        
        tp_logging_data = {}
        if self.use_tensor_parallel:
            tp_logging_data = {
                "tp_strategy": self.parallel_strategy,
                "tp_mesh_shape": self.tp_mesh_shape,
            }

        zero_logging_data = {}
        if self.parallel_strategy == ParallelStrategy.ZERO:
            zero_logging_data = {
                "optimizer_class": self.zero_optimizer_class.__name__,
                "overlap_comm": self.zero_overlap_comm,
                "sharded_optimizer_memory": sum(p.numel() for group in optimizer.param_groups for p in group['params']) * 4 / 1e9,  # in GB
            }
        
        return {
            "strategy": self.parallel_strategy,
            "throughput": throughput,
            "total_time": end_time - start_time,
            "gpu_memory_usage": torch.cuda.max_memory_allocated(self.device) / 1e9,  # in GB
            "fsdp_logging_data": fsdp_logging_data,
            "device_mesh_logging_data": device_mesh_logging_data,
            "fsdp_enabled": self.enable_fsdp,
            "enhanced_fsdp": self.use_enhanced_fsdp,
            "use_device_mesh": self.use_device_mesh,
            "tensor_parallel_data": tp_logging_data,
            "use_tensor_parallel": self.use_tensor_parallel,
            "zero_logging_data": zero_logging_data,
            "enable_zero": self.enable_zero,
            "amp_enabled": self.enable_amp,
            "amp_dtype": str(self.amp_dtype),
            "amp_scale": self.scaler.get_scale() if self.enable_amp else None,
            "gradient_clipping_enabled": self.enable_gradient_clipping,
            "gradient_clipping_threshold": self.gradient_clipping_threshold if self.enable_gradient_clipping else None,
            "gradient_accumulation_enabled": self.enable_gradient_accumulation,
            "gradient_accumulation_steps": self.gradient_accumulation_steps if self.enable_gradient_accumulation else None,
            "gradient_penalty_enabled": self.enable_gradient_penalty,
            "gradient_penalty_lambda": self.gradient_penalty_lambda if self.enable_gradient_penalty else None,
        }

class DataPartitioner:
    def __init__(self, data: Dataset, sizes: List[float]) -> None:
        self.data: Dataset = data
        self.partitions: List[List[int]] = self.partition_dataset(sizes)

    def partition_dataset(self, sizes: List[float]) -> List[List[int]]:
        data_len: int = len(self.data)
        indexes: List[int] = list(range(data_len))
        return [indexes[int(sum(sizes[:i])*data_len):int(sum(sizes[:i+1])*data_len)] for i in range(len(sizes))]

    def use(self, partition: int) -> Dataset:
        return torch.utils.data.Subset(self.data, self.partitions[partition])

class ModelParallelModule(nn.Module):
    def __init__(self, module: nn.Module, num_gpus: int):
        super().__init__()
        self.num_gpus: int = num_gpus
        self.layers: List[nn.Sequential] = self._split_model(module)

    def _split_model(self, module: nn.Module) -> List[nn.Sequential]:
        layers: List[nn.Module] = list(module.children())
        split_size: int = len(layers) // self.num_gpus
        return [nn.Sequential(*layers[i * split_size:(i + 1) * split_size]).to(f'cuda:{i}')
                for i in range(self.num_gpus)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x.to(f'cuda:{i}'))
        return x

class PipelineParallelModule(nn.Module):
    def __init__(self, module: nn.Module, num_gpus: int, split_size: int):
        super().__init__()
        self.num_gpus: int = num_gpus
        self.split_size: int = split_size
        self.layers: List[nn.Sequential] = self._split_model(module)

    def _split_model(self, module: nn.Module) -> List[nn.Sequential]:
        layers: List[nn.Module] = list(module.children())
        split_size: int = len(layers) // self.num_gpus
        return [nn.Sequential(*layers[i * split_size:(i + 1) * split_size]).to(f'cuda:{i}')
                for i in range(self.num_gpus)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits: iter = iter(x.split(self.split_size, dim=0))
        s_next: torch.Tensor = next(splits)
        s_prev: torch.Tensor = self.layers[0](s_next).to('cuda:1')
        ret: List[torch.Tensor] = []

        for s_next in splits:
            s_prev = self.layers[1](s_prev)
            ret.append(s_prev)
            s_prev = self.layers[0](s_next).to('cuda:1')

        s_prev = self.layers[1](s_prev)
        ret.append(s_prev)

        return torch.cat(ret)