from typing import Dict, Any, Optional, List, Union, Type
from pydantic import BaseModel as PBaseModel, Field
from torch.utils.data import DataLoader, Dataset
from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.runner import Runner, BaseLoop
from mmengine.hooks import Hook
from mmengine.registry import LOOPS, HOOKS, RUNNERS, MODELS, METRICS, DATASETS, PARAM_SCHEDULERS, OPTIM_WRAPPERS, VISUALIZERS
from mmengine.logging import MessageHub, MMLogger
from mmengine.config import Config

class RunnerConfig(PBaseModel):
    project_name: str
    work_dir: str
    seed: int
    model: Dict[str, Any]
    train_dataloader: Dict[str, Any]
    val_dataloader: Dict[str, Any]
    test_dataloader: Optional[Dict[str, Any]] = None
    train_cfg: Dict[str, Any]
    val_cfg: Dict[str, Any]
    test_cfg: Optional[Dict[str, Any]] = None
    val_evaluator: Dict[str, Any]
    test_evaluator: Optional[Dict[str, Any]] = None
    optim_wrapper: Dict[str, Any]
    param_schedulers: Dict[str, Any]
    hooks: List[Dict[str, Any]]
    feature_flags: Dict[str, bool] = Field(default_factory=dict)
    checkpoint: Dict[str, Any]
    visualizer: Optional[Dict[str, Any]] = Field(default_factory=dict)
    resume_from: Optional[str] = None
    load_from: Optional[str] = None
    cudnn_benchmark: bool = False
    mp_start_method: str = 'fork'
    dist_params: Dict[str, Any] = Field(default_factory=lambda: {'backend': 'nccl'})
    log: Dict[str, Any] = Field(default_factory=dict)
    log_level: str = 'INFO'
    default_scope: str = 'mmengine'
    log_processor: Dict[str, Any] = Field(default_factory=dict)
    default_hooks: Dict[str, Any] = Field(default_factory=dict)
    launcher: str = 'none'
    env_cfg: Dict[str, Any] = Field(default_factory=dict)
    resume: bool = False
    cfg: Dict[str, Any] = Field(default_factory=dict)

class RunnerManager:
    def __init__(self, config: Union[str, Dict[str, Any]]):
        self.logger = MMLogger.get_current_instance()
        self.logger.info("Initializing RunnerManager")
        
        if isinstance(config, str):
            self.config = Config.fromfile(config)
        else:
            self.config = Config(config)
        self.runner_config = RunnerConfig(**self.config)
        
        self.runner = self._build_runner()
        self.message_hub = MessageHub.get_current_instance()
        self.logger.info("RunnerManager initialized successfully")

    def _build_runner(self) -> Runner:
        self.logger.debug("Building Runner")
        try:
            return Runner(
                model=self.build_model(),
                work_dir=self.runner_config.work_dir,
                train_dataloader=self.build_dataloader(self.runner_config.train_dataloader, is_train=True),
                val_dataloader=self.build_dataloader(self.runner_config.val_dataloader, is_train=False),
                test_dataloader=self.build_dataloader(self.runner_config.test_dataloader, is_train=False) if self.runner_config.test_dataloader else None,
                train_cfg=self.runner_config.train_cfg,
                val_cfg=self.runner_config.val_cfg,
                test_cfg=self.runner_config.test_cfg,
                auto_scale_lr=None,
                optim_wrapper=self.runner_config.optim_wrapper,
                param_scheduler=self.runner_config.param_schedulers,
                val_evaluator=self.build_evaluator(self.runner_config.val_evaluator),
                test_evaluator=self.build_evaluator(self.runner_config.test_evaluator) if self.runner_config.test_evaluator else None,
                default_hooks=self.runner_config.default_hooks,
                custom_hooks=self.runner_config.hooks,
                data_preprocessor=self.runner_config.model.get('data_preprocessor'),
                load_from=self.runner_config.load_from,
                resume=self.runner_config.resume,
                launcher=self.runner_config.launcher,
                env_cfg=self.runner_config.env_cfg,
                log_processor=self.runner_config.log_processor,
                log_level=self.runner_config.log_level,
                visualizer=self.runner_config.visualizer,
                default_scope=self.runner_config.default_scope,
                randomness={'seed': self.runner_config.seed},
                experiment_name=self.runner_config.project_name,
                cfg=self.runner_config.cfg
            )
        except ValueError as ve:
            self.logger.error(f"Invalid configuration value: {str(ve)}")
            raise
        except KeyError as ke:
            self.logger.error(f"Missing required configuration key: {str(ke)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while building Runner: {str(e)}")
            raise

    def train(self) -> None:
        self.logger.info("Starting training process")
        self.runner.train()

    def validate(self) -> None:
        self.logger.info("Starting validation process")
        self.runner.val()

    def test(self) -> None:
        self.logger.info("Starting testing process")
        if self.runner_config.test_dataloader and self.runner_config.test_evaluator:
            self.runner.test()
        else:
            self.logger.warning("Test dataloader or evaluator not configured. Skipping test.")

    def configure_logging(self, custom_cfg: Optional[List[Dict[str, Any]]] = None) -> None:
        self.logger.info("Configuring logging")
        log_cfg = self.runner_config.log.copy()
        if custom_cfg:
            log_cfg.update(custom_cfg)
        self.runner.logger = MMLogger.get_instance(
            name=self.runner_config.project_name,
            log_file=f"{self.runner_config.work_dir}/{self.runner_config.project_name}.log",
            log_level=self.runner_config.log_level,
            **log_cfg
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.runner.load_checkpoint(checkpoint_path)

    def save_checkpoint(self, save_path: str) -> None:
        self.logger.info(f"Saving checkpoint to {save_path}")
        self.runner.save_checkpoint(save_path)

    def register_custom_hook(self, hook: Union[Hook, Dict[str, Any]]) -> None:
        self.logger.info(f"Registering custom hook: {hook}")
        self.runner.register_hook(hook)

    def register_custom_loop(self, loop_name: str, loop: Union[BaseLoop, Dict[str, Any]]) -> None:
        self.logger.info(f"Registering custom loop: {loop_name}")
        LOOPS.register_module(name=loop_name, module=loop)

    def update_custom_log(self, name: str, value: Any) -> None:
        self.logger.debug(f"Updating custom log: {name} = {value}")
        self.message_hub.update_scalar(f'{self.runner.mode}/{name}', value)

    @staticmethod
    def register_custom_runner(runner_name: str, runner_class: Type[Runner]) -> None:
        RUNNERS.register_module(name=runner_name, module=runner_class)

    def build_model(self) -> BaseModel:
        self.logger.debug("Building model")
        try:
            return MODELS.build(self.runner_config.model)
        except Exception as e:
            self.logger.error(f"Failed to build model: {str(e)}")
            raise

    def build_dataloader(self, dataloader_cfg: Optional[Dict[str, Any]], is_train: bool = True) -> Optional[DataLoader]:
        if dataloader_cfg is None:
            return None
        self.logger.debug(f"Building {'train' if is_train else 'val/test'} dataloader")
        try:
            return Runner.build_dataloader(
                dataloader_cfg,
                seed=self.runner_config.seed,
                diff_rank_seed=is_train
            )
        except Exception as e:
            self.logger.error(f"Failed to build dataloader: {str(e)}")
            raise

    def build_evaluator(self, evaluator_cfg: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]) -> Optional[Union[BaseMetric, List[BaseMetric]]]:
        if evaluator_cfg is None:
            return None
        self.logger.debug("Building evaluator")
        try:
            if isinstance(evaluator_cfg, dict):
                return Evaluator(METRICS.build(evaluator_cfg))
            elif isinstance(evaluator_cfg, list):
                return [Evaluator(METRICS.build(cfg)) for cfg in evaluator_cfg]
            else:
                raise TypeError(f"Unsupported evaluator config type: {type(evaluator_cfg)}")
        except Exception as e:
            self.logger.error(f"Failed to build evaluator: {str(e)}")
            raise

    def set_feature_flags(self) -> None:
        self.logger.debug("Setting feature flags")
        for flag, value in self.runner_config.feature_flags.items():
            setattr(self.runner, flag, value)

    def configure_optimizers(self) -> None:
        self.logger.debug("Configuring optimizers")
        if isinstance(self.runner_config.optim_wrapper, dict):
            self.runner.optim_wrapper = OPTIM_WRAPPERS.build(self.runner_config.optim_wrapper)
        elif isinstance(self.runner_config.optim_wrapper, list):
            self.runner.optim_wrapper = [OPTIM_WRAPPERS.build(o) for o in self.runner_config.optim_wrapper]
        else:
            raise TypeError(f"Unsupported optim_wrapper type: {type(self.runner_config.optim_wrapper)}")

    def configure_schedulers(self) -> None:
        self.logger.debug("Configuring schedulers")
        if isinstance(self.runner_config.param_schedulers, dict):
            self.runner.param_scheduler = PARAM_SCHEDULERS.build(self.runner_config.param_schedulers)
        elif isinstance(self.runner_config.param_schedulers, list):
            self.runner.param_scheduler = [PARAM_SCHEDULERS.build(s) for s in self.runner_config.param_schedulers]
        else:
            raise TypeError(f"Unsupported param_schedulers type: {type(self.runner_config.param_schedulers)}")

    def configure_hooks(self) -> None:
        self.logger.debug("Configuring hooks")
        for hook_cfg in self.runner_config.hooks:
            if isinstance(hook_cfg, dict):
                self.runner.register_hook(HOOKS.build(hook_cfg))
            elif isinstance(hook_cfg, Hook):
                self.runner.register_hook(hook_cfg)
            else:
                raise TypeError(f"Unsupported hook type: {type(hook_cfg)}")

    def configure_checkpoint(self) -> None:
        self.logger.debug("Configuring checkpoint")
        self.runner.default_hooks['checkpoint'].interval = self.runner_config.checkpoint['interval']

    def configure_distributed(self) -> None:
        self.logger.debug("Configuring distributed training")
        if self.runner_config.launcher != 'none':
            self.runner.launcher = self.runner_config.launcher
            self.runner.distributed = True

    def configure_visualizer(self) -> None:
        self.logger.debug("Configuring visualizer")
        if self.runner_config.visualizer:
            self.runner.visualizer = VISUALIZERS.build(self.runner_config.visualizer)

    def run(self, tasks: List[str] = ['train', 'val', 'test']) -> None:
        self.logger.info(f"Running tasks: {tasks}")
        self.set_feature_flags()
        self.configure_optimizers()
        self.configure_schedulers()
        self.configure_hooks()
        self.configure_checkpoint()
        self.configure_distributed()
        self.configure_visualizer()
        
        for task in tasks:
            if task == 'train':
                self.train()
            elif task == 'val':
                self.validate()
            elif task == 'test':
                self.test()
            else:
                self.logger.warning(f"Unsupported task: {task}")

    def evaluate(self, data: Any) -> Dict[str, Any]:
        self.logger.info("Evaluating model")
        return self.runner.val_evaluator.evaluate(data)

    def inference(self, data: Any) -> Any:
        self.logger.info("Performing inference")
        return self.runner.model(data, mode='predict')

    @staticmethod
    def register_custom_dataset(dataset_name: str, dataset_class: Type[Dataset]) -> None:
        DATASETS.register_module(name=dataset_name, module=dataset_class)

    def add_custom_metric(self, metric_name: str, metric_class: Type[BaseMetric]) -> None:
        self.logger.info(f"Adding custom metric: {metric_name}")
        METRICS.register_module(name=metric_name, module=metric_class)