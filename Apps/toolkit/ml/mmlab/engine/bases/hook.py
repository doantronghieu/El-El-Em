from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.optim as optim

from mmengine.runner import Runner
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.optim as optim

from mmengine.runner import Runner
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.structures import BaseDataElement

class BaseHook(Hook, ABC):
    """
    An abstract base class for hooks that can be used in various applications.
    
    This class defines a set of methods that can be called at different stages
    of a process (e.g., before/after run, epoch, or iteration). Subclasses should
    implement these methods as needed for specific use cases.

    Attributes:
        priority (int): The priority of the hook. Lower numbers indicate higher priority.
    """

    def __init__(self, priority: int = 50) -> None:
        super().__init__()
        self.priority = priority

    @abstractmethod
    def before_run(self, runner: Runner) -> None:
        """
        Called before the main process starts.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    @abstractmethod
    def after_run(self, runner: Runner) -> None:
        """
        Called after the main process ends.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def before_train(self, runner: Runner) -> None:
        """
        Called before the training process.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def after_train(self, runner: Runner) -> None:
        """
        Called after the training process.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def before_val(self, runner: Runner) -> None:
        """
        Called before the validation process.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def after_val(self, runner: Runner) -> None:
        """
        Called after the validation process.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def before_test(self, runner: Runner) -> None:
        """
        Called before the test process.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def after_test(self, runner: Runner) -> None:
        """
        Called after the test process.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def before_train_epoch(self, runner: Runner) -> None:
        """
        Called before each training epoch.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def after_train_epoch(self, runner: Runner) -> None:
        """
        Called after each training epoch.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def before_val_epoch(self, runner: Runner) -> None:
        """
        Called before each validation epoch.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def after_val_epoch(self, runner: Runner) -> None:
        """
        Called after each validation epoch.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def before_train_iter(self, runner: Runner, batch: BaseDataElement, data_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Called before each training iteration.

        Args:
            runner (Runner): The runner managing the process.
            batch (BaseDataElement): The input data for this iteration.
            data_info (Optional[Dict[str, Any]]): Additional information about the data.
        """
        pass

    def after_train_iter(self, runner: Runner, batch: BaseDataElement, data_info: Optional[Dict[str, Any]] = None, outputs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called after each training iteration.

        Args:
            runner (Runner): The runner managing the process.
            batch (BaseDataElement): The input data for this iteration.
            data_info (Optional[Dict[str, Any]]): Additional information about the data.
            outputs (Optional[Dict[str, Any]]): The outputs of the model for this iteration.
        """
        pass

    def before_val_iter(self, runner: Runner, batch: BaseDataElement, data_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Called before each validation iteration.

        Args:
            runner (Runner): The runner managing the process.
            batch (BaseDataElement): The input data for this iteration.
            data_info (Optional[Dict[str, Any]]): Additional information about the data.
        """
        pass

    def after_val_iter(self, runner: Runner, batch: BaseDataElement, data_info: Optional[Dict[str, Any]] = None, outputs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called after each validation iteration.

        Args:
            runner (Runner): The runner managing the process.
            batch (BaseDataElement): The input data for this iteration.
            data_info (Optional[Dict[str, Any]]): Additional information about the data.
            outputs (Optional[Dict[str, Any]]): The outputs of the model for this iteration.
        """
        pass

    def before_save_checkpoint(self, runner: Runner, checkpoint: Dict[str, Any]) -> None:
        """
        Called before saving a checkpoint.

        Args:
            runner (Runner): The runner managing the process.
            checkpoint (Dict[str, Any]): The checkpoint to be saved.
        """
        pass

    def after_load_checkpoint(self, runner: Runner, checkpoint: Dict[str, Any]) -> None:
        """
        Called after loading a checkpoint.

        Args:
            runner (Runner): The runner managing the process.
            checkpoint (Dict[str, Any]): The loaded checkpoint.
        """
        pass

    def every_n_epochs(self, runner: Runner, n: int) -> bool:
        """
        A helper method to determine if the current epoch is every n epochs.

        Args:
            runner (Runner): The runner managing the process.
            n (int): The interval of epochs.

        Returns:
            bool: True if the current epoch is every n epochs, False otherwise.
        """
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner: Runner, n: int) -> bool:
        """
        A helper method to determine if the current iteration is every n iterations.

        Args:
            runner (Runner): The runner managing the process.
            n (int): The interval of iterations.

        Returns:
            bool: True if the current iteration is every n iterations, False otherwise.
        """
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def get_triggered_stages(self) -> Dict[str, bool]:
        """
        Get a dict indicating which stages this hook should be triggered.

        Returns:
            Dict[str, bool]: A dict indicating which stages this hook should be triggered.
        """
        return {name: getattr(self, name) is not Hook.__dict__[name]
                for name in Hook.__dict__
                if not name.startswith('__') and callable(getattr(self, name))}

    @staticmethod
    def set_random_seed(seed: int) -> None:
        """
        Set random seed for reproducibility.

        Args:
            seed (int): The random seed to set.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def update_params(self, runner: Runner) -> None:
        """
        Update parameters of the model or optimizer.

        Args:
            runner (Runner): The runner managing the process.
        """
        pass

    def log(self, runner: Runner, log_dict: Dict[str, Any]) -> None:
        """
        Log information during the process.

        Args:
            runner (Runner): The runner managing the process.
            log_dict (Dict[str, Any]): The dictionary containing log information.
        """
        runner.logger.info(log_dict)
