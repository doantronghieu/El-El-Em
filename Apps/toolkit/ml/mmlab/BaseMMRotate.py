from tqdm import tqdm
import os
from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Literal, Optional, Union, List, Tuple
from loguru import logger
import torch

class BaseMMRotate(ABC):
  def __init__(self) -> None:
    """
    More information at: https://mmrotate.readthedocs.io/en/latest/
    """
    super().__init__()