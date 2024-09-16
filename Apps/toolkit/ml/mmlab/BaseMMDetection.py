from tqdm import tqdm
import os
from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Literal, Optional, Union, List, Tuple
from loguru import logger
import torch

class BaseMMDetection(ABC):
  def __init__(self) -> None:
    """
    More information at: https://mmdetection.readthedocs.io/en/latest/
    """
    super().__init__()