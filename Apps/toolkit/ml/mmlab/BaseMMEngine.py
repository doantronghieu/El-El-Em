# My code starts from here
import torch
from loguru import logger
import hydra

class BaseMMEngine():
  def __init__(self) -> None:
    """
    https://mmengine.readthedocs.io/en/latest/
    """