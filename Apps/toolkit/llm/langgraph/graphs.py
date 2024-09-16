from abc import ABC, classmethod
from typing import Union, Literal

class BaseGraph(ABC):
  def __init__(
    self,
    agents: list,
  ) -> None:
    super().__init__()
  
  
  def create_graph(self):
    pass