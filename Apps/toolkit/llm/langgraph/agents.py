import os, dotenv, yaml
from abc import ABC, classmethod
from typing import Union, Literal

dotenv.load_dotenv()

MODEL_PROVIDER = Literal["openai", "gemini", "anthropic"]
COLORS = Literal["red", "blue"]

class BaseAgent(ABC):
  def __init__(
    self,
    name: str = None,
    state = None,
    state_key: str = None, 
    prompt_tpl: str = None,
    llm_temperature: float = 0,
    llm_provider: MODEL_PROVIDER = "openai",
    llm_version: str = None,
    llm_endpoint: str = None,
    stop_token: str = None,
    mode_json: bool = False,
    tools: list = None,
  ) -> None:
    super().__init__()
    
    self.llm = ...

  def get_llm(self):
    ...
  
  @classmethod
  def invoke(self):
    pass
  
  @classmethod
  def update_state(self):
    pass
  
  @classmethod
  def print_response(
    self, response: str, color: str,
  ):
    pass
  
  
